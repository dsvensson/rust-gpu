//! Raw-Vulkan compute runner for the `physical-ptr-shader` linked-list-sum
//! shader. Builds the SPIR-V, creates a Vulkan 1.2 device with
//! `bufferDeviceAddress`, writes a linked list whose `next` fields hold
//! `vkGetBufferDeviceAddress` results, and pushes the head address plus an
//! output address (both physical pointers) as push constants. The output is
//! deliberately placed *above* 4 GiB so the shader's store exercises 64-bit
//! addressing; the result is read back and checked against a CPU reference.

#![feature(f16)]

use anyhow::{Context, Result, anyhow};
use ash::util::read_spv;
use ash::{Entry, vk};
use physical_ptr_shader::Node;
use spirv_builder::{Capability, SpirvBuilder};
use spirv_std::ptr::PhysicalPtr;
use std::ffi::{CStr, c_char};
use std::fs::File;
use std::path::PathBuf;

const NODE_COUNT: usize = 8;
const PAYLOADS: [f32; NODE_COUNT] = [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5];

/// Byte offset at which the 2-byte output buffer is bound inside its (larger
/// than 4 GiB) allocation, forcing its device address above 4 GiB.
const HIGH_OFFSET: u64 = 4 * 1024 * 1024 * 1024; // 0x1_0000_0000

fn main() -> Result<()> {
    let spv_words = compile_shader()?;
    println!("Built shader: {} SPIR-V words", spv_words.len());

    // Mirror the shader's `f16` accumulation (payload stored + summed as `f16`)
    // so the reference matches the GPU result bit-for-bit.
    let cpu_sum: f16 = PAYLOADS.iter().fold(0.0, |acc, &p| acc + p as f16);
    println!("CPU reference sum = {}", cpu_sum as f32);

    unsafe { dispatch(&spv_words, cpu_sum) }
}

fn compile_shader() -> Result<Vec<u32>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let crate_path: PathBuf = [manifest_dir, "..", "..", "shaders", "physical-ptr-shader"]
        .iter()
        .collect();

    let compile_result = SpirvBuilder::new(crate_path, "spirv-unknown-vulkan1.2")
        .capability(Capability::PhysicalStorageBufferAddresses)
        .capability(Capability::Int64)
        .capability(Capability::Float16)
        .extension("SPV_KHR_physical_storage_buffer")
        .build()?;
    let spv_path = compile_result.module.unwrap_single();
    Ok(read_spv(&mut File::open(spv_path)?)?)
}

unsafe fn dispatch(spv_words: &[u32], cpu_sum: f16) -> Result<()> { unsafe {
    let entry = Entry::load()?;

    // ── Instance ────────────────────────────────────────────────────────
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"physical-ptr-runner")
        .api_version(vk::make_api_version(0, 1, 2, 0));

    let layer_names: &[*const c_char] = &[c"VK_LAYER_KHRONOS_validation".as_ptr()];
    let mut create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // Validation layers are optional; try with them first, then without.
    create_info = create_info.enabled_layer_names(layer_names);
    let instance = entry.create_instance(&create_info, None).or_else(|_| {
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        entry.create_instance(&create_info, None)
    })?;

    // ── Physical device ────────────────────────────────────────────────
    let physical_devices = instance.enumerate_physical_devices()?;
    let (physical_device, queue_family) = physical_devices
        .into_iter()
        .find_map(|pd| {
            let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut bda);
            instance.get_physical_device_features2(pd, &mut features);
            if bda.buffer_device_address == 0 {
                return None;
            }
            let queue_props = instance.get_physical_device_queue_family_properties(pd);
            queue_props
                .iter()
                .enumerate()
                .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| (pd, i as u32))
        })
        .ok_or_else(|| anyhow!("no Vulkan device with bufferDeviceAddress + compute queue"))?;

    let props = instance.get_physical_device_properties(physical_device);
    let name = CStr::from_ptr(props.device_name.as_ptr())
        .to_string_lossy()
        .into_owned();
    println!("Using GPU: {name}");

    // ── Logical device ─────────────────────────────────────────────────
    let queue_create = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];

    // `shaderInt64` for `PhysicalPtr`'s u64 unpacking; `bufferDeviceAddress`
    // for `vkGetBufferDeviceAddress`; `vulkanMemoryModel` because
    // `rustc_codegen_spirv` declares it in the emitted SPIR-V; `shaderFloat16`
    // for the `f16` payload/sum.
    let core_feat = vk::PhysicalDeviceFeatures::default().shader_int64(true);
    // `storageBuffer16BitAccess` for the `f16` values in the storage/physical
    // buffers.
    let mut vk11_feat =
        vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
    let mut vk12_feat = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .vulkan_memory_model(true)
        .shader_float16(true);
    let device_extensions: &[*const c_char] = &[];

    let device_create = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create)
        .enabled_features(&core_feat)
        .enabled_extension_names(device_extensions)
        .push_next(&mut vk11_feat)
        .push_next(&mut vk12_feat);

    let device = instance.create_device(physical_device, &device_create, None)?;
    let queue = device.get_device_queue(queue_family, 0);

    let mem_props = instance.get_physical_device_memory_properties(physical_device);

    // ── Nodes buffer (SHADER_DEVICE_ADDRESS + HOST_VISIBLE) ────────────
    let node_size = size_of::<Node>() as u64;
    let nodes_size = node_size * NODE_COUNT as u64;
    let (nodes_buf, nodes_mem) = create_buffer(
        &device,
        &mem_props,
        nodes_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        true,
    )?;
    let nodes_base_addr =
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(nodes_buf));

    // Build the linked list; last node's `next` is `null()`.
    let nodes: Vec<Node> = (0..NODE_COUNT)
        .map(|i| Node {
            next: if i + 1 < NODE_COUNT {
                PhysicalPtr::<Node>::from_addr(nodes_base_addr + (i as u64 + 1) * node_size)
            } else {
                PhysicalPtr::<Node>::null()
            },
            payload: PAYLOADS[i] as f16,
        })
        .collect();
    let mapped = device.map_memory(nodes_mem, 0, nodes_size, vk::MemoryMapFlags::empty())?;
    std::ptr::copy_nonoverlapping(nodes.as_ptr(), mapped.cast(), nodes.len());
    device.unmap_memory(nodes_mem);

    // ── Output: a single `f16` placed ABOVE 4 GiB ──────────────────────
    // Reserve just over 4 GiB and bind the 2-byte output buffer at offset
    // 4 GiB; its device address is then `allocation_base + 4 GiB`, which is
    // guaranteed to have a non-zero high dword regardless of where the driver
    // placed the allocation. The shader writes the sum through this address.
    let output_buf = device.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(2)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    )?;
    let out_req = device.get_buffer_memory_requirements(output_buf);
    let out_alloc_size = HIGH_OFFSET + out_req.size;
    let out_mem_type = find_memory_type_with_heap(
        &mem_props,
        out_req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        out_alloc_size,
    )
    .ok_or_else(|| {
        anyhow!(
            "no host-visible memory type with a heap ≥ {out_alloc_size} bytes \
             (need >4 GiB — enable Resizable BAR or ensure enough system RAM)"
        )
    })?;
    let mut out_alloc_flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let output_mem = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(out_alloc_size)
                .memory_type_index(out_mem_type)
                .push_next(&mut out_alloc_flags),
            None,
        )
        .context("allocating >4 GiB for the output buffer")?;
    device
        .bind_buffer_memory(output_buf, output_mem, HIGH_OFFSET)
        .context("binding output buffer at the 4 GiB offset")?;
    let output_addr =
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(output_buf));
    println!("Nodes buffer  @ {nodes_base_addr:#018x}");
    println!(
        "Output buffer @ {output_addr:#018x}  (high 32 bits = {:#x})",
        output_addr >> 32
    );
    assert!(
        output_addr >> 32 != 0,
        "output address {output_addr:#x} is not above 4 GiB"
    );

    // ── Pipeline (push constants only: two 8-byte physical addresses) ──
    let push_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(16)];
    let pipeline_layout = device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_range),
        None,
    )?;

    let shader_module = device.create_shader_module(
        &vk::ShaderModuleCreateInfo::default().code(spv_words),
        None,
    )?;

    let pipeline = device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader_module)
                        .name(c"main"),
                )
                .layout(pipeline_layout)],
            None,
        )
        .map_err(|(_, e)| e)?[0];

    // ── Command pool / buffer ──────────────────────────────────────────
    let command_pool = device.create_command_pool(
        &vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT),
        None,
    )?;
    let cmd_buf = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1),
    )?[0];

    device.begin_command_buffer(
        cmd_buf,
        &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )?;
    device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline);
    // Push the head address (bytes 0..8) and the output address (bytes 8..16),
    // matching the shader's `Params { root, output }` push-constant layout.
    let mut push = [0u8; 16];
    push[0..8].copy_from_slice(&nodes_base_addr.to_le_bytes());
    push[8..16].copy_from_slice(&output_addr.to_le_bytes());
    device.cmd_push_constants(
        cmd_buf,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &push,
    );
    device.cmd_dispatch(cmd_buf, 1, 1, 1);
    device.end_command_buffer(cmd_buf)?;

    // ── Submit + wait ──────────────────────────────────────────────────
    let cmd_bufs = [cmd_buf];
    let submits = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
    device.queue_submit(queue, &submits, fence)?;
    device.wait_for_fences(&[fence], true, u64::MAX)?;

    // ── Read back (single `f16` at the >4 GiB offset) ──────────────────
    let mapped = device.map_memory(output_mem, HIGH_OFFSET, 2, vk::MemoryMapFlags::empty())?;
    let mut bytes = [0u8; 2];
    std::ptr::copy_nonoverlapping(mapped.cast::<u8>(), bytes.as_mut_ptr(), 2);
    device.unmap_memory(output_mem);
    let gpu_sum = f16::from_bits(u16::from_le_bytes(bytes));
    println!("GPU sum = {}", gpu_sum as f32);

    let ok = (gpu_sum as f32 - cpu_sum as f32).abs() < 1e-2;
    println!(
        "{}",
        if ok {
            "PASS: GPU matches CPU reference (written above 4 GiB)."
        } else {
            "FAIL: GPU and CPU disagree."
        }
    );

    // ── Cleanup ────────────────────────────────────────────────────────
    device.destroy_fence(fence, None);
    device.destroy_command_pool(command_pool, None);
    device.destroy_pipeline(pipeline, None);
    device.destroy_shader_module(shader_module, None);
    device.destroy_pipeline_layout(pipeline_layout, None);
    device.destroy_buffer(output_buf, None);
    device.free_memory(output_mem, None);
    device.destroy_buffer(nodes_buf, None);
    device.free_memory(nodes_mem, None);
    device.destroy_device(None);
    instance.destroy_instance(None);

    if ok { Ok(()) } else { Err(anyhow!("verification failed")) }
}}

/// Creates a buffer, allocates memory for it, and binds it at offset 0. When
/// `device_address` is set, `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` is passed
/// so the buffer can be queried via `vkGetBufferDeviceAddress`.
unsafe fn create_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    required_flags: vk::MemoryPropertyFlags,
    device_address: bool,
) -> Result<(vk::Buffer, vk::DeviceMemory)> { unsafe {
    let buf = device.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    )?;
    let mem_req = device.get_buffer_memory_requirements(buf);
    let mem_type = find_memory_type_with_heap(mem_props, mem_req.memory_type_bits, required_flags, 0)
        .ok_or_else(|| anyhow!("no memory type with required flags"))?;

    let mut alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req.size)
        .memory_type_index(mem_type);
    let mut alloc_flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    if device_address {
        alloc_info = alloc_info.push_next(&mut alloc_flags);
    }
    let mem = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(buf, mem, 0).context("bind_buffer_memory")?;
    Ok((buf, mem))
}}

/// Finds a memory type satisfying `type_bits` and `required` flags whose heap
/// is at least `min_heap_size` bytes (pass 0 to ignore heap size).
fn find_memory_type_with_heap(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
    min_heap_size: u64,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| {
        let ty = mem_props.memory_types[i as usize];
        type_bits & (1 << i) != 0
            && ty.property_flags.contains(required)
            && mem_props.memory_heaps[ty.heap_index as usize].size >= min_heap_size
    })
}
