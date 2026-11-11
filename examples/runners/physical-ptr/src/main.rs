//! Runs `physical-ptr-shader` with Vulkan: builds a linked list in a buffer, passes the
//! device addresses of the list and of an output buffer as push constants, and checks the
//! sum the shader writes against the CPU's.

use anyhow::{Context, Result, ensure};
use ash::util::read_spv;
use ash::{Entry, vk};
use physical_ptr_shader::{Node, Params};
use spirv_builder::{Capability, SpirvBuilder};
use spirv_std::ptr::PhysicalPtr;
use std::fs::File;
use std::path::PathBuf;

const PAYLOADS: [f32; 8] = [1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5];

/// Where the output buffer is bound in its allocation, to give it an address above 4 GiB.
const HIGH_OFFSET: u64 = 1 << 32;

fn main() -> Result<()> {
    let spv = compile_shader()?;
    println!("Built shader: {} SPIR-V words", spv.len());

    let cpu_sum: f32 = PAYLOADS.iter().sum();
    println!("CPU sum = {cpu_sum}");
    let gpu_sum = run(&spv)?;
    println!("GPU sum = {gpu_sum}");
    ensure!(gpu_sum == cpu_sum, "GPU and CPU sums differ");
    Ok(())
}

fn compile_shader() -> Result<Vec<u32>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let crate_path: PathBuf = [manifest_dir, "..", "..", "shaders", "physical-ptr-shader"]
        .iter()
        .collect();
    let result = SpirvBuilder::new(crate_path, "spirv-unknown-vulkan1.2")
        .capability(Capability::PhysicalStorageBufferAddresses)
        .capability(Capability::Int64)
        .build()?;
    Ok(read_spv(&mut File::open(result.module.unwrap_single())?)?)
}

fn run(spv: &[u32]) -> Result<f32> {
    unsafe {
        let entry = Entry::load()?;
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // Use the validation layer if it's installed.
        let (instance, validation) =
            match entry.create_instance(&instance_info.enabled_layer_names(&layers), None) {
                Ok(instance) => (instance, true),
                Err(_) => (entry.create_instance(&instance_info, None)?, false),
            };

        let (physical_device, queue_family) = instance
            .enumerate_physical_devices()?
            .into_iter()
            .find_map(|pd| {
                if instance.get_physical_device_properties(pd).api_version < vk::API_VERSION_1_2 {
                    return None;
                }
                let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
                let mut features =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut features12);
                instance.get_physical_device_features2(pd, &mut features);
                if features.features.shader_int64 == vk::FALSE
                    || features12.buffer_device_address == vk::FALSE
                    || features12.vulkan_memory_model == vk::FALSE
                {
                    return None;
                }
                let queue_family = instance
                    .get_physical_device_queue_family_properties(pd)
                    .iter()
                    .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))?;
                Some((pd, queue_family as u32))
            })
            .context(
                "no Vulkan 1.2 GPU with shaderInt64, bufferDeviceAddress and vulkanMemoryModel",
            )?;

        let mut props11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut props11);
        instance.get_physical_device_properties2(physical_device, &mut props);
        let name = props.properties.device_name_as_c_str()?.to_string_lossy();
        println!(
            "Using GPU: {name} (validation layer {})",
            if validation { "on" } else { "off" }
        );
        let max_alloc = props11.max_memory_allocation_size;
        let mem_props = instance.get_physical_device_memory_properties(physical_device);

        // `vulkanMemoryModel` because rust-gpu's Vulkan targets use it.
        let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .vulkan_memory_model(true);
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0])];
        let device = instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_features(&features)
                .push_next(&mut features12),
            None,
        )?;
        let queue = device.get_device_queue(queue_family, 0);

        // The list, with its nodes in order in one buffer.
        let nodes_buf = create_buffer(&device, (size_of::<Node>() * PAYLOADS.len()) as u64)?;
        let req = device.get_buffer_memory_requirements(nodes_buf);
        let nodes_mem = allocate(&device, &mem_props, req.memory_type_bits, req.size)?;
        device.bind_buffer_memory(nodes_buf, nodes_mem, 0)?;
        let nodes_addr = buffer_address(&device, nodes_buf);
        let head = PhysicalPtr::<Node>::from_addr(nodes_addr);
        let nodes: Vec<Node> = PAYLOADS
            .iter()
            .enumerate()
            .map(|(i, &payload)| Node {
                next: if i + 1 < PAYLOADS.len() {
                    head.add(i as u64 + 1)
                } else {
                    PhysicalPtr::null()
                },
                payload,
            })
            .collect();
        let mapped =
            device.map_memory(nodes_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?;
        mapped
            .cast::<Node>()
            .copy_from_nonoverlapping(nodes.as_ptr(), nodes.len());
        device.unmap_memory(nodes_mem);

        // The output, bound at `HIGH_OFFSET` in a larger allocation if there is room for one.
        let output_buf = create_buffer(&device, size_of::<f32>() as u64)?;
        let req = device.get_buffer_memory_requirements(output_buf);
        let high_size = HIGH_OFFSET + req.size;
        let high_mem = if high_size <= max_alloc {
            allocate(&device, &mem_props, req.memory_type_bits, high_size).ok()
        } else {
            None
        };
        let (output_mem, output_offset) = if let Some(mem) = high_mem {
            (mem, HIGH_OFFSET)
        } else {
            println!(
                "Couldn't allocate >4 GiB of host-visible memory; the output isn't above 4 GiB"
            );
            let mem = allocate(&device, &mem_props, req.memory_type_bits, req.size)?;
            (mem, 0)
        };
        device.bind_buffer_memory(output_buf, output_mem, output_offset)?;
        let output_addr = buffer_address(&device, output_buf);
        println!("Nodes at {nodes_addr:#x}, output at {output_addr:#x}");

        let params = Params {
            root: head,
            output: PhysicalPtr::from_addr(output_addr),
        };
        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<Params>() as u32)];
        let layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_range),
            None,
        )?;
        let module =
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)?;
        let pipeline = device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main"),
                    )
                    .layout(layout)],
                None,
            )
            .map_err(|(_, e)| e)?[0];

        let pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
            None,
        )?;
        let cmd = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(1),
        )?[0];
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        let push =
            std::slice::from_raw_parts((&raw const params).cast::<u8>(), size_of::<Params>());
        device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, push);
        device.cmd_dispatch(cmd, 1, 1, 1);
        // Makes the shader's write visible to the host.
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)],
            &[],
            &[],
        );
        device.end_command_buffer(cmd)?;

        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cmds = [cmd];
        device.queue_submit(
            queue,
            &[vk::SubmitInfo::default().command_buffers(&cmds)],
            fence,
        )?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;

        let mapped = device.map_memory(
            output_mem,
            output_offset,
            size_of::<f32>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        let gpu_sum = mapped.cast::<f32>().read();
        device.unmap_memory(output_mem);

        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(module, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_buffer(output_buf, None);
        device.free_memory(output_mem, None);
        device.destroy_buffer(nodes_buf, None);
        device.free_memory(nodes_mem, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
        Ok(gpu_sum)
    }
}

/// Creates a buffer used through its device address.
unsafe fn create_buffer(device: &ash::Device, size: u64) -> Result<vk::Buffer> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    Ok(unsafe { device.create_buffer(&info, None)? })
}

/// Allocates `size` bytes of host-visible, coherent memory of one of `type_bits`, for buffers
/// used through their device address.
unsafe fn allocate(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    size: u64,
) -> Result<vk::DeviceMemory> {
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let memory_type = (0..mem_props.memory_type_count)
        .find(|&i| {
            let ty = mem_props.memory_types[i as usize];
            type_bits & (1 << i) != 0
                && ty.property_flags.contains(flags)
                && mem_props.memory_heaps[ty.heap_index as usize].size >= size
        })
        .context("no host-visible memory type with a large enough heap")?;
    let mut flags_info =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(memory_type)
        .push_next(&mut flags_info);
    Ok(unsafe { device.allocate_memory(&info, None)? })
}

/// Returns the device address of `buffer`.
unsafe fn buffer_address(device: &ash::Device, buffer: vk::Buffer) -> u64 {
    unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    }
}
