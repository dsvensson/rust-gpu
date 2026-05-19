// `PhysicalPtr` is assumed valid at codegen time (the app enables the
// capability), so without `PhysicalStorageBufferAddresses` declared the
// emitted SPIR-V is rejected by `spirv-val`. This test locks in those
// validator diagnostics (`OpConvertUToPtr` / `StorageClass.PhysicalStorageBuffer`).

// build-fail
// normalize-stderr-test "\\\\" -> "/"

use spirv_std::ptr::PhysicalPtr;
use spirv_std::spirv;

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(push_constant)] head_addr: &u64,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] out: &mut u32,
) {
    let ptr: PhysicalPtr<u32> = PhysicalPtr::from_addr(*head_addr);
    unsafe {
        *out = *ptr.get();
    }
}
