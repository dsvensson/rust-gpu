// Without the `PhysicalStorageBufferAddresses` capability, calls into the
// `PhysicalPtr` API must zombie with a clear diagnostic, not silently
// produce invalid SPIR-V. This test locks in the wording emitted by
// `builder_methods.rs::zombie_convert_{u_to_ptr,ptr_to_u}`.

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
