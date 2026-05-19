// Compiles a fragment shader that round-trips a value through a
// `PhysicalPtr<u32>`. Exercises the end-to-end PhysicalStorageBuffer
// addressing-model path: capability + extension enabled, pointer cast through
// `inttoptr`, `OpTypePointer PhysicalStorageBuffer u32`, load/store with
// `MemoryAccess::Aligned`.

// build-pass
// compile-flags: -C target-feature=+PhysicalStorageBufferAddresses,+Int64,+ext:SPV_KHR_physical_storage_buffer

use spirv_std::ptr::PhysicalPtr;
use spirv_std::spirv;

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] addr: &u64,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out: &mut u32,
) {
    let ptr: PhysicalPtr<u32> = PhysicalPtr::from_addr(*addr);
    unsafe {
        *out = *ptr.get();
    }
}
