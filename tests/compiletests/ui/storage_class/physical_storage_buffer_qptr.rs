// Same as `physical_storage_buffer.rs`, but routed through the SPIR-T `qptr`
// pipeline (opt-in via `--no-infer-storage-classes` + `--spirt-passes=qptr`).
// Verifies that the PhysicalStorageBuffer addressing-model path survives the
// qptr lower/lift roundtrip.

// build-pass
// compile-flags: -C target-feature=+PhysicalStorageBufferAddresses,+Int64,+ext:SPV_KHR_physical_storage_buffer -C llvm-args=--no-infer-storage-classes -C llvm-args=--spirt-passes=qptr

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
