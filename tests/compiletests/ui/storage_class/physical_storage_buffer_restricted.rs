// Cover `RestrictedPhysicalPtr<T>::get()` as a typed-marker wrapper around
// `PhysicalPtr<T>`. The SPIR-V `RestrictPointer` decoration is not yet
// emitted (the spec only allows it on memory-object declarations, and the
// runtime-constructed pointer that backs the API doesn't survive
// optimization as a stable variable); this test just makes sure the
// wrapper API compiles end-to-end through the qptr-free pipeline.

// build-pass
// compile-flags: -C target-feature=+PhysicalStorageBufferAddresses,+Int64,+ext:SPV_KHR_physical_storage_buffer

use spirv_std::ptr::{PhysicalPtr, RestrictedPhysicalPtr};
use spirv_std::spirv;

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] addr_in: &u64,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] addr_out: &u64,
) {
    let src = unsafe { RestrictedPhysicalPtr::new(PhysicalPtr::<u32>::from_addr(*addr_in)) };
    let dst = unsafe { RestrictedPhysicalPtr::new(PhysicalPtr::<u32>::from_addr(*addr_out)) };
    unsafe {
        *dst.get() = *src.get();
    }
}
