// Cover `RestrictedPhysicalPtr<T>::get()`: emits `OpDecorate %ptr Restrict`
// against the OpBitcast result. Driver can then assume non-aliasing.

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
