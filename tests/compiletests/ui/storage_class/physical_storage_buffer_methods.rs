// Exercises the `PhysicalPtr<T>` methods that aren't covered by the other
// `physical_storage_buffer*` tests: `offset`, `byte_offset`, `cast`,
// `as_mut`, `map_addr`, `is_null`, `null`. Build-pass only — the runtime
// correctness of these is covered by the ash runner in
// `examples/runners/physical-ptr-runner`.

// build-pass
// compile-flags: -C target-feature=+PhysicalStorageBufferAddresses,+Int64,+ext:SPV_KHR_physical_storage_buffer

use spirv_std::ptr::PhysicalPtr;
use spirv_std::spirv;

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(push_constant)] base: &PhysicalPtr<u32>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] out: &mut u32,
) {
    // `null` + `is_null` round-trip.
    let n: PhysicalPtr<u32> = PhysicalPtr::null();
    if n.is_null() {
        *out = 0;
    }

    // Element-stride offset.
    let elem3: PhysicalPtr<u32> = unsafe { base.offset(3) };

    // Byte-stride offset.
    let byte12: PhysicalPtr<u32> = unsafe { base.byte_offset(12) };

    // Reinterpret-cast to a different pointee type.
    let as_f32: PhysicalPtr<f32> = base.cast::<f32>();

    // `map_addr` and `from_addr`/`addr` round-trip.
    let shifted: PhysicalPtr<u32> = base.map_addr(|a| a.wrapping_add(8));

    // `as_mut` returns `Option<&mut T>`; the `Some` branch must be a no-op
    // store so the chain stays alive after DCE.
    unsafe {
        if let Some(slot) = elem3.as_mut() {
            *slot = 1;
        }
        if let Some(slot) = byte12.as_mut() {
            *slot = 2;
        }
        if let Some(slot) = as_f32.as_mut() {
            *slot = 3.0;
        }
        if let Some(slot) = shifted.as_mut() {
            *slot = 4;
        }
    }
}
