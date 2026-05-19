// Test the new `Image::texel_pointer` API: it emits `OpImageTexelPointer`,
// which produces an `Image`-storage-class pointer. The pointer is then
// passed to `atomic_i_add`, whose `&u32` parameter compiles to a generic
// pointer that must be specialized to the `Image` storage class via the
// specializer's `Instance` vs `Concrete` unification (see the
// "Specializer: propagate concrete storage class into generic pointer
// instances" commit).

// build-pass

use spirv_std::arch::atomic_i_add;
use spirv_std::image::Image;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::spirv;

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = r32ui, sampled = false),
) {
    unsafe {
        let texel = image.texel_pointer(glam::IVec2::new(0, 0), 0);
        let _previous = atomic_i_add::<u32, { Scope::Workgroup as u32 }, { Semantics::NONE.bits() }>(
            texel, 1,
        );
    }
}
