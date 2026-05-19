// Same scenario as `image_texel_pointer.rs`, but run through the SPIR-T
// `qptr` pipeline (`--no-infer-storage-classes` skips the specializer
// pass; the qptr passes take over storage-class assignment instead).
// This verifies that explicit-storage-class pointer types produced by
// the new `Image::texel_pointer` API survive the qptr lowering and
// lifting passes without breaking.

// build-pass
// compile-flags: -C llvm-args=--no-infer-storage-classes -C llvm-args=--spirt-passes=qptr

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
