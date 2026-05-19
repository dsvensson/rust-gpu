// Test that `asm!` now accepts an explicit (non-`Generic`) storage class on
// `OpTypePointer`. Before the `Add an optional explicit storage class to
// pointers` change, `asm!` rejected anything other than `Generic` here with
// "TypePointer in asm! requires `Generic` storage class".
//
// Declaring the pointer type is sufficient to exercise the asm! parser path;
// we don't need to *use* it (which would run into separate restrictions of
// Vulkan's Logical addressing mode).

// build-pass

use spirv_std::spirv;

#[spirv(compute(threads(1)))]
pub fn main(#[spirv(workgroup)] wg_var: &mut u32) {
    unsafe {
        core::arch::asm! {
            // Explicit `Workgroup` storage class - the regression target.
            "%ptr_ty = OpTypePointer Workgroup typeof*{wg_var}",
            // Touch the variable so it isn't dead-stripped before validation.
            "%c = OpConstant typeof*{wg_var} 7",
            "OpStore {wg_var} %c",
            wg_var = in(reg) wg_var,
        }
    }
}
