// Verify that the explicit-storage-class asm! path is compatible with the
// SPIR-T `qptr` pipeline (opt-in via `--no-infer-storage-classes` +
// `--spirt-passes=qptr`). This is the same test as
// `asm_explicit_storage_class.rs`, run through the qptr passes that
// would otherwise run if the user opted in.

// build-pass
// compile-flags: -C llvm-args=--no-infer-storage-classes -C llvm-args=--spirt-passes=qptr

use spirv_std::spirv;

#[spirv(compute(threads(1)))]
pub fn main(#[spirv(workgroup)] wg_var: &mut u32) {
    unsafe {
        core::arch::asm! {
            "%ptr_ty = OpTypePointer Workgroup typeof*{wg_var}",
            "%c = OpConstant typeof*{wg_var} 7",
            "OpStore {wg_var} %c",
            wg_var = in(reg) wg_var,
        }
    }
}
