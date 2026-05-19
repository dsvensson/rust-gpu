//! Compute shader that sums a `PhysicalStorageBuffer`-backed linked list.
//! The host pushes the head address as a [`PhysicalPtr<Node>`] push
//! constant; the shader chases `next` until null. See
//! `examples/runners/physical-ptr-runner` for the live-fire dispatcher.

#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use spirv_std::ptr::PhysicalPtr;
use spirv_std::spirv;

/// Two-field node accessed through a physical pointer. `next` is itself a
/// `PhysicalPtr<Node>` — the recursive shape from the PR description, made
/// possible because `PhysicalPtr` is a fixed-size address wrapper (two
/// `u32`s) and `PhantomData<*mut T>` carries the `T` only at the type level.
/// A null `next` terminates the list.
#[repr(C)]
pub struct Node {
    pub next: PhysicalPtr<Node>,
    pub payload: f32,
}

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(push_constant)] root_node: &PhysicalPtr<Node>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] output: &mut f32,
) {
    // Ideally this would read:
    //
    //     while let Some(node) = current.as_ref() {
    //         *output += node.payload;
    //         current = node.next;
    //     }
    //
    // …but Rust's null-pointer-optimised `Option<&T>` lowers the `None` case
    // to `OpConstantNull` of the `&T` pointer type, and `spirv-val` forbids
    // null constants for pointers in the `PhysicalStorageBuffer` storage
    // class. Wiring up a SPIR-T pass that rewrites those into a
    // function-local `OpConvertUToPtr 0` is left as a follow-up; in the
    // meantime, the explicit `is_null` form below produces clean SPIR-V.
    let mut current = *root_node;
    *output = 0.0;
    while !current.is_null() {
        let node = unsafe { current.as_ref_unchecked() };
        *output += node.payload;
        current = node.next;
    }
}
