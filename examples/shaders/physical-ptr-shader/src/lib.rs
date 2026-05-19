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
    let mut current = *root_node;
    let mut sum: f32 = 0.0;
    // Bound the loop so we never run away on a corrupted list. 4096 nodes is
    // plenty for examples; a real workload would size this to the maximum
    // expected list length.
    let mut steps = 0u32;
    while !current.is_null() && steps < 4096 {
        let node = unsafe { &*current.get() };
        sum += node.payload;
        current = node.next;
        steps += 1;
    }
    *output = sum;
}
