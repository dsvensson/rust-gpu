//! Compute shader that sums a `PhysicalStorageBuffer`-backed linked list.
//! The host pushes the head address as a [`PhysicalPtr<Node>`] push
//! constant; the shader chases `next` until null. See
//! `examples/runners/physical-ptr-runner` for the live-fire dispatcher.

#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(f16)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use spirv_std::ptr::PhysicalPtr;
use spirv_std::spirv;

/// Recursive linked-list node accessed through a physical pointer. `#[repr(C)]`
/// so the CPU host and the GPU agree on the layout; the host builds a
/// `Vec<Node>` and uploads the raw bytes. `PhysicalPtr` (8 bytes) + `f16`
/// (2 bytes) leaves 2 bytes of implicit tail padding — that's fine here (the
/// GPU never reads it), which is why `Node` isn't `bytemuck::Pod`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Node {
    pub next: PhysicalPtr<Node>,
    pub payload: f16,
}

#[spirv(compute(threads(1)))]
pub fn main(
    #[spirv(push_constant)] root_node: &PhysicalPtr<Node>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] output: &mut f16,
) {
    let mut current = *root_node;
    *output = 0.0;
    unsafe {
        while let Some(node) = current.as_ref() {
            *output += node.payload;
            current = node.next;
        }
    }
}
