//! Compute shader that sums a `PhysicalStorageBuffer`-backed linked list.
//! The host pushes the head address as a [`PhysicalPtr<Node>`] push
//! constant; the shader chases `next` until null. See
//! `examples/runners/physical-ptr-runner` for the live-fire dispatcher.

#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use bytemuck::{Pod, Zeroable};
use spirv_std::ptr::PhysicalPtr;
use spirv_std::spirv;

/// Recursive linked-list node accessed through a physical pointer. `Pod`
/// so a CPU host can build a `Vec<Node>` and upload it via
/// `bytemuck::cast_slice`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
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
    *output = 0.0;
    unsafe {
        while let Some(node) = current.as_ref() {
            *output += node.payload;
            current = node.next;
        }
    }
}
