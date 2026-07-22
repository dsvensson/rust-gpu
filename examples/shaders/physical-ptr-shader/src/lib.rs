//! Compute shader that sums a `PhysicalStorageBuffer`-backed linked list.
//! The host pushes the head address and an output address (both physical
//! pointers) as a push constant; the shader chases `next` until null and
//! writes the `f16` sum through the output pointer. The host deliberately
//! places that output above 4 GiB to exercise 64-bit addressing. See
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

/// Push-constant parameters: the head of the list and where to write the sum,
/// both as physical (device-address) pointers.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Params {
    pub root: PhysicalPtr<Node>,
    pub output: PhysicalPtr<f16>,
}

#[spirv(compute(threads(1)))]
pub fn main(#[spirv(push_constant)] params: &Params) {
    let mut current = params.root;
    let mut sum = 0.0f16;
    unsafe {
        while let Some(node) = current.as_ref() {
            sum += node.payload;
            current = node.next;
        }
        // Store the result through the physical output pointer (a >4 GiB
        // device address), exercising a 64-bit `OpConvertUToPtr` + store.
        *params.output.get() = sum;
    }
}
