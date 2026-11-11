//! Sums a linked list in `PhysicalStorageBuffer` memory and writes the sum through
//! another physical pointer. The host side is `examples/runners/physical-ptr`.

#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use spirv_std::ptr::PhysicalPtr;
use spirv_std::spirv;

/// A list node, also built by the host.
#[repr(C)]
pub struct Node {
    pub next: PhysicalPtr<Node>,
    pub payload: f32,
}

/// The push constants.
#[repr(C)]
pub struct Params {
    pub root: PhysicalPtr<Node>,
    pub output: PhysicalPtr<f32>,
}

#[spirv(compute(threads(1)))]
pub fn main(#[spirv(push_constant)] params: &Params) {
    let mut cur = params.root;
    let mut sum = 0.0;
    unsafe {
        while !cur.is_null() {
            let node = cur.as_ref_unchecked();
            sum += node.payload;
            cur = node.next;
        }
        *params.output.as_mut_unchecked() = sum;
    }
}
