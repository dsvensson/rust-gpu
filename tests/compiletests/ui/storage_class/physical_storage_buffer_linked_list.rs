// Verify the recursive linked-list shape from the `PhysicalPtr` examples
// compiles end-to-end: `struct Node { next: PhysicalPtr<Node>, payload }`
// works because `PhysicalPtr` is a fixed-size address wrapper, so the
// recursive `Node` -> `PhysicalPtr<Node>` reference doesn't actually inline
// a `Node`.

// build-pass
// compile-flags: -C target-feature=+PhysicalStorageBufferAddresses,+Int64,+ext:SPV_KHR_physical_storage_buffer

use spirv_std::ptr::PhysicalPtr;
use spirv_std::spirv;

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
    *output = 0.0;
    while !current.is_null() {
        let node = unsafe { current.as_ref_unchecked() };
        *output += node.payload;
        current = node.next;
    }
}
