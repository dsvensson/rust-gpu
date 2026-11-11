// Test the null-checked `PhysicalPtr::as_ref` and `PhysicalPtr::as_mut`.

// build-pass
// compile-flags: -C target-feature=+PhysicalStorageBufferAddresses,+Int64,+ext:SPV_KHR_physical_storage_buffer
// compile-flags: -C llvm-args=--disassemble-entry=main
// normalize-stderr-test "\n\W*OpLine .*" -> ""

use spirv_std::ptr::PhysicalPtr;
use spirv_std::spirv;

#[repr(C)]
pub struct Node {
    next: PhysicalPtr<Node>,
    value: u32,
}

#[repr(C)]
pub struct Stats {
    count: u32,
    sum: u32,
}

#[repr(C)]
pub struct Roots {
    list: PhysicalPtr<Node>,
    stats: PhysicalPtr<Stats>,
    total: PhysicalPtr<u32>,
}

#[spirv(compute(threads(1)))]
pub fn main(#[spirv(push_constant)] roots: &Roots) {
    unsafe {
        let (mut count, mut sum) = (0, 0);
        let mut cur = roots.list;
        while let Some(node) = cur.as_ref() {
            count += 1;
            sum += node.value;
            cur = node.next;
        }
        if let Some(mut stats) = roots.stats.as_mut() {
            stats.count = count;
            stats.sum = sum;
        }
        if let Some(mut total) = roots.total.as_mut() {
            *total += sum;
        }
    }
}
