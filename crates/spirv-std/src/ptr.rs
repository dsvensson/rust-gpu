//! Physical pointers

#[cfg(target_arch = "spirv")]
use core::arch::asm;
use core::marker::PhantomData;

use crate::glam;

/// A physical pointer in the `PhysicalStorageBuffer` storage class
/// with semantics similar to `*mut T`.
///
/// This is similar to a raw pointer retrieved through `u64 as *mut T`, but
/// provides utilities for pointer manipulation that are currently not
/// supported on raw pointers due to the otherwise logical addressing model
/// and 32-bit pointer size.
pub struct PhysicalPtr<T> {
    // Stored as `UVec2` to avoid depending on the `Int64` SPIR-V capability.
    addr: glam::UVec2,
    _marker: PhantomData<*mut T>,
}

impl<T> Copy for PhysicalPtr<T> {}

impl<T> Clone for PhysicalPtr<T> {
    fn clone(&self) -> Self {
        Self {
            addr: self.addr,
            _marker: PhantomData,
        }
    }
}

impl<T> PhysicalPtr<T> {
    /// Get a mutable pointer to the physical address.
    /// The same aliasing rules that apply to FFI, apply to the returned pointer.
    #[crate::macros::gpu_only]
    pub fn get(self) -> *mut T {
        // Recombine via shift/mask (transmute would pull in `Int8`).
        let addr_u64: u64 = (self.addr.x as u64) | ((self.addr.y as u64) << 32);
        let result: *mut T;
        unsafe {
            asm!(
                "%ptr_type = OpTypePointer PhysicalStorageBuffer typeof**{result}",
                "{result} = OpConvertUToPtr %ptr_type {addr}",
                addr = in(reg) addr_u64,
                result = out(reg) result,
            );
            result
        }
    }

    /// Creates a null physical pointer.
    pub fn null() -> Self {
        Self {
            addr: glam::UVec2::ZERO,
            _marker: PhantomData,
        }
    }

    /// Returns `true` if the pointer is null.
    pub fn is_null(self) -> bool {
        self.addr == glam::UVec2::ZERO
    }

    /// Casts to a pointer of another type.
    pub fn cast<U>(self) -> PhysicalPtr<U> {
        PhysicalPtr { addr: self.addr, _marker: PhantomData }
    }

    /// Returns `None` if the pointer is null, or else returns a shared reference to the value wrapped in `Some`.
    pub unsafe fn as_ref<'a>(self) -> Option<&'a T> {
        self.is_null().then_some(unsafe { self.as_ref_unchecked() })
    }

    /// Returns `None` if the pointer is null, or else returns a mutable reference to the value wrapped in `Some`.
    pub unsafe fn as_mut<'a>(self) -> Option<&'a mut T> {
        self.is_null().then_some(unsafe { self.as_mut_unchecked() })
    }

    /// Returns a shared reference to the value behind the pointer.
    pub unsafe fn as_ref_unchecked<'a>(self) -> &'a T {
        unsafe { &*self.get() }
    }

    /// Returns a mutable reference to the value behind the pointer.
    pub unsafe fn as_mut_unchecked<'a>(self) -> &'a mut T {
        unsafe { &mut *self.get() }
    }

    /// Gets the address portion of the pointer. All physical pointers are considered to have global provenance.
    pub fn addr(self) -> u64 {
        // Shift/mask (transmute would pull in `Int8`).
        (self.addr.x as u64) | ((self.addr.y as u64) << 32)
    }

    /// Forms a physical pointer from an address. All physical pointers are considered to have global provenance.
    pub fn from_addr(addr: u64) -> Self {
        Self {
            addr: glam::UVec2::new(addr as u32, (addr >> 32) as u32),
            _marker: PhantomData,
        }
    }

    /// Creates a new pointer by mapping `self`’s address to a new one.
    pub fn map_addr(self, f: impl FnOnce(u64) -> u64) -> Self {
        Self::from_addr(f(self.addr()))
    }

    /// Adds a signed offset to a pointer.
    pub unsafe fn offset(self, count: i64) -> Self {
        unsafe { self.byte_offset(count * core::mem::size_of::<T>() as i64) }
    }

    /// Adds a signed offset in bytes to a pointer.
    pub unsafe fn byte_offset(self, count: i64) -> Self {
        self.map_addr(|addr| addr.overflowing_add_signed(count).0)
    }
}

/// A physical pointer in the `PhysicalStorageBuffer` storage class that the
/// caller asserts does not alias any other pointer accessed by the same
/// shader. The wrapper is a typed marker today; emitting the SPIR-V
/// `Restrict` decoration on the actual access still requires codegen-level
/// plumbing because the bitcast result is stored through a Rust local before
/// being dereferenced and decorations don't propagate through that store/load
/// pair. Until that lands, the wrapper documents the invariant and lets
/// consumer APIs require the restrict promise via the type system.
///
/// # Safety
/// Aliasing reads or writes through a restricted pointer are undefined
/// behavior at the driver level once the decoration is emitted; treating the
/// type as a contract from the start avoids API churn later.
pub struct RestrictedPhysicalPtr<T> {
    inner: PhysicalPtr<T>,
}

impl<T> Copy for RestrictedPhysicalPtr<T> {}

impl<T> Clone for RestrictedPhysicalPtr<T> {
    fn clone(&self) -> Self {
        Self { inner: self.inner }
    }
}

impl<T> RestrictedPhysicalPtr<T> {
    /// Wraps an existing [`PhysicalPtr`] without changing its address. The
    /// caller is responsible for ensuring that the pointer does not alias
    /// any other pointer that may be loaded from or stored to during the
    /// same shader invocation.
    ///
    /// # Safety
    /// The non-aliasing contract is the caller's responsibility.
    pub const unsafe fn new(inner: PhysicalPtr<T>) -> Self {
        Self { inner }
    }

    /// Returns the wrapped pointer, dropping the restrict guarantee.
    pub fn into_inner(self) -> PhysicalPtr<T> {
        self.inner
    }

    /// Get a mutable pointer to the physical address. The address is the same
    /// as the wrapped [`PhysicalPtr::get`] result; the wrapper's restrict
    /// promise is documentary today (see the type-level docs).
    pub fn get(self) -> *mut T {
        self.inner.get()
    }
}
