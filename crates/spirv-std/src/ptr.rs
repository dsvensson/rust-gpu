//! Pointers to `PhysicalStorageBuffer` memory, given by 64-bit device addresses
//! (Vulkan's `bufferDeviceAddress`).
//!
//! Requires the `PhysicalStorageBufferAddresses` and `Int64` capabilities, and before
//! SPIR-V 1.5 the `SPV_KHR_physical_storage_buffer` extension:
//! ```text
//! -C target-feature=+PhysicalStorageBufferAddresses,+Int64,+ext:SPV_KHR_physical_storage_buffer
//! ```
//!
//! ```no_run
//! use spirv_std::ptr::PhysicalPtr;
//! use spirv_std::spirv;
//!
//! #[repr(C)]
//! pub struct PushConstants {
//!     input: PhysicalPtr<u32>,
//! }
//!
//! #[spirv(compute(threads(1)))]
//! pub fn main_cs(
//!     #[spirv(push_constant)] constants: &PushConstants,
//!     #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] output: &mut u32,
//! ) {
//!     if !constants.input.is_null() {
//!         *output = unsafe { *constants.input.as_ref_unchecked() };
//!     }
//! }
//! ```
//!
//! Loads and stores through a `PhysicalPtr<T>` are emitted with `Aligned` set to the size
//! of the largest scalar in `T`, the minimum Vulkan requires of the address; an access
//! needing a stricter claim can be written with `asm!` (`OpLoad … Aligned n`).
//!
//! # Limitations
//! - A reference obtained from a `PhysicalPtr` is like any other reference in rust-gpu:
//!   it can't be stored in memory, so keep the `PhysicalPtr` (not the reference) in
//!   structs and across loop iterations.
//! - Such a reference can't share a variable or the result of a branch with a reference
//!   to non-physical memory (e.g. `if c { p.as_ref_unchecked() } else { &local }`):
//!   storage classes are inferred per pointer, so compilation fails with an
//!   "inference conflict" between the two.
//! - There are no null physical references, and `Option<&T>` won't work for them; use
//!   [`PhysicalPtr::as_ref`] or [`PhysicalPtr::as_mut`] instead, or test
//!   [`PhysicalPtr::is_null`].
//! - Pointer arithmetic and casts on the raw pointer from [`PhysicalPtr::as_ptr`] aren't
//!   supported; do them on the `PhysicalPtr`.
#[cfg(target_arch = "spirv")]
use core::arch::asm;
use core::fmt;
use core::marker::PhantomData;
use core::mem;
use core::num::NonZeroU64;
use core::ops::{Deref, DerefMut};

/// The address of a `T` in `PhysicalStorageBuffer` memory, laid out like a `u64` (`VkDeviceAddress`).
///
/// Unlike the raw pointer methods they're named after, [`add`](Self::add),
/// [`offset`](Self::offset), [`byte_add`](Self::byte_add) and
/// [`byte_offset`](Self::byte_offset) are plain (wrapping) address arithmetic, and so safe.
#[repr(transparent)]
pub struct PhysicalPtr<T> {
    addr: u64,
    // `fn(T) -> T` makes this invariant in `T`, and `Send + Sync` for any `T`.
    _marker: PhantomData<fn(T) -> T>,
}

impl<T> PhysicalPtr<T> {
    /// The null pointer, with address `0`.
    #[inline]
    pub const fn null() -> Self {
        Self::from_addr(0)
    }

    /// Creates a pointer from a device address.
    #[inline]
    pub const fn from_addr(addr: u64) -> Self {
        Self {
            addr,
            _marker: PhantomData,
        }
    }

    /// Returns the device address.
    #[inline]
    pub const fn addr(self) -> u64 {
        self.addr
    }

    /// Returns `true` if the address is `0`.
    #[inline]
    pub const fn is_null(self) -> bool {
        self.addr == 0
    }

    /// Changes the pointee type, keeping the address.
    #[inline]
    pub const fn cast<U>(self) -> PhysicalPtr<U> {
        PhysicalPtr::from_addr(self.addr)
    }

    /// Adds `count * size_of::<T>()` to the address.
    #[inline]
    pub const fn add(self, count: u64) -> Self {
        self.byte_add(count.wrapping_mul(size_of::<T>() as u64))
    }

    /// Adds `count * size_of::<T>()`, which may be negative, to the address.
    #[inline]
    pub const fn offset(self, count: i64) -> Self {
        self.byte_offset(count.wrapping_mul(size_of::<T>() as i64))
    }

    /// Adds `count` to the address.
    #[inline]
    pub const fn byte_add(self, count: u64) -> Self {
        Self::from_addr(self.addr.wrapping_add(count))
    }

    /// Adds `count`, which may be negative, to the address.
    #[inline]
    pub const fn byte_offset(self, count: i64) -> Self {
        Self::from_addr(self.addr.wrapping_add_signed(count))
    }

    /// Converts the address to a raw pointer.
    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpConvertUToPtr")]
    #[inline]
    pub fn as_ptr(self) -> *mut T {
        let result: *mut T;
        unsafe {
            asm!(
                "{result} = OpConvertUToPtr typeof{result} {addr}",
                addr = in(reg) self.addr,
                result = out(reg) result,
            );
        }
        result
    }

    /// Returns a shared reference to the `T` at the address.
    ///
    /// # Safety
    /// The address must be that of an initialized `T`, aligned to `align_of::<T>()`,
    /// which isn't mutated (other than through an `UnsafeCell`) for `'a`.
    #[inline]
    pub unsafe fn as_ref_unchecked<'a>(self) -> &'a T {
        unsafe { &*self.as_ptr() }
    }

    /// Returns a unique reference to the `T` at the address.
    ///
    /// # Safety
    /// The address must be that of an initialized `T`, aligned to `align_of::<T>()`,
    /// which isn't accessed other than through the returned reference for `'a`.
    #[inline]
    pub unsafe fn as_mut_unchecked<'a>(self) -> &'a mut T {
        unsafe { &mut *self.as_ptr() }
    }

    /// Returns `None` if the address is `0`, or else a handle that dereferences to the `T`
    /// at the address.
    ///
    /// # Safety
    /// A non-zero address must meet the requirements of
    /// [`as_ref_unchecked`](Self::as_ref_unchecked).
    #[inline]
    pub unsafe fn as_ref<'a>(self) -> Option<PhysicalRef<'a, T>> {
        // SAFETY: `Option` of a `repr(transparent)` `NonZeroU64` is a `u64` with `0` for `None`.
        // (`NonZeroU64::new(addr).map(…)` would emit a second null check.)
        unsafe { mem::transmute::<u64, Option<PhysicalRef<'a, T>>>(self.addr) }
    }

    /// Returns `None` if the address is `0`, or else a handle that mutably dereferences to
    /// the `T` at the address.
    ///
    /// # Safety
    /// A non-zero address must meet the requirements of
    /// [`as_mut_unchecked`](Self::as_mut_unchecked).
    #[inline]
    pub unsafe fn as_mut<'a>(self) -> Option<PhysicalMut<'a, T>> {
        // SAFETY: as in `as_ref`.
        unsafe { mem::transmute::<u64, Option<PhysicalMut<'a, T>>>(self.addr) }
    }
}

// Written by hand, as `#[derive]` would require the same traits of `T`.
impl<T> Clone for PhysicalPtr<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for PhysicalPtr<T> {}

impl<T> PartialEq for PhysicalPtr<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.addr == other.addr
    }
}

impl<T> Eq for PhysicalPtr<T> {}

impl<T> Default for PhysicalPtr<T> {
    #[inline]
    fn default() -> Self {
        Self::null()
    }
}

impl<T> fmt::Debug for PhysicalPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PhysicalPtr({:#x})", self.addr)
    }
}

// SAFETY: `repr(transparent)` over a `u64`, for which all zeroes is valid.
#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Zeroable for PhysicalPtr<T> {}

// SAFETY: `repr(transparent)` over a `u64`, so there is no padding and every bit
// pattern is valid.
#[cfg(feature = "bytemuck")]
unsafe impl<T: 'static> bytemuck::Pod for PhysicalPtr<T> {}

/// A non-null [`PhysicalPtr`] that dereferences to the `T`, returned by [`PhysicalPtr::as_ref`].
///
/// It exists because the `None` of `Option<&T>` is a null pointer, which SPIR-V doesn't allow
/// for `PhysicalStorageBuffer` pointers, while the `None` of `Option<PhysicalRef<T>>` is the
/// address `0`.
#[repr(transparent)]
pub struct PhysicalRef<'a, T> {
    addr: NonZeroU64,
    _marker: PhantomData<&'a T>,
}

impl<T> Clone for PhysicalRef<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for PhysicalRef<'_, T> {}

impl<T> Deref for PhysicalRef<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: guaranteed by the caller of `PhysicalPtr::as_ref`.
        unsafe { PhysicalPtr::from_addr(self.addr.get()).as_ref_unchecked() }
    }
}

/// A non-null [`PhysicalPtr`] that mutably dereferences to the `T`, returned by
/// [`PhysicalPtr::as_mut`].
///
/// It exists instead of `Option<&mut T>` for the same reason as [`PhysicalRef`].
#[repr(transparent)]
pub struct PhysicalMut<'a, T> {
    addr: NonZeroU64,
    _marker: PhantomData<&'a mut T>,
}

impl<T> Deref for PhysicalMut<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: guaranteed by the caller of `PhysicalPtr::as_mut`.
        unsafe { PhysicalPtr::from_addr(self.addr.get()).as_ref_unchecked() }
    }
}

impl<T> DerefMut for PhysicalMut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: guaranteed by the caller of `PhysicalPtr::as_mut`.
        unsafe { PhysicalPtr::from_addr(self.addr.get()).as_mut_unchecked() }
    }
}

#[cfg(test)]
mod tests {
    use super::{PhysicalMut, PhysicalPtr, PhysicalRef};

    #[test]
    fn layout() {
        assert_eq!(size_of::<PhysicalPtr<u8>>(), size_of::<u64>());
        assert_eq!(align_of::<PhysicalPtr<u8>>(), align_of::<u64>());
        assert_eq!(size_of::<Option<PhysicalRef<'_, u32>>>(), 8);
        assert_eq!(size_of::<Option<PhysicalMut<'_, u32>>>(), 8);
    }

    #[test]
    fn null() {
        assert!(PhysicalPtr::<u32>::null().is_null());
        assert!(PhysicalPtr::<u32>::default().is_null());
        assert!(!PhysicalPtr::<u32>::from_addr(4).is_null());
    }

    #[test]
    fn as_ref_as_mut() {
        // Only checks for null, as dereferencing needs a GPU.
        unsafe {
            assert!(PhysicalPtr::<u32>::null().as_ref().is_none());
            assert!(PhysicalPtr::<u32>::null().as_mut().is_none());
            assert!(PhysicalPtr::<u32>::from_addr(4).as_ref().is_some());
            assert!(PhysicalPtr::<u32>::from_addr(4).as_mut().is_some());
        }
    }

    #[test]
    fn arithmetic() {
        let p = PhysicalPtr::<[u32; 3]>::from_addr(0x1000);
        assert_eq!(p.add(2).addr(), 0x1018);
        assert_eq!(p.offset(2).addr(), 0x1018);
        assert_eq!(p.offset(-2).addr(), 0x0fe8);
        assert_eq!(p.byte_add(4).addr(), 0x1004);
        assert_eq!(p.byte_offset(-4).addr(), 0x0ffc);
        assert_eq!(p.cast::<u8>().add(2).addr(), 0x1002);
        assert_eq!(p.cast::<u8>(), PhysicalPtr::from_addr(0x1000));

        let p = PhysicalPtr::<u32>::from_addr(4);
        assert_eq!(p.offset(-2).addr(), u64::MAX - 3);
        assert_eq!(p.byte_offset(-5).addr(), u64::MAX);
        assert_eq!(PhysicalPtr::<u64>::from_addr(u64::MAX).add(1).addr(), 7);
    }

    #[cfg(feature = "bytemuck")]
    #[test]
    fn pod() {
        fn is_pod<T: bytemuck::Pod>() {}
        is_pod::<PhysicalPtr<u32>>();
    }
}
