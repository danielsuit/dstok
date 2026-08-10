//! Byte-category lookup-table helpers.
//!
//! The table builders remain useful for tests and future verified kernels. The
//! former experimental two-stage AVX2 gather was not representable with one
//! 256-bit register and has been replaced by a compatibility wrapper around
//! the exact comparison kernel.

/// Build the 256-byte stage table: `table[byte] = Cat as u8`. This is the
/// ground truth both the LUT and the scalar `classify_byte` must match.
pub fn stage_table() -> [u8; 256] {
    let mut t = [0u8; 256];
    for b in 0..=255u8 {
        t[b as usize] = crate::classify_byte(b) as u8;
    }
    t
}

/// Pack the 256-byte table into 16 pages of 16 bytes (row-major: page = hi
/// nibble, index within page = lo nibble). Returns a 256-byte array usable
/// as the broadcast source for `vpshufb` page-gather.
pub fn pack_pages() -> [u8; 256] {
    let t = stage_table();
    let mut pages = [0u8; 256];
    for b in 0..=255u8 {
        let hi = (b >> 4) as usize;
        let lo = (b & 0x0F) as usize;
        // store so that page `hi` lives at bytes [hi*16 .. hi*16+16],
        // indexed by `lo` — matches the broadcast+gather order.
        pages[hi * 16 + lo] = t[b as usize];
    }
    pages
}

#[cfg(target_arch = "x86_64")]
mod x86_lut {
    use crate::Cat;

    /// Compatibility entry point for the former experimental LUT kernel.
    ///
    /// The previous two-stage `vpshufb` implementation could not gather all 16
    /// high-nibble pages with one register and produced incorrect categories.
    /// Keep the public symbol for compatibility, but delegate to the verified
    /// AVX2 comparison kernel. `pages` remains accepted so callers do not need
    /// to change.
    ///
    /// # Safety
    ///
    /// The caller must invoke this only when AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn classify_avx2_lut(bytes: &[u8], out: &mut Vec<Cat>, _pages: &[u8; 256]) -> usize {
        // SAFETY: this function has the same AVX2 precondition as the delegated
        // kernel.
        unsafe { crate::x86::classify_avx2(bytes, out) }
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86_lut::classify_avx2_lut;
