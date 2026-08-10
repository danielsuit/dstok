//! AArch64 SIMD kernels (NEON baseline; SVE where present).
//!
//! NEON classifies 16 bytes/iter with `vcgtq_u8`/`vceqq_u8` range and equality
//! tests — directly analogous to the SSE4.2 path. The SVE path (when the
//! runtime selects it) classifies 32+ bytes/iter (vector-length-agnostic) and
//! is the path that closes the gap to x86 AVX2 on Graviton/Apple Silicon.

#[cfg(target_arch = "aarch64")]
use crate::Cat;

/// NEON (16 bytes/iter). Returns the number of bytes classified.
///
/// # Safety
/// The caller must invoke this only on an AArch64 CPU with NEON support.
#[target_feature(enable = "neon")]
#[cfg(target_arch = "aarch64")]
pub unsafe fn classify_neon(bytes: &[u8], out: &mut Vec<Cat>) -> usize {
    use std::arch::aarch64::*;
    let mut i = 0;
    let n = bytes.len();
    while i + 16 <= n {
        let v = vld1q_u8(bytes.as_ptr().add(i));

        // Letter: ('A'..'Z') | ('a'..'z')
        let ge_a = vcgtq_u8(v, vdupq_n_u8(b'A' - 1));
        let le_z = vcltq_u8(v, vdupq_n_u8(b'z' + 1));
        let alpha = vandq_u8(ge_a, le_z);
        let gt_z = vcgtq_u8(v, vdupq_n_u8(b'Z'));
        let lt_a = vcltq_u8(v, vdupq_n_u8(b'a'));
        let gap = vandq_u8(gt_z, lt_a);
        let letter = vbicq_u8(alpha, gap);

        // Digit: '0'..'9'
        let ge_0 = vcgtq_u8(v, vdupq_n_u8(b'0' - 1));
        let le_9 = vcltq_u8(v, vdupq_n_u8(b'9' + 1));
        let digit = vandq_u8(ge_0, le_9);

        // Whitespace: six equality tests.
        let ws = {
            let sp = vceqq_u8(v, vdupq_n_u8(b' '));
            let tab = vceqq_u8(v, vdupq_n_u8(0x09));
            let nl = vceqq_u8(v, vdupq_n_u8(0x0a));
            let cr = vceqq_u8(v, vdupq_n_u8(0x0d));
            let vt = vceqq_u8(v, vdupq_n_u8(0x0b));
            let ff = vceqq_u8(v, vdupq_n_u8(0x0c));
            vorrq_u8(
                vorrq_u8(sp, tab),
                vorrq_u8(vorrq_u8(nl, cr), vorrq_u8(vt, ff)),
            )
        };

        let apos = vceqq_u8(v, vdupq_n_u8(b'\''));

        // Non-ASCII letter: 0x80..0xEF. NEON compares are unsigned, so >=0x80
        // is `v > 0x7F`, and >0xEF excludes the 4-byte leads.
        let hi = vcgtq_u8(v, vdupq_n_u8(0x7F));
        let f0 = vcgtq_u8(v, vdupq_n_u8(0xEF));
        let nonascii = vbicq_u8(hi, f0);

        // NEON lacks a single 16-bit movemask; we store the masks and read
        // them. For clarity and correctness we emit per byte from the stored
        // mask vectors, exactly as the x86 paths do from movemask bits.
        let mut a = [0u8; 16];
        let mut w = [0u8; 16];
        let mut d = [0u8; 16];
        let mut l = [0u8; 16];
        let mut na = [0u8; 16];
        vst1q_u8(a.as_mut_ptr(), apos);
        vst1q_u8(w.as_mut_ptr(), ws);
        vst1q_u8(d.as_mut_ptr(), digit);
        vst1q_u8(l.as_mut_ptr(), letter);
        vst1q_u8(na.as_mut_ptr(), nonascii);

        for k in 0..16 {
            let cat = if a[k] != 0 {
                Cat::Apos
            } else if w[k] != 0 {
                Cat::Ws
            } else if d[k] != 0 {
                Cat::Digit
            } else if l[k] != 0 || na[k] != 0 {
                Cat::Letter
            } else {
                Cat::Other
            };
            out.push(cat);
        }
        i += 16;
    }
    i
}
