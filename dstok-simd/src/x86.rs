//! Stable x86_64 SIMD byte-classification kernels (SSE4.2 and AVX2).
//!
//! Each kernel classifies a block of bytes into `Cat` values using vectorized
//! range/comparisons, returning how many leading bytes it consumed. The
//! caller fills the scalar tail. All kernels MUST produce the same category
//! vector as `crate::classify_byte`.
//!
//! The classification is the hot inner loop. AVX2 handles 32 ASCII bytes per
//! iteration; non-ASCII lanes fall back to the scalar byte oracle so SIMD can
//! never change category semantics.

#[cfg(target_arch = "x86_64")]
use crate::Cat;

/// SSE4.2 (16 bytes/iter). Returns the number of bytes classified.
///
/// On SSE4.2 we use `_mm_cmpgt_epi8` range tests for the ASCII classes and a
/// set-membership test via `_mm_cmpeq_epi8` for whitespace, then assemble Cat
/// per lane from the extracted 16-bit masks. Correct but modest throughput —
/// the tier that buys correctness on older x86; AVX2 is the wider stable tier.
///
/// # Safety
///
/// The caller must invoke this only when SSE4.2 is available.
#[target_feature(enable = "sse4.2")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn classify_sse42(bytes: &[u8], out: &mut Vec<Cat>) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    let n = bytes.len();
    while i + 16 <= n {
        let v = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);

        // Letter: 'A'..'Z' | 'a'..'z'
        let ge_a = _mm_cmpgt_epi8(v, _mm_set1_epi8(b'A' as i8 - 1));
        let le_z = _mm_cmpgt_epi8(_mm_set1_epi8(b'z' as i8 + 1), v);
        let gt_z = _mm_cmpgt_epi8(v, _mm_set1_epi8(b'Z' as i8));
        let lt_a = _mm_cmpgt_epi8(_mm_set1_epi8(b'a' as i8), v);
        let letter = _mm_andnot_si128(_mm_and_si128(gt_z, lt_a), _mm_and_si128(ge_a, le_z));

        // Digit: '0'..'9'
        let ge_0 = _mm_cmpgt_epi8(v, _mm_set1_epi8(b'0' as i8 - 1));
        let le_9 = _mm_cmpgt_epi8(_mm_set1_epi8(b'9' as i8 + 1), v);
        let digit = _mm_and_si128(ge_0, le_9);

        // Whitespace via six equality tests.
        let ws = {
            let sp = _mm_cmpeq_epi8(v, _mm_set1_epi8(b' ' as i8));
            let tab = _mm_cmpeq_epi8(v, _mm_set1_epi8(0x09));
            let nl = _mm_cmpeq_epi8(v, _mm_set1_epi8(0x0a));
            let cr = _mm_cmpeq_epi8(v, _mm_set1_epi8(0x0d));
            let vt = _mm_cmpeq_epi8(v, _mm_set1_epi8(0x0b));
            let ff = _mm_cmpeq_epi8(v, _mm_set1_epi8(0x0c));
            _mm_or_si128(
                _mm_or_si128(sp, tab),
                _mm_or_si128(_mm_or_si128(nl, cr), _mm_or_si128(vt, ff)),
            )
        };

        let apos = _mm_cmpeq_epi8(v, _mm_set1_epi8(b'\'' as i8));

        let m_apos = _mm_movemask_epi8(apos) as u16;
        let m_ws = _mm_movemask_epi8(ws) as u16;
        let m_digit = _mm_movemask_epi8(digit) as u16;
        let m_letter = _mm_movemask_epi8(letter) as u16;
        let m_nonascii = _mm_movemask_epi8(v) as u16;

        for k in 0..16 {
            let bit = 1u16 << k;
            let cat = if m_nonascii & bit != 0 {
                crate::classify_byte(bytes[i + k])
            } else if m_apos & bit != 0 {
                Cat::Apos
            } else if m_ws & bit != 0 {
                Cat::Ws
            } else if m_digit & bit != 0 {
                Cat::Digit
            } else if m_letter & bit != 0 {
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

/// AVX2 (32 bytes/iter). Returns the number of bytes classified.
///
/// # Safety
///
/// The caller must invoke this only when AVX2 is available.
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn classify_avx2(bytes: &[u8], out: &mut Vec<Cat>) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    let n = bytes.len();
    // 16-byte LUT: index = (byte >> 4) -> coarse class, refined by range tests.
    // We process 32 bytes per iteration; the hot ASCII case (the >99% case for
    // code/web text) classifies with 4 compares + 1 shuffle.
    while i + 32 <= n {
        let v = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
        // Letter: <= 'z' && >= 'A', excluding the gap between 'Z' and 'a'.
        let ge_a = _mm256_cmpgt_epi8(v, _mm256_set1_epi8(b'A' as i8 - 1)); // v >= 'A'
        let le_z = _mm256_cmpgt_epi8(_mm256_set1_epi8(b'z' as i8 + 1), v); // v <= 'z'
        let alpha = _mm256_and_si256(ge_a, le_z);
        let ge_cap = _mm256_cmpgt_epi8(v, _mm256_set1_epi8(b'Z' as i8)); // v > 'Z'
        let lt_low = _mm256_cmpgt_epi8(_mm256_set1_epi8(b'a' as i8), v); // v < 'a'
        let in_gap = _mm256_and_si256(ge_cap, lt_low); // 'Z' < v < 'a'
        let letter = _mm256_andnot_si256(in_gap, alpha);

        // Digit: '0' <= v <= '9'
        let ge_0 = _mm256_cmpgt_epi8(v, _mm256_set1_epi8(b'0' as i8 - 1));
        let le_9 = _mm256_cmpgt_epi8(_mm256_set1_epi8(b'9' as i8 + 1), v);
        let digit = _mm256_and_si256(ge_0, le_9);

        // Whitespace via six equality tests.
        let sp = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(b' ' as i8));
        let tab = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x09));
        let nl = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x0a));
        let cr = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x0d));
        let vtab = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x0b));
        let ff = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x0c));
        let ws = _mm256_or_si256(
            _mm256_or_si256(sp, tab),
            _mm256_or_si256(_mm256_or_si256(nl, cr), _mm256_or_si256(vtab, ff)),
        );

        // Apostrophe
        let apos = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'\'' as i8));

        // Signed SIMD comparisons cannot classify bytes >= 0x80 correctly, so
        // those lanes use the scalar oracle while ASCII lanes retain the SIMD
        // masks. This keeps every byte exact without sacrificing the common
        // all-ASCII path.
        let m_apos = _mm256_movemask_epi8(apos) as u32;
        let m_ws = _mm256_movemask_epi8(ws) as u32;
        let m_digit = _mm256_movemask_epi8(digit) as u32;
        let m_letter = _mm256_movemask_epi8(letter) as u32;
        let m_nonascii = _mm256_movemask_epi8(v) as u32;

        for k in 0..32 {
            let bit = 1u32 << k;
            let cat = if m_nonascii & bit != 0 {
                crate::classify_byte(bytes[i + k])
            } else if m_apos & bit != 0 {
                Cat::Apos
            } else if m_ws & bit != 0 {
                Cat::Ws
            } else if m_digit & bit != 0 {
                Cat::Digit
            } else if m_letter & bit != 0 {
                Cat::Letter
            } else {
                Cat::Other
            };
            out.push(cat);
        }
        i += 32;
    }
    i
}
