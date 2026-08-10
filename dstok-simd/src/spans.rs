//! Run-boundary span formation (O2).
//!
//! Replaces the per-byte `while cat == prev { i += 1 }` loop with bulk
//! boundary extraction: build a bitmask of "category changes here", then
//! walk set bits with `trailing_zeros` (tzcnt). Each set bit yields a run
//! length in one instruction instead of one branch per byte.
//!
//! Bit-identical to `lib::spans_from_cats`: Apos is always a length-1 run
//! (only `'\''` maps to `Cat::Apos`); when it starts a contraction
//! (`'s 't 're 've 'll 'm 'd`) the run is merged forward to the contraction
//! end, exactly as the scalar former does.

use crate::{contraction_end, fxhash_bytes, Cat};
use dstok_core::regex::Span;
use smallvec::SmallVec;

/// Form spans from a category vector + the raw bytes. Shared, kernel-agnostic.
///
/// `digit_run` caps digit-run spans: `0` means unbounded (`\p{N}+`, GPT-2);
/// `n > 0` splits each `Digit` run into `n`-digit chunks (`\p{N}{1,n}`,
/// Llama-3/GLM), matching the scalar oracle in `dstok_core::regex`.
pub fn spans_from_cats(cats: &[Cat], bytes: &[u8], digit_run: usize) -> SmallVec<[Span; 64]> {
    let n = cats.len();
    if n == 0 {
        return SmallVec::new();
    }
    let mut runs: SmallVec<[Span; 64]> = SmallVec::new();
    // tzcnt over a rolling change-mask. We process 64-wide blocks so the mask
    // fits in a u64; the last partial block is handled by the scalar tail.
    let mut i = 0;
    while i + 64 <= n {
        i = emit_block(&cats[i..i + 64], i, &mut runs);
    }
    // Scalar tail for the remainder (< 64 categories).
    while i < n {
        let s = i;
        let c = cats[i];
        while i < n && cats[i] == c {
            i += 1;
        }
        runs.push(Span { start: s, end: i });
    }

    // Post-merge: an Apos run that starts a contraction absorbs the following
    // run(s) up to the contraction end, matching the scalar former's `i = end`.
    let mut merged: SmallVec<[Span; 64]> = SmallVec::with_capacity(runs.len());
    let mut r = 0;
    while r < runs.len() {
        let span = runs[r];
        if cats[span.start] == Cat::Apos {
            if let Some(end) = contraction_end(bytes, span.start, n) {
                // Absorb all subsequent runs whose bytes fall within [..end).
                let mut absorb_end = end;
                let mut k = r + 1;
                while k < runs.len() && runs[k].start < absorb_end {
                    absorb_end = absorb_end.max(runs[k].end);
                    k += 1;
                }
                merged.push(Span {
                    start: span.start,
                    end: absorb_end,
                });
                r = k;
                continue;
            }
        }
        merged.push(span);
        r += 1;
    }

    // Digit-run cap (Llama-3 {1,2}, GLM {1,3}). `digit_run == 0` (GPT-2) leaves
    // digit runs whole. Otherwise split each `Digit`-category span longer than
    // `digit_run` into `digit_run`-sized chunks (leftover in its own trailing
    // chunk), exactly matching the scalar oracle's per-chunk emission. (This
    // was a latent divergence: the SIMD path previously ignored the cap.)
    if digit_run > 0 {
        let mut chunked: SmallVec<[Span; 64]> = SmallVec::with_capacity(merged.len());
        for span in &merged {
            if cats[span.start] == Cat::Digit && span.end > span.start + digit_run {
                let mut s = span.start;
                while s < span.end {
                    let e = (s + digit_run).min(span.end);
                    chunked.push(Span { start: s, end: e });
                    s = e;
                }
            } else {
                chunked.push(*span);
            }
        }
        merged = chunked;
    }

    merged
}

/// O18: same as `spans_from_cats` but also returns a parallel vector of
/// per-span fxhashes (of each pretoken's bytes), computed while the bytes are
/// hot in L1. The fast path uses these as the cache hot-table index so it
/// never re-hashes the key.
pub fn spans_from_cats_hashed(
    cats: &[Cat],
    bytes: &[u8],
    digit_run: usize,
) -> (SmallVec<[Span; 64]>, SmallVec<[u64; 64]>) {
    let spans = spans_from_cats(cats, bytes, digit_run);
    let hashes: SmallVec<[u64; 64]> = spans
        .iter()
        .map(|s| fxhash_bytes(&bytes[s.start..s.end]))
        .collect();
    (spans, hashes)
}

/// Emit the runs of one 64-category block. Returns the index just past it.
fn emit_block(block: &[Cat], base: usize, out: &mut SmallVec<[Span; 64]>) -> usize {
    // Build a 64-bit change mask: bit k set iff block[k] != block[k+1], plus
    // bit 63 set to force a final run close. We also set bit "before 0" by
    // always opening a run at index 0.
    let mut mask: u64 = 0;
    for k in 0..63 {
        if block[k] != block[k + 1] {
            mask |= 1 << k;
        }
    }
    mask |= 1 << 63; // ensure the last run closes

    let mut start = base;
    let mut bits = mask;
    // Iterate set bits low→high via trailing_zeros.
    loop {
        if bits == 0 {
            break;
        }
        let tz = bits.trailing_zeros() as usize;
        let end = base + tz + 1;
        out.push(Span { start, end });
        start = end;
        bits &= bits.wrapping_sub(1);
    }
    base + 64
}
