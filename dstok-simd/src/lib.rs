//! # dstok-simd — experimental runtime-dispatched byte classifiers
//!
//! The only crate with `unsafe`. Kernels are selected once at construction via
//! CPU feature detection; the scalar path is always compiled in as the
//! fallback and differential target.
//!
//! This crate is unpublished and is not used by the supported `dstok` package.
//! Its byte-category model is narrower than the pinned GLM Oniguruma regex; a
//! future production integration would first need exact span and complete-ID
//! differential gates.

#![cfg_attr(not(test), deny(unsafe_code))]
// `unsafe` is permitted here and only here.
#![allow(unsafe_code)]

pub mod arm;
pub mod lut;
pub mod spans;
pub mod x86;

use dstok_core::regex::Span;
use smallvec::SmallVec;

type HashedSpans = (SmallVec<[Span; 64]>, SmallVec<[u64; 64]>);

/// The selected SIMD feature tier for pretokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    Scalar,
    Sse42,
    Avx2,
    Neon,
    Sve,
}

impl Feature {
    /// Detect the best available tier at runtime. On non-x86 non-aarch64
    /// targets this is `Scalar`.
    pub fn detect() -> Feature {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return Feature::Avx2;
            }
            if std::is_x86_feature_detected!("sse4.2") {
                return Feature::Sse42;
            }
            return Feature::Scalar;
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SVE detection via std is nightly-only; we conservatively pick
            // NEON (always present on aarch64) and let the SVE kernel be
            // selected by an explicit `with_feature` constructor.
            return Feature::Neon;
        }
        #[allow(unreachable_code)]
        {
            Feature::Scalar
        }
    }

    /// O59: the detected tier, cached for the process lifetime. CPU feature
    /// probing (`is_x86_feature_detected!`) is cheap per call but called on
    /// every encode; caching turns it into a single atomic load. The result
    /// never changes during a process, so a `OnceLock` is sound.
    pub fn detect_cached() -> Feature {
        static CACHED: std::sync::OnceLock<Feature> = std::sync::OnceLock::new();
        *CACHED.get_or_init(Feature::detect)
    }
}

/// Byte category produced by the vectorized classifier. Matches the
/// scalar pretokenizer's notion of "what kind of run am I in".
///
/// - `Ws`    : ASCII whitespace (space, \t, \n, \r, \x0b, \x0c)
/// - `Letter`: ASCII letter OR a non-ASCII lead/continuation byte (the
///             scalar oracle treats CJK/multibyte as a letter run)
/// - `Digit` : ASCII digit 0-9
/// - `Other` : everything else (punctuation, emoji lead bytes per the oracle)
/// - `Apos`  : ASCII apostrophe `'` (the contraction starter)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Cat {
    Other = 0,
    Ws = 1,
    Letter = 2,
    Digit = 3,
    Apos = 4,
}

/// Classify one byte (scalar reference; also the per-byte fallback for the
/// tail that doesn't fill a vector). MUST match the SIMD kernels exactly.
#[inline]
pub fn classify_byte(b: u8) -> Cat {
    match b {
        b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c => Cat::Ws,
        b'0'..=b'9' => Cat::Digit,
        b'A'..=b'Z' | b'a'..=b'z' => Cat::Letter,
        b'\'' => Cat::Apos,
        // Non-ASCII: the oracle's lead-byte rule. Continuation bytes (0x80..0xBF)
        // and 2/3-byte leads (0xC2..0xEF) are Letter runs; 4-byte leads
        // (0xF0..0xF4, emoji/historic) are Other. This matches
        // `dstok_core::unicode_class::class_of_lead`.
        0x80..=0xEF => Cat::Letter,
        0xF0..=0xF4 => Cat::Other,
        _ => Cat::Other,
    }
}

/// Form experimental category-run spans with the best available SIMD tier.
///
/// This is not a replacement for the production GLM Oniguruma pre-tokenizer.
/// `digit_run` caps digit runs: `0` is unbounded and `n > 0` emits chunks of at
/// most `n` digit bytes.
pub fn pretokenize(bytes: &[u8], digit_run: usize) -> SmallVec<[Span; 64]> {
    let feat = Feature::detect_cached();
    let cats = classify_all(bytes, feat);
    // O2: run-boundary span formation (tzcnt bulk extraction), bit-identical
    // to the scalar `spans_from_cats` below — kept as the differential target.
    spans::spans_from_cats(&cats, bytes, digit_run)
}

/// O18: pretokenize and also emit a per-span **precomputed fxhash** of each
/// pretoken's bytes. The hash is computed here, while the pretoken bytes are
/// hot in L1 during span formation — so the fast path's cache lookup reuses
/// it instead of re-hashing the key. The mixing constant matches
/// `dstok_fast::cache::fxhash_key` so the hot-table index is consistent.
///
/// Output spans are byte-identical to `pretokenize`; the hashes are a pure
/// side product (a property test asserts `hash == fxhash(bytes[start..end])`).
pub fn pretokenize_hashed(
    bytes: &[u8],
    digit_run: usize,
) -> (SmallVec<[Span; 64]>, SmallVec<[u64; 64]>) {
    let feat = Feature::detect_cached();
    let cats = classify_all(bytes, feat);
    spans::spans_from_cats_hashed(&cats, bytes, digit_run)
}

/// The FxHash multiplier, shared with the cache so precomputed hashes index
/// the hot table identically.
pub const FXHASH_MUL: u64 = 0x517cc1b727220a95;

/// O18: the exact hash the cache's hot table uses, exposed so span formation
/// and the cache agree. Public so tests can assert parity.
#[inline]
pub fn fxhash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0u64;
    for chunk in bytes.chunks_exact(8) {
        h = (h.rotate_left(5) ^ u64::from_le_bytes(chunk.try_into().unwrap()))
            .wrapping_mul(FXHASH_MUL);
    }
    for &b in bytes.chunks_exact(8).remainder() {
        h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(FXHASH_MUL);
    }
    h
}

/// O74: prefetch one cache line into L1/L2 (`PREFETCHT0`). Safe wrapper over
/// the SSE intrinsic — `dstok-fast` is `#![forbid(unsafe_code)]`, so all
/// prefetch goes through here. No-op on non-x86 (the hardware prefetcher
/// covers sequential access there). `ptr` need not be dereferenceable.
#[inline(always)]
pub fn prefetch_read_data(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: PREFETCHT0 does not dereference; it only stages a cache line.
        // The pointer may be arbitrary; the instruction never faults on it.
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

/// O20: classify every byte in parallel across a rayon pool.
///
/// Classification is per-byte independent (each byte's category depends only
/// on that byte), so it parallelizes trivially: split `bytes` into ~`threads`
/// contiguous chunks, classify each with the best SIMD tier on its own core,
/// concatenate in order. Span formation then runs once over the full
/// `Vec<Cat>` (serial, but O(n) and tzcnt-cheap), so chunk boundaries never
/// affect spans — correctness is identical to the serial `classify_all`.
///
/// This removes the one serial pass (SIMD classify) that capped total
/// multi-core throughput: at ~10–30 GB/s per core of SIMD classify, the
/// serial pass bottlenecked a 1 GB file at ~0.1 s regardless of core count.
/// Parallel classify lifts that ceiling to ~(per-core classify) × cores.
pub fn classify_parallel(bytes: &[u8], threads: usize) -> Vec<Cat> {
    let feat = Feature::detect_cached();
    classify_parallel_with(bytes, feat, threads)
}

/// Same, with an explicit feature tier (for testing/differential against the
/// scalar path regardless of the host CPU).
pub fn classify_parallel_with(bytes: &[u8], feat: Feature, threads: usize) -> Vec<Cat> {
    // Small inputs or single-thread: don't pay the pool overhead.
    if threads <= 1 || bytes.len() < (1 << 16) {
        return classify_all(bytes, feat);
    }
    use rayon::prelude::*;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    let chunk = bytes.len().div_ceil(threads);
    let parts: Vec<Vec<Cat>> = pool.install(|| {
        bytes
            .par_chunks(chunk)
            .map(|c| classify_all(c, feat))
            .collect()
    });
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts {
        out.extend(p);
    }
    out
}

/// O20: pretokenize with a parallel classify pass, then serial (cheap) span
/// formation + hashing. Output byte-identical to `pretokenize_hashed`.
pub fn pretokenize_parallel_hashed(
    bytes: &[u8],
    digit_run: usize,
    threads: usize,
) -> (SmallVec<[Span; 64]>, SmallVec<[u64; 64]>) {
    let cats = classify_parallel(bytes, threads);
    spans::spans_from_cats_hashed(&cats, bytes, digit_run)
}

/// O48: pretokenize with **both** passes parallel — parallel classify, then
/// parallel span formation. This removes the serial span-formation pass that
/// capped huge-file throughput at high core counts (classify was already
/// parallel via O20, but span formation ran on one thread).
///
/// Correctness rests on the **safe-split rule**: we only split the category
/// vector at category changes that are NOT `Apos → Letter`. The `Apos → Letter`
/// boundary may lie inside a contraction (`'s`, `'re`, …) — splitting there
/// would separate an apostrophe from its suffix, and the chunk-local former
/// (which can't see the next chunk) would emit a standalone apostrophe instead
/// of the contraction, breaking `incremental == batch`. Every other category
/// change is a true pretoken boundary (the scalar former consumes one
/// category run at a time, except for the absorbed Apos+letters), so chunks
/// bounded by safe splits are pretoken-aligned and encode independently.
///
/// Output byte-identical to `pretokenize_hashed` (property-tested).
/// O48: pretokenize with **both** passes parallel — parallel classify, then
/// parallel span formation. Span formation is otherwise the one serial O(n)
/// pass left after O20, which caps many-core throughput (at high core counts
/// the serial former dominates wall-clock). This lifts it.
///
/// Correctness: span formation is parallelized by splitting the category
/// vector at *safe* points only. A split before index `k` is safe iff it is a
/// pretoken boundary, i.e. `cats[k-1] != cats[k]` AND not an apostrophe that
/// begins a contraction (`cats[k-1] == Apos` and `contraction_end` matches —
/// that Apos+letters is one pretoken and must stay in one chunk). Every other
/// category change is a true pretoken boundary, so chunks are pretoken-aligned
/// and each chunk's spans are independent and identical to the serial result.
/// The chunk-local former is the *same* `spans_from_cats_hashed` run on the
/// slice (relative indices, shifted by the chunk start) — no second former to
/// drift. Output is byte-identical to `pretokenize_hashed`.
pub fn pretokenize_full_parallel(
    bytes: &[u8],
    digit_run: usize,
    threads: usize,
) -> (SmallVec<[Span; 64]>, SmallVec<[u64; 64]>) {
    if threads <= 1 || bytes.len() < (1 << 18) {
        return pretokenize_parallel_hashed(bytes, digit_run, threads);
    }
    let cats = classify_parallel(bytes, threads);
    let n = bytes.len();
    let splits = safe_split_points(&cats, bytes, n, threads);

    use rayon::prelude::*;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    let ranges: Vec<(usize, usize)> = splits.windows(2).map(|w| (w[0], w[1])).collect();
    let parts: Vec<HashedSpans> = pool.install(|| {
        ranges
            .par_iter()
            .map(|&(s, e)| {
                let (rel_spans, rel_hashes) =
                    spans::spans_from_cats_hashed(&cats[s..e], &bytes[s..e], digit_run);
                let abs_spans: SmallVec<[Span; 64]> = rel_spans
                    .iter()
                    .map(|sp| Span {
                        start: sp.start + s,
                        end: sp.end + s,
                    })
                    .collect();
                (abs_spans, rel_hashes)
            })
            .collect()
    });
    let total_spans: usize = parts.iter().map(|p| p.0.len()).sum();
    let mut spans: SmallVec<[Span; 64]> = SmallVec::with_capacity(total_spans);
    let mut hashes: SmallVec<[u64; 64]> = SmallVec::with_capacity(total_spans);
    for (sp, hs) in parts {
        spans.extend(sp);
        hashes.extend(hs);
    }
    (spans, hashes)
}

/// Find `~target` safe (pretoken-aligned) split points across `cats`,
/// including 0 and `n`. Returns sorted split offsets that partition `[0, n)`.
/// Public so `dstok-fast`'s streaming encoder can drive parallel streaming
/// encode over the same safe chunks.
pub fn safe_split_points(cats: &[Cat], bytes: &[u8], n: usize, target: usize) -> Vec<usize> {
    let mut splits = Vec::with_capacity(target + 1);
    splits.push(0);
    if target == 0 || n == 0 {
        splits.push(n);
        return splits;
    }
    let stride = n / target;
    let mut want = stride;
    let mut k = 1usize;
    while splits.len() < target && want < n {
        while k < n && !is_safe_split(cats, bytes, k) {
            k += 1;
        }
        if k >= n {
            break;
        }
        if k >= want {
            splits.push(k);
            want += stride;
        }
        k += 1;
    }
    splits.push(n);
    splits.sort_unstable();
    splits
}

/// Is splitting *before* index `k` (between `k-1` and `k`) safe? Safe iff
/// `cats[k-1] != cats[k]` and `k-1` is not an apostrophe starting a contraction
/// (that Apos + following letters form one pretoken).
#[inline]
fn is_safe_split(cats: &[Cat], bytes: &[u8], k: usize) -> bool {
    if k == 0 || k >= cats.len() {
        return false;
    }
    if cats[k - 1] == cats[k] {
        return false;
    }
    if cats[k - 1] == Cat::Apos && contraction_end(bytes, k - 1, bytes.len()).is_some() {
        return false;
    }
    true
}

/// O53: find safe (pretoken-aligned) split points roughly every `block` bytes
/// **without building the full `Cat` vector** — only the two boundary bytes at
/// each candidate split are scalar-classified, and `contraction_end` is checked
/// to avoid splitting inside a contraction. Returns boundaries from 0 to `n`;
/// every interior split `k` satisfies `classify_byte(bytes[k-1]) !=
/// classify_byte(bytes[k])` and `k-1` is not an apostrophe starting a
/// contraction, i.e. `k` is a true pretoken boundary. Each block is therefore
/// self-contained for `for_each_pretoken_in`, so per-block streaming output
/// concatenates to the whole-file result.
///
/// This is what lets block-streaming keep the `Cat` working set L2-sized
/// (one `block`-sized buffer per worker) instead of allocating an
/// input-sized `Vec<Cat>` — removing the cat-vector DRAM traffic that would
/// cap a 32 GB/s streaming path on large files. The split scan walks at most
/// one pretoken past each target, so total work is O(number of blocks ×
/// avg pretoken length) — negligible.
pub fn find_block_splits(bytes: &[u8], block: usize) -> Vec<usize> {
    let n = bytes.len();
    if n == 0 {
        return vec![0];
    }
    let block = block.max(1);
    let mut splits = vec![0usize];
    let mut target = block.min(n);
    while target < n {
        let mut k = target;
        while k < n {
            let prev = classify_byte(bytes[k - 1]);
            let cur = classify_byte(bytes[k]);
            if prev != cur && !(prev == Cat::Apos && contraction_end(bytes, k - 1, n).is_some()) {
                break;
            }
            k += 1;
        }
        let split = if k >= n { n } else { k };
        if split > *splits.last().unwrap() {
            splits.push(split);
        }
        // advance past this split; saturating add guards the loop exit
        target = split.saturating_add(block);
        if target >= n {
            break;
        }
    }
    if *splits.last().unwrap() != n {
        splits.push(n);
    }
    splits
}

/// Classify every byte of `bytes` into a category vector using the best
/// available SIMD tier (pub version for `dstok-fast`'s streaming encoder,
/// which classifies once then streams). Identical to the per-byte path used
/// by `pretokenize`.
pub fn classify_all_pub(bytes: &[u8]) -> Vec<Cat> {
    let mut out = Vec::with_capacity(bytes.len());
    classify_all_into(bytes, Feature::detect_cached(), &mut out);
    out
}

/// Classify every byte of `bytes` into a caller-supplied, reused buffer
/// (O53 block-streaming): `out` is cleared and refilled, keeping its capacity
/// so a per-worker L2-sized buffer is reused across blocks with zero
/// allocation. Identical output to `classify_all_pub`.
pub fn classify_all_into_pub(bytes: &[u8], out: &mut Vec<Cat>) {
    classify_all_into(bytes, Feature::detect_cached(), out);
}

/// Classify every byte of `bytes` into `out` (cleared first) using `feat`.
fn classify_all_into(bytes: &[u8], feat: Feature, out: &mut Vec<Cat>) {
    out.clear();
    out.reserve(bytes.len());
    let mut i = 0;
    match feat {
        #[cfg(target_arch = "x86_64")]
        Feature::Avx2 => {
            i = unsafe { x86::classify_avx2(bytes, out) };
        }
        #[cfg(target_arch = "x86_64")]
        Feature::Sse42 => {
            i = unsafe { x86::classify_sse42(bytes, out) };
        }
        #[cfg(target_arch = "aarch64")]
        Feature::Neon => {
            i = unsafe { arm::classify_neon(bytes, out) };
        }
        _ => {}
    }
    // Scalar tail (also the full path when no SIMD / unhandled tier).
    while i < bytes.len() {
        out.push(classify_byte(bytes[i]));
        i += 1;
    }
}

fn classify_all(bytes: &[u8], feat: Feature) -> Vec<Cat> {
    let mut out = Vec::with_capacity(bytes.len());
    classify_all_into(bytes, feat, &mut out);
    out
}

/// O47/O51: a **streaming** run-former. Mirrors `spans_from_cats_scalar`
/// exactly but invokes `emit(start, end)` per pretoken instead of building a
/// span vector — so a fused encoder can drive the cache lookup + id emission
/// directly, with no `SmallVec` span array and no second pass over it. The
/// indices are relative to the provided `cats`/`bytes` slice.
///
/// This is the 32 GB/s hot path: classify (parallel) → stream-form →
/// cache-lookup → emit ids, with only the 1-byte/byte `Cat` vector resident
/// (no per-span records). The caller computes the pretoken's hash from
/// `bytes[start..end]` (a few hot-L1 bytes) and calls `cache.get_hashed`.
pub fn for_each_pretoken_in(
    cats: &[Cat],
    bytes: &[u8],
    digit_run: usize,
    mut emit: impl FnMut(usize, usize),
) {
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        let start = i;
        let c = cats[i];
        match c {
            Cat::Ws => {
                while i < n && cats[i] == Cat::Ws {
                    i += 1;
                }
            }
            Cat::Letter => {
                while i < n && cats[i] == Cat::Letter {
                    i += 1;
                }
            }
            Cat::Digit => {
                // `\p{N}+` (GPT-2, digit_run == 0) or `\p{N}{1,n}` (Llama-3/GLM).
                // Capping at `digit_run` and looping again on the still-Digit
                // next byte emits `digit_run`-sized chunks, matching the scalar
                // oracle and the bulk `spans_from_cats` chunker.
                if digit_run > 0 {
                    let mut cnt = 0;
                    while i < n && cats[i] == Cat::Digit && cnt < digit_run {
                        i += 1;
                        cnt += 1;
                    }
                } else {
                    while i < n && cats[i] == Cat::Digit {
                        i += 1;
                    }
                }
            }
            Cat::Apos => {
                if let Some(end) = contraction_end(bytes, i, n) {
                    i = end;
                } else {
                    i += 1;
                }
            }
            Cat::Other => {
                while i < n && cats[i] == Cat::Other {
                    i += 1;
                }
            }
        }
        if i > start {
            emit(start, i);
        }
    }
}

/// Find the next pretoken span starting at `*i`, advancing `*i` past it. Shared
/// by `for_each_pretoken_in` and the hashed/prefetching variant so span logic
/// never drifts between them.
fn find_span(
    cats: &[Cat],
    bytes: &[u8],
    n: usize,
    i: &mut usize,
    digit_run: usize,
) -> Option<(usize, usize)> {
    if *i >= n {
        return None;
    }
    let start = *i;
    let c = cats[*i];
    match c {
        Cat::Ws => {
            while *i < n && cats[*i] == Cat::Ws {
                *i += 1;
            }
        }
        Cat::Letter => {
            while *i < n && cats[*i] == Cat::Letter {
                *i += 1;
            }
        }
        Cat::Digit => {
            if digit_run > 0 {
                let mut cnt = 0;
                while *i < n && cats[*i] == Cat::Digit && cnt < digit_run {
                    *i += 1;
                    cnt += 1;
                }
            } else {
                while *i < n && cats[*i] == Cat::Digit {
                    *i += 1;
                }
            }
        }
        Cat::Apos => {
            if let Some(end) = contraction_end(bytes, *i, n) {
                *i = end;
            } else {
                *i += 1;
            }
        }
        Cat::Other => {
            while *i < n && cats[*i] == Cat::Other {
                *i += 1;
            }
        }
    }
    if *i > start {
        Some((start, *i))
    } else {
        None
    }
}

/// O74: span iterator that also hands the caller each span's precomputed
/// `fxhash` (so `emit` skips re-hashing) **and** a one-span look-ahead
/// `(start, end, hash)` of the *next* span — letting the caller prefetch that
/// span's hot-table cache line while emitting the current one. The last span
/// receives `None` as its look-ahead. Produces the same spans as
/// `for_each_pretoken_in` (shared `find_span`).
pub fn for_each_pretoken_in_hashed(
    cats: &[Cat],
    bytes: &[u8],
    digit_run: usize,
    mut emit: impl FnMut(usize, usize, u64, Option<(usize, usize, u64)>),
) {
    let n = bytes.len();
    let mut i = 0;
    let mut cur = find_span(cats, bytes, n, &mut i, digit_run);
    while let Some((s, e)) = cur {
        let h = fxhash_bytes(&bytes[s..e]);
        let next = find_span(cats, bytes, n, &mut i, digit_run);
        let next_hashed = next.map(|(ns, ne)| (ns, ne, fxhash_bytes(&bytes[ns..ne])));
        emit(s, e, h, next_hashed);
        cur = next;
    }
}

/// Form spans from the per-byte category vector + the raw bytes. This is the
/// shared, kernel-agnostic run-forming logic. It MUST match the oracle's
/// pretoken regex semantics.
/// The scalar run-former. Kept as the **differential target** for the SIMD
/// `spans::spans_from_cats` (O2): property tests assert the two agree on
/// every input. Not on the hot path.
#[cfg(test)]
#[allow(dead_code)]
fn spans_from_cats_scalar(cats: &[Cat], bytes: &[u8], digit_run: usize) -> SmallVec<[Span; 64]> {
    let mut spans = SmallVec::new();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        let start = i;
        let c = cats[i];
        match c {
            Cat::Ws => {
                // The regex distinguishes `\s+(?!\S)` (trailing ws) from `\s+`,
                // but for span *boundaries* (which is all dstok needs — the BPE
                // merge is local to a pretoken), maximal whitespace runs are
                // correct: a trailing-ws run and a following non-ws run are
                // already separate spans. So: consume the whole ws run.
                while i < n && cats[i] == Cat::Ws {
                    i += 1;
                }
            }
            Cat::Letter => {
                while i < n && cats[i] == Cat::Letter {
                    i += 1;
                }
            }
            Cat::Digit => {
                // `\p{N}+` (GPT-2, digit_run == 0) or `\p{N}{1,n}` (Llama-3/GLM).
                if digit_run > 0 {
                    let mut cnt = 0;
                    while i < n && cats[i] == Cat::Digit && cnt < digit_run {
                        i += 1;
                        cnt += 1;
                    }
                } else {
                    while i < n && cats[i] == Cat::Digit {
                        i += 1;
                    }
                }
            }
            Cat::Apos => {
                // Apostrophe: the oracle attempts a contraction match
                // ('s 't 're 've 'll 'm 'd). If no contraction, the apostrophe
                // is a standalone "other" pretoken. We replicate by checking
                // the bytes ahead.
                if let Some(end) = contraction_end(bytes, i, n) {
                    i = end;
                } else {
                    i += 1; // standalone apostrophe
                }
            }
            Cat::Other => {
                while i < n && cats[i] == Cat::Other {
                    i += 1;
                }
            }
        }
        if i > start {
            spans.push(Span { start, end: i });
        }
    }
    spans
}

/// Mirror of `dstok_core::regex::contraction_len`: returns the end offset
/// (exclusive) of a contraction starting at apostrophe `i`, or None.
pub(crate) fn contraction_end(bytes: &[u8], i: usize, n: usize) -> Option<usize> {
    if i + 1 >= n {
        return None;
    }
    let b1 = bytes[i + 1].to_ascii_lowercase();
    // Two-letter suffixes: 're 've 'll
    let two_form = |suf: &[u8; 2]| -> bool {
        i + 3 <= n
            && bytes[i + 1].to_ascii_lowercase() == suf[0]
            && bytes[i + 2].to_ascii_lowercase() == suf[1]
    };
    if two_form(b"re") || two_form(b"ve") || two_form(b"ll") {
        return Some(i + 3);
    }
    // One-letter suffixes: 's 't 'm 'd
    if b1 == b's' || b1 == b't' || b1 == b'm' || b1 == b'd' {
        return Some(i + 2);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(len: usize) -> Vec<u8> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index < 256 {
                    index as u8
                } else {
                    state as u8
                }
            })
            .collect()
    }

    fn assert_tier_matches_scalar(bytes: &[u8], feature: Feature) {
        let expected = classify_all(bytes, Feature::Scalar);
        let actual = classify_all(bytes, feature);
        assert_eq!(actual, expected, "{feature:?} mismatch");
    }

    #[test]
    fn detected_tier_matches_scalar_for_all_bytes_and_vector_tails() {
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 255, 256, 257, 4097] {
            let bytes = corpus(len);
            assert_tier_matches_scalar(&bytes, Feature::detect());
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_available_x86_tier_matches_scalar() {
        let bytes = corpus(8193);
        if std::is_x86_feature_detected!("sse4.2") {
            assert_tier_matches_scalar(&bytes, Feature::Sse42);
        }
        if std::is_x86_feature_detected!("avx2") {
            assert_tier_matches_scalar(&bytes, Feature::Avx2);
            let mut actual = Vec::new();
            // SAFETY: guarded by runtime AVX2 detection.
            let consumed =
                unsafe { lut::classify_avx2_lut(&bytes, &mut actual, &lut::pack_pages()) };
            actual.extend(bytes[consumed..].iter().copied().map(classify_byte));
            assert_eq!(actual, classify_all(&bytes, Feature::Scalar));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar() {
        assert_tier_matches_scalar(&corpus(8193), Feature::Neon);
    }
}
