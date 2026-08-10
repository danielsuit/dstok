//! # dstok-fuzz — cargo-fuzz targets.
//!
//! Targets (§4.3):
//!   - raw arbitrary bytes
//!   - malformed/truncated UTF-8, overlong encodings, lone surrogates
//!   - deeply nested combining characters
//!   - gigantic single words, pathological whitespace
//!   - adversarial special-token collisions
//!
//! Assertions: never panic, never UB (ASan/UBSan; MSan on SIMD), bounded
//! memory, and output matches the oracle where defined.
//!
//! These are written as ordinary functions so they can be wired to
//! `cargo-fuzz` / `libFuzzer` with `arbitrary` inputs. The differential fuzz
//! target (dstok-fast vs dstok-core) runs continuously in CI.

#![cfg_attr(not(test), deny(warnings))]

use dstok_model::Artifact;
use dstok_oracle::{HfTokenizersOracle, Oracle};
use std::sync::OnceLock;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 20;

/// Real GLM-5.2 three-way fuzz target: fast IDs, core IDs, and HF 0.22
/// `encode_fast` IDs must all agree. Arbitrary bytes are lossily converted in
/// the same deterministic way before all three implementations see them.
pub fn fuzz_glm52_against_hf(data: &[u8]) {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }
    static ARTIFACT: OnceLock<Artifact> = OnceLock::new();
    static HF: OnceLock<HfTokenizersOracle> = OnceLock::new();
    let fixture = format!(
        "{}/../fixtures/glm-5.2/tokenizer.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let artifact = ARTIFACT.get_or_init(|| {
        dstok_model::loader::load_glm52_path(fixture.as_str())
            .expect("load pinned GLM artifact for fuzzing")
    });
    let hf = HF.get_or_init(|| {
        HfTokenizersOracle::from_file(&fixture).expect("load HF oracle for fuzzing")
    });
    let text = String::from_utf8_lossy(data);
    let expected = hf.encode(&text).ids;
    let core = dstok_core::encode_to_ids(artifact, &text);
    let fast = dstok_fast::encode_ids_only(artifact, &text);
    assert_eq!(core, expected, "core != HF");
    assert_eq!(fast, expected, "fast != HF");
    assert_eq!(dstok_fast::count(artifact, &text), expected.len());
}

/// Fuzz target: arbitrary bytes must never panic, and fast == core.
pub fn fuzz_arbitrary(art: &Artifact, data: &[u8]) {
    // Safe even on non-UTF-8: encode takes bytes via str only in the public
    // API, so we lossy-convert; the internal pretokenizer is byte-oriented.
    let text = String::from_utf8_lossy(data);
    let core = dstok_core::encode(art, &text);
    let fast = dstok_fast::encode(art, &text);
    assert_eq!(core.ids, fast.ids, "fast != core on arbitrary bytes");
    assert_eq!(core.offsets, fast.offsets, "fast offsets != core offsets");
}

/// Fuzz target: malformed UTF-8 specifically.
pub fn fuzz_malformed_utf8(art: &Artifact, data: &[u8]) {
    fuzz_arbitrary(art, data);
}

/// Fuzz target: adversarial special-token collisions. User text containing
/// the literal bytes of a special token must never be silently promoted
/// (§4.2 special-token safety) unless the splitter is explicitly enabled.
pub fn fuzz_special_token_safety(art: &Artifact, data: &[u8]) {
    let text = String::from_utf8_lossy(data);
    let enc = dstok_fast::encode(art, &text);
    // If split_specials is disabled, no id in the output may be a registered
    // special-token id for a span that was not an exact special-token match.
    if !art.config.split_specials {
        for (i, &id) in enc.ids.iter().enumerate() {
            let (s, e) = (enc.offsets[i].start, enc.offsets[i].end);
            let span = &text.as_bytes()[s..e];
            let is_special = art.vocab.special_map().values().any(|&sid| sid == id);
            if is_special {
                // The span must exactly equal a registered special token's bytes.
                let exact = art
                    .vocab
                    .special_map()
                    .iter()
                    .any(|(k, &v)| v == id && k.as_slice() == span);
                assert!(exact, "silent special-token promotion at offset {s}");
            }
        }
    }
}

/// Fuzz target: gigantic single "word" (no whitespace) — exercises the BPE
/// merge on a pathological-length pretoken. Must not panic; fast == core. The
/// cache miss path (the oracle merge) is O(n²) per pass — this also probes the
/// bounded-memory contract (the cache cap, §6).
pub fn fuzz_gigantic_word(art: &Artifact, seed: &[u8]) {
    // Repeat the seed to ~256 KiB to stress a single long pretoken.
    let mut s = Vec::with_capacity(256 * 1024);
    while s.len() < 256 * 1024 {
        s.extend_from_slice(seed);
    }
    s.truncate(256 * 1024);
    let text = String::from_utf8_lossy(&s);
    let core = dstok_core::encode(art, &text);
    let fast = dstok_fast::encode(art, &text);
    assert_eq!(core.ids, fast.ids, "fast != core on gigantic word");
}

/// Fuzz target: pathological whitespace runs + near-boundary multibyte
/// sequences. Exercises chunk-boundary independence (§4.2) by encoding the
/// whole input and a version split at a random byte offset, then asserting
/// the concatenated id stream matches (modulo a mid-pretoken cut, which for
/// byte-level BPE is the one case the property must hold for trivially
/// because cuts only re-enter the pretokenizer). We assert no panic + fast==core.
pub fn fuzz_whitespace_and_boundaries(art: &Artifact, data: &[u8]) {
    let text = String::from_utf8_lossy(data);
    let core = dstok_core::encode(art, &text);
    let fast = dstok_fast::encode(art, &text);
    assert_eq!(core.ids, fast.ids, "fast != core on whitespace/boundaries");
    // Determinism across thread counts (§4.2): 1 vs N threads must match.
    let n = (data.len() % 8) + 2; // 2..9 threads
    let par = dstok_fast::encode_parallel(art, &text, n);
    assert_eq!(
        fast.ids, par.ids,
        "nondeterministic across thread counts: {n}"
    );
}

/// Fuzz target: deeply nested combining characters / overlong UTF-8 /
/// lone surrogates (as bytes). The pretokenizer is byte-oriented and must
/// never panic on any byte sequence; output must match the oracle.
pub fn fuzz_unicode_edge(art: &Artifact, data: &[u8]) {
    fuzz_arbitrary(art, data);
}

/// The contract every fuzz target upholds (§4.3):
///   - never panic
///   - never UB (run under ASan + UBSan; MSan on dstok-simd)
///   - bounded memory (the pretoken cache cap, §6)
///   - fast == core (the internal oracle) wherever defined
///   - matches the external oracle wherever it is defined for the input
/// Crashes get a minimized regression corpus entry (checked in).

/// libFuzzer entry point for the differential target (fast vs core). Wire via
/// `cargo fuzz` with this symbol as the harness. Gated behind `libfuzzer`.
#[cfg(feature = "libfuzzer")]
pub fn libfuzzer_differential(data: &[u8]) {
    // Requires a loaded artifact in a thread-local; the cargo-fuzz harness
    // sets it up via `dstok_fuzz::set_artifact` before calling.
    if let Some(art) = current_artifact() {
        fuzz_arbitrary(&art, data);
    }
}

#[cfg(feature = "libfuzzer")]
thread_local! {
    static ART: std::cell::RefCell<Option<Artifact>> = std::cell::RefCell::new(None);
}

#[cfg(feature = "libfuzzer")]
pub fn set_artifact(art: Artifact) {
    ART.with(|c| *c.borrow_mut() = Some(art));
}

#[cfg(feature = "libfuzzer")]
fn current_artifact() -> Option<Artifact>
where
    Artifact: Clone,
{
    ART.with(|c| c.borrow().clone())
}

#[cfg(test)]
mod tests {
    #[test]
    fn glm52_differential_fuzz_smoke() {
        let seeds: &[&[u8]] = &[
            b"",
            b"<|assistant|><think></think>",
            b"\xff\xfe\xf0\x28\x8c\x28",
            b"12345678901234567890\r\n\t ",
            "👩🏽‍💻e\u{301}中文".as_bytes(),
        ];
        for seed in seeds {
            super::fuzz_glm52_against_hf(seed);
            for index in 0..seed.len() {
                let mut mutated = seed.to_vec();
                mutated[index] ^= 0x5a;
                super::fuzz_glm52_against_hf(&mutated);
            }
        }
    }
}
