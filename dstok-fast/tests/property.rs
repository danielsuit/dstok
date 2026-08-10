//! Property-based tests for the correctness contract.
//!
//! All properties hold on *any* input, independent of any oracle (except the
//! `fast_eq_oracle` differential). They run against
//! a **self-contained synthetic byte-level BPE artifact** built in `synth()`
//! (no external tokenizer files, no network) so the property suite is hermetic.
//!
//! The synthetic artifact is a real byte-level BPE tokenizer: the 256
//! GPT-2 `byte_to_unicode` single-byte tokens (ids 0..255) plus a few merges
//! (" t", "he", " the", " a"). It exercises the *actual* `dstok_core` oracle,
//! `dstok_fast` SIMD+cache path, pretokenizer, merge, decode, and offset
//! machinery — so a passing suite means the real invariants hold.
//!
//! Not run here (per project constraint); written to compile against the
//! public API and to fail loudly if an invariant is violated.

use proptest::prelude::*;

use dstok_core as core;
use dstok_fast as fast;
use dstok_model::config::TokenizerConfig;
use dstok_model::family::ModelId;
use dstok_model::merges::Merges;
use dstok_model::vocab::{Symbol, Vocab};
use dstok_model::{Artifact, ArtifactFormat, Provenance};

/// Build a hermetic byte-level BPE artifact (GPT-2 family, no specials):
///   ids 0..255 : the 256 single-byte `byte_to_unicode` tokens (byte b → id b)
///   256 = " t"   merge (32, 116) → 256   (rank 1)
///   257 = "he"   merge (104, 101) → 257  (rank 0)
///   258 = " the" merge (256, 257) → 258  (rank 2)
///   259 = " a"   merge (32, 97) → 259    (rank 3)
/// `finalize_byte_table` + `finalize_decode_table` are run, exactly as the
/// loader does, so the O3 seed and O76 decode fast paths are exercised.
fn synth() -> Artifact {
    let table = Vocab::byte_to_unicode_table();

    // Map a raw byte string to its byte-mapped symbol (UTF-8 of each mapped char).
    let mapped = |raw: &[u8]| -> Symbol {
        let mut sym: Symbol = Symbol::new();
        let mut buf = [0u8; 4];
        for &b in raw {
            let ch = table[b as usize];
            for &c in ch.encode_utf8(&mut buf).as_bytes() {
                sym.push(c);
            }
        }
        sym
    };

    let mut vocab = Vocab::default();
    // 256 single-byte tokens, in byte order → id == byte.
    for b in 0u8..=255 {
        vocab.add(mapped(&[b]), false);
    }
    // Merged tokens (added in this order → these ids).
    vocab.add(mapped(b" t"), false); // 256
    vocab.add(mapped(b"he"), false); // 257
    vocab.add(mapped(b" the"), false); // 258
    vocab.add(mapped(b" a"), false); // 259

    // Merges in rank order (rank == index). Lower rank = applied first.
    // (h,e)→"he" first so " the" forms as (" t")+("he").
    let merges = Merges::from_triples([
        (104u32, 101, 257), // (h,e)→"he"   rank 0
        (32, 116, 256),     // (sp,t)→" t"  rank 1
        (256, 257, 258),    // (" t","he")→" the" rank 2
        (32, 97, 259),      // (sp,a)→" a"  rank 3
    ]);

    let config = TokenizerConfig {
        family: ModelId::Gpt2,
        ..TokenizerConfig::default()
    };

    let mut art = Artifact {
        vocab,
        merges,
        config,
        provenance: Provenance {
            format: ArtifactFormat::HfTokenizersJson,
            format_version: "synth".to_string(),
            blake3: [0; 32],
            source_path: None,
        },
    };
    // Mirror the loader: finalize the direct-indexed tables before any use.
    art.vocab.finalize_byte_table();
    art.vocab.finalize_decode_table();
    art
}

// ---------------------------------------------------------------------------
// Differential: dstok-fast == dstok-core (the internal oracle) on every
// input. THE central correctness invariant — the fast path must produce
// bit-identical ids to the oracle.
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn fast_eq_oracle(s in any::<String>()) {
        let art = synth();
        let fast_ids = fast::encode(&art, &s).ids;
        let oracle_ids = core::encode(&art, &s).ids;
        prop_assert_eq!(fast_ids, oracle_ids);
        prop_assert_eq!(fast::count(&art, &s), core::count(&art, &s));
    }

    // Offsets too: the differential covers ids and offsets.
    #[test]
    fn fast_eq_oracle_offsets(s in any::<String>()) {
        let art = synth();
        let f = fast::encode(&art, &s);
        let o = core::encode(&art, &s);
        prop_assert_eq!(f.offsets, o.offsets);
    }
}

// ---------------------------------------------------------------------------
// Determinism + chunk-boundary independence: the parallel encoder (which
// snaps cuts to pretoken boundaries) is bit-identical to the serial encoder
// for any thread count. Thread count never affects the token stream.
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn determinism(s in any::<String>()) {
        let art = synth();
        let a = fast::encode(&art, &s).ids;
        let b = fast::encode(&art, &s).ids;
        prop_assert_eq!(a, b);
    }

    #[test]
    fn parallel_eq_serial(s in any::<String>()) {
        let art = synth();
        let serial = fast::encode(&art, &s).ids;
        for n in [1usize, 2, 4, 8] {
            let par = fast::encode_parallel(&art, &s, n).ids;
            prop_assert!(par == serial, "thread count {} changed output", n);
        }
    }

    // The block-streaming 32 GB/s path must also match the serial path.
    #[test]
    fn block_stream_eq_serial(s in any::<String>()) {
        let art = synth();
        let serial = fast::encode(&art, &s).ids;
        let bs = fast::encode_ids_block_stream(&art, s.as_bytes(), 4);
        prop_assert_eq!(bs, serial);
    }
}

// ---------------------------------------------------------------------------
// Roundtrip: byte-level BPE is lossless (every byte has a token, no
// normalization drops), so decode(encode(x)) == x for ALL inputs.
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn roundtrip_lossless(s in any::<String>()) {
        let art = synth();
        let ids = fast::encode(&art, &s).ids;
        let decoded = core::decode(&art, &ids);
        prop_assert_eq!(decoded, s.as_bytes());
    }

    // The single-pass `decode_into` (O80) must agree with `decode`.
    #[test]
    fn decode_into_eq_decode(s in any::<String>()) {
        let art = synth();
        let ids = fast::encode(&art, &s).ids;
        let mut buf = Vec::new();
        core::decode_into(&art, &ids, &mut buf);
        prop_assert_eq!(buf, core::decode(&art, &ids));
    }
}

// ---------------------------------------------------------------------------
// Offset coverage: offsets are monotonic, non-overlapping, and exactly
// cover the input bytes (byte-level BPE has no normalization drops).
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn offset_coverage(s in any::<String>()) {
        let art = synth();
        let enc = fast::encode(&art, &s);
        let n = s.len();
        let mut prev_end: usize = 0;
        prop_assert!(enc.offsets.len() == enc.ids.len(), "offsets/ids length mismatch");
        for o in enc.offsets.iter() {
            prop_assert!(o.start == prev_end, "offset not contiguous: prev_end={} start={}", prev_end, o.start);
            prop_assert!(o.end >= o.start, "offset inverted");
            prop_assert!(o.end <= n, "offset past input end");
            prev_end = o.end;
        }
        prop_assert!(prev_end == n, "offsets do not cover the whole input");
    }
}

// ---------------------------------------------------------------------------
// IDs-only equivalence: the allocation-reduced path must match full encoding,
// including every token ID.
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn ids_only_eq_encode(s in any::<String>()) {
        let art = synth();
        let full = fast::encode(&art, &s).ids;
        let only = fast::encode_ids_only(&art, &s);
        prop_assert_eq!(only, full);
    }
}

// ---------------------------------------------------------------------------
// A focused unit test with a known answer (non-randomized) so a single
// regression is immediately legible: " the" → [258], " a" → [259], "x" → [120].
// ---------------------------------------------------------------------------
#[test]
fn known_answers() {
    let art = synth();
    assert_eq!(fast::encode(&art, " the").ids, vec![258u32], " the → [258]");
    assert_eq!(fast::encode(&art, "a").ids, vec![97u32], "a → [97]");
    assert_eq!(
        core::encode(&art, " the").ids,
        vec![258u32],
        "oracle agrees on the"
    );
    // roundtrip a non-trivial string
    let s = " the a he";
    let ids = fast::encode(&art, s).ids;
    assert_eq!(core::decode(&art, &ids), s.as_bytes(), "roundtrip");
}
