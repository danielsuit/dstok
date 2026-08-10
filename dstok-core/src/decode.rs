//! Decode: token-id sequence → original bytes.
//!
//! The oracle decode. The heavy lifting (inverse `byte_to_unicode` map, the
//! flat id→decoded-bytes table) lives in `dstok_model::vocab` and is built once
//! at load (`Vocab::finalize_decode_table`); this module is the thin caller so
//! both the oracle and `dstok-fast` share one decode implementation. That makes
//! `decode(encode(x)) == x` a single code path to
//! test, not two implementations to keep in sync.

use dstok_model::{Artifact, TokenId};

/// Decode a token-id sequence to the original bytes.
pub fn decode(artifact: &Artifact, ids: &[TokenId]) -> Vec<u8> {
    let mut out = Vec::new();
    decode_into(artifact, ids, &mut out);
    out
}

/// Decode into a caller-provided buffer in a single pass. The plain
/// `decode` double-reads `decode_id` (once to sum lengths for the exact
/// reservation, once to copy); this skips the summing pass and reserves a
/// cheap heuristic (`ids.len() * 4` — most tokens are ≤4 bytes), letting the
/// Vec grow amortized. Faster for large id streams, same output.
pub fn decode_into(artifact: &Artifact, ids: &[TokenId], out: &mut Vec<u8>) {
    out.reserve(ids.len() * 4);
    for &id in ids {
        out.extend_from_slice(artifact.vocab.decode_id(id));
    }
}

/// Decode an `Encoding` (its `ids` field) — convenience so callers don't pull
/// `.ids` out by hand.
pub fn decode_encoding(artifact: &Artifact, enc: &crate::Encoding) -> Vec<u8> {
    decode(artifact, &enc.ids)
}

/// Decode to a `String`, replacing invalid UTF-8 with the replacement char.
/// Most token streams decode to valid UTF-8 (they came from text); special
/// tokens decode to their literal text.
pub fn decode_to_string(artifact: &Artifact, ids: &[TokenId]) -> String {
    let bytes = decode(artifact, ids);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Decode to a `String` reusing the caller buffer (avoids a second
/// allocation when the caller already owns a `String`/`Vec`).
pub fn decode_to_string_into(artifact: &Artifact, ids: &[TokenId], out: &mut String) {
    let mut buf = Vec::new();
    decode_into(artifact, ids, &mut buf);
    out.push_str(&String::from_utf8_lossy(&buf));
}
