//! Byte→symbol mapping (the oracle version).
//!
//! Maps raw input bytes to the GPT-2 `byte_to_unicode` symbol bytes. The
//! oracle builds the table per call; correctness, not speed. The fast path
//! caches the table in a thread-local.

use smallvec::SmallVec;

/// Map a byte slice to its symbol-byte representation.
pub fn bytes_to_symbols(bytes: &[u8]) -> SmallVec<[u8; 32]> {
    let t = dstok_model::vocab::Vocab::byte_to_unicode_table();
    let mut out = SmallVec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        let mut buf = [0u8; 4];
        let s = t[b as usize].encode_utf8(&mut buf);
        out.extend_from_slice(s.as_bytes());
    }
    out
}
