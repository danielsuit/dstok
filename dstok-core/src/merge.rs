//! The BPE merge — the oracle algorithm, factored so `dstok-fast`'s cache
//! reuses the *exact* oracle on a cache miss (bit-identical by construction:
//! `bpe_merge_pretoken` is literally `seed_word` + `bpe_merge`).
//!
//! Algorithm (textbook byte-level BPE, matching `dstok_core::encode`):
//!   1. `seed_word`: byte-map each pretoken byte through the GPT-2
//!      `byte_to_unicode` table and seed one token per byte using the
//!      direct-indexed `id_of_byte` table.
//!   2. `bpe_merge`: repeatedly find the adjacent pair with the **minimum
//!      merge rank** and merge **all** its occurrences in one left-to-right
//!      pass, recompute, repeat until no mergeable pair remains.
//!   3. The surviving symbols' ids (in order) are the pretoken's tokens;
//!      each token's byte length is the sum of its constituent input bytes
//!      (tracked in `byte_lens` for offset reconstruction).
//!
//! The "merge all occurrences of the minimal pair per pass" is the canonical
//! BPE definition; it is what makes the output independent of merge order,
//! which is what makes `dstok-fast`'s pair-heap merge (O7) able to match it.

use dstok_model::{Artifact, TokenId};
use smallvec::SmallVec;

type Word = SmallVec<[TokenId; 32]>;
type Lens = SmallVec<[usize; 32]>;

/// Seed the BPE word: one token per pretoken byte. Each byte is mapped
/// through `byte_to_unicode` and looked up in the vocab (array-indexed for
/// single-byte symbols; hashmap fallback for the rare multibyte-mapped
/// char). Returns the seed word and the per-symbol byte lengths.
pub fn seed_word(artifact: &Artifact, span_bytes: &[u8]) -> (Word, Lens) {
    let table = dstok_model::Vocab::byte_to_unicode_table();
    let mut word: Word = SmallVec::with_capacity(span_bytes.len());
    let mut byte_lens: Lens = SmallVec::with_capacity(span_bytes.len());
    let mut buf = [0u8; 4];
    for &b in span_bytes {
        let ch = table[b as usize];
        let id = artifact.vocab.id_of_byte(b).unwrap_or_else(|| {
            let mapped = ch.encode_utf8(&mut buf);
            let mut s = dstok_model::vocab::Symbol::new();
            for &c in mapped.as_bytes() {
                s.push(c);
            }
            artifact.vocab.id_of(&s).unwrap_or(0)
        });
        word.push(id);
        byte_lens.push(1);
    }
    (word, byte_lens)
}

/// The oracle merge: mutate `word`/`byte_lens` in place by repeatedly merging
/// the minimal-rank adjacent pair (all occurrences per pass) until none remain.
pub fn bpe_merge(artifact: &Artifact, word: &mut Word, byte_lens: &mut Lens) {
    loop {
        // Find the adjacent pair with the smallest rank.
        let mut best_rank: u32 = u32::MAX;
        let mut best_pair: Option<(TokenId, TokenId)> = None;
        for i in 0..word.len().saturating_sub(1) {
            let rank = match artifact.merges.rank(word[i], word[i + 1]) {
                Some(r) => r,
                None => continue,
            };
            if rank < best_rank {
                best_rank = rank;
                best_pair = Some((word[i], word[i + 1]));
            }
        }
        let Some((a, b)) = best_pair else { break };
        let result = merge_lookup(artifact, a, b);
        merge_all(word, byte_lens, a, b, result);
    }
}

/// Look up the result token id of merging adjacent (a, b). Used by the oracle
/// and by `dstok-fast`'s pair-heap merge (O7) so both consult the same table.
#[inline]
pub fn merge_lookup(artifact: &Artifact, a: TokenId, b: TokenId) -> TokenId {
    artifact.merges.result_of(a, b).unwrap_or(a)
}

/// Merge every adjacent occurrence of the pair `(a, b)` into `result` in one
/// left-to-right pass. After merging a pair, the new symbol cannot pair with
/// the next symbol in the same pass (its right neighbor changed), so we skip
/// past it — the standard left-to-right "merge all occurrences" pass.
pub fn merge_all(word: &mut Word, byte_lens: &mut Lens, a: TokenId, b: TokenId, result: TokenId) {
    let mut i = 0;
    while i + 1 < word.len() {
        if word[i] == a && word[i + 1] == b {
            // Merge i and i+1: result takes the combined byte length.
            let combined = byte_lens[i] + byte_lens[i + 1];
            word[i] = result;
            byte_lens[i] = combined;
            word.remove(i + 1);
            byte_lens.remove(i + 1);
            // Skip past the merged symbol; its new right neighbor is fresh.
            i += 1;
        } else {
            i += 1;
        }
    }
}

/// Combined seed + merge for one pretoken — the function `dstok-fast`'s cache
/// calls on a miss. Identical to `seed_word` + `bpe_merge` by construction,
/// except it honors `ignore_merges` first (GLM): when set, a pretoken whose
/// byte-mapped symbol is already a whole vocab token is returned verbatim
/// with no merge applied — matching HF `model.ignore_merges`.
pub fn bpe_merge_pretoken(
    artifact: &Artifact,
    pretoken: &[u8],
) -> (SmallVec<[TokenId; 32]>, SmallVec<[usize; 32]>) {
    if artifact.config.ignore_merges {
        if let Some(id) = whole_word_token(artifact, pretoken) {
            return (smallvec::smallvec![id], smallvec::smallvec![pretoken.len()]);
        }
    }
    let (mut word, mut byte_lens) = seed_word(artifact, pretoken);
    bpe_merge(artifact, &mut word, &mut byte_lens);
    (word, byte_lens)
}

/// `ignore_merges` short-circuit: byte-map `pretoken` through the GPT-2
/// `byte_to_unicode` table (exactly as `seed_word` does) and look the resulting
/// symbol up in the vocab. If it is already a whole token, return its id —
/// BPE merges are skipped for this pretoken.
pub fn whole_word_token(artifact: &Artifact, pretoken: &[u8]) -> Option<TokenId> {
    let table = dstok_model::Vocab::byte_to_unicode_table();
    let mut sym = dstok_model::vocab::Symbol::new();
    let mut buf = [0u8; 4];
    for &b in pretoken {
        let ch = table[b as usize];
        for &c in ch.encode_utf8(&mut buf).as_bytes() {
            sym.push(c);
        }
    }
    artifact.vocab.id_of(&sym)
}
