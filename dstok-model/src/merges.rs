//! The merge table: adjacent (id_a, id_b) → (rank, result_id).
//! Lower rank = merged first. `result_id` is the token id the pair merges into.
//!
//! Stored as a hashmap from the packed pair key `(a as u64) | ((b as u64) <<
//! 32)` to `(rank, result_id)`. Lookups are the inner loop of the BPE merge;
//! we keep the map flat and the key a single u64 so hashing is one word.
//! O4-note: id-pair lookup means the merge loop never reconstructs symbol
//! bytes — it works entirely in id space, which is far cheaper.

use std::collections::HashMap;

/// The merge rules for a BPE artifact.
#[derive(Debug, Clone, Default)]
pub struct Merges {
    /// packed (a,b) -> (rank, result_id)
    map: HashMap<u64, (u32, crate::TokenId)>,
}

impl Merges {
    /// Build from `(a, b, result_id)` triples in rank order (rank = index).
    pub fn from_triples<
        I: IntoIterator<Item = (crate::TokenId, crate::TokenId, crate::TokenId)>,
    >(
        iter: I,
    ) -> Self {
        let mut m = Merges::default();
        for (rank, (a, b, c)) in iter.into_iter().enumerate() {
            m.map.insert(pack(a, b), (rank as u32, c));
        }
        m
    }

    /// Build from `(a, b)` pairs in rank order; result_id is inferred from the
    /// vocab at load time by the loader (this convenience leaves it as `a`'s
    /// fallback and should be patched via `set_result`).
    pub fn from_pairs<I: IntoIterator<Item = (crate::TokenId, crate::TokenId)>>(iter: I) -> Self {
        let mut m = Merges::default();
        for (rank, (a, b)) in iter.into_iter().enumerate() {
            m.map.insert(pack(a, b), (rank as u32, a)); // placeholder; loader sets real result
        }
        m
    }

    /// Set the result_id for an existing (a,b) rule (used by the loader once
    /// it has resolved the merged symbol's id).
    pub fn set_result(&mut self, a: crate::TokenId, b: crate::TokenId, result: crate::TokenId) {
        if let Some(slot) = self.map.get_mut(&pack(a, b)) {
            slot.1 = result;
        }
    }

    /// The rank of merging the adjacent pair (a, b), or None if no merge rule.
    #[inline]
    pub fn rank(&self, a: crate::TokenId, b: crate::TokenId) -> Option<u32> {
        self.map.get(&pack(a, b)).map(|(r, _)| *r)
    }

    /// The result token id of merging (a, b), or None if no merge rule.
    #[inline]
    pub fn result_of(&self, a: crate::TokenId, b: crate::TokenId) -> Option<crate::TokenId> {
        self.map.get(&pack(a, b)).map(|(_, c)| *c)
    }

    /// The (rank, result_id) of merging (a, b), or None if no merge rule.
    #[inline]
    pub fn rule(&self, a: crate::TokenId, b: crate::TokenId) -> Option<(u32, crate::TokenId)> {
        self.map.get(&pack(a, b)).copied()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[inline]
fn pack(a: crate::TokenId, b: crate::TokenId) -> u64 {
    (a as u64) | ((b as u64) << 32)
}
