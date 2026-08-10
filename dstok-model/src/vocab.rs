//! The vocab: symbol-bytes ↔ token-id, plus the byte_to_unicode table.

use smallvec::SmallVec;

/// A token's symbol bytes (the byte-mapped representation, or the literal
/// special-token bytes). Stored as a small vector for cache locality.
pub type Symbol = SmallVec<[u8; 16]>;

/// The full vocabulary: forward (symbol → id) and reverse (id → symbol),
/// plus the registered special tokens.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// symbol bytes → id
    fwd: std::collections::HashMap<Symbol, crate::TokenId>,
    /// id → symbol bytes (dense; index == id)
    rev: Vec<Symbol>,
    /// special-token symbol bytes → flagged id. A separate map so ordinary
    /// text that *contains* a special token's bytes is not promoted unless
    /// the splitter is enabled.
    specials: std::collections::HashMap<Symbol, crate::TokenId>,
    /// Every HF added token, including non-special entries. Added tokens are
    /// extracted before pre-tokenization; `special` only controls decoding
    /// skip semantics in HF.
    added: std::collections::HashMap<Symbol, crate::TokenId>,
    /// Direct-indexed `byte → id` table for the 256 single-byte symbols.
    /// Computed once at load (`finalize_byte_table`), so the BPE seed loop
    /// (the hottest part of a cache miss) is a flat array index instead of a
    /// hashmap lookup per byte. `None` entries mean that byte has no
    /// single-token id (the common case for non-mapped bytes).
    byte_to_id: [Option<crate::TokenId>; 256],
    /// Flat `id → decoded bytes` table for fast decode. Built once by
    /// `finalize_decode_table`. `decode_bytes[id]` is the original input
    /// bytes that token represents: inverse-`byte_to_unicode`-mapped for BPE
    /// tokens, literal for special tokens. Decode is then one
    /// `extend_from_slice` per id — no per-token UTF-8 decode or hashmap probe.
    decode_bytes: Vec<Box<[u8]>>,
}

// Manual `Default`: `#[derive(Default)]` can't handle `byte_to_id`, because
// `[T; 256]: Default` is not implemented in std (array `Default` only covers
// lengths up to 32). `[None; 256]` is constructible directly (Copy element).
impl Default for Vocab {
    fn default() -> Self {
        Self {
            fwd: Default::default(),
            rev: Default::default(),
            specials: Default::default(),
            added: Default::default(),
            byte_to_id: [None; 256],
            decode_bytes: Default::default(),
        }
    }
}

impl Vocab {
    /// Build a vocab from an iterator of (symbol bytes, is_special).
    pub fn from_entries<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Symbol, bool)>,
    {
        let mut v = Vocab::default();
        for (sym, special) in iter {
            v.add(sym, special);
        }
        v
    }

    /// Insert a token. Ids are assigned densely in insertion order.
    pub fn add(&mut self, sym: Symbol, special: bool) -> crate::TokenId {
        if let Some(&id) = self.fwd.get(&sym) {
            return id;
        }
        let id = self.rev.len() as crate::TokenId;
        self.fwd.insert(sym.clone(), id);
        self.rev.push(sym.clone());
        if special {
            self.added.insert(sym.clone(), id);
            self.specials.insert(sym, id);
        }
        id
    }

    /// Insert an HF added token at the next dense id.
    pub fn add_added(&mut self, sym: Symbol, special: bool) -> crate::TokenId {
        let id = self.add(sym.clone(), false);
        self.added.insert(sym.clone(), id);
        if special {
            self.specials.insert(sym, id);
        }
        id
    }

    /// Strict loader insertion preserving the artifact's numeric ids.
    pub fn insert_at(
        &mut self,
        id: crate::TokenId,
        sym: Symbol,
        added: bool,
        special: bool,
    ) -> Result<(), String> {
        if id as usize != self.rev.len() {
            return Err(format!(
                "non-dense token id {id}; expected {}",
                self.rev.len()
            ));
        }
        if let Some(existing) = self.fwd.get(&sym) {
            return Err(format!("duplicate token symbol at ids {existing} and {id}"));
        }
        self.fwd.insert(sym.clone(), id);
        self.rev.push(sym.clone());
        if added {
            self.added.insert(sym.clone(), id);
        }
        if special {
            self.specials.insert(sym, id);
        }
        Ok(())
    }

    pub fn id_of(&self, sym: &Symbol) -> Option<crate::TokenId> {
        self.fwd.get(sym).copied()
    }

    pub fn token_of(&self, id: crate::TokenId) -> Option<&Symbol> {
        self.rev.get(id as usize)
    }

    pub fn special_map(&self) -> &std::collections::HashMap<Symbol, crate::TokenId> {
        &self.specials
    }

    pub fn added_map(&self) -> &std::collections::HashMap<Symbol, crate::TokenId> {
        &self.added
    }

    pub fn len(&self) -> usize {
        self.rev.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rev.is_empty()
    }

    /// Precompute the `byte → id` direct-indexed table. Call once after
    /// all entries are loaded. For each of the 256 byte values, look up the
    /// single-byte symbol (via the GPT-2 `byte_to_unicode` mapping, since vocab
    /// symbols are stored byte-mapped) and store its id. After this, the BPE
    /// seed loop is array-indexed.
    pub fn finalize_byte_table(&mut self) {
        let table = Self::byte_to_unicode_table();
        for b in 0..=255u8 {
            let ch = table[b as usize];
            let mut sym = Symbol::new();
            sym.push(ch as u8); // symbols are byte-mapped single bytes for the base vocab
                                // For multibyte-mapped chars, encode the char's UTF-8 form into the
                                // symbol (GPT-2 stores the mapped char, not the raw byte).
            let mut buf = [0u8; 4];
            sym.clear();
            for &c in ch.encode_utf8(&mut buf).as_bytes() {
                sym.push(c);
            }
            self.byte_to_id[b as usize] = self.fwd.get(&sym).copied();
        }
    }

    /// Direct-indexed single-byte → id lookup. Returns None if that byte
    /// has no single-token id.
    pub fn id_of_byte(&self, b: u8) -> Option<crate::TokenId> {
        // b is 0..=255, a valid index into the 256-entry table.
        self.byte_to_id[b as usize]
    }

    /// Build the flat `id → decoded bytes` table. Call once after all
    /// entries are loaded (and after `finalize_byte_table`). For each token,
    /// store the original input bytes it decodes to: inverse-`byte_to_unicode`
    /// for BPE tokens, literal symbol bytes for special tokens (so a special
    /// token decodes to its own text, not byte-mapped gibberish).
    pub fn finalize_decode_table(&mut self) {
        let inv = Self::unicode_to_byte_map();
        let mut out: Vec<Box<[u8]>> = Vec::with_capacity(self.rev.len());
        for sym in &self.rev {
            let decoded: SmallVec<[u8; 16]> = if self.added.contains_key(sym) {
                // Added token: decode to its literal bytes verbatim.
                sym.iter().copied().collect()
            } else {
                // Byte-level BPE: inverse-map each mapped char back to its byte.
                let mut d: SmallVec<[u8; 16]> = SmallVec::new();
                match std::str::from_utf8(sym) {
                    Ok(s) => {
                        for ch in s.chars() {
                            match inv.get(&ch) {
                                Some(&b) => d.push(b),
                                None => {
                                    let mut buf = [0u8; 4];
                                    for &c in ch.encode_utf8(&mut buf).as_bytes() {
                                        d.push(c);
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // Non-UTF-8 symbol (shouldn't happen for byte-level BPE):
                        // fall back to literal bytes.
                        d.extend_from_slice(sym);
                    }
                }
                d
            };
            out.push(decoded.into_vec().into_boxed_slice());
        }
        self.decode_bytes = out;
    }

    /// The decoded bytes for a single token id (the inverse-mapped or literal
    /// bytes it represents). Returns empty for an out-of-range id.
    #[inline]
    pub fn decode_id(&self, id: crate::TokenId) -> &[u8] {
        match self.decode_bytes.get(id as usize) {
            Some(b) => b,
            None => &[],
        }
    }

    /// Decode a token-id sequence back to bytes by concatenating each
    /// token's precomputed decoded bytes. `ids` and `specials` are parallel;
    /// specials are decoded from the same table (their entries hold literal
    /// bytes), so the flag is not strictly needed here but is accepted for API
    /// symmetry with `encode`.
    pub fn decode(&self, ids: &[crate::TokenId]) -> Vec<u8> {
        let total: usize = ids.iter().map(|&id| self.decode_id(id).len()).sum();
        let mut out = Vec::with_capacity(total);
        for &id in ids {
            out.extend_from_slice(self.decode_id(id));
        }
        out
    }

    /// The inverse of `byte_to_unicode_table`: each mapped character maps to its
    /// original byte. Used by `finalize_decode_table`.
    fn unicode_to_byte_map() -> std::collections::HashMap<char, u8> {
        let table = Self::byte_to_unicode_table();
        let mut m = std::collections::HashMap::with_capacity(256);
        for (b, &ch) in table.iter().enumerate() {
            m.insert(ch, b as u8);
        }
        m
    }

    /// The GPT-2 `byte_to_unicode` table: a fixed mapping from each of the 256
    /// byte values to a Unicode char. Remaps control bytes and the few
    /// "awkward" Latin-1 codepoints to U+0100.. so every byte round-trips
    /// through a printable, single-(mapped)char symbol. This is the table both
    /// the oracle and the fast path use to seed the BPE word.
    pub fn byte_to_unicode_table() -> [char; 256] {
        let mut table = ['\0'; 256];
        let mut n = 0u32;
        for b in 0..=255u8 {
            let ch = if (33..=126).contains(&u32::from(b))
                || (161..=172).contains(&u32::from(b))
                || (174..=255).contains(&u32::from(b))
            {
                char::from_u32(u32::from(b)).unwrap_or('?')
            } else {
                let c = char::from_u32(256 + n).unwrap_or('?');
                n += 1;
                c
            };
            table[b as usize] = ch;
        }
        table
    }
}
