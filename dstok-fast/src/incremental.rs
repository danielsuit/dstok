//! Conservative incremental facade.
//!
//! Append-boundary tokenization is deliberately not claimed for GLM-5.2:
//! BPE and the regex are non-additive. Chunks are buffered and the exact
//! tokenizer runs at `flush`, so correctness is preserved at the cost of
//! deferred output.

use dstok_core::Offset;
use dstok_model::{Artifact, TokenId};

pub struct IncrementalEncoder<'a> {
    artifact: &'a Artifact,
    input: Vec<u8>,
    ids: Vec<TokenId>,
    offsets: Vec<Offset>,
    specials: Vec<bool>,
    flushed: bool,
}

impl<'a> IncrementalEncoder<'a> {
    pub fn new(artifact: &'a Artifact) -> Self {
        Self {
            artifact,
            input: Vec::new(),
            ids: Vec::new(),
            offsets: Vec::new(),
            specials: Vec::new(),
            flushed: false,
        }
    }

    /// Buffer one chunk. No prefix is emitted because a later append may
    /// change the final pretoken and BPE result.
    pub fn push(&mut self, chunk: &[u8]) -> &[TokenId] {
        assert!(!self.flushed, "cannot push after flush");
        self.input.extend_from_slice(chunk);
        &self.ids[self.ids.len()..]
    }

    pub fn flush(&mut self) -> &[TokenId] {
        if self.flushed {
            return &self.ids[self.ids.len()..];
        }
        let text = std::str::from_utf8(&self.input)
            .expect("incremental tokenizer input must be valid UTF-8 at flush");
        let encoding = crate::encode(self.artifact, text);
        self.ids = encoding.ids;
        self.offsets = encoding.offsets.into_vec();
        self.specials = encoding.specials;
        self.flushed = true;
        &self.ids
    }

    pub fn ids(&self) -> &[TokenId] {
        &self.ids
    }

    pub fn offsets(&self) -> &[Offset] {
        &self.offsets
    }

    pub fn specials(&self) -> &[bool] {
        &self.specials
    }

    pub fn consumed_bytes(&self) -> usize {
        if self.flushed {
            self.input.len()
        } else {
            0
        }
    }
}
