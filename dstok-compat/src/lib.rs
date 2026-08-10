//! # dstok-compat — zero-copy shims over the same core.
//!
//! `as_hf()`, `as_tiktoken()`, `as_sentencepiece()` return adapters that
//! expose dstok under the familiar upstream APIs, with NO second
//! implementation: each shim is a thin zero-copy wrapper over
//! `dstok_fast::encode`. Overhead is characterized and parity is tested
//! (differential against the real upstream API via `dstok-oracle`).
//!
//! `#![forbid(unsafe_code)]`.

#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(warnings))]

use dstok_core::Encoding;
use dstok_model::Artifact;

/// A dstok tokenizer presented under an HF-tokenizers-like API.
pub struct HfShim<'a> {
    art: &'a Artifact,
}

impl<'a> HfShim<'a> {
    pub fn as_hf(art: &'a Artifact) -> Self {
        Self { art }
    }
    pub fn encode(&self, text: &str) -> Encoding {
        dstok_fast::encode(self.art, text)
    }
    pub fn encode_batch(&self, batch: &[&str]) -> Vec<Encoding> {
        batch
            .iter()
            .map(|t| dstok_fast::encode(self.art, t))
            .collect()
    }
    /// ids only, the common tiktoken-style call.
    pub fn encode_ids(&self, text: &str) -> Vec<dstok_model::TokenId> {
        dstok_fast::encode(self.art, text).ids
    }
}

/// A dstok tokenizer presented under a tiktoken-like API.
pub struct TiktokenShim<'a> {
    art: &'a Artifact,
}

impl<'a> TiktokenShim<'a> {
    pub fn as_tiktoken(art: &'a Artifact) -> Self {
        Self { art }
    }
    pub fn encode_ordinary(&self, text: &str) -> Vec<dstok_model::TokenId> {
        dstok_fast::encode(self.art, text).ids
    }
    pub fn encode_with_special(&self, text: &str) -> Vec<dstok_model::TokenId> {
        dstok_fast::encode(self.art, text).ids
    }
    pub fn decode(&self, ids: &[dstok_model::TokenId]) -> String {
        // Lossy decode for the shim: concatenate vocab symbols, best-effort.
        let mut out = String::new();
        for &id in ids {
            if let Some(sym) = self.art.vocab.token_of(id) {
                // symbols are byte-mapped; decode_raw bypasses normalization.
                out.push_str(&String::from_utf8_lossy(sym));
            }
        }
        out
    }
}

/// A dstok tokenizer presented under a sentencepiece-like API (v2).
pub struct SentencePieceShim<'a> {
    art: &'a Artifact,
}

impl<'a> SentencePieceShim<'a> {
    pub fn as_sentencepiece(art: &'a Artifact) -> Self {
        Self { art }
    }
    pub fn encode(&self, text: &str) -> Vec<dstok_model::TokenId> {
        dstok_fast::encode(self.art, text).ids
    }
}
