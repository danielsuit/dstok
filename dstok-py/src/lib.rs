//! # dstok-py — PyO3 bindings, PyPI package.
//!
//! Thin PyO3 wrapper over `dstok_fast`. The Python API mirrors the compat
//! shims: `dstok.from_artifact(path)` returns an `Encoding` object with
//! `.encode(text)` / `.encode_batch(texts)` returning lists of ids + offsets.
//!
//! (Sketched as the module signature; the `maturin` build wires the PyO3
//! `#[pymodule]`.)

#![cfg_attr(not(test), deny(warnings))]

use dstok_core::Encoding;
use dstok_model::{loader, Artifact};

/// A loaded tokenizer exposed to Python.
pub struct PyTokenizer {
    art: Artifact,
}

impl PyTokenizer {
    /// Load + hash-verify an artifact. (Python-facing wrapper.)
    pub fn from_path(path: &str, hash: [u8; 32]) -> Result<Self, String> {
        let art = loader::load_path(path, hash).map_err(|e| format!("{e:?}"))?;
        Ok(Self { art })
    }

    pub fn encode(&self, text: &str) -> Encoding {
        dstok_fast::encode(&self.art, text)
    }

    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        dstok_fast::encode(&self.art, text).ids
    }

    pub fn encode_batch(&self, texts: &[String]) -> Vec<Encoding> {
        texts
            .iter()
            .map(|t| dstok_fast::encode(&self.art, t))
            .collect()
    }
}
