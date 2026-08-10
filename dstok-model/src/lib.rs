//! # dstok-model — verified tokenizer artifacts and configuration
//!
//! The production loader supports one artifact: the exact pinned GLM-5.2
//! Hugging Face tokenizer JSON. Internal types are structured for other
//! formats, but those formats are not part of the supported package contract.
//!
//! Every artifact records its source format, version, and BLAKE3 digest so a
//! silently mutated vocabulary can be rejected.
//!
//! `#![forbid(unsafe_code)]` — this crate touches untrusted config files but
//! never does anything that needs `unsafe`.

#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(warnings))]

pub mod config;
pub mod family;
pub mod loader;
pub mod merges;
pub mod vocab;

pub use config::{
    AddedToken, BpeConfig, ByteLevelConfig, DecoderKind, Padding, PostProcessorKind,
    PreTokenizerKind, SplitBehavior, SplitConfig, SplitPattern, TokenizerConfig, TokenizerProfile,
    Truncation,
};
pub use family::{Family, ModelId, GPT2_REGEX, LLAMA3_REGEX};
pub use merges::Merges;
pub use vocab::{Symbol, Vocab};

/// A token id. NonZero would be nice for niche optimization but many real
/// vocabs use id 0 for `<pad>`/`<unk>`, so a plain u32.
pub type TokenId = u32;

/// A loaded, hash-verified tokenizer artifact: the vocab, the merge ranks,
/// the pretokenizer config, and special tokens. Shared read-only by the
/// oracle and the fast path.
///
/// The model family is reached via `config.family: ModelId` (which carries the
/// algorithm `Family` via `ModelId::family()`). There is no separate `family`
/// field on `Artifact` — it would only duplicate `config.family`.
#[derive(Debug, Clone)]
pub struct Artifact {
    /// The full vocab: symbol bytes → id, and id → symbol bytes.
    pub vocab: Vocab,
    /// Adjacent-pair → merge rank (lower = applied first).
    pub merges: Merges,
    /// Pretokenizer + special-token + pad/truncation config (incl. `family`).
    pub config: TokenizerConfig,
    /// Provenance: source format, version, BLAKE3 of the source bytes.
    pub provenance: Provenance,
}

#[derive(Debug, Clone)]
pub struct Provenance {
    pub format: ArtifactFormat,
    pub format_version: String,
    pub blake3: [u8; 32],
    pub source_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactFormat {
    HfTokenizersJson,
    Tiktoken,
    SentencePiece,
}

impl Artifact {
    /// Number of tokens in the vocab.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
