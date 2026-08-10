//! Public API for exact GLM-5.2 token counting.
//!
//! `dstok` supports one production tokenizer artifact: the pinned GLM-5.2
//! `tokenizer.json` identified in the repository's conformance manifest. The
//! loader fails closed for any other artifact, and the optimized count path is
//! continuously compared with Hugging Face `tokenizers` 0.23.1.
//!
//! # Example
//!
//! ```no_run
//! use dstok::Tokenizer;
//!
//! let tokenizer = Tokenizer::from_file("tokenizer.json")?;
//! let tokens = tokenizer.count("<|user|>Count this exactly.");
//! assert!(tokens > 0);
//! # Ok::<(), dstok::LoadError>(())
//! ```
//!
//! Input must already be rendered with the model's chat template. This crate
//! deliberately does not render chat requests.

#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::Arc;

pub use dstok_fast::{CacheStats, Encoding};
pub use dstok_model::loader::LoadError;
pub use dstok_model::{Artifact, TokenId};

/// An exact tokenizer for the pinned GLM-5.2 artifact.
///
/// Clones share the immutable artifact and are safe to use across threads.
/// Count caches are process-wide, bounded, and keyed by the artifact digest
/// plus complete input bytes.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    artifact: Arc<Artifact>,
}

impl Tokenizer {
    /// Load and verify a GLM-5.2 `tokenizer.json` file.
    ///
    /// Files with any digest or structural difference from the supported
    /// artifact are rejected.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let artifact = dstok_model::loader::load_glm52_path(path)?;
        Ok(Self {
            artifact: Arc::new(artifact),
        })
    }

    /// Load and verify GLM-5.2 tokenizer JSON bytes.
    ///
    /// This is useful when an application embeds or downloads the artifact
    /// through its own asset pipeline.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let artifact = dstok_model::loader::load_glm52_bytes(bytes)?;
        Ok(Self {
            artifact: Arc::new(artifact),
        })
    }

    /// Return the verified low-level artifact.
    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    /// Count tokens without materializing token IDs or offsets.
    ///
    /// Tokenizer-level truncation is applied once after the exact count.
    pub fn count(&self, rendered_prompt: &str) -> usize {
        dstok_fast::count(&self.artifact, rendered_prompt)
    }

    /// Count tokens and return per-call cache statistics.
    pub fn count_with_stats(&self, rendered_prompt: &str) -> (usize, CacheStats) {
        dstok_fast::count_with_stats(&self.artifact, rendered_prompt)
    }

    /// Encode the complete token IDs, byte offsets, and special-token mask.
    ///
    /// Count-only segment-cache entries are not used for this API because they
    /// intentionally retain counts rather than complete token runs.
    pub fn encode(&self, rendered_prompt: &str) -> Encoding {
        dstok_fast::encode(&self.artifact, rendered_prompt)
    }

    /// Encode only token IDs.
    pub fn encode_ids(&self, rendered_prompt: &str) -> Vec<TokenId> {
        dstok_fast::encode_ids_only(&self.artifact, rendered_prompt)
    }

    /// Invalidate all process-wide dstok count caches.
    ///
    /// This affects every [`Tokenizer`] instance in the process.
    pub fn clear_caches(&self) {
        dstok_fast::clear_global_cache();
    }
}
