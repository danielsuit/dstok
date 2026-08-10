//! Correctness-gated GLM-5.2 fast/count-only APIs.
//!
//! Optimized APIs consume the exact piece boundaries produced by
//! `dstok-core`; no SIMD scanner is permitted to bypass the HF differential
//! gate. Globally bounded raw-pretoken and complete text-segment caches retain
//! full key bytes, so hash collisions cannot change token IDs or counts.

#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(warnings))]

pub mod cache;
pub mod encode;
pub mod incremental;

pub use cache::{clear_global_cache, CacheStats, CachedTokens, TokenRun};
pub use dstok_core::decode::{
    decode, decode_encoding, decode_into, decode_to_string, decode_to_string_into,
};
pub use encode::{
    count, count_parallel, count_with_stats, encode, encode_batch, encode_batch_with_stats,
    encode_ids_only, encode_ids_only_batch, encode_ids_only_into, encode_ids_only_parallel,
    encode_parallel, encode_with_stats, Encoding,
};
pub use incremental::IncrementalEncoder;

/// Compatibility name for the former block-stream implementation. Work is
/// distributed only at exact pretoken boundaries.
pub fn encode_ids_block_stream(
    artifact: &dstok_model::Artifact,
    bytes: &[u8],
    threads: usize,
) -> Vec<dstok_model::TokenId> {
    let text = std::str::from_utf8(bytes).expect("tokenizer input must be valid UTF-8");
    encode_ids_only_parallel(artifact, text, threads)
}
