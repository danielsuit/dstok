//! Globally bounded exact count caches.
//!
//! Both caches key on the artifact digest and complete raw bytes. Hash-table
//! collisions therefore trigger byte equality checks and cannot alter token
//! IDs or counts. `DSTOK_CACHE_MAX_BYTES` controls the pretoken cache (64 MiB
//! by default), while `DSTOK_SEGMENT_CACHE_MAX_BYTES` controls the text-segment
//! count cache (128 MiB by default). Either cache can be disabled with a zero
//! budget.

use dstok_model::{Artifact, TokenId, TokenizerProfile};
use moka::sync::Cache;
use moka::Equivalent;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

const DEFAULT_CACHE_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_SEGMENT_CACHE_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct TokenRun {
    pub ids: Box<[TokenId]>,
    pub byte_lens: Box<[usize]>,
}

impl TokenRun {
    fn compute(artifact: &Artifact, pretoken: &[u8]) -> Self {
        let (ids, byte_lens) = dstok_core::bpe_merge_pretoken(artifact, pretoken);
        Self {
            ids: ids.into_vec().into_boxed_slice(),
            byte_lens: byte_lens.into_vec().into_boxed_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

pub type CachedTokens = Arc<TokenRun>;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Raw-pretoken cache hits.
    pub hits: u64,
    /// Raw-pretoken cache misses.
    pub misses: u64,
    /// Complete ordinary-text-segment count cache hits.
    pub segment_hits: u64,
    /// Complete ordinary-text-segment count cache misses.
    pub segment_misses: u64,
}

impl CacheStats {
    /// Hit rate for the raw-pretoken cache.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Hit rate for the complete text-segment count cache.
    pub fn segment_hit_rate(&self) -> f64 {
        let total = self.segment_hits + self.segment_misses;
        if total == 0 {
            0.0
        } else {
            self.segment_hits as f64 / total as f64
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct CacheKey {
    artifact: [u8; 32],
    bytes: Box<[u8]>,
}

struct CacheKeyRef<'a> {
    artifact: &'a [u8; 32],
    bytes: &'a [u8],
}

impl Hash for CacheKeyRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.artifact.hash(state);
        self.bytes.hash(state);
    }
}

impl Equivalent<CacheKey> for CacheKeyRef<'_> {
    fn equivalent(&self, key: &CacheKey) -> bool {
        self.artifact == &key.artifact && self.bytes == key.bytes.as_ref()
    }
}

fn cache() -> Option<&'static Cache<CacheKey, Arc<TokenRun>>> {
    static CACHE: OnceLock<Option<Cache<CacheKey, Arc<TokenRun>>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let budget = std::env::var("DSTOK_CACHE_MAX_BYTES")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(DEFAULT_CACHE_BYTES);
            if budget == 0 {
                return None;
            }
            Some(
                Cache::builder()
                    .max_capacity(budget)
                    .weigher(|key: &CacheKey, value: &Arc<TokenRun>| {
                        let weight = key.bytes.len()
                            + value.ids.len() * std::mem::size_of::<TokenId>()
                            + value.byte_lens.len() * std::mem::size_of::<usize>()
                            + std::mem::size_of::<CacheKey>();
                        u32::try_from(weight).unwrap_or(u32::MAX)
                    })
                    .build(),
            )
        })
        .as_ref()
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SegmentCacheKey {
    artifact: [u8; 32],
    bytes: Box<[u8]>,
}

struct SegmentCacheKeyRef<'a> {
    artifact: &'a [u8; 32],
    bytes: &'a [u8],
}

impl Hash for SegmentCacheKeyRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.artifact.hash(state);
        self.bytes.hash(state);
    }
}

impl Equivalent<SegmentCacheKey> for SegmentCacheKeyRef<'_> {
    fn equivalent(&self, key: &SegmentCacheKey) -> bool {
        self.artifact == &key.artifact && self.bytes == key.bytes.as_ref()
    }
}

fn segment_count_cache() -> Option<&'static Cache<SegmentCacheKey, usize>> {
    static CACHE: OnceLock<Option<Cache<SegmentCacheKey, usize>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let budget = std::env::var("DSTOK_SEGMENT_CACHE_MAX_BYTES")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(DEFAULT_SEGMENT_CACHE_BYTES);
            if budget == 0 {
                return None;
            }
            Some(
                Cache::builder()
                    .max_capacity(budget)
                    .weigher(|key: &SegmentCacheKey, _count: &usize| {
                        let weight = key.bytes.len()
                            + std::mem::size_of::<usize>()
                            + std::mem::size_of::<SegmentCacheKey>();
                        u32::try_from(weight).unwrap_or(u32::MAX)
                    })
                    .build(),
            )
        })
        .as_ref()
}

pub(crate) fn get_or_compute(
    artifact: &Artifact,
    pretoken: &[u8],
    stats: Option<&mut CacheStats>,
) -> Arc<TokenRun> {
    // Synthetic artifacts do not carry a content digest. Caching them by a
    // zero digest could cross-contaminate independent property-test vocabs.
    if artifact.provenance.blake3 == [0; 32] {
        if let Some(stats) = stats {
            stats.misses += 1;
        }
        return Arc::new(TokenRun::compute(artifact, pretoken));
    }
    let Some(cache) = cache() else {
        if let Some(stats) = stats {
            stats.misses += 1;
        }
        return Arc::new(TokenRun::compute(artifact, pretoken));
    };
    let borrowed = CacheKeyRef {
        artifact: &artifact.provenance.blake3,
        bytes: pretoken,
    };
    if let Some(run) = cache.get(&borrowed) {
        if let Some(stats) = stats {
            stats.hits += 1;
        }
        return run;
    }
    if let Some(stats) = stats {
        stats.misses += 1;
    }
    let key = CacheKey {
        artifact: artifact.provenance.blake3,
        bytes: pretoken.into(),
    };
    cache.get_with(key, || Arc::new(TokenRun::compute(artifact, pretoken)))
}

/// Return the exact count for one text segment produced by added-token
/// extraction. Only the hash-verified GLM profile is eligible: its loader
/// enforces no normalization, context-free ByteLevel pretokenization, and no
/// tokenizer-level truncation or padding.
pub(crate) fn get_or_compute_segment_count(
    artifact: &Artifact,
    segment: &[u8],
    stats: &mut CacheStats,
    compute: impl FnOnce(&mut CacheStats) -> usize,
) -> usize {
    if segment.is_empty() {
        return 0;
    }
    if artifact.config.profile != TokenizerProfile::Glm52 || artifact.provenance.blake3 == [0; 32] {
        stats.segment_misses += 1;
        return compute(stats);
    }
    let Some(cache) = segment_count_cache() else {
        stats.segment_misses += 1;
        return compute(stats);
    };
    let borrowed = SegmentCacheKeyRef {
        artifact: &artifact.provenance.blake3,
        bytes: segment,
    };
    if let Some(count) = cache.get(&borrowed) {
        stats.segment_hits += 1;
        return count;
    }

    stats.segment_misses += 1;
    let count = compute(stats);
    let key = SegmentCacheKey {
        artifact: artifact.provenance.blake3,
        bytes: segment.into(),
    };
    cache.get_with(key, || count)
}

pub fn clear_global_cache() {
    if let Some(cache) = cache() {
        cache.invalidate_all();
        cache.run_pending_tasks();
    }
    if let Some(cache) = segment_count_cache() {
        cache.invalidate_all();
        cache.run_pending_tasks();
    }
}
