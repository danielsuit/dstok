//! # dstok-oracle — FFI/subprocess wrappers around the canonical references.
//!
//! Wraps HF `tokenizers` (Rust, callable directly), OpenAI `tiktoken`, and
//! Google `sentencepiece` so differential tests (§4.1) can assert
//! dstok == oracle on token IDs *and* offsets *and* special-token placement.
//!
//! Each oracle pins an exact reference version. A model isn't "supported"
//! until its conformance manifest is green (§3).

#![cfg_attr(not(test), deny(warnings))]

use dstok_model::TokenId;

pub const HF_TOKENIZERS_VERSION: &str = "0.23.1";

/// The reference tool + version pinned for a given artifact family.
#[derive(Debug, Clone)]
pub struct PinnedOracle {
    pub tool: OracleTool,
    pub version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleTool {
    HfTokenizers,
    Tiktoken,
    SentencePiece,
}

/// The oracle's ground-truth output for one input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleOutput {
    pub ids: Vec<TokenId>,
    /// per-token byte offsets (start, end)
    pub offsets: Vec<(usize, usize)>,
    /// which ids are special tokens
    pub special_positions: Vec<usize>,
}

/// Trait every oracle wrapper implements. The differential harness calls
/// `encode` on the oracle and on dstok and compares.
pub trait Oracle {
    fn pinned(&self) -> &PinnedOracle;
    fn encode(&self, text: &str) -> OracleOutput;
}

/// A no-op oracle used in the scaffold when the native reference isn't linked.
pub struct StubOracle {
    pinned: PinnedOracle,
}

impl StubOracle {
    pub fn hf(version: &str) -> Self {
        Self {
            pinned: PinnedOracle {
                tool: OracleTool::HfTokenizers,
                version: version.to_string(),
            },
        }
    }
    pub fn tiktoken(version: &str) -> Self {
        Self {
            pinned: PinnedOracle {
                tool: OracleTool::Tiktoken,
                version: version.to_string(),
            },
        }
    }
    pub fn sentencepiece(version: &str) -> Self {
        Self {
            pinned: PinnedOracle {
                tool: OracleTool::SentencePiece,
                version: version.to_string(),
            },
        }
    }
}

impl Oracle for StubOracle {
    fn pinned(&self) -> &PinnedOracle {
        &self.pinned
    }
    fn encode(&self, _text: &str) -> OracleOutput {
        OracleOutput {
            ids: vec![],
            offsets: vec![],
            special_positions: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Real HF `tokenizers` wrapper (§3 Layer 1). Enabled with the `hf` feature.
// ---------------------------------------------------------------------------

/// A differential oracle backed by the real HF `tokenizers` crate (loaded from
/// a `tokenizer.json` artifact). This is the ground truth for the GPT-2 /
/// Llama / Qwen / DeepSeek / GLM / Mistral / Gemma families.
#[cfg(feature = "hf")]
pub struct HfTokenizersOracle {
    pinned: PinnedOracle,
    tk: tokenizers::Tokenizer,
}

#[cfg(feature = "hf")]
impl HfTokenizersOracle {
    /// Load from a `tokenizer.json` file. `version` is the pinned
    /// `tokenizers` crate version used for conformance (§3).
    pub fn from_file(path: &str) -> Result<Self, String> {
        let tk = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("load tokenizer.json: {e}"))?;
        Ok(Self {
            pinned: PinnedOracle {
                tool: OracleTool::HfTokenizers,
                version: HF_TOKENIZERS_VERSION.to_string(),
            },
            tk,
        })
    }
}

#[cfg(feature = "hf")]
impl Oracle for HfTokenizersOracle {
    fn pinned(&self) -> &PinnedOracle {
        &self.pinned
    }
    fn encode(&self, text: &str) -> OracleOutput {
        let enc = self
            .tk
            .encode_fast(text, false)
            .unwrap_or_else(|e| panic!("hf tokenizers encode failed: {e}"));
        let ids: Vec<TokenId> = enc.get_ids().iter().map(|&u| u as TokenId).collect();
        let offsets: Vec<(usize, usize)> = enc.get_offsets().iter().map(|o| (o.0, o.1)).collect();
        // special tokens: HF marks them via `Encoding::get_special_tokens_mask`
        // or by id matching the special-token set; we record positions whose id
        // is in the artifact's special set. Here we approximate via the mask
        // if available, else empty.
        let special_positions: Vec<usize> = enc
            .get_special_tokens_mask()
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m != 0 { Some(i) } else { None })
            .collect();
        OracleOutput {
            ids,
            offsets,
            special_positions,
        }
    }
}

/// tiktoken / sentencepiece are reached via a subprocess (they are Python /
/// C++ libraries), pinned by environment. The wrapper shells out once per
/// batch and caches results; see `SubprocessOracle`.
#[cfg(not(feature = "hf"))]
pub struct HfTokenizersOracle {
    _pinned: PinnedOracle,
}

// ---------------------------------------------------------------------------
// Conformance manifest (§3): a model isn't "supported" until green in CI.
// ---------------------------------------------------------------------------

/// A frozen golden vector + the pinned oracle + the corpus slice validated.
#[derive(Debug, Clone)]
pub struct ConformanceManifest {
    pub model: String,
    pub oracle: PinnedOracle,
    /// BLAKE3 of the artifact bytes (§7 supply-chain verification).
    pub artifact_blake3: [u8; 32],
    /// (input → expected ids) golden vectors.
    pub golden: Vec<GoldenCase>,
    /// The corpus slice the parity was validated on.
    pub corpus_slice: String,
}

#[derive(Debug, Clone)]
pub struct GoldenCase {
    pub input: String,
    pub ids: Vec<TokenId>,
    pub offsets: Vec<(usize, usize)>,
}

impl ConformanceManifest {
    /// Check dstok output against every golden case. Returns the first failure.
    /// A model is "supported" iff this returns Ok (§3).
    pub fn check(
        &self,
        dstok_encode: impl Fn(&str) -> OracleOutput,
    ) -> Result<(), ManifestFailure> {
        for (i, g) in self.golden.iter().enumerate() {
            let got = dstok_encode(&g.input);
            if got.ids != g.ids {
                return Err(ManifestFailure {
                    case: i,
                    kind: Divergence::Ids,
                });
            }
            if got.offsets != g.offsets {
                return Err(ManifestFailure {
                    case: i,
                    kind: Divergence::Offsets,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ManifestFailure {
    pub case: usize,
    pub kind: Divergence,
}

/// Assert dstok == oracle on ids, offsets, and special-token positions.
/// Returns the first divergence, if any. Used by the differential harness.
pub fn diff(dstok: &OracleOutput, oracle: &OracleOutput) -> Option<Divergence> {
    if dstok.ids != oracle.ids {
        return Some(Divergence::Ids);
    }
    if dstok.offsets != oracle.offsets {
        return Some(Divergence::Offsets);
    }
    if dstok.special_positions != oracle.special_positions {
        return Some(Divergence::Specials);
    }
    None
}

/// Three-way differential: dstok-fast vs dstok-core (internal oracle) vs the
/// external oracle, on one input (§4.1). Returns the first layer that diverges.
pub fn diff3(
    fast: &OracleOutput,
    core: &OracleOutput,
    external: &OracleOutput,
) -> Option<ThreeWayDivergence> {
    if let Some(d) = diff(fast, core) {
        return Some(ThreeWayDivergence::FastVsCore(d));
    }
    if let Some(d) = diff(core, external) {
        return Some(ThreeWayDivergence::CoreVsExternal(d));
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Divergence {
    Ids,
    Offsets,
    Specials,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeWayDivergence {
    /// dstok-fast disagrees with dstok-core — a SIMD/cache/parallel bug.
    FastVsCore(Divergence),
    /// dstok-core disagrees with the external reference — the oracle itself is
    /// wrong, which is a P0 (the reference implementation is the oracle, §0).
    CoreVsExternal(Divergence),
}

/// BLAKE3-hash an artifact file for the conformance manifest (§7).
pub fn artifact_hash(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}
