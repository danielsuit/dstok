//! Structural tokenizer configuration for the pinned GLM-5.2 artifact.
//!
//! Configuration is data, not a family-name hint. The loader records every
//! field that affects token IDs and rejects unsupported shapes before an
//! [`Artifact`](crate::Artifact) can be constructed.

use crate::{family::ModelId, TokenId};

/// Exact Split regex from the pinned gateway GLM-5.2 `tokenizer.json`.
///
/// This pattern is not original to GLM-5.2: it is the byte-level
/// pre-tokenizer regex introduced by Llama 3 and reused unchanged by many
/// later BPE tokenizers. It is reproduced here verbatim because any
/// deviation changes token IDs. See `THIRD_PARTY_NOTICES.md`.
pub const GLM52_SPLIT_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// SHA-256 of the pinned gateway tokenizer artifact.
pub const GLM52_TOKENIZER_SHA256: [u8; 32] = [
    0x19, 0xe7, 0x73, 0x64, 0x8c, 0xb4, 0xe6, 0x5d, 0xe8, 0x66, 0x0e, 0xa6, 0x36, 0x5e, 0x10, 0xac,
    0xca, 0x11, 0x2d, 0x42, 0xa8, 0x54, 0x92, 0x3d, 0xf9, 0x3d, 0xb4, 0xa6, 0xf3, 0x33, 0xa8, 0x2d,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenizerProfile {
    /// In-memory artifacts used by unit/property tests.
    #[default]
    Synthetic,
    /// The exact gateway GLM-5.2 artifact identified by digest and structure.
    Glm52,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedToken {
    pub id: TokenId,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitPattern {
    String(String),
    Regex(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitConfig {
    pub pattern: SplitPattern,
    pub behavior: SplitBehavior,
    pub invert: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteLevelConfig {
    pub add_prefix_space: bool,
    pub trim_offsets: bool,
    pub use_regex: bool,
}

/// A faithful structural representation of the HF pre-tokenizer graph.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PreTokenizerKind {
    #[default]
    None,
    ByteLevel(ByteLevelConfig),
    Split(SplitConfig),
    Sequence(Vec<PreTokenizerKind>),
    Whitespace,
    WhitespaceSplit,
    BertPreTokenizer,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PostProcessorKind {
    #[default]
    None,
    ByteLevel(ByteLevelConfig),
    Template,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DecoderKind {
    #[default]
    None,
    ByteLevel(ByteLevelConfig),
    Other(String),
}

/// BPE fields that can alter token IDs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BpeConfig {
    pub dropout: Option<f32>,
    pub unk_token: Option<String>,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
    pub fuse_unk: bool,
    pub byte_fallback: bool,
    pub ignore_merges: bool,
}

#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub profile: TokenizerProfile,
    pub family: ModelId,
    pub added_tokens: Vec<AddedToken>,
    pub special_tokens: Vec<String>,
    /// Extract all declared added tokens before pre-tokenization. Despite the
    /// historical name, this includes non-special added tokens too.
    pub split_specials: bool,
    /// Compatibility mirror of `bpe.ignore_merges`; the loader keeps them
    /// equal and encoding code reads this hot-path field.
    pub ignore_merges: bool,
    pub bpe: BpeConfig,
    pub pre_tokenizer: Option<PreTokenizerKind>,
    pub post_processor: Option<PostProcessorKind>,
    pub decoder: Option<DecoderKind>,
    pub normalizer_is_none: bool,
    pub truncation: Truncation,
    pub padding: Option<Padding>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            profile: TokenizerProfile::Synthetic,
            family: ModelId::Gpt2,
            added_tokens: Vec::new(),
            special_tokens: Vec::new(),
            split_specials: false,
            ignore_merges: false,
            bpe: BpeConfig::default(),
            pre_tokenizer: None,
            post_processor: None,
            decoder: None,
            normalizer_is_none: true,
            truncation: Truncation::None,
            padding: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub enum Truncation {
    #[default]
    None,
    LongestFirst {
        max_length: usize,
    },
    OnlyFirst {
        max_length: usize,
    },
}

#[derive(Debug, Clone)]
pub struct Padding {
    pub strategy: PadStrategy,
    pub max_length: usize,
    pub pad_token: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadStrategy {
    BatchLongest,
    FixedLength,
}
