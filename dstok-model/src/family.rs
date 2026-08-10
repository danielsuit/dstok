//! Internal tokenizer-family and model descriptors.
//!
//! These enums support synthetic tests and future profile work. Their presence
//! does not advertise production support: the strict public loader currently
//! accepts only the pinned GLM-5.2 artifact.

/// The algorithm class of a tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Family {
    /// Byte-level BPE with a regex pretokenizer.
    #[default]
    ByteBPE,
    /// SentencePiece unigram language model descriptor.
    SentencePieceUnigram,
    /// SentencePiece BPE descriptor.
    SentencePieceBPE,
    /// WordPiece greedy-longest-match descriptor.
    WordPiece,
}

impl Family {
    /// Whether this family is byte-level BPE.
    pub fn is_byte_level(self) -> bool {
        matches!(self, Family::ByteBPE)
    }
}

/// A model descriptor used by tokenizer profiles and synthetic tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelId {
    #[default]
    Gpt2,
    Llama3,
    Qwen2,
    Qwen3,
    DeepSeekV3,
    DeepSeekR1,
    Glm4,
    Glm5,
    Mistral,
    Gemma,
}

/// The GPT-2 byte-level pretokenizer regex. The canonical pattern (whitespace
/// runs, letters, digits, apostrophe contractions, single other chars).
pub const GPT2_REGEX: &str =
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;

/// The Llama-3 byte-level pretokenizer regex: like GPT-2 but numbers form their
/// own runs (digits split from adjacent letters) and a larger other-class.
pub const LLAMA3_REGEX: &str =
    r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{R}|\p{N}{1,128}| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;

impl ModelId {
    /// The algorithm family for this model.
    pub fn family(self) -> Family {
        match self {
            ModelId::Gpt2
            | ModelId::Llama3
            | ModelId::Qwen2
            | ModelId::Qwen3
            | ModelId::DeepSeekV3
            | ModelId::DeepSeekR1
            | ModelId::Glm4
            | ModelId::Glm5
            | ModelId::Mistral
            | ModelId::Gemma => Family::ByteBPE,
        }
    }

    /// The pretokenizer template (regex source) for this model, if it uses a
    /// regex pretokenizer.
    pub fn pretokenizer_template(self) -> Option<&'static str> {
        match self {
            ModelId::Llama3
            | ModelId::Qwen2
            | ModelId::Qwen3
            | ModelId::DeepSeekV3
            | ModelId::DeepSeekR1
            | ModelId::Glm4
            | ModelId::Glm5 => Some(LLAMA3_REGEX),
            ModelId::Gpt2 | ModelId::Mistral | ModelId::Gemma => Some(GPT2_REGEX),
        }
    }

    /// Whether this descriptor caps consecutive digit pretoken runs.
    ///
    /// This is a back-compat bool view of [`ModelId::digit_run`]: true iff the
    /// model caps digit runs (i.e. `digit_run() != 0`). New code should call
    /// `digit_run()` directly.
    pub fn llama3_digits(self) -> bool {
        self.digit_run() != 0
    }

    /// The maximum number of digits in one pretoken run (`\p{N}{1,n}`).
    ///
    /// - `0` means **unbounded** (`\p{N}+`): GPT-2 / Mistral / Gemma.
    /// - `2` preserves dstok's existing Llama-3-family oracle behavior
    ///   (`\p{N}{1,2}`). NOTE: the canonical Llama-3 regex is `{1,128}`; dstok
    ///   uses `{1,2}` as its current oracle behavior — a known parity gap vs
    ///   HF, tracked separately.
    /// - `3` is GLM's custom digit splitting (`{1,3}`): Glm4 / Glm5.
    ///
    /// Both the scalar oracle (`dstok_core::regex`) and the SIMD scanner
    /// (`dstok_simd`) consume exactly this cap so the two cannot diverge on
    /// digit runs (a previous latent bug: the SIMD path ignored the cap).
    pub fn digit_run(self) -> usize {
        match self {
            ModelId::Llama3
            | ModelId::Qwen2
            | ModelId::Qwen3
            | ModelId::DeepSeekV3
            | ModelId::DeepSeekR1 => 2,
            ModelId::Glm4 | ModelId::Glm5 => 3,
            ModelId::Gpt2 | ModelId::Mistral | ModelId::Gemma => 0,
        }
    }
}
