//! Strict, hash-verified artifact loading.
//!
//! The HF JSON path intentionally supports one production profile: the exact
//! GLM-5.2 tokenizer pinned by the gateway. Unknown or partially-understood
//! shapes fail closed instead of being assigned a guessed tokenizer family.

use crate::config::{
    AddedToken, BpeConfig, ByteLevelConfig, DecoderKind, PostProcessorKind, PreTokenizerKind,
    SplitBehavior, SplitConfig, SplitPattern, TokenizerConfig, TokenizerProfile, GLM52_SPLIT_REGEX,
    GLM52_TOKENIZER_SHA256,
};
use crate::family::ModelId;
use crate::vocab::{Symbol, Vocab};
use crate::{Artifact, ArtifactFormat, Merges, Provenance, TokenId};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

/// Failure to load or validate a tokenizer artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// The artifact could not be read.
    Io(String),
    /// The artifact format was recognized but malformed.
    Parse(String),
    /// The caller-supplied BLAKE3 digest did not match.
    HashMismatch,
    /// The artifact is valid data but outside dstok's supported contract.
    Unsupported(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message) => write!(f, "I/O error: {message}"),
            Self::Parse(message) => write!(f, "invalid tokenizer artifact: {message}"),
            Self::HashMismatch => f.write_str("artifact BLAKE3 mismatch"),
            Self::Unsupported(message) => write!(f, "unsupported tokenizer artifact: {message}"),
        }
    }
}

impl std::error::Error for LoadError {}

pub fn load_path(path: impl AsRef<Path>, expected_hash: [u8; 32]) -> Result<Artifact, LoadError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|error| LoadError::Io(error.to_string()))?;
    load_bytes(
        &bytes,
        expected_hash,
        Some(path.to_string_lossy().into_owned()),
    )
}

/// Load the pinned GLM-5.2 tokenizer while enforcing its embedded SHA-256.
pub fn load_glm52_path(path: impl AsRef<Path>) -> Result<Artifact, LoadError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|error| LoadError::Io(error.to_string()))?;
    let expected = blake3(&bytes);
    load_bytes(&bytes, expected, Some(path.to_string_lossy().into_owned()))
}

/// Load the pinned GLM-5.2 tokenizer from memory.
///
/// The artifact's embedded SHA-256 must match the supported tokenizer exactly;
/// callers do not need to calculate or supply a digest.
pub fn load_glm52_bytes(bytes: &[u8]) -> Result<Artifact, LoadError> {
    load_bytes(bytes, blake3(bytes), None)
}

pub fn load_bytes(
    bytes: &[u8],
    expected_hash: [u8; 32],
    source: Option<String>,
) -> Result<Artifact, LoadError> {
    let actual = blake3(bytes);
    if actual != expected_hash {
        return Err(LoadError::HashMismatch);
    }
    let (format, mut artifact) = parse(bytes)?;
    artifact.vocab.finalize_byte_table();
    artifact.vocab.finalize_decode_table();
    artifact.provenance = Provenance {
        format,
        format_version: extract_version(bytes).unwrap_or_default(),
        blake3: actual,
        source_path: source,
    };
    Ok(artifact)
}

fn parse(bytes: &[u8]) -> Result<(ArtifactFormat, Artifact), LoadError> {
    if bytes.first().copied() == Some(b'{') {
        return parse_hf_json(bytes).map(|artifact| (ArtifactFormat::HfTokenizersJson, artifact));
    }
    if bytes.iter().take(64).any(|&byte| byte == b' ') && bytes.iter().all(|&byte| byte != b'{') {
        return parse_tiktoken(bytes).map(|artifact| (ArtifactFormat::Tiktoken, artifact));
    }
    Err(LoadError::Unsupported("unknown artifact format".into()))
}

#[derive(Debug)]
struct VocabEntry {
    symbol: Symbol,
    added: bool,
    special: bool,
}

fn parse_hf_json(bytes: &[u8]) -> Result<Artifact, LoadError> {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    if digest != GLM52_TOKENIZER_SHA256 {
        return Err(LoadError::Unsupported(format!(
            "HF JSON is not the pinned GLM-5.2 artifact (sha256={})",
            hex(&digest)
        )));
    }

    let root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| LoadError::Parse(error.to_string()))?;
    let object = root
        .as_object()
        .ok_or_else(|| LoadError::Parse("root must be an object".into()))?;

    if object
        .get("normalizer")
        .is_some_and(|value| !value.is_null())
    {
        return Err(LoadError::Unsupported(
            "pinned GLM normalizer must be null".into(),
        ));
    }
    if object
        .get("truncation")
        .is_some_and(|value| !value.is_null())
    {
        return Err(LoadError::Unsupported(
            "pinned GLM truncation must be null".into(),
        ));
    }
    if object.get("padding").is_some_and(|value| !value.is_null()) {
        return Err(LoadError::Unsupported(
            "pinned GLM padding must be null".into(),
        ));
    }

    let pre_tokenizer = parse_pre_tokenizer(required(object, "pre_tokenizer")?)?;
    validate_glm_pre_tokenizer(&pre_tokenizer)?;
    let post_processor = parse_post_processor(required(object, "post_processor")?)?;
    let decoder = parse_decoder(required(object, "decoder")?)?;

    let model = required_object(object, "model")?;
    require_string(model, "type", "BPE")?;
    let bpe = BpeConfig {
        dropout: optional_f32(model, "dropout")?,
        unk_token: optional_string(model, "unk_token")?,
        continuing_subword_prefix: optional_string(model, "continuing_subword_prefix")?,
        end_of_word_suffix: optional_string(model, "end_of_word_suffix")?,
        fuse_unk: required_bool(model, "fuse_unk")?,
        byte_fallback: required_bool(model, "byte_fallback")?,
        ignore_merges: required_bool(model, "ignore_merges")?,
    };
    if bpe.dropout.is_some()
        || bpe.unk_token.is_some()
        || bpe.continuing_subword_prefix.is_some()
        || bpe.end_of_word_suffix.is_some()
        || bpe.fuse_unk
        || bpe.byte_fallback
        || !bpe.ignore_merges
    {
        return Err(LoadError::Unsupported(
            "unexpected BPE flags for pinned GLM-5.2".into(),
        ));
    }

    let added_tokens = parse_added_tokens(required(object, "added_tokens")?)?;
    let mut entries = parse_vocab(required(model, "vocab")?)?;
    for token in &added_tokens {
        insert_entry(
            &mut entries,
            token.id,
            VocabEntry {
                symbol: Symbol::from_slice(token.content.as_bytes()),
                added: true,
                special: token.special,
            },
        )?;
    }
    let mut vocab = Vocab::default();
    for (expected, (id, entry)) in entries.into_iter().enumerate() {
        if id as usize != expected {
            return Err(LoadError::Parse(format!(
                "vocabulary id gap: expected {expected}, found {id}"
            )));
        }
        vocab
            .insert_at(id, entry.symbol, entry.added, entry.special)
            .map_err(LoadError::Parse)?;
    }

    let merges = parse_merges(required(model, "merges")?, &vocab)?;
    if vocab.len() != 154_856 || merges.len() != 321_649 || added_tokens.len() != 36 {
        return Err(LoadError::Unsupported(format!(
            "unexpected GLM artifact shape: vocab={}, merges={}, added={}",
            vocab.len(),
            merges.len(),
            added_tokens.len()
        )));
    }

    let special_tokens = added_tokens
        .iter()
        .filter(|token| token.special)
        .map(|token| token.content.clone())
        .collect();
    let config = TokenizerConfig {
        profile: TokenizerProfile::Glm52,
        family: ModelId::Glm5,
        special_tokens,
        split_specials: !added_tokens.is_empty(),
        ignore_merges: bpe.ignore_merges,
        bpe,
        added_tokens,
        pre_tokenizer: Some(pre_tokenizer),
        post_processor: Some(post_processor),
        decoder: Some(decoder),
        normalizer_is_none: true,
        truncation: crate::Truncation::None,
        padding: None,
    };

    Ok(Artifact {
        vocab,
        merges,
        config,
        provenance: empty_provenance(ArtifactFormat::HfTokenizersJson),
    })
}

fn parse_vocab(value: &serde_json::Value) -> Result<BTreeMap<TokenId, VocabEntry>, LoadError> {
    let object = value
        .as_object()
        .ok_or_else(|| LoadError::Unsupported("GLM vocab must be an object".into()))?;
    let mut entries = BTreeMap::new();
    for (token, id_value) in object {
        let id = as_token_id(id_value, "model.vocab id")?;
        insert_entry(
            &mut entries,
            id,
            VocabEntry {
                symbol: Symbol::from_slice(token.as_bytes()),
                added: false,
                special: false,
            },
        )?;
    }
    Ok(entries)
}

fn insert_entry(
    entries: &mut BTreeMap<TokenId, VocabEntry>,
    id: TokenId,
    entry: VocabEntry,
) -> Result<(), LoadError> {
    if entries.insert(id, entry).is_some() {
        return Err(LoadError::Parse(format!("duplicate token id {id}")));
    }
    Ok(())
}

fn parse_merges(value: &serde_json::Value, vocab: &Vocab) -> Result<Merges, LoadError> {
    let array = value
        .as_array()
        .ok_or_else(|| LoadError::Parse("model.merges must be an array".into()))?;
    let mut triples = Vec::with_capacity(array.len());
    for (rank, value) in array.iter().enumerate() {
        let (left, right) = match value {
            serde_json::Value::String(pair) => pair
                .split_once(' ')
                .ok_or_else(|| LoadError::Parse(format!("invalid merge string at rank {rank}")))?,
            serde_json::Value::Array(pair) if pair.len() == 2 => {
                let left = pair[0].as_str().ok_or_else(|| {
                    LoadError::Parse(format!("invalid left merge token at rank {rank}"))
                })?;
                let right = pair[1].as_str().ok_or_else(|| {
                    LoadError::Parse(format!("invalid right merge token at rank {rank}"))
                })?;
                (left, right)
            }
            _ => {
                return Err(LoadError::Parse(format!(
                    "invalid merge entry at rank {rank}"
                )));
            }
        };
        let left_symbol = Symbol::from_slice(left.as_bytes());
        let right_symbol = Symbol::from_slice(right.as_bytes());
        let mut merged = left_symbol.clone();
        merged.extend_from_slice(&right_symbol);
        let left_id = vocab
            .id_of(&left_symbol)
            .ok_or_else(|| LoadError::Parse(format!("unknown left merge token at rank {rank}")))?;
        let right_id = vocab
            .id_of(&right_symbol)
            .ok_or_else(|| LoadError::Parse(format!("unknown right merge token at rank {rank}")))?;
        let result_id = vocab
            .id_of(&merged)
            .ok_or_else(|| LoadError::Parse(format!("unknown merge result at rank {rank}")))?;
        triples.push((left_id, right_id, result_id));
    }
    Ok(Merges::from_triples(triples))
}

fn parse_added_tokens(value: &serde_json::Value) -> Result<Vec<AddedToken>, LoadError> {
    let array = value
        .as_array()
        .ok_or_else(|| LoadError::Parse("added_tokens must be an array".into()))?;
    array
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let object = value.as_object().ok_or_else(|| {
                LoadError::Parse(format!("added_tokens[{index}] must be an object"))
            })?;
            Ok(AddedToken {
                id: as_token_id(required(object, "id")?, "added token id")?,
                content: required_string(object, "content")?.to_string(),
                single_word: required_bool(object, "single_word")?,
                lstrip: required_bool(object, "lstrip")?,
                rstrip: required_bool(object, "rstrip")?,
                normalized: required_bool(object, "normalized")?,
                special: required_bool(object, "special")?,
            })
        })
        .collect()
}

fn parse_pre_tokenizer(value: &serde_json::Value) -> Result<PreTokenizerKind, LoadError> {
    let object = value
        .as_object()
        .ok_or_else(|| LoadError::Parse("pre_tokenizer must be an object".into()))?;
    match required_string(object, "type")? {
        "Sequence" => {
            let children = required(object, "pretokenizers")?
                .as_array()
                .ok_or_else(|| LoadError::Parse("pretokenizers must be an array".into()))?;
            children
                .iter()
                .map(parse_pre_tokenizer)
                .collect::<Result<Vec<_>, _>>()
                .map(PreTokenizerKind::Sequence)
        }
        "Split" => {
            let pattern_object = required_object(object, "pattern")?;
            let pattern = if let Some(regex) = pattern_object.get("Regex") {
                SplitPattern::Regex(
                    regex
                        .as_str()
                        .ok_or_else(|| LoadError::Parse("Split Regex must be a string".into()))?
                        .to_string(),
                )
            } else if let Some(string) = pattern_object.get("String") {
                SplitPattern::String(
                    string
                        .as_str()
                        .ok_or_else(|| LoadError::Parse("Split String must be a string".into()))?
                        .to_string(),
                )
            } else {
                return Err(LoadError::Unsupported(
                    "unknown Split pattern representation".into(),
                ));
            };
            Ok(PreTokenizerKind::Split(SplitConfig {
                pattern,
                behavior: parse_split_behavior(required_string(object, "behavior")?)?,
                invert: required_bool(object, "invert")?,
            }))
        }
        "ByteLevel" => Ok(PreTokenizerKind::ByteLevel(parse_byte_level(object)?)),
        other => Err(LoadError::Unsupported(format!(
            "pre-tokenizer type {other}"
        ))),
    }
}

fn validate_glm_pre_tokenizer(pre_tokenizer: &PreTokenizerKind) -> Result<(), LoadError> {
    let PreTokenizerKind::Sequence(children) = pre_tokenizer else {
        return Err(LoadError::Unsupported(
            "GLM pre-tokenizer must be a Sequence".into(),
        ));
    };
    if children.len() != 2 {
        return Err(LoadError::Unsupported(
            "GLM pre-tokenizer Sequence must have two children".into(),
        ));
    }
    match (&children[0], &children[1]) {
        (
            PreTokenizerKind::Split(SplitConfig {
                pattern: SplitPattern::Regex(pattern),
                behavior: SplitBehavior::Isolated,
                invert: false,
            }),
            PreTokenizerKind::ByteLevel(ByteLevelConfig {
                add_prefix_space: false,
                trim_offsets: true,
                use_regex: false,
            }),
        ) if pattern == GLM52_SPLIT_REGEX => Ok(()),
        _ => Err(LoadError::Unsupported(
            "GLM Split/ByteLevel configuration differs from the pinned profile".into(),
        )),
    }
}

fn parse_post_processor(value: &serde_json::Value) -> Result<PostProcessorKind, LoadError> {
    let object = value
        .as_object()
        .ok_or_else(|| LoadError::Parse("post_processor must be an object".into()))?;
    match required_string(object, "type")? {
        "ByteLevel" => Ok(PostProcessorKind::ByteLevel(parse_byte_level(object)?)),
        other => Err(LoadError::Unsupported(format!(
            "post-processor type {other}"
        ))),
    }
}

fn parse_decoder(value: &serde_json::Value) -> Result<DecoderKind, LoadError> {
    let object = value
        .as_object()
        .ok_or_else(|| LoadError::Parse("decoder must be an object".into()))?;
    match required_string(object, "type")? {
        "ByteLevel" => Ok(DecoderKind::ByteLevel(parse_byte_level(object)?)),
        other => Err(LoadError::Unsupported(format!("decoder type {other}"))),
    }
}

fn parse_byte_level(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<ByteLevelConfig, LoadError> {
    Ok(ByteLevelConfig {
        add_prefix_space: required_bool(object, "add_prefix_space")?,
        trim_offsets: required_bool(object, "trim_offsets")?,
        use_regex: required_bool(object, "use_regex")?,
    })
}

fn parse_split_behavior(value: &str) -> Result<SplitBehavior, LoadError> {
    match value {
        "Removed" => Ok(SplitBehavior::Removed),
        "Isolated" => Ok(SplitBehavior::Isolated),
        "MergedWithPrevious" => Ok(SplitBehavior::MergedWithPrevious),
        "MergedWithNext" => Ok(SplitBehavior::MergedWithNext),
        "Contiguous" => Ok(SplitBehavior::Contiguous),
        other => Err(LoadError::Unsupported(format!("Split behavior {other}"))),
    }
}

fn required<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'a serde_json::Value, LoadError> {
    object
        .get(field)
        .ok_or_else(|| LoadError::Parse(format!("missing field {field}")))
}

fn required_object<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, LoadError> {
    required(object, field)?
        .as_object()
        .ok_or_else(|| LoadError::Parse(format!("{field} must be an object")))
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'a str, LoadError> {
    required(object, field)?
        .as_str()
        .ok_or_else(|| LoadError::Parse(format!("{field} must be a string")))
}

fn require_string(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    expected: &str,
) -> Result<(), LoadError> {
    let actual = required_string(object, field)?;
    if actual == expected {
        Ok(())
    } else {
        Err(LoadError::Unsupported(format!(
            "{field}={actual}, expected {expected}"
        )))
    }
}

fn required_bool(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<bool, LoadError> {
    required(object, field)?
        .as_bool()
        .ok_or_else(|| LoadError::Parse(format!("{field} must be a boolean")))
}

fn optional_string(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<Option<String>, LoadError> {
    match required(object, field)? {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::String(value) => Ok(Some(value.clone())),
        _ => Err(LoadError::Parse(format!("{field} must be string or null"))),
    }
}

fn optional_f32(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<Option<f32>, LoadError> {
    match required(object, field)? {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(value) => value
            .as_f64()
            .map(|value| Some(value as f32))
            .ok_or_else(|| LoadError::Parse(format!("{field} must be numeric or null"))),
        _ => Err(LoadError::Parse(format!("{field} must be numeric or null"))),
    }
}

fn as_token_id(value: &serde_json::Value, context: &str) -> Result<TokenId, LoadError> {
    let id = value
        .as_u64()
        .ok_or_else(|| LoadError::Parse(format!("{context} must be an unsigned integer")))?;
    TokenId::try_from(id).map_err(|_| LoadError::Parse(format!("{context} exceeds u32")))
}

fn parse_tiktoken(bytes: &[u8]) -> Result<Artifact, LoadError> {
    let mut vocab = Vocab::default();
    for line in bytes.split(|&byte| byte == b'\n') {
        let line = line.split(|&byte| byte == b'\r').next().unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let Some(space) = line.iter().position(|&byte| byte == b' ') else {
            return Err(LoadError::Parse("malformed tiktoken line".into()));
        };
        let token = base64_decode_table::b64decode(&line[..space])
            .map_err(|_| LoadError::Parse("invalid tiktoken base64".into()))?;
        vocab.add(Symbol::from_slice(&token), false);
    }
    if vocab.is_empty() {
        return Err(LoadError::Parse("no entries in .tiktoken file".into()));
    }
    Ok(Artifact {
        vocab,
        merges: Merges::default(),
        config: TokenizerConfig::default(),
        provenance: empty_provenance(ArtifactFormat::Tiktoken),
    })
}

fn empty_provenance(format: ArtifactFormat) -> Provenance {
    Provenance {
        format,
        format_version: String::new(),
        blake3: [0; 32],
        source_path: None,
    }
}

fn extract_version(bytes: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    value
        .get("version")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

fn blake3(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

mod base64_decode_table {
    pub fn b64decode(input: &[u8]) -> Result<Vec<u8>, ()> {
        let mut output = Vec::with_capacity(input.len() * 3 / 4);
        let mut buffer = 0u32;
        let mut bits = 0u32;
        for &byte in input {
            let value = match byte {
                b'A'..=b'Z' => byte - b'A',
                b'a'..=b'z' => byte - b'a' + 26,
                b'0'..=b'9' => byte - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => break,
                _ => return Err(()),
            } as u32;
            buffer = (buffer << 6) | value;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                output.push((buffer >> bits) as u8);
                buffer &= (1 << bits) - 1;
            }
        }
        Ok(output)
    }
}
