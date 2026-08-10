use dstok_model::config::{
    ByteLevelConfig, DecoderKind, PostProcessorKind, PreTokenizerKind, SplitBehavior, SplitConfig,
    SplitPattern, TokenizerProfile, GLM52_SPLIT_REGEX,
};
use dstok_model::family::ModelId;
use dstok_model::loader::{load_bytes, load_glm52_path, LoadError};

fn fixture_path() -> String {
    format!(
        "{}/../fixtures/glm-5.2/tokenizer.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn pinned_glm52_structure_is_loaded_exactly() {
    let artifact = load_glm52_path(fixture_path()).expect("load pinned GLM fixture");
    assert_eq!(artifact.config.profile, TokenizerProfile::Glm52);
    assert_eq!(artifact.config.family, ModelId::Glm5);
    assert!(artifact.config.ignore_merges);
    assert_eq!(artifact.vocab.len(), 154_856);
    assert_eq!(artifact.merges.len(), 321_649);
    assert_eq!(artifact.config.added_tokens.len(), 36);
    assert_eq!(artifact.config.special_tokens.len(), 18);

    assert_eq!(
        artifact.config.pre_tokenizer,
        Some(PreTokenizerKind::Sequence(vec![
            PreTokenizerKind::Split(SplitConfig {
                pattern: SplitPattern::Regex(GLM52_SPLIT_REGEX.to_string()),
                behavior: SplitBehavior::Isolated,
                invert: false,
            }),
            PreTokenizerKind::ByteLevel(ByteLevelConfig {
                add_prefix_space: false,
                trim_offsets: true,
                use_regex: false,
            }),
        ]))
    );
    assert_eq!(
        artifact.config.post_processor,
        Some(PostProcessorKind::ByteLevel(ByteLevelConfig {
            add_prefix_space: true,
            trim_offsets: false,
            use_regex: true,
        }))
    );
    assert_eq!(
        artifact.config.decoder,
        Some(DecoderKind::ByteLevel(ByteLevelConfig {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }))
    );

    for token in &artifact.config.added_tokens {
        assert!(!token.single_word);
        assert!(!token.lstrip);
        assert!(!token.rstrip);
        assert!(!token.normalized);
        assert_eq!(
            artifact.vocab.added_map().get(token.content.as_bytes()),
            Some(&token.id)
        );
    }
}

#[test]
fn hf_json_with_a_different_digest_fails_closed() {
    let mut bytes = std::fs::read(fixture_path()).expect("fixture");
    let index = bytes
        .windows(b"GLM-5.2".len())
        .position(|window| window == b"GLM-5.2")
        .unwrap_or(0);
    bytes[index] ^= 1;
    let expected = *blake3::hash(&bytes).as_bytes();
    let error = load_bytes(&bytes, expected, None).expect_err("digest must be pinned");
    assert!(matches!(error, LoadError::Unsupported(_)));
}
