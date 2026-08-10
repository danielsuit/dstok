use dstok_model::loader::load_glm52_path;
use dstok_model::Artifact;
use dstok_oracle::{HfTokenizersOracle, Oracle};
use proptest::prelude::*;
use std::sync::OnceLock;

fn fixture(relative: &str) -> String {
    format!(
        "{}/../fixtures/glm-5.2/{relative}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn artifact() -> &'static Artifact {
    static ARTIFACT: OnceLock<Artifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| {
        load_glm52_path(fixture("tokenizer.json")).expect("load pinned GLM-5.2 artifact")
    })
}

fn hf() -> &'static HfTokenizersOracle {
    static HF: OnceLock<HfTokenizersOracle> = OnceLock::new();
    HF.get_or_init(|| {
        HfTokenizersOracle::from_file(&fixture("tokenizer.json"))
            .expect("load HF tokenizers oracle")
    })
}

fn assert_ids(text: &str) {
    let expected = hf().encode(text).ids;
    let core = dstok_core::encode_to_ids(artifact(), text);
    let fast = dstok_fast::encode_ids_only(artifact(), text);
    let parallel = dstok_fast::encode_ids_only_parallel(artifact(), text, 4);
    assert_eq!(core, expected, "core/HF mismatch for {text:?}");
    assert_eq!(fast, expected, "fast/HF mismatch for {text:?}");
    assert_eq!(parallel, expected, "parallel/HF mismatch for {text:?}");
    assert_eq!(dstok_core::count(artifact(), text), expected.len());
    assert_eq!(dstok_fast::count(artifact(), text), expected.len());
    assert_eq!(
        dstok_fast::count_parallel(artifact(), text, 4),
        expected.len()
    );
}

#[test]
fn frozen_gateway_rendered_prompts_match_hf_ids() {
    let prompts: Vec<String> =
        serde_json::from_slice(&std::fs::read(fixture("rendered-prompts.json")).expect("prompts"))
            .expect("valid prompt fixture");
    for prompt in prompts {
        assert_ids(&prompt);
    }
}

#[test]
fn overlapping_added_tokens_and_ignore_merges_match_hf() {
    for text in [
        "<think>",
        "x<think>y</think>z",
        "<tool_call></tool_call>",
        "<|assistant|><|assistant|>",
        "<|code_prefix|><|code_middle|><|code_suffix|>",
        " имущества",
    ] {
        assert_ids(text);
    }
}

#[test]
fn exact_segment_count_cache_matches_hf_and_reuses_repeated_segments() {
    dstok_fast::clear_global_cache();
    let repeated = "A👩🏽‍💻e\u{301}\r\nfn cached_segment(x: usize) -> usize { x + 1234 }\n";
    let text = format!(
        "<|user|>{repeated}<|assistant|>dstok-unique-cache-middle-7f83c1\
         <|user|>{repeated}<|assistant|>"
    );
    let expected = hf().encode(&text).ids.len();

    let (cold_count, cold_stats) = dstok_fast::count_with_stats(artifact(), &text);
    assert_eq!(cold_count, expected);
    assert!(
        cold_stats.segment_misses >= 1,
        "at least one unique text segment must be computed"
    );
    assert!(
        cold_stats.segment_hits >= 1,
        "an identical complete text segment should be reused within one count"
    );

    let (warm_count, warm_stats) = dstok_fast::count_with_stats(artifact(), &text);
    assert_eq!(warm_count, expected);
    assert!(
        warm_stats.segment_hits >= 3,
        "every non-empty text segment should be reusable on a warm count"
    );
    assert_eq!(
        dstok_fast::encode_ids_only(artifact(), &text).len(),
        warm_count
    );
}

#[test]
fn segment_cache_never_reuses_changed_or_newly_formed_boundaries() {
    for text in [
        "<|user",
        "<|user|>",
        "<|user|>alpha  ",
        "<|user|>alpha  x",
        "<|user|>alpha  x<|assistant",
        "<|user|>alpha  x<|assistant|>",
        "<|user|>alpha  x<|assistant|><|user|>",
        "<|user|>alpha  x<|assistant|><think>partial",
        "<|user|>alpha  x<|assistant|><think>partial</think>",
        "<|user|>中文👩🏽‍💻\r\n  x<|assistant|>done",
    ] {
        let expected = hf().encode(text).ids;
        let (actual, _) = dstok_fast::count_with_stats(artifact(), text);
        assert_eq!(actual, expected.len(), "cached count mismatch for {text:?}");
        assert_eq!(
            dstok_fast::encode_ids_only(artifact(), text),
            expected,
            "ID mismatch for {text:?}"
        );
    }
}

#[test]
fn append_only_cached_histories_match_full_hf_recounts() {
    let mut history =
        String::from("<|system|>You are a coding agent.\n<|user|>Preserve exact behavior.\n");
    for turn in 0..24 {
        history.push_str("<|assistant|><think>inspect</think><tool_call>{\"path\":\"src/lib.rs\"}</tool_call><|observation|>");
        history.push_str(&format!(
            "turn={turn}\r\nfn parse_{turn}(input: &str) {{ /* 中文 👩🏽‍💻 */ }}\n"
        ));
        history.push_str("<|assistant|>Done.<|user|>Continue with the next exact change.\n");

        let expected = hf().encode(&history).ids;
        let (actual, _) = dstok_fast::count_with_stats(artifact(), &history);
        assert_eq!(
            actual,
            expected.len(),
            "append-only cached count mismatch after turn {turn}"
        );
    }
}

#[test]
fn incremental_flush_is_equal_at_arbitrary_byte_boundaries() {
    let text = "A👩🏽‍💻é<tool_call>{\"x\":1234}</tool_call>\r\n中文";
    for split in 0..=text.len() {
        let mut incremental = dstok_fast::IncrementalEncoder::new(artifact());
        incremental.push(&text.as_bytes()[..split]);
        incremental.push(&text.as_bytes()[split..]);
        let ids = incremental.flush().to_vec();
        assert_eq!(ids, hf().encode(text).ids, "split at byte {split}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        max_shrink_iters: 2048,
        .. ProptestConfig::default()
    })]

    #[test]
    fn arbitrary_unicode_matches_hf(text in "\\PC{0,512}") {
        assert_ids(&text);
    }

    #[test]
    fn serial_parallel_and_count_are_deterministic(
        chunks in prop::collection::vec("\\PC{0,64}", 0..32)
    ) {
        let text = chunks.concat();
        let serial = dstok_fast::encode_ids_only(artifact(), &text);
        prop_assert_eq!(
            dstok_fast::encode_ids_only_parallel(artifact(), &text, 2),
            serial.clone()
        );
        prop_assert_eq!(
            dstok_fast::encode_ids_only_parallel(artifact(), &text, 8),
            serial.clone()
        );
        prop_assert_eq!(dstok_fast::count(artifact(), &text), serial.len());
        prop_assert_eq!(dstok_fast::count_parallel(artifact(), &text, 4), serial.len());
    }

    #[test]
    fn cached_append_sequences_match_full_hf(
        chunks in prop::collection::vec("\\PC{0,64}", 0..8)
    ) {
        let mut text = String::from("<|system|>exact append parity");
        for (turn, chunk) in chunks.iter().enumerate() {
            text.push_str("<|user|>");
            text.push_str(chunk);
            text.push_str("<|assistant|>");
            text.push_str(if turn % 2 == 0 { "ok\r\n  " } else { "<think>ok</think>" });

            let expected = hf().encode(&text).ids;
            let actual = dstok_fast::count(artifact(), &text);
            prop_assert_eq!(actual, expected.len(), "append turn {}", turn);
        }
    }
}
