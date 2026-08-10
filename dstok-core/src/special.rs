//! Special-token splitting — the oracle version.
//!
//! Scans the input for the literal bytes of any registered special token and
//! splits the input into alternating `Text` and `Special` segments. The
//! encoder then pretokenizes only the `Text` segments and emits specials
//! verbatim.
//!
//! Safety property: when `split` is false, or when
//! `allow_special_in_text` is false, user text containing a special token's
//! bytes is *never* promoted to that special — it is tokenized as ordinary
//! bytes. The oracle is the reference behavior for this property.

use crate::regex::Span;
use dstok_model::vocab::Symbol;
use dstok_model::TokenId;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::sync::OnceLock;

/// A segment of the input produced by special-token splitting.
#[derive(Debug, Clone)]
pub enum Segment<'a> {
    /// Ordinary text to be pretokenized + BPE-merged.
    Text { bytes: &'a [u8], offset: usize },
    /// A verbatim special token: its id + the byte span it occupied.
    Special {
        id: TokenId,
        span: Span,
        is_special: bool,
    },
}

/// Split `input` into Text/Special segments.
///
/// `split`: whether to recognize special tokens at all. When false, the whole
/// input is one Text segment (the special-token-safety property: literals are
/// never promoted unless explicitly enabled).
pub fn split_specials<'a>(
    vocab: &dstok_model::Vocab,
    input: &'a [u8],
    split: bool,
    profile: dstok_model::TokenizerProfile,
) -> SmallVec<[Segment<'a>; 8]> {
    let mut out = SmallVec::new();
    if !split {
        out.push(Segment::Text {
            bytes: input,
            offset: 0,
        });
        return out;
    }

    if profile == dstok_model::TokenizerProfile::Glm52 {
        return split_with_glm_matcher(vocab, input);
    }

    split_with_fallback(vocab, input)
}

struct AddedMatcher {
    automaton: aho_corasick::AhoCorasick,
    ids: Vec<TokenId>,
    special_ids: HashSet<TokenId>,
}

fn split_with_glm_matcher<'a>(
    vocab: &dstok_model::Vocab,
    input: &'a [u8],
) -> SmallVec<[Segment<'a>; 8]> {
    static MATCHER: OnceLock<AddedMatcher> = OnceLock::new();
    let matcher = MATCHER.get_or_init(|| {
        let mut added: Vec<_> = vocab
            .added_map()
            .iter()
            .map(|(symbol, id)| (*id, symbol.as_slice().to_vec()))
            .collect();
        added.sort_by_key(|(id, _)| *id);
        let ids = added.iter().map(|(id, _)| *id).collect();
        let patterns: Vec<&[u8]> = added.iter().map(|(_, symbol)| symbol.as_slice()).collect();
        AddedMatcher {
            automaton: aho_corasick::AhoCorasickBuilder::new()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(patterns)
                .expect("GLM added-token automaton"),
            ids,
            special_ids: vocab.special_map().values().copied().collect(),
        }
    });
    let mut output = SmallVec::new();
    let mut previous = 0;
    for matched in matcher.automaton.find_iter(input) {
        if previous < matched.start() {
            output.push(Segment::Text {
                bytes: &input[previous..matched.start()],
                offset: previous,
            });
        }
        let id = matcher.ids[matched.pattern().as_usize()];
        output.push(Segment::Special {
            id,
            span: Span {
                start: matched.start(),
                end: matched.end(),
            },
            is_special: matcher.special_ids.contains(&id),
        });
        previous = matched.end();
    }
    if previous < input.len() {
        output.push(Segment::Text {
            bytes: &input[previous..],
            offset: previous,
        });
    }
    if output.is_empty() {
        output.push(Segment::Text {
            bytes: &[],
            offset: 0,
        });
    }
    output
}

fn split_with_fallback<'a>(
    vocab: &dstok_model::Vocab,
    input: &'a [u8],
) -> SmallVec<[Segment<'a>; 8]> {
    let mut out = SmallVec::new();
    // Collect (symbol bytes, flagged id) for each special, sorted by length
    // descending so the longest match wins at any position (HF semantics: the
    // tokenizer is configured with a specific special-token set; ties broken
    // by longest, then by first registration).
    let mut specials: Vec<(&Symbol, TokenId)> =
        vocab.added_map().iter().map(|(s, id)| (s, *id)).collect();
    specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut i = 0usize;
    let n = input.len();
    let mut text_start = 0usize;

    while i < n {
        let mut matched: Option<(usize, TokenId)> = None;
        for (sym, id) in &specials {
            let m = sym.len();
            if m > 0 && i + m <= n && &input[i..i + m] == sym.as_slice() {
                matched = Some((m, *id));
                break;
            }
        }
        if let Some((m, id)) = matched {
            // flush preceding text
            if i > text_start {
                out.push(Segment::Text {
                    bytes: &input[text_start..i],
                    offset: text_start,
                });
            }
            let is_special = vocab
                .special_map()
                .values()
                .any(|candidate| *candidate == id);
            out.push(Segment::Special {
                id,
                span: Span {
                    start: i,
                    end: i + m,
                },
                is_special,
            });
            i += m;
            text_start = i;
        } else {
            i += 1;
        }
    }
    if text_start < n {
        out.push(Segment::Text {
            bytes: &input[text_start..n],
            offset: text_start,
        });
    }
    if out.is_empty() {
        out.push(Segment::Text {
            bytes: &[],
            offset: 0,
        });
    }
    out
}
