//! Exact scalar encoding for the pinned GLM-5.2 profile.
//!
//! Full-ID and count-only APIs share one piece planner and one BPE merge path;
//! count-only differs solely in whether emitted IDs are retained.

use crate::merge::bpe_merge_pretoken;
use crate::offsets::{Offset, OffsetList};
use crate::regex::{for_each_glm52_span, pretokenize};
use crate::special::{split_specials, Segment};
use dstok_model::{Artifact, TokenId, TokenizerProfile};
use smallvec::SmallVec;

#[derive(Debug, Clone, Default)]
pub struct Encoding {
    pub ids: Vec<TokenId>,
    pub offsets: OffsetList,
    pub specials: Vec<bool>,
}

impl Encoding {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// A fully ordered unit of work after added-token extraction and exact
/// pre-tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    Pretoken {
        start: usize,
        end: usize,
    },
    Added {
        id: TokenId,
        start: usize,
        end: usize,
        is_special: bool,
    },
}

trait TokenSink {
    fn push(&mut self, id: TokenId, offset: Offset, special: bool);
    fn len(&self) -> usize;
    fn truncate(&mut self, max_length: usize);
}

#[derive(Default)]
struct EncodingSink {
    ids: Vec<TokenId>,
    offsets: OffsetList,
    specials: Vec<bool>,
}

impl TokenSink for EncodingSink {
    #[inline]
    fn push(&mut self, id: TokenId, offset: Offset, special: bool) {
        self.ids.push(id);
        self.offsets.push(offset);
        self.specials.push(special);
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn truncate(&mut self, max_length: usize) {
        self.ids.truncate(max_length);
        self.offsets.truncate(max_length);
        self.specials.truncate(max_length);
    }
}

#[derive(Default)]
struct IdSink(Vec<TokenId>);

impl TokenSink for IdSink {
    #[inline]
    fn push(&mut self, id: TokenId, _offset: Offset, _special: bool) {
        self.0.push(id);
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn truncate(&mut self, max_length: usize) {
        self.0.truncate(max_length);
    }
}

pub fn encode(artifact: &Artifact, text: &str) -> Encoding {
    encode_with_offsets(artifact, text)
}

pub fn encode_with_offsets(artifact: &Artifact, text: &str) -> Encoding {
    let mut sink = EncodingSink {
        ids: Vec::with_capacity(text.len() / 3 + 8),
        offsets: SmallVec::with_capacity(text.len() / 3 + 8),
        specials: Vec::with_capacity(text.len() / 3 + 8),
    };
    emit_all(artifact, text, &mut sink);
    Encoding {
        ids: sink.ids,
        offsets: sink.offsets,
        specials: sink.specials,
    }
}

pub fn encode_to_ids(artifact: &Artifact, text: &str) -> Vec<TokenId> {
    let mut sink = IdSink(Vec::with_capacity(text.len() / 3 + 8));
    emit_all(artifact, text, &mut sink);
    sink.0
}

/// Count tokens without allocating an output ID/offset/mask encoding.
pub fn count(artifact: &Artifact, text: &str) -> usize {
    let mut count = 0usize;
    for_each_piece(artifact, text, |piece| {
        let emitted = match piece {
            Piece::Added { .. } => 1,
            Piece::Pretoken { start, end } => {
                bpe_merge_pretoken(artifact, &text.as_bytes()[start..end])
                    .0
                    .len()
            }
        };
        count = count
            .checked_add(emitted)
            .expect("token count exceeds usize");
    });
    match artifact.config.truncation {
        dstok_model::Truncation::None => count,
        dstok_model::Truncation::LongestFirst { max_length }
        | dstok_model::Truncation::OnlyFirst { max_length } => count.min(max_length),
    }
}

fn emit_all<S: TokenSink>(artifact: &Artifact, text: &str, sink: &mut S) {
    for_each_piece(artifact, text, |piece| match piece {
        Piece::Added {
            id,
            start,
            end,
            is_special,
        } => sink.push(id, Offset { start, end }, is_special),
        Piece::Pretoken { start, end } => {
            let (ids, byte_lens) = bpe_merge_pretoken(artifact, &text.as_bytes()[start..end]);
            let mut byte_position = start;
            for (&id, &length) in ids.iter().zip(byte_lens.iter()) {
                sink.push(
                    id,
                    Offset {
                        start: byte_position,
                        end: byte_position + length,
                    },
                    false,
                );
                byte_position += length;
            }
        }
    });
    apply_truncation(artifact, sink);
}

/// Plan exact pretoken boundaries. Parallel implementations may distribute
/// these independent pieces, but must preserve this order when collecting IDs.
pub fn plan_pieces(artifact: &Artifact, text: &str) -> Vec<Piece> {
    let mut pieces = Vec::with_capacity(text.len() / 4 + 8);
    for_each_piece(artifact, text, |piece| pieces.push(piece));
    pieces
}

/// Stream exact pretoken spans for one ordinary-text segment.
///
/// The caller must perform added-token extraction first. Keeping this step
/// separate lets count-only implementations cache a complete text segment
/// without changing the pretokenizer or BPE semantics.
pub fn for_each_pretoken_span(
    artifact: &Artifact,
    text: &str,
    mut emit: impl FnMut(crate::regex::Span),
) {
    match artifact.config.profile {
        TokenizerProfile::Glm52 => for_each_glm52_span(text, emit),
        TokenizerProfile::Synthetic => {
            for span in pretokenize(text.as_bytes(), artifact.config.family.digit_run()) {
                emit(span);
            }
        }
    }
}

/// Stream exact pieces without retaining an input-sized planning vector.
pub fn for_each_piece(artifact: &Artifact, text: &str, mut emit: impl FnMut(Piece)) {
    let input = text.as_bytes();
    for segment in split_specials(
        &artifact.vocab,
        input,
        artifact.config.split_specials,
        artifact.config.profile,
    ) {
        match segment {
            Segment::Text { bytes, offset } => {
                if bytes.is_empty() {
                    continue;
                }
                let segment_text = std::str::from_utf8(bytes)
                    .expect("segments of a valid UTF-8 input remain UTF-8");
                for_each_pretoken_span(artifact, segment_text, |span| {
                    emit(Piece::Pretoken {
                        start: offset + span.start,
                        end: offset + span.end,
                    });
                });
            }
            Segment::Special {
                id,
                span,
                is_special,
            } => emit(Piece::Added {
                id,
                start: span.start,
                end: span.end,
                is_special,
            }),
        }
    }
}

fn apply_truncation<S: TokenSink>(artifact: &Artifact, sink: &mut S) {
    let max_length = match artifact.config.truncation {
        dstok_model::Truncation::None => return,
        dstok_model::Truncation::LongestFirst { max_length }
        | dstok_model::Truncation::OnlyFirst { max_length } => max_length,
    };
    if sink.len() > max_length {
        sink.truncate(max_length);
    }
}
