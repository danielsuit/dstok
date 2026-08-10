//! Exact cached and pretoken-boundary-parallel encoding.

use crate::cache::{get_or_compute, get_or_compute_segment_count, CacheStats, TokenRun};
use dstok_core::{Offset, Piece, Segment};
use dstok_model::{Artifact, TokenId, Truncation};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::Arc;

pub use dstok_core::Encoding;

#[derive(Clone)]
enum EncodedPiece {
    Added {
        id: TokenId,
        start: usize,
        end: usize,
        is_special: bool,
    },
    Pretoken {
        start: usize,
        run: Arc<TokenRun>,
    },
}

pub fn encode(artifact: &Artifact, text: &str) -> Encoding {
    encode_exact(artifact, text, 1, None)
}

pub fn encode_parallel(artifact: &Artifact, text: &str, threads: usize) -> Encoding {
    encode_exact(artifact, text, threads.max(1), None)
}

pub fn encode_with_stats(artifact: &Artifact, text: &str) -> (Encoding, CacheStats) {
    let mut stats = CacheStats::default();
    let encoding = encode_exact(artifact, text, 1, Some(&mut stats));
    (encoding, stats)
}

fn encode_exact(
    artifact: &Artifact,
    text: &str,
    threads: usize,
    mut stats: Option<&mut CacheStats>,
) -> Encoding {
    let pieces = dstok_core::plan_pieces(artifact, text);
    let encoded = if threads <= 1 || pieces.len() <= 1 {
        pieces
            .iter()
            .map(|piece| encode_piece(artifact, text, *piece, stats.as_deref_mut()))
            .collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        pool.install(|| {
            pieces
                .par_iter()
                .map(|piece| encode_piece(artifact, text, *piece, None))
                .collect()
        })
    };
    collect_encoding(artifact, encoded)
}

fn encode_piece(
    artifact: &Artifact,
    text: &str,
    piece: Piece,
    stats: Option<&mut CacheStats>,
) -> EncodedPiece {
    match piece {
        Piece::Added {
            id,
            start,
            end,
            is_special,
        } => EncodedPiece::Added {
            id,
            start,
            end,
            is_special,
        },
        Piece::Pretoken { start, end } => EncodedPiece::Pretoken {
            start,
            run: get_or_compute(artifact, &text.as_bytes()[start..end], stats),
        },
    }
}

fn collect_encoding(artifact: &Artifact, pieces: Vec<EncodedPiece>) -> Encoding {
    let capacity = pieces
        .iter()
        .map(|piece| match piece {
            EncodedPiece::Added { .. } => 1,
            EncodedPiece::Pretoken { run, .. } => run.ids.len(),
        })
        .sum();
    let mut ids = Vec::with_capacity(capacity);
    let mut offsets: SmallVec<[Offset; 64]> = SmallVec::with_capacity(capacity);
    let mut specials = Vec::with_capacity(capacity);
    for piece in pieces {
        match piece {
            EncodedPiece::Added {
                id,
                start,
                end,
                is_special,
            } => {
                ids.push(id);
                offsets.push(Offset { start, end });
                specials.push(is_special);
            }
            EncodedPiece::Pretoken { start, run } => {
                let mut position = start;
                for (&id, &length) in run.ids.iter().zip(run.byte_lens.iter()) {
                    ids.push(id);
                    offsets.push(Offset {
                        start: position,
                        end: position + length,
                    });
                    specials.push(false);
                    position += length;
                }
            }
        }
    }
    truncate(artifact, &mut ids, &mut offsets, &mut specials);
    Encoding {
        ids,
        offsets,
        specials,
    }
}

pub fn encode_ids_only(artifact: &Artifact, text: &str) -> Vec<TokenId> {
    encode_ids_streaming(artifact, text, None)
}

pub fn encode_ids_only_into(artifact: &Artifact, text: &str, output: &mut Vec<TokenId>) {
    let ids = encode_ids_streaming(artifact, text, None);
    output.extend_from_slice(&ids);
}

pub fn encode_ids_only_parallel(artifact: &Artifact, text: &str, threads: usize) -> Vec<TokenId> {
    if threads <= 1 {
        return encode_ids_only(artifact, text);
    }
    let pieces = dstok_core::plan_pieces(artifact, text);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    let encoded: Vec<EncodedPiece> = pool.install(|| {
        pieces
            .par_iter()
            .map(|piece| encode_piece(artifact, text, *piece, None))
            .collect()
    });
    collect_ids(artifact, encoded)
}

fn encode_ids_streaming(
    artifact: &Artifact,
    text: &str,
    mut stats: Option<&mut CacheStats>,
) -> Vec<TokenId> {
    let mut ids = Vec::with_capacity(text.len() / 3 + 8);
    dstok_core::for_each_piece(artifact, text, |piece| match piece {
        Piece::Added { id, .. } => ids.push(id),
        Piece::Pretoken { start, end } => {
            let run = get_or_compute(artifact, &text.as_bytes()[start..end], stats.as_deref_mut());
            ids.extend_from_slice(&run.ids);
        }
    });
    let max = match artifact.config.truncation {
        Truncation::None => ids.len(),
        Truncation::LongestFirst { max_length } | Truncation::OnlyFirst { max_length } => {
            max_length
        }
    };
    ids.truncate(max);
    ids
}

fn collect_ids(artifact: &Artifact, pieces: Vec<EncodedPiece>) -> Vec<TokenId> {
    let capacity = pieces
        .iter()
        .map(|piece| match piece {
            EncodedPiece::Added { .. } => 1,
            EncodedPiece::Pretoken { run, .. } => run.ids.len(),
        })
        .sum();
    let mut ids = Vec::with_capacity(capacity);
    for piece in pieces {
        match piece {
            EncodedPiece::Added { id, .. } => ids.push(id),
            EncodedPiece::Pretoken { run, .. } => ids.extend_from_slice(&run.ids),
        }
    }
    let max = match artifact.config.truncation {
        Truncation::None => ids.len(),
        Truncation::LongestFirst { max_length } | Truncation::OnlyFirst { max_length } => {
            max_length
        }
    };
    ids.truncate(max);
    ids
}

pub fn encode_batch(artifact: &Artifact, texts: &[&str], threads: usize) -> Vec<Encoding> {
    if threads <= 1 || texts.len() <= 1 {
        return texts.iter().map(|text| encode(artifact, text)).collect();
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    pool.install(|| {
        texts
            .par_iter()
            .map(|text| encode(artifact, text))
            .collect()
    })
}

pub fn encode_ids_only_batch(
    artifact: &Artifact,
    texts: &[&str],
    threads: usize,
) -> Vec<Vec<TokenId>> {
    encode_batch(artifact, texts, threads)
        .into_iter()
        .map(|encoding| encoding.ids)
        .collect()
}

pub fn encode_batch_with_stats(artifact: &Artifact, texts: &[&str]) -> Vec<(Encoding, CacheStats)> {
    texts
        .iter()
        .map(|text| encode_with_stats(artifact, text))
        .collect()
}

/// True count-only API: no full output ID/offset/mask vectors are retained.
pub fn count(artifact: &Artifact, text: &str) -> usize {
    count_with_stats(artifact, text).0
}

pub fn count_with_stats(artifact: &Artifact, text: &str) -> (usize, CacheStats) {
    let mut stats = CacheStats::default();
    let mut count = 0usize;
    for segment in dstok_core::split_specials(
        &artifact.vocab,
        text.as_bytes(),
        artifact.config.split_specials,
        artifact.config.profile,
    ) {
        count = count
            .checked_add(match segment {
                Segment::Special { .. } => 1,
                Segment::Text { bytes, .. } => {
                    let segment_text = std::str::from_utf8(bytes)
                        .expect("segments of a valid UTF-8 input remain UTF-8");
                    get_or_compute_segment_count(artifact, bytes, &mut stats, |stats| {
                        count_text_segment(artifact, segment_text, stats)
                    })
                }
            })
            .expect("token count exceeds usize");
    }
    (truncate_count(artifact, count), stats)
}

fn count_text_segment(artifact: &Artifact, text: &str, stats: &mut CacheStats) -> usize {
    let bytes = text.as_bytes();
    let mut count = 0usize;
    dstok_core::for_each_pretoken_span(artifact, text, |span| {
        count = count
            .checked_add(get_or_compute(artifact, &bytes[span.start..span.end], Some(stats)).len())
            .expect("token count exceeds usize");
    });
    count
}

pub fn count_parallel(artifact: &Artifact, text: &str, threads: usize) -> usize {
    let pieces = dstok_core::plan_pieces(artifact, text);
    if threads <= 1 || pieces.len() <= 1 {
        return count(artifact, text);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    let count = pool.install(|| {
        pieces
            .par_iter()
            .map(|piece| match *piece {
                Piece::Added { .. } => 1,
                Piece::Pretoken { start, end } => {
                    get_or_compute(artifact, &text.as_bytes()[start..end], None).len()
                }
            })
            .sum()
    });
    truncate_count(artifact, count)
}

fn truncate(
    artifact: &Artifact,
    ids: &mut Vec<TokenId>,
    offsets: &mut SmallVec<[Offset; 64]>,
    specials: &mut Vec<bool>,
) {
    let max = match artifact.config.truncation {
        Truncation::None => return,
        Truncation::LongestFirst { max_length } | Truncation::OnlyFirst { max_length } => {
            max_length
        }
    };
    ids.truncate(max);
    offsets.truncate(max);
    specials.truncate(max);
}

fn truncate_count(artifact: &Artifact, count: usize) -> usize {
    match artifact.config.truncation {
        Truncation::None => count,
        Truncation::LongestFirst { max_length } | Truncation::OnlyFirst { max_length } => {
            count.min(max_length)
        }
    }
}
