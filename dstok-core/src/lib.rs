//! # dstok-core — exact safe scalar tokenization
//!
//! The safe scalar implementation shared by full-ID and count-only APIs.
//! `dstok-fast` delegates cache misses and tokenization decisions here, and
//! differential tests compare its output with the pinned Hugging Face oracle.

#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(warnings))]

pub mod byte_map;
pub mod decode;
pub mod encode;
pub mod merge;
pub mod offsets;
pub mod regex;
pub mod special;

pub use decode::{decode, decode_encoding, decode_into, decode_to_string, decode_to_string_into};
pub use encode::{
    count, encode, encode_to_ids, encode_with_offsets, for_each_piece, for_each_pretoken_span,
    plan_pieces, Encoding, Piece,
};
pub use merge::{bpe_merge, bpe_merge_pretoken, merge_all, merge_lookup, seed_word};
pub use offsets::{Offset, OffsetList};
pub use regex::{for_each_glm52_span, pretokenize, pretokenize_glm52};
pub use special::{split_specials, Segment};
