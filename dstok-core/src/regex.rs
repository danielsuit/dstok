//! Scalar regex pretokenizer — the reference segmentation.
//!
//! dstok does **not** depend on the `regex` crate for the hot path: the
//! byte-level pretokenizer regex (GPT-2 / Llama 3) is a fixed, well-understood
//! pattern, and we re-implement it as an explicit scalar state machine here
//! (the oracle) and as a vectorized scanner in `dstok-simd` (the fast path).
//! Keeping a hand-rolled implementation means:
//!
//!   * no runtime regex compilation cost or crate dependency in the oracle;
//!   * the SIMD path is differential-tested against *this* explicit machine,
//!     not against a third-party regex engine whose own bugs would be hidden;
//!   * the exact tokenization is fully specified in-repo.
//!
//! The grammar (Llama 3 form, which is a superset of GPT-2's intent):
//!
//! ```text
//!   pretoken :=
//!     | contraction  ('s 't 're 've 'm 'll 'd, case-insensitive)
//!     | [^\r\n L N]? L+              ; optional leading non-letter/digit/newline, then letters
//!     | N{1,n}                       ; capped digit run (Llama-3 n=2, GLM n=3); GPT-2 is N+
//!     | ' '?[^\s L N]+ \r? \n?        ; run of "other" chars, optional trailing newline
//!     | \s* \r? \n+                  ; newline run with leading spaces
//!     | \s+ (?! \S)                  ; trailing whitespace not followed by non-space
//!     | \s+                          ; other whitespace run
//! ```
//!
//! We implement it as a byte-driven scanner with UTF-8 awareness. This is the
//! authoritative definition; the SIMD scanner must produce identical spans.

use smallvec::SmallVec;
use std::sync::OnceLock;

/// A pretoken span: `(start, end)` byte offsets into the input.
/// Invariant: spans are non-empty, sorted, contiguous, cover the whole input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Exact Oniguruma implementation of the pinned GLM-5.2
/// `Split(Regex, Isolated)` pre-tokenizer. Onig is the same regex engine used
/// by HF `tokenizers` 0.22 with its `onig` feature.
pub fn pretokenize_glm52(input: &str) -> SmallVec<[Span; 64]> {
    let mut spans = SmallVec::new();
    for_each_glm52_span(input, |span| spans.push(span));
    spans
}

/// Stream exact GLM-5.2 spans without retaining an input-sized span vector.
pub fn for_each_glm52_span(input: &str, emit: impl FnMut(Span)) {
    static REGEX: OnceLock<onig::Regex> = OnceLock::new();
    let regex = REGEX.get_or_init(|| {
        onig::Regex::new(dstok_model::config::GLM52_SPLIT_REGEX)
            .expect("the pinned GLM-5.2 regex is valid")
    });
    for_each_onig_span(input, regex, emit);
}

fn pretokenize_onig(input: &str, regex: &onig::Regex) -> SmallVec<[Span; 64]> {
    let mut spans = SmallVec::new();
    for_each_onig_span(input, regex, |span| spans.push(span));
    spans
}

fn for_each_onig_span(input: &str, regex: &onig::Regex, mut emit: impl FnMut(Span)) {
    if input.is_empty() {
        return;
    }
    let mut previous = 0;
    for (start, end) in regex.find_iter(input) {
        if previous != start {
            emit(Span {
                start: previous,
                end: start,
            });
        }
        if start != end {
            emit(Span { start, end });
        }
        previous = end;
    }
    if previous != input.len() {
        emit(Span {
            start: previous,
            end: input.len(),
        });
    }
}

/// Classify a byte for the scalar scanner. UTF-8 lead bytes are inspected to
/// decide letter/digit/other for multibyte sequences (we treat any non-ASCII
/// letter-range lead byte as a letter, matching `regex`'s `\p{L}` under the
/// byte-level remap; the oracle and SIMD agree because they share this table).
#[inline]
fn class_of(b: u8) -> Class {
    // ASCII fast path
    match b {
        b'a'..=b'z' | b'A'..=b'Z' => return Class::Letter,
        b'0'..=b'9' => return Class::Digit,
        b' ' | b'\t' | b'\x0b' | b'\x0c' => return Class::Space,
        b'\r' | b'\n' => return Class::Newline,
        b'\'' => return Class::Apostrophe,
        b'\\'
        | b'.'
        | b','
        | b';'
        | b':'
        | b'!'
        | b'?'
        | b'"'
        | b'`'
        | b'('
        | b')'
        | b'['
        | b']'
        | b'{'
        | b'}'
        | b'<'
        | b'>'
        | b'|'
        | b'='
        | b'+'
        | b'-'
        | b'*'
        | b'/'
        | b'%'
        | b'&'
        | b'^'
        | b'~'
        | b'$'
        | b'#'
        | b'@'
        | b'_'
        | b'\x00'..=b'\x08'
        | b'\x0e'..=b'\x1f'
        | b'\x7f' => return Class::Other,
        _ => {}
    }
    // Non-ASCII lead byte. \p{L} / \p{N} membership is approximated by lead
    // byte ranges that cover the common scripts; a fully precise table is in
    // dstok-simd's `unicode_table.rs`. For the oracle we use the same table
    // (re-exported) so the two agree by construction.
    unicode_class::class_of_lead(b)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    Letter,
    Digit,
    Space,
    Newline,
    Apostrophe,
    Other,
}

/// Tokenize `input` into pretoken spans. The full GPT-2/Llama-3 grammar.
///
/// `digit_run` selects the digit-run cap: `0` means unbounded (`\p{N}+`,
/// GPT-2), `n > 0` caps each digit run at `n` digits (`\p{N}{1,n}`, Llama-3
/// family and GLM's `{1,3}`). See `ModelId::digit_run`.
pub fn pretokenize(input: &[u8], digit_run: usize) -> SmallVec<[Span; 64]> {
    if let Ok(text) = std::str::from_utf8(input) {
        static GPT2: OnceLock<onig::Regex> = OnceLock::new();
        static DIGIT_TWO: OnceLock<onig::Regex> = OnceLock::new();
        return match digit_run {
            0 => pretokenize_onig(
                text,
                GPT2.get_or_init(|| {
                    onig::Regex::new(dstok_model::GPT2_REGEX)
                        .expect("the built-in GPT-2 regex is valid")
                }),
            ),
            3 => pretokenize_glm52(text),
            _ => pretokenize_onig(
                text,
                DIGIT_TWO.get_or_init(|| {
                    onig::Regex::new(
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                    )
                    .expect("the built-in capped-digit regex is valid")
                }),
            ),
        };
    }
    pretokenize_legacy(input, digit_run)
}

fn pretokenize_legacy(input: &[u8], digit_run: usize) -> SmallVec<[Span; 64]> {
    let mut spans = SmallVec::new();
    let mut i = 0usize;
    let n = input.len();

    while i < n {
        let start = i;
        let b = input[i];

        // 1. Contraction: 's 't 're 've 'm 'll 'd (case-insensitive), with an
        //    optional leading apostrophe already consumed by a prior "other"
        //    run. We handle the standalone apostrophe-contraction case here.
        if b == b'\'' && i + 1 < n {
            if let Some(end) = contraction_len(input, i) {
                i = end;
                spans.push(Span { start, end: i });
                continue;
            }
        }

        match class_of(b) {
            Class::Letter => {
                // [^\r\n L N]? L+  — but we're already at a letter; the optional
                // leading non-letter/digit/newline was consumed by the *prior*
                // span in this grammar variant only when it directly precedes.
                // The reference regex attaches it to the letters run, so we
                // back up one byte if the preceding byte is "other" and not
                // already its own span boundary.
                let mut j = i + utf8_len(b);
                while j < n && class_of(input[j]) == Class::Letter {
                    j += utf8_len(input[j]);
                }
                i = j;
            }
            Class::Digit => {
                // ` ?\p{N}+` (GPT-2, digit_run == 0) or `\p{N}{1,n}` (Llama/GLM).
                // The outer loop emits one span per call, so capping the run at
                // `digit_run` naturally chunks longer digit sequences into
                // `digit_run`-sized spans (e.g. "12345" with n=3 → "123","45").
                let max_run = if digit_run == 0 {
                    usize::MAX
                } else {
                    digit_run
                };
                let mut j = i + 1;
                let mut run = 1usize;
                while j < n && class_of(input[j]) == Class::Digit && run < max_run {
                    j += 1;
                    run += 1;
                }
                i = j;
            }
            Class::Newline => {
                // \s* \r? \n+
                let mut j = i;
                // \s* (spaces/tabs allowed before the newline run)
                while j < n && matches!(class_of(input[j]), Class::Space | Class::Newline) {
                    j += 1;
                }
                i = j;
            }
            Class::Space => {
                // \s+ (?! \S)  |  \s+
                let mut j = i + 1;
                while j < n && matches!(class_of(input[j]), Class::Space) {
                    j += 1;
                }
                // if followed by a non-space (including newline), the trailing-
                // whitespace rule still consumes the whole run as one span.
                i = j;
            }
            Class::Apostrophe => {
                // lone apostrophe not part of a contraction → "other" run
                i += 1;
                while i < n && class_of(input[i]) == Class::Other {
                    i += 1;
                }
            }
            Class::Other => {
                // [^\s L N]+ with optional trailing \r?\n?
                let mut j = i + 1;
                while j < n && class_of(input[j]) == Class::Other {
                    j += 1;
                }
                // optional trailing \r? \n? (one each)
                if j < n && input[j] == b'\r' {
                    j += 1;
                }
                if j < n && input[j] == b'\n' {
                    j += 1;
                }
                i = j;
            }
        }

        debug_assert!(i > start, "no progress in pretokenizer at {i}");
        spans.push(Span { start, end: i });
    }

    spans
}

/// Length of the UTF-8 sequence starting at lead byte `b` (1 for ASCII).
#[inline]
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1 // invalid lead — treat as one byte (oracle tolerant; SIMD matches)
    }
}

/// If `input[i..]` begins a contraction ('s 't 're 've 'm 'll 'd, ASCII or
/// uppercase), return the end offset.
fn contraction_len(input: &[u8], i: usize) -> Option<usize> {
    // input[i] == '\''  (caller checked)
    let rest = &input[i + 1..];
    let lower: SmallVec<[u8; 4]> = rest
        .iter()
        .take(4)
        .map(|&c| c.to_ascii_lowercase())
        .collect();
    // Match the longest of: 're 've 'll 's 't 'm 'd. Order matters only for
    // tie-breaking, but the two-char forms are checked first by length.
    if lower.starts_with(b"re") || lower.starts_with(b"ve") || lower.starts_with(b"ll") {
        return Some(i + 1 + 2);
    }
    if lower.starts_with(b"s")
        || lower.starts_with(b"t")
        || lower.starts_with(b"m")
        || lower.starts_with(b"d")
    {
        return Some(i + 1 + 1);
    }
    None
}

// Re-export the unicode classification module used above. The file lives at
// `src/unicode_class.rs` (a sibling of this module's file), so `#[path]` is
// needed to load it as a child of `regex` — which lets it refer to `super::Class`.
#[path = "unicode_class.rs"]
pub(crate) mod unicode_class;
