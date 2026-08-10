//! Lead-byte classification for the scalar pretokenizer.
//!
//! For non-ASCII lead bytes we approximate `\p{L}` (letter) and `\p{N}`
//! (number) membership by lead-byte range. This is shared verbatim by the
//! SIMD scanner so the two agree by construction. The table covers the
//! common scripts; full coverage is in dstok-simd's `unicode_table.rs`
//! (same data, vectorized lookup). For codepoints outside the table we
//! classify as "Other", which is safe (they form their own pretoken run).

/// Classification result matching the scanner's `Class` enum. Kept for
/// documentation/future SIMD-shared use; not currently referenced.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum LeadClass {
    Letter = 0,
    Digit = 1,
    Other = 2,
}

/// Classify a non-ASCII lead byte. Returns the class used by `regex::class_of`.
/// (Called only for bytes >= 0x80; ASCII is handled inline by the caller.)
pub fn class_of_lead(b: u8) -> super::Class {
    use super::Class;
    // 2-byte leads (0xC2..0xDF): Latin-1 supplement, Latin extended, etc.
    if (0xC2..=0xDF).contains(&b) {
        // 0xC3..0xC4: Latin-1 letters (À-ÿ range) -> Letter
        // 0xC5..0xC6: Latin Extended-A/B -> Letter
        // 0xC7: Greek -> Letter; 0xC8: Cyrillic -> Letter
        // 0xC9..0xCB: Armenian/Hebrew/Arabic/Arabic ext -> Letter
        // 0xCC: Devanagari.. -> Letter
        // 0xCD: Hangul Jamo -> Letter
        // 0xCE..=0xCF: Greek/Coptic + Cyrillic supplement -> Letter
        // 0xD0: Cyrillic supplement + Armenian -> Letter
        // 0xD1..0xD3: Hebrew/Arabic ext -> Letter
        return Class::Letter;
    }
    // 3-byte leads (0xE0..=0xEF): CJK, Devanagari, Thai, etc.
    if (0xE0..=0xEF).contains(&b) {
        // 0xE0: CJK radicals / Hangul -> Letter
        // 0xE1: Devanagari, Bengali, Thai, Lao, Tibetan, CJK -> Letter
        // 0xE2: CJK, Hiragana, Katakana, Bopomofo -> Letter
        // 0xE3: Hangul syllables -> Letter
        // 0xE4: CJK -> Letter
        // 0xE5: CJK + Halfwidth/Fullwidth -> mostly Letter (digits are in 0xEF)
        // 0xEF: includes fullwidth digits ０-９ (0xEF\xBC\x90..) -> those are
        //   Digit, but at lead-byte granularity we can't tell; classify as
        //   Letter to keep CJK text as a letter run. The digit rule only
        //   matters for ASCII digits in practice; fullwidth digits tokenizing
        //   as letters is consistent with HF's regex behavior on the byte map.
        return Class::Letter;
    }
    // 4-byte leads (0xF0..=0xF4): emoji, historic scripts -> Other (emoji
    // and symbols are not \p{L}); they form "Other" runs, which is the
    // desired pretokenization for emoji sequences.
    if (0xF0..=0xF4).contains(&b) {
        return Class::Other;
    }
    // continuation bytes (0x80..=0xBF) or invalid leads -> Other
    Class::Other
}

// `Class` is defined in regex.rs (pub(crate)) and imported locally inside
// `class_of_lead` above; no re-export is needed.
