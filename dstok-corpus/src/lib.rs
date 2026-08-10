//! # dstok-corpus — corpus manifests + adversarial generators (no data checked in).
//!
//! No data in the repo (§4.5). Manifests point to upstream slices; fetchers
//! download them; generators synthesize adversarial inputs. Each slice is
//! tagged with its oracle (§3).

#![cfg_attr(not(test), deny(warnings))]

/// A corpus slice descriptor.
#[derive(Debug, Clone)]
pub struct CorpusSlice {
    pub name: &'static str,
    pub flavor: SliceFlavor,
    /// Where to fetch it (URL or generator seed).
    pub source: SliceSource,
    /// The oracle this slice's parity is validated against.
    pub oracle: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceFlavor {
    Web,
    Code,
    Multilingual,
    Chat,
    Adversarial,
}

#[derive(Debug, Clone)]
pub enum SliceSource {
    Url(&'static str),
    Generated(u64),
}

/// The curated v1 corpus manifest. Used by `xtask` to fetch + run parity.
pub fn manifest() -> Vec<CorpusSlice> {
    vec![
        CorpusSlice {
            name: "openwebtext-sample",
            flavor: SliceFlavor::Web,
            source: SliceSource::Url("https://huggingface.co/datasets/Skylion007/openwebtext"),
            oracle: "hf-tokenizers",
        },
        CorpusSlice {
            name: "fineweb-sample",
            flavor: SliceFlavor::Web,
            source: SliceSource::Url("https://huggingface.co/datasets/HuggingFaceFW/fineweb"),
            oracle: "hf-tokenizers",
        },
        CorpusSlice {
            name: "the-stack-mono",
            flavor: SliceFlavor::Code,
            source: SliceSource::Url("https://huggingface.co/datasets/bigcode/the-stack"),
            oracle: "hf-tokenizers",
        },
        CorpusSlice {
            name: "flores",
            flavor: SliceFlavor::Multilingual,
            source: SliceSource::Url("https://huggingface.co/datasets/facebook/flores"),
            oracle: "hf-tokenizers",
        },
        CorpusSlice {
            name: "agent-transcripts",
            flavor: SliceFlavor::Chat,
            source: SliceSource::Generated(0xC0FFEE),
            oracle: "hf-tokenizers",
        },
        CorpusSlice {
            name: "special-token-soup",
            flavor: SliceFlavor::Adversarial,
            source: SliceSource::Generated(0xDEADBEEF),
            oracle: "hf-tokenizers",
        },
    ]
}

/// Adversarial generator: deterministic from a seed (no Date/random — §script).
pub fn adversarial(seed: u64, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed;
    while out.len() < n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let b = (s >> 33) as u8;
        // Skew toward whitespace / multibyte / special-token bytes.
        out.push(match b % 8 {
            0 => b' ',
            1 => b'\n',
            2 => 0xE2, // a UTF-8 lead byte (forces multibyte handling)
            3 => 0x80, // a continuation byte (forces mid-sequence cuts)
            4 => b'<', // special-token-ish
            5 => b'\\',
            6 => b'\'',
            _ => b,
        });
    }
    out
}
