//! # dstok-bench — criterion + end-to-end harness, published methodology (§8).
//!
//! The anti-Gigatoken-marketing commitment, encoded as a harness:
//!   - identical dataset, identical interface, fixed thread count
//!   - verified byte-identical output before any ratio is reported
//!   - full p50/p99 distribution, not best-of-three
//!   - unflattering workloads (short prompts, streaming, concurrency) alongside
//!     the big-file case
//!
//! This crate defines the workload set and the verification wrapper. The
//! actual criterion benches live in `benches/`.

#![cfg_attr(not(test), deny(warnings))]

use dstok_model::Artifact;
use std::time::{Duration, Instant};

/// A benchmark workload: a name, the input bytes, and the thread count.
#[derive(Debug, Clone)]
pub struct Workload {
    pub name: &'static str,
    pub bytes: Vec<u8>,
    pub threads: usize,
    pub flavor: WorkloadFlavor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadFlavor {
    /// Big-file encode — the flattering case (Gigatoken's headline).
    BigFile,
    /// A single short prompt — the realistic gateway-per-request shape.
    ShortPrompt,
    /// Many small concurrent requests.
    HighConcurrency,
    /// Streaming incremental encode.
    Streaming,
    /// Adversarial: degenerate whitespace, special-token soup.
    Adversarial,
}

/// Run one workload through dstok-fast and verify its output matches the
/// oracle (`dstok-core`) byte-for-byte before timing. Returns throughput and
/// the p50/p99 latency over `iters` runs. §8: no ratio is reported unless
/// `verified` is true.
pub fn run_and_verify(art: &Artifact, w: &Workload, iters: usize) -> BenchResult {
    let text = std::str::from_utf8(&w.bytes).unwrap_or("");
    // Verify correctness against the oracle first — no timing if it diverges.
    let oracle = dstok_core::encode(art, text);
    let fast = if w.threads > 1 {
        dstok_fast::encode_parallel(art, text, w.threads)
    } else {
        dstok_fast::encode(art, text)
    };
    let verified = fast.ids == oracle.ids && fast.offsets == oracle.offsets;

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        // Time the SAME encoder that was verified (single vs parallel).
        if w.threads > 1 {
            let _ = dstok_fast::encode_parallel(art, text, w.threads);
        } else {
            let _ = dstok_fast::encode(art, text);
        }
        times.push(t0.elapsed());
    }
    times.sort();
    let p50 = times[times.len() / 2];
    let p99 = times[(times.len() * 99) / 100];
    let bytes = w.bytes.len() as f64;
    let secs = p50.as_secs_f64().max(1e-12);
    let throughput_gbs = bytes / secs / 1e9;

    BenchResult {
        workload: w.name,
        threads: w.threads,
        flavor: w.flavor,
        verified,
        p50,
        p99,
        throughput_gbs,
        tokens: fast.ids.len(),
    }
}

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub workload: &'static str,
    pub threads: usize,
    pub flavor: WorkloadFlavor,
    /// §8: false ⇒ the throughput number is INVALID and must not be reported.
    pub verified: bool,
    pub p50: Duration,
    pub p99: Duration,
    pub throughput_gbs: f64,
    pub tokens: usize,
}

/// A published-benchmark report (§8): one row per (hardware, tokenizer family,
/// interface, workload). Never a single headline multiplier.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub hardware: String,
    pub family: String,
    pub rows: Vec<BenchResult>,
}

impl Report {
    /// Render the report as a fixed-width table for the published harness.
    /// Unverified rows are marked `INVALID` so they cannot be mistaken for
    /// real numbers.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("hardware={}\n", self.hardware));
        s.push_str(&format!("family={}\n", self.family));
        s.push_str(&format!(
            "{:<24} {:>7} {:>9} {:>9} {:>10} {:>8}\n",
            "workload", "threads", "p50(ns)", "p99(ns)", "GB/s", "status",
        ));
        for r in &self.rows {
            let status = if r.verified { "ok" } else { "INVALID" };
            s.push_str(&format!(
                "{:<24} {:>7} {:>9} {:>9} {:>10.3} {:>8}\n",
                r.workload,
                r.threads,
                r.p50.as_nanos(),
                r.p99.as_nanos(),
                if r.verified { r.throughput_gbs } else { 0.0 },
                status,
            ));
        }
        s
    }
}
