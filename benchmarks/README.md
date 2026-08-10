# GLM-5.2 benchmark evidence

## Apple Silicon

Measured on Apple Silicon (`aarch64-apple-darwin`, 11 logical CPUs) with the
pinned GLM-5.2 artifact and HF `tokenizers` 0.22.2. Every workload passed full
token-ID equality before timing.

These artifacts were recorded before the reference implementation was moved to
`tokenizers` 0.23.1, and their `hf-0.22.2` labels are kept as the record of
what was actually measured. Token IDs are identical under both versions, so
the comparison remains meaningful, but a re-run is needed before any of these
timings are attributed to 0.23.1.

The full run used one timed sample per size, so these are latency observations,
not statistically stable p95/p99 claims:

- 1.50M actual tokens: HF 1.049 s; dstok cold count 0.437 s and exact
  segment-warm count 5.69 ms.
- 4.98M actual tokens: HF 3.329 s; dstok cold count 1.334 s and exact
  segment-warm count 17.86 ms.
- 9.95M actual tokens: HF 6.754 s; dstok cold count 2.772 s and exact
  segment-warm count 37.70 ms.
- 19.89M actual tokens: HF 15.558 s; dstok cold count 5.191 s and exact
  segment-warm count 70.26 ms.
- 19.89M append-only full recounts at 25/50/75/100%, starting cold: HF
  37.094 s; dstok 5.717 s (6.49×).

An isolated 19.89M-token process measured:

- HF `encode_fast`: 16.276 s and 8.640 GB peak RSS.
- dstok first count with 64 MiB pretoken and 128 MiB segment budgets:
  5.970 s and 0.294 GB peak RSS.
- dstok first count with both caches disabled: 11.516 s and 0.231 GB peak RSS.

The count-only segment cache reduced a repeated exact 19.89M-token count from
the previous 4.925 s pretoken-cache path to 70.26 ms. The append-only sequence
fell from the previous 12.905 s to 5.717 s because each newly added quarter is
tokenized once while unchanged added-token-delimited text segments are reused.
Keys contain complete raw bytes plus the artifact digest, so hash collisions
cannot change results.

The first unseen 19.89M-token prompt still requires a full scan and took
5.191 s. Full-ID encoding also cannot use a count-only segment cache. The
existing 2/4/8-worker path remains slower and bypasses segment-count reuse, so
the serial `count` API remains the recommended gateway path.

## Linux AArch64

`glm52-full-linux-aarch64.json` repeats the same matrix on a 20-core bare-metal
`aarch64-unknown-linux-gnu` host with Rust 1.85.0. Unlike the Apple Silicon
artifacts, this run was recorded against the current `tokenizers` 0.23.1 pin,
and every workload passed full token-ID equality before timing.

| Tokens | HF `encode_fast` | dstok cold | dstok segment-warm | Append-only: HF vs dstok |
|--------|------------------|------------|--------------------|--------------------------|
| 1.50M  | 1.004 s | 0.417 s | 5.21 ms | 3.101 s vs 0.445 s (6.97×) |
| 4.98M  | 4.175 s | 1.468 s | 17.56 ms | 10.362 s vs 1.494 s (6.93×) |
| 9.95M  | 8.182 s | 2.863 s | 34.63 ms | 20.230 s vs 2.937 s (6.89×) |
| 19.89M | 16.547 s | 5.748 s | 69.68 ms | 41.107 s vs 5.871 s (7.00×) |

The host was shared and under concurrent load during the run, so these are
conservative rather than best-case timings. The relative shape matches the
Apple Silicon matrix closely — segment-warm counts land within a few percent at
every size — which is the property worth checking across platforms. Absolute
times are not comparable between the two hosts.

Correctness gates (`cargo test --workspace --all-targets`, Clippy with
warnings denied, and `xtask parity-glm52`) also pass on this host and on Linux
x86_64 in CI. See `docs/CORRECTNESS.md` for the platform matrix.

Artifacts:

- `glm52-full.json`: full 1.5M/5M/10M/20M matrix.
- `glm52-full-linux-aarch64.json`: the same matrix on Linux AArch64, recorded
  against `tokenizers` 0.23.1.
- `glm52-quick.json`: five-sample 100K-token smoke benchmark.
- `glm52-rss-hf.json`: isolated HF 20M probe.
- `glm52-rss-dstok.json`: isolated bounded-cache dstok 20M probe.
- `glm52-rss-dstok-cache-off.json`: isolated cache-off dstok 20M probe.

The per-row `peak_rss_bytes` in `glm52-full.json` is the high-water mark of the
entire mixed process and must not be used for implementation-level RSS
comparison. Use the isolated RSS probe files for that comparison.
