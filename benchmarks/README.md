# GLM-5.2 benchmark evidence

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

Artifacts:

- `glm52-full.json`: full 1.5M/5M/10M/20M matrix.
- `glm52-quick.json`: five-sample 100K-token smoke benchmark.
- `glm52-rss-hf.json`: isolated HF 20M probe.
- `glm52-rss-dstok.json`: isolated bounded-cache dstok 20M probe.
- `glm52-rss-dstok-cache-off.json`: isolated cache-off dstok 20M probe.

The per-row `peak_rss_bytes` in `glm52-full.json` is the high-water mark of the
entire mixed process and must not be used for implementation-level RSS
comparison. Use the isolated RSS probe files for that comparison.
