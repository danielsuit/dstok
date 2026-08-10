# dstok

Exact, memory-efficient token counting for a pinned production GLM-5.2
tokenizer.

`dstok` is deliberately narrow: it supports one hash-verified tokenizer
artifact and treats Hugging Face `tokenizers` 0.22.2 with Oniguruma as the
correctness oracle. Unknown or modified artifacts fail closed.

## Status

`0.1.0` is the first release. The public API is small and is expected to stay
stable within `0.1.x`.

```toml
[dependencies]
dstok = "0.1.0"
```

The public package uses the safe scalar/Onig tokenizer path plus bounded exact
caches; the experimental `dstok-simd` crate is not part of the dependency
graph.

## Supported contract

- Tokenizer: only the exact GLM-5.2 artifact recorded in
  `fixtures/glm-5.2/conformance.json`.
- Input: an already-rendered UTF-8 prompt.
- Output: token counts or token IDs exactly equal to HF 0.22.2
  `encode_fast(..., add_special_tokens=false)`.
- Minimum Rust version: 1.85.
- Tested targets: Linux x86_64, Linux AArch64, and macOS AArch64.

Chat-template rendering is intentionally outside this package. Generic
GPT/Llama/Qwen support, arbitrary HF tokenizer JSON, and incremental token
emission are not advertised.

## Quick start

The tokenizer JSON is not embedded in the crate. Supply the pinned artifact
through your existing model-asset pipeline:

```rust
use dstok::Tokenizer;

fn main() -> Result<(), dstok::LoadError> {
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    let rendered = "<|user|>Count this rendered prompt.<|assistant|>";

    let count = tokenizer.count(rendered);
    let ids = tokenizer.encode_ids(rendered);
    assert_eq!(count, ids.len());
    Ok(())
}
```

Applications that already hold the JSON in memory can use
`Tokenizer::from_bytes`.

The repository also includes a minimal CLI example:

```sh
cargo run --example count -- \
  fixtures/glm-5.2/tokenizer.json '<|user|>hello<|assistant|>'
```

## Why segment caching is exact

Byte-pair encoding is not additive across arbitrary strings. `dstok` therefore
does not cache logical messages, fixed-size chunks, or guessed prefixes.

The count path first performs the same added-token extraction as HF. It caches
only the complete ordinary-text segments produced by that tokenizer step.
Each key contains:

- the verified artifact digest; and
- the complete raw segment bytes.

Hash-table collisions still require complete byte equality. If an append
changes trailing whitespace, completes an added token, or modifies any prior
text, that segment misses and is recomputed. Global truncation is applied once
after all exact segment counts are combined.

The process-wide budgets are:

- `DSTOK_CACHE_MAX_BYTES`: raw-pretoken cache, 64 MiB by default;
- `DSTOK_SEGMENT_CACHE_MAX_BYTES`: count-only text-segment cache, 128 MiB by
  default.

Set either variable to `0` before the first tokenizer call to disable that
cache. Cached keys retain prompt bytes within the configured bound, which
should be considered in multi-tenant privacy reviews.

## Correctness

The production gate compares complete IDs—not only counts—with HF 0.22.2:

```sh
cargo run -p xtask -- parity-glm52
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

Coverage includes frozen gateway prompts, arbitrary Unicode, malformed and
overlapping added-token boundaries, append-only histories, serial/parallel
determinism, cache reuse, and differential fuzz smoke tests. The generated
manifest is checked in at `fixtures/glm-5.2/conformance.json`.

For longer differential fuzzing:

```sh
cargo run -p dstok-fuzz --features libfuzzer \
  --bin glm52-parity-fuzz -- -runs=10000
```

See `docs/CORRECTNESS.md` for the invariants and release policy.

## Performance

For the checked-in 19.89M-token Apple Silicon workload:

- HF `encode_fast`: 15.558 s.
- dstok cold count: 5.191 s.
- dstok exact segment-warm count: 70.26 ms.
- Isolated peak RSS: 8.640 GB for HF and 0.294 GB for dstok with the default
  cache budgets.

These are observations from specific machines, not portable latency promises.
Linux x86_64 and AArch64 are separate correctness/compile gates. Every
benchmark verifies token IDs before timing. Reproduce the checked-in matrix
with:

```sh
cargo run --release -p dstok-bench -- \
  --full --iterations 1 --threads 1,2,4,8 \
  --cache-bytes 67108864 --segment-cache-bytes 134217728 \
  --output benchmarks/glm52-full.json
```

See `benchmarks/README.md` for methodology and raw artifacts.

## Documentation

- `docs/ARCHITECTURE.md`: crate boundaries and request flow.
- `docs/CORRECTNESS.md`: exactness invariants and parity gates.
- `docs/INTEGRATION.md`: gateway integration, concurrency, and rollout.
- `docs/RELEASING.md`: package verification and publication order.
- `SECURITY.md`: vulnerability reporting and prompt-retention considerations.

## Workspace layout

- `src/`: stable public `dstok` facade.
- `dstok-model/`: strict artifact parsing and verification.
- `dstok-core/`: exact scalar tokenization.
- `dstok-fast/`: count-only caches and optimized APIs.
Published to crates.io: `dstok`, `dstok-model`, `dstok-core`, `dstok-fast`.
The remaining crates are `publish = false`:

- `dstok-oracle/`: private HF conformance oracle.
- `dstok-bench/`: private benchmark harness.
- `dstok-fuzz/`: differential fuzz targets.
- `dstok-corpus/`: corpus manifests and fetchers; no corpus data is checked in.
- `dstok-py/`: prototype Python-facing API.
- `dstok-simd/`: unpublished experimental classifiers, not used by `dstok`.
- `xtask/`: release and conformance automation.

## License

Licensed under either the Apache License, Version 2.0 or the MIT license, at
your option. Repository fixtures are covered by `THIRD_PARTY_NOTICES.md`.
