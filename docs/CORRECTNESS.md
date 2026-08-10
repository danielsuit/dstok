# Correctness contract

## Reference

The production oracle is Hugging Face `tokenizers` 0.23.1 using the Oniguruma
backend and:

```text
encode_fast(rendered_prompt, add_special_tokens=false)
```

Correctness means complete token-ID equality. Equal counts alone are
insufficient because two incorrect token streams can have the same length.

The supported artifact is identified by the SHA-256 and structural facts in
`fixtures/glm-5.2/conformance.json`. Loading any modified or merely similar
tokenizer JSON fails.

## Shared implementation

ID, offset, and count APIs share:

- added-token extraction;
- exact pretoken boundaries;
- byte mapping;
- BPE merge ranks and `ignore_merges` behavior; and
- global truncation semantics.

Count-only operation changes the output sink, not tokenization decisions.

## Why cached counts remain exact

BPE is not additive across arbitrary boundaries. A cached prefix count plus a
suffix count is generally wrong.

HF extracts added tokens before pre-tokenization. For the pinned artifact,
ordinary text on each side of an extracted added token is processed
independently. The GLM loader also proves:

- no normalizer;
- no BPE dropout;
- no `lstrip`, `rstrip`, `single_word`, or normalized added tokens;
- `ByteLevel.add_prefix_space=false`; and
- no tokenizer-level padding or truncation.

The segment cache operates only after that exact extraction step. Its key
contains the artifact digest and complete segment bytes. Therefore:

- unchanged segments reuse the same deterministic count;
- changed segments cannot compare equal;
- a newly completed added token changes extraction and cannot reuse the old
  text segment; and
- hash collisions still require complete key equality.

Truncation is applied once after summing, never independently per segment.

## Required gates

Every release must pass:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo run -p xtask -- parity-glm52
```

The parity suite includes:

- frozen rendered gateway prompts;
- arbitrary Unicode;
- emoji, combining marks, CRLF, and trailing whitespace;
- overlapping and partially formed added tokens;
- append-only and edited histories with warm caches;
- scalar, ID-only, count-only, and parallel determinism; and
- a differential fuzz smoke test.

Longer fuzz runs use:

```sh
cargo run -p dstok-fuzz --features libfuzzer \
  --bin glm52-parity-fuzz -- -runs=10000
```

## Platform verification

The gates are architecture-sensitive in principle, so they are run on more than
the development host. No published crate contains architecture-gated code —
every `target_arch` and `target_feature` in the workspace lives in
`dstok-simd`, which is `publish = false` and absent from the `dstok` dependency
graph — but the C dependencies (`onig_sys`, `blake3`) and the parallel paths
still differ per platform.

| Platform | How | Scope |
|----------|-----|-------|
| macOS AArch64 | CI job + development host | full gates |
| Linux AArch64 | CI job + a 20-core bare-metal host | full gates, plus the 20M benchmark matrix |
| Linux x86_64 | CI job | full gates |

Cross-compiling from macOS to either Linux target does **not** work without a
cross C toolchain, because `onig_sys` and `blake3` build C. `cargo check
--target x86_64-apple-darwin` is a useful stand-in for catching x86-only
compile errors without leaving an Apple Silicon host, and it exercises
`dstok-simd`'s SSE4.2 and AVX2 paths.

## Changing the artifact or algorithm

Do not update an artifact digest merely to make a loader test pass. A tokenizer
change requires:

1. reviewing every configuration field that can affect IDs;
2. adding a new explicit profile or version;
3. regenerating frozen fixtures;
4. running differential tests against the exact upstream version;
5. publishing a new conformance manifest; and
6. rerunning large latency and RSS probes.

An optimized scanner must first prove exact span equality with Oniguruma and
then complete ID equality with HF before it can enter the production path.
