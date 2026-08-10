# Contributing

## Toolchain

The workspace pins Rust 1.85, the minimum supported Rust version. Install the
components declared in `rust-toolchain.toml`; Cargo selects the pinned
toolchain automatically.

The Hugging Face oracle builds the Oniguruma backend and therefore requires a
working native C toolchain.

## Required checks

Before opening a pull request:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo run -p xtask -- parity-glm52
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

Changes to hot paths should also run the quick benchmark. Do not present a
speedup unless complete IDs were checked before timing:

```sh
cargo run --release -p dstok-bench -- \
  --targets 100000 --iterations 5 \
  --output benchmarks/glm52-quick.json
```

## Correctness-first rules

- Never infer support from a model family name.
- Reject unknown tokenizer configuration instead of guessing.
- Compare complete token IDs with the pinned HF oracle.
- Do not sum counts across arbitrary strings or logical messages.
- Keep optimization code behind differential tests.
- Preserve a safe exact fallback.
- Do not update the pinned digest without reviewing and versioning the
  artifact contract.

## Repository scope

The supported package is the root `dstok` crate and its `dstok-model`,
`dstok-core`, and `dstok-fast` dependencies. Other workspace crates are private
development tools or prototypes and must not be advertised as supported
packages without their own release review.

Corpus payloads, generated fuzz artifacts, credentials, and build output must
not be committed.
