# dstok-core

Exact scalar GLM-5.2 tokenization for [`dstok`](https://crates.io/crates/dstok).

This crate is the reference implementation: special-token splitting,
Oniguruma pre-tokenization, byte mapping, and BPE merging, written for
clarity over speed. Every faster path in the workspace is differentially
tested against it, and it is in turn compared with Hugging Face
`tokenizers` 0.22.2.

Most users should depend on the [`dstok`](https://crates.io/crates/dstok)
facade instead of this crate directly.

Licensed under either Apache-2.0 or MIT, at your option.
