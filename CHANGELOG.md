# Changelog

All notable changes to dstok are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-08-10

### Added

- Public `dstok::Tokenizer` API for loading the pinned GLM-5.2 artifact from a
  file or bytes.
- Exact count-only text-segment cache for append-heavy conversations.
- Hugging Face 0.23.1 differential, property, append-boundary, and fuzz gates.
- Benchmark evidence up to 20 million tokens on macOS AArch64 and Linux
  AArch64, plus correctness gates on Linux x86_64.

### Changed

- Stable x86 SIMD dispatch uses verified SSE4.2 and AVX2 kernels; experimental
  unstable AVX-512 code is no longer part of the build.
- The pinned conformance oracle is `tokenizers` 0.23.1. Token IDs are identical
  under 0.22.2, against which the checked-in benchmark artifacts were recorded.
- `dstok-model` requires `sha2` 0.11. Dependents still on `sha2` 0.10 will
  compile both versions.

[Unreleased]: https://github.com/danielsuit/dstok/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/danielsuit/dstok/releases/tag/v0.1.0
