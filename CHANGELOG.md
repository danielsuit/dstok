# Changelog

All notable changes to dstok are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Public `dstok::Tokenizer` API for loading the pinned GLM-5.2 artifact from a
  file or bytes.
- Exact count-only text-segment cache for append-heavy conversations.
- Hugging Face 0.22.2 differential, property, append-boundary, and fuzz gates.
- Linux x86_64 and AArch64 benchmark evidence up to 20 million tokens.

### Changed

- Stable x86 SIMD dispatch uses verified SSE4.2 and AVX2 kernels; experimental
  unstable AVX-512 code is no longer part of the build.

[Unreleased]: https://github.com/danielsuit/dstok/commits/main
