# Third-party notices

## GLM-5.2 tokenizer fixture

The repository's `fixtures/glm-5.2/tokenizer.json` is derived without
modification from
[`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) at
revision `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`.

That repository's Hub metadata declares `license: mit` and carries the
`license:mit` tag; it contains no separate license file asserting different
terms. Its base model,
[`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2), likewise declares
`license: mit` and ships a `LICENSE` file.

The fixture is used for conformance testing and is not included in any
published `dstok` crate archive.

Exact provenance and the local digest are recorded in
`fixtures/glm-5.2/SOURCE.md`.

## Pre-tokenizer regex

`dstok_model::config::GLM52_SPLIT_REGEX` reproduces the pinned artifact's
`Split` pattern verbatim, because any deviation changes token IDs. The pattern
is the byte-level pre-tokenizer regex introduced by Llama 3 and reused
unchanged by many later BPE tokenizers; it is not original to GLM-5.2. This is
the only upstream literal contained in the published crate archives.

## Conformance oracle

`dstok-oracle` and `dstok-bench` compare against Hugging Face
[`tokenizers`](https://github.com/huggingface/tokenizers) 0.23.1, which is
licensed under Apache-2.0. Both crates are `publish = false`, so `tokenizers`
is not a dependency of any published `dstok` crate. "Hugging Face" is used
descriptively to identify the reference implementation; this project is not
affiliated with or endorsed by Hugging Face.
