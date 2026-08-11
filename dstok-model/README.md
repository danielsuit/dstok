# dstok-model

Strict artifact parsing and verification for [`dstok`](https://crates.io/crates/dstok).

Loads a Hugging Face tokenizer JSON, records every structural field that can
affect token IDs, and fails closed on unsupported normalizers, pre-tokenizers,
or BPE options. The pinned GLM-5.2 artifact is identified by SHA-256, not by a
family-name hint.

Most users should depend on the [`dstok`](https://crates.io/crates/dstok)
facade instead of this crate directly. The tokenizer JSON is not bundled;
supply it through your own model-asset pipeline.

Licensed under either Apache-2.0 or MIT, at your option.
