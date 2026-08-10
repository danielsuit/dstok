# Integration guide

## Dependency

Before the first crates.io release, pin a Git commit:

```toml
[dependencies]
dstok = { git = "https://github.com/danielsuit/dstok", rev = "<commit-sha>" }
```

Do not track `main` in a production lockfile.

## Artifact lifecycle

Load one `Tokenizer` during process startup and share clones of it:

```rust
use dstok::Tokenizer;
use std::sync::Arc;

let tokenizer = Arc::new(Tokenizer::from_file("/assets/glm-5.2/tokenizer.json")?);
# Ok::<(), dstok::LoadError>(())
```

Loading verifies the exact artifact. A failure should stop startup rather than
fall back to an unverified tokenizer.

The caller must render the GLM chat template first. Cache keys and counts apply
to that rendered UTF-8 string, not the request JSON.

## Async servers

Tokenization is CPU-bound. Do not run a cold multi-million-token count directly
on a Tokio worker. Use the application's bounded blocking pool or
`spawn_blocking`, and enforce both:

- a concurrency permit; and
- a weighted permit based on rendered/request bytes.

The concurrency limit should be established with production-hardware load
tests. A 20M-token first count remains seconds of CPU even though warm segment
reuse is much faster.

## Cache configuration

Set these before constructing or using any tokenizer:

```text
DSTOK_CACHE_MAX_BYTES=67108864
DSTOK_SEGMENT_CACHE_MAX_BYTES=134217728
```

The values are process-wide byte weights. Zero disables a cache. Configuration
changes require a process restart because global caches initialize once.

The segment cache stores complete raw text-segment keys. For a multi-tenant
service, review:

- prompt retention in process memory;
- whether shared-cache timing is an acceptable side channel;
- cache invalidation during incident response; and
- whether the cache should be disabled until tenant namespaces exist.

## Recommended rollout

1. Integrate behind a feature flag.
2. In shadow mode, compute HF and dstok counts for a bounded sample.
3. Emit mismatch, duration, cache-hit, timeout, and byte-admission metrics.
4. Treat any mismatch as a release blocker and continue using HF.
5. Canary one replica while keeping immediate rollback.
6. Increase traffic only after cold/warm latency and RSS match the tested
   operating envelope.
7. Remove HF only after a sustained zero-mismatch window.

Shadow comparison should compare complete IDs on sampled manageable prompts,
not only counts. For very large prompts, compare counts after the same artifact
and template have already passed complete-ID conformance gates.

## Operational defaults

- Keep the tokenizer artifact and chat template version coupled.
- Include the artifact digest in metrics and cache schema versions.
- Keep request-body and token-count limits independent.
- Set tokenization timeouts above measured cold p99, not warm-cache latency.
- Preserve a full-recount fallback when introducing future incremental logic.
