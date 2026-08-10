# Security policy

## Reporting

Please report vulnerabilities through the private security-advisory feature of
the `danielsuit/dstok` GitHub repository. Do not open a public issue for an
unpatched vulnerability or include customer prompts, tokenizer assets, API
keys, or infrastructure credentials in a report.

## Supported versions

Until the first stable release, only the latest tagged `0.1.x` version is
supported. The `main` branch is development code.

## Security model

Tokenizer JSON is untrusted input. Production loading:

- verifies the exact pinned SHA-256;
- rejects unsupported structure and behavior;
- uses safe Rust in the public model, core, fast, and facade crates; and
- does not execute tokenizer-provided code or regex.

The unpublished `dstok-simd` development crate contains architecture
intrinsics and is not linked into the public `dstok` package.

## Prompt retention

The count-only segment cache retains complete raw prompt segments in process
memory so a hash collision cannot alter correctness. Retention is bounded by
`DSTOK_SEGMENT_CACHE_MAX_BYTES`, but shared cache timing and memory residency
may matter in multi-tenant deployments.

Operators should:

- choose a budget consistent with their data-retention policy;
- set the budget to zero when raw prompt retention is unacceptable;
- restrict process memory inspection and core dumps;
- clear caches during relevant incident response; and
- use tenant-scoped caching before relying on this optimization where
  cross-tenant timing is in scope.

## Denial of service

Large inputs consume CPU and memory even when output IDs are not retained.
Integrators must enforce request-byte limits, weighted concurrency admission,
deadlines, and process memory limits outside this crate. A cache is an
optimization, not an admission-control mechanism.
