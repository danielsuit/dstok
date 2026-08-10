# Architecture

## Production data flow

The public `dstok::Tokenizer` owns an immutable, hash-verified `Artifact`.
Counting an already-rendered prompt follows this order:

1. Extract declared added tokens with HF-compatible leftmost-longest matching.
2. Split the remaining input into ordinary-text segments.
3. Look up each complete segment in the bounded exact count cache.
4. On a miss, run the pinned Oniguruma pre-tokenizer over that segment.
5. Look up each raw pretoken in the bounded BPE-result cache.
6. On a miss, run exact byte-level BPE.
7. Sum added-token and text-segment counts.
8. Apply tokenizer-level truncation once to the global result.

Full-ID encoding follows the same extraction, pre-tokenization, and BPE rules,
but does not use count-only segment entries because those intentionally omit
token IDs and offsets.

## Crate boundaries

### `dstok`

The supported public facade. It exposes verified loading, count-only encoding,
ID encoding, cache statistics, and explicit cache invalidation.

### `dstok-model`

Parses and validates tokenizer artifacts. The GLM-5.2 loader verifies the
artifact SHA-256 and rejects unsupported normalizers, pre-tokenizers, BPE
flags, post-processors, padding, truncation, vocabulary shapes, and added-token
flags.

### `dstok-core`

The safe scalar implementation. Added-token extraction, exact Oniguruma spans,
BPE merge behavior, IDs, offsets, and counts share this implementation.

### `dstok-fast`

Adds bounded process-wide caches and output-specialized sinks. Cache misses
delegate to the exact core operations rather than maintaining another
tokenization algorithm.

### Private development crates

`dstok-oracle`, `dstok-bench`, `dstok-fuzz`, `dstok-corpus`, `dstok-compat`,
`dstok-py`, and `dstok-simd` are not published. In particular, the public
package does not depend on the experimental SIMD pre-tokenizer.

## Ownership and concurrency

`Tokenizer` clones share an `Arc<Artifact>`. The artifact is immutable after
loading. Moka provides synchronized process-wide caches; values are pure
functions of the artifact digest and complete raw key bytes.

Cache capacities are process-wide, not per tokenizer, thread, tenant, or
request. Environment variables are read when each global cache is initialized,
so applications must set them before the first tokenizer call.

## Memory model

Count-only operation avoids materializing the complete ID, offset, and special
mask vectors. The segment cache retains raw prompt bytes by design to make
hash collisions incapable of changing a count. Its configured weight bounds
key and value payloads, but allocator and cache bookkeeping mean the setting is
not an exact RSS limit.

Large callers should additionally bound:

- concurrent request-body bytes;
- concurrent rendered-prompt bytes;
- CPU-bound tokenizer tasks; and
- request and tokenization deadlines.
