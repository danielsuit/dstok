# dstok-fast

Count-only caches and optimized encoding APIs for
[`dstok`](https://crates.io/crates/dstok).

Adds a bounded pretoken cache and an exact count-only text-segment cache on
top of [`dstok-core`](https://crates.io/crates/dstok-core). Caching is keyed
on the verified artifact digest plus complete raw segment bytes, so a changed
or newly formed boundary misses rather than returning a stale count. No path
here may bypass the differential gate against `dstok-core`.

Most users should depend on the [`dstok`](https://crates.io/crates/dstok)
facade instead of this crate directly.

Licensed under either Apache-2.0 or MIT, at your option.
