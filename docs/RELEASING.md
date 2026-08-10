# Releasing

`dstok` is a facade over three publishable crates. Publish them in dependency
order:

1. `dstok-model`
2. `dstok-core`
3. `dstok-fast`
4. `dstok`

The remaining workspace crates are private and have `publish = false`.

## Pre-release checklist

1. Confirm the working tree contains only intended changes.
2. Update `CHANGELOG.md`.
3. Set the same semantic version in `[workspace.package]` and every internal
   dependency requirement.
4. Run all correctness gates on Linux x86_64 and Linux/macOS AArch64.
5. Inspect each package archive.
6. Publish with `--dry-run`.
7. Publish in order, waiting for each crate to become available in the index.
8. Tag the exact release commit as `vX.Y.Z`.

Commands:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo run -p xtask -- parity-glm52
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

cargo package -p dstok-model
cargo package -p dstok-core --no-verify \
  --config "patch.crates-io.dstok-model.path=\"$PWD/dstok-model\""
cargo package -p dstok-fast --no-verify \
  --config "patch.crates-io.dstok-model.path=\"$PWD/dstok-model\"" \
  --config "patch.crates-io.dstok-core.path=\"$PWD/dstok-core\""
cargo package -p dstok --no-verify \
  --config "patch.crates-io.dstok-model.path=\"$PWD/dstok-model\"" \
  --config "patch.crates-io.dstok-core.path=\"$PWD/dstok-core\"" \
  --config "patch.crates-io.dstok-fast.path=\"$PWD/dstok-fast\""
```

The temporary patches let Cargo resolve unpublished dependencies while it
builds each archive; packaged manifests still contain registry versions, not
the local paths. `--no-verify` is necessary until the preceding package exists
in the registry. After publication, repeat `cargo package` without patches or
`--no-verify` in a clean checkout.

Before the first release, only `dstok-model` can complete a crates.io dry-run;
the other dry-runs require their newly published dependencies to be visible in
the index. During an approved release, process one crate at a time:

```sh
cargo publish -p dstok-model --dry-run
# Review, then: cargo publish -p dstok-model
# Wait for dstok-model to appear in the index.

cargo publish -p dstok-core --dry-run
# Review, publish, and wait for the index.

cargo publish -p dstok-fast --dry-run
# Review, publish, and wait for the index.

cargo publish -p dstok --dry-run
# Review, then publish the facade.
```

Never use `--allow-dirty` for an actual release.

## Package contents

The public `dstok` crate does not bundle the 20 MB tokenizer artifact. Users
must obtain the exact artifact through their model-asset pipeline. This keeps
the Rust package small and makes artifact updates explicit.

Verify archives under `target/package/` contain source, README, changelog, and
license files but no benchmark outputs, corpora, credentials, or build
artifacts.

## Versioning policy

- Patch: fixes that preserve accepted artifacts and token IDs.
- Minor: additive APIs, cache changes, or a new explicitly versioned profile.
- Major: public API removals or changed behavior for an existing profile.

Any change that alters token IDs for the existing pinned artifact is not a
normal semantic-version update; it is a correctness regression unless the
reference artifact itself is intentionally versioned.
