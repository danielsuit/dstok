use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let command = std::env::args().nth(1).unwrap_or_else(|| "ci".into());
    match command.as_str() {
        "ci" => run_ci(),
        "parity" | "parity-glm52" => run_parity_glm52(),
        "bench" => run(&["run", "-p", "dstok-bench", "--release", "--"]),
        other => {
            eprintln!("unknown xtask command: {other}");
            std::process::exit(2);
        }
    }
}

fn run_ci() {
    run(&["check", "--workspace", "--all-targets"]);
    run(&["test", "--workspace", "--all-targets"]);
    run(&[
        "clippy",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ]);
    run_parity_glm52();
}

fn run_parity_glm52() {
    run(&["test", "-p", "dstok-oracle", "--test", "glm52_parity"]);

    let root = workspace_root();
    let artifact_path = root.join("fixtures/glm-5.2/tokenizer.json");
    let bytes = std::fs::read(&artifact_path).unwrap_or_else(|error| {
        panic!("read {}: {error}", artifact_path.display());
    });
    let artifact = dstok_model::loader::load_glm52_path(
        artifact_path
            .to_str()
            .expect("workspace paths must be valid UTF-8"),
    )
    .expect("load pinned artifact after parity");

    let manifest = ConformanceManifest {
        schema_version: 1,
        profile: "glm-5.2",
        status: "passing",
        artifact: ArtifactManifest {
            sha256: hex(&Sha256::digest(&bytes)),
            blake3: hex(blake3::hash(&bytes).as_bytes()),
            bytes: bytes.len(),
            vocab_size: artifact.vocab.len(),
            merge_rules: artifact.merges.len(),
            added_tokens: artifact.config.added_tokens.len(),
        },
        oracle: OracleManifest {
            implementation: "huggingface/tokenizers",
            version: "0.23.1",
            regex_backend: "onig",
            interface: "encode_fast(add_special_tokens=false)",
        },
        assertions: vec![
            "dstok-core token IDs equal HF",
            "dstok-fast token IDs equal HF",
            "count-only equals HF ID length",
            "serial and 2/4/8-worker paths agree",
            "frozen gateway-rendered prompts pass",
            "adversarial Unicode and added-token cases pass",
            "exact segment-cache append and boundary cases equal full HF recounts",
            "property-generated Unicode cases pass",
        ],
    };
    let path = root.join("fixtures/glm-5.2/conformance.json");
    let mut json = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
    json.push('\n');
    std::fs::write(&path, json).unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    println!("{}", path.display());
}

fn run(arguments: &[&str]) {
    let status = Command::new("cargo")
        .args(arguments)
        .current_dir(workspace_root())
        .status()
        .unwrap_or_else(|error| panic!("run cargo {}: {error}", arguments.join(" ")));
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a workspace parent")
        .to_path_buf()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            let _ = write!(output, "{byte:02x}");
            output
        },
    )
}

#[derive(Serialize)]
struct ConformanceManifest {
    schema_version: u32,
    profile: &'static str,
    status: &'static str,
    artifact: ArtifactManifest,
    oracle: OracleManifest,
    assertions: Vec<&'static str>,
}

#[derive(Serialize)]
struct ArtifactManifest {
    sha256: String,
    blake3: String,
    bytes: usize,
    vocab_size: usize,
    merge_rules: usize,
    added_tokens: usize,
}

#[derive(Serialize)]
struct OracleManifest {
    implementation: &'static str,
    version: &'static str,
    regex_backend: &'static str,
    interface: &'static str,
}
