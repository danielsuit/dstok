use dstok::Tokenizer;

fn artifact_path() -> String {
    format!(
        "{}/../fixtures/glm-5.2/tokenizer.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn tokenizer_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tokenizer>();
}

#[test]
fn public_count_and_encoding_apis_agree() {
    let tokenizer = Tokenizer::from_file(artifact_path()).expect("load pinned artifact");
    let rendered = "<|user|>fn main() { println!(\"hello\"); }\n<|assistant|>";
    assert_eq!(
        tokenizer.count(rendered),
        tokenizer.encode_ids(rendered).len()
    );
    assert_eq!(tokenizer.count(rendered), tokenizer.encode(rendered).len());
}

#[test]
fn in_memory_loader_enforces_the_pinned_artifact() {
    let bytes = std::fs::read(artifact_path()).expect("read pinned artifact");
    Tokenizer::from_bytes(&bytes).expect("load pinned bytes");

    let mut changed = bytes;
    changed.push(b'\n');
    assert!(Tokenizer::from_bytes(&changed).is_err());
}
