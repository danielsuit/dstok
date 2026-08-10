use dstok::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: count <tokenizer.json> <rendered-prompt>",
        )
    };
    let mut args = std::env::args().skip(1);
    let artifact = args.next().ok_or_else(usage)?;
    let prompt = args.next().ok_or_else(usage)?;
    if args.next().is_some() {
        return Err(usage().into());
    }

    let tokenizer = Tokenizer::from_file(artifact)?;
    println!("{}", tokenizer.count(&prompt));
    Ok(())
}
