use dstok_model::loader::load_glm52_path;
use serde::Serialize;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Label for the reference implementation in emitted measurements. Kept as a
/// single constant so a `tokenizers` bump cannot leave some call sites
/// reporting a version that was not the one measured.
const HF_LABEL: &str = "hf-0.23.1";

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(pointer, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    let args = Args::parse();
    std::env::set_var("DSTOK_CACHE_MAX_BYTES", args.cache_bytes.to_string());
    std::env::set_var(
        "DSTOK_SEGMENT_CACHE_MAX_BYTES",
        args.segment_cache_bytes.to_string(),
    );

    let root = workspace_root();
    let tokenizer_path = root.join("fixtures/glm-5.2/tokenizer.json");
    if let Some(probe) = args.probe.as_deref() {
        run_isolated_rss_probe(probe, &args, &tokenizer_path);
        return;
    }
    let artifact =
        load_glm52_path(path_string(&tokenizer_path)).expect("load pinned GLM-5.2 artifact");
    let hf = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|error| panic!("load HF tokenizer: {error}"));

    let mut workloads = Vec::new();
    for target_tokens in &args.targets {
        let rendered = render_coding_history(target_tokens.saturating_mul(4), args.turns);
        let hf_ids = hf
            .encode_fast(rendered.as_str(), false)
            .expect("HF verification encode")
            .get_ids()
            .to_vec();
        let core_ids = dstok_core::encode_to_ids(&artifact, &rendered);
        let fast_ids = dstok_fast::encode_ids_only(&artifact, &rendered);
        assert_eq!(core_ids, hf_ids, "core IDs must match HF before timing");
        assert_eq!(fast_ids, hf_ids, "fast IDs must match HF before timing");
        assert_eq!(dstok_fast::count(&artifact, &rendered), hf_ids.len());
        drop(core_ids);
        drop(fast_ids);

        let mut measurements = Vec::new();
        measurements.push(measure(
            HF_LABEL,
            "encode-fast",
            1,
            args.iterations,
            || {},
            || {
                hf.encode_fast(black_box(rendered.as_str()), false)
                    .expect("HF timed encode")
                    .len()
            },
        ));
        measurements.push(measure(
            HF_LABEL,
            "render-plus-encode",
            1,
            args.iterations,
            || {},
            || {
                let prompt = render_coding_history(target_tokens.saturating_mul(4), args.turns);
                hf.encode_fast(black_box(prompt.as_str()), false)
                    .expect("HF render-plus-encode")
                    .len()
            },
        ));
        measurements.push(measure(
            "dstok-core",
            "count-only",
            1,
            args.iterations,
            || {},
            || dstok_core::count(&artifact, black_box(&rendered)),
        ));
        measurements.push(measure(
            "dstok-fast",
            "count-cold-cache",
            1,
            args.iterations,
            dstok_fast::clear_global_cache,
            || dstok_fast::count(&artifact, black_box(&rendered)),
        ));
        let _ = dstok_fast::count(&artifact, &rendered);
        measurements.push(measure(
            "dstok-fast",
            "count-warm-cache",
            1,
            args.iterations,
            || {},
            || dstok_fast::count(&artifact, black_box(&rendered)),
        ));
        measurements.push(measure(
            "dstok-fast",
            "ids-only-warm-cache",
            1,
            args.iterations,
            || {},
            || dstok_fast::encode_ids_only(&artifact, black_box(&rendered)).len(),
        ));
        measurements.push(measure(
            "dstok-fast",
            "render-plus-count-warm-cache",
            1,
            args.iterations,
            || {},
            || {
                let prompt = render_coding_history(target_tokens.saturating_mul(4), args.turns);
                dstok_fast::count(&artifact, black_box(&prompt))
            },
        ));
        for &threads in &args.threads {
            if threads > 1 {
                let _ = dstok_fast::count_parallel(&artifact, &rendered, threads);
                measurements.push(measure(
                    "dstok-fast",
                    "count-parallel-warm-cache",
                    threads,
                    args.iterations,
                    || {},
                    || dstok_fast::count_parallel(&artifact, black_box(&rendered), threads),
                ));
            }
        }

        let prefix_lengths = [25usize, 50, 75, 100].map(|percent| rendered.len() * percent / 100);
        measurements.push(measure(
            HF_LABEL,
            "append-only-full-recount",
            1,
            args.iterations,
            || {},
            || {
                prefix_lengths
                    .iter()
                    .map(|&length| {
                        hf.encode_fast(&rendered[..length], false)
                            .expect("HF append encode")
                            .len()
                    })
                    .sum()
            },
        ));
        measurements.push(measure(
            "dstok-fast",
            "append-only-full-recount",
            1,
            args.iterations,
            dstok_fast::clear_global_cache,
            || {
                prefix_lengths
                    .iter()
                    .map(|&length| dstok_fast::count(&artifact, &rendered[..length]))
                    .sum()
            },
        ));

        let hf_encode_p50 = measurements
            .iter()
            .find(|measurement| {
                measurement.implementation == HF_LABEL && measurement.scenario == "encode-fast"
            })
            .map(|measurement| measurement.p50_ms)
            .expect("HF reference measurement");
        let hf_render_p50 = measurements
            .iter()
            .find(|measurement| {
                measurement.implementation == HF_LABEL
                    && measurement.scenario == "render-plus-encode"
            })
            .map(|measurement| measurement.p50_ms)
            .expect("HF render reference measurement");
        let hf_append_p50 = measurements
            .iter()
            .find(|measurement| {
                measurement.implementation == HF_LABEL
                    && measurement.scenario == "append-only-full-recount"
            })
            .map(|measurement| measurement.p50_ms)
            .expect("HF append reference measurement");
        for measurement in &mut measurements {
            let baseline = match measurement.scenario {
                "append-only-full-recount" => hf_append_p50,
                "render-plus-encode" | "render-plus-count-warm-cache" => hf_render_p50,
                _ => hf_encode_p50,
            };
            measurement.speedup_vs_hf_p50 =
                Some(baseline / measurement.p50_ms.max(f64::MIN_POSITIVE));
        }
        workloads.push(WorkloadReport {
            requested_tokens: *target_tokens,
            actual_tokens: hf_ids.len(),
            rendered_bytes: rendered.len(),
            ids_verified: true,
            measurements,
        });
    }

    let report = Report {
        schema_version: 2,
        profile: "glm-5.2",
        oracle: "huggingface/tokenizers 0.23.1 encode_fast",
        hardware: Hardware {
            architecture: std::env::consts::ARCH,
            operating_system: std::env::consts::OS,
            logical_cpus: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        },
        cache_max_bytes: args.cache_bytes,
        segment_cache_max_bytes: args.segment_cache_bytes,
        iterations: args.iterations,
        methodology: vec![
            "All token IDs are compared with HF before timing.",
            "Inputs are already-rendered coding-agent histories.",
            "Cold cache is invalidated before every timed sample.",
            "Append-only starts cold, then fully recounts 25/50/75/100 percent while reusing exact unchanged text segments.",
            "Allocation counters are process-global deltas during each row.",
            "Peak RSS is the process high-water mark at row completion.",
            "Speedups use the matching HF encode, render, or append scenario.",
        ],
        workloads,
    };
    let mut json = serde_json::to_string_pretty(&report).expect("serialize benchmark report");
    json.push('\n');
    if let Some(path) = args.output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create benchmark output directory");
        }
        std::fs::write(&path, &json).expect("write benchmark report");
    }
    print!("{json}");
}

fn run_isolated_rss_probe(probe: &str, args: &Args, tokenizer_path: &Path) {
    let requested_tokens = *args.targets.first().expect("one probe target");
    let rendered = render_coding_history(requested_tokens.saturating_mul(4), args.turns);
    let started = Instant::now();
    let tokens = match probe {
        "dstok" => {
            let artifact =
                load_glm52_path(path_string(tokenizer_path)).expect("load dstok artifact");
            black_box(dstok_fast::count(&artifact, black_box(&rendered)))
        }
        "hf" => {
            let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
                .unwrap_or_else(|error| panic!("load HF tokenizer: {error}"));
            let encoding = tokenizer
                .encode_fast(black_box(rendered.as_str()), false)
                .expect("HF RSS probe encode");
            black_box(encoding.len())
        }
        other => panic!("unknown RSS probe: {other}; expected dstok or hf"),
    };
    let report = RssProbeReport {
        schema_version: 2,
        implementation: probe,
        requested_tokens,
        actual_tokens: tokens,
        rendered_bytes: rendered.len(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        peak_rss_bytes: peak_rss_bytes(),
        cache_max_bytes: args.cache_bytes,
        segment_cache_max_bytes: args.segment_cache_bytes,
        parity_manifest: "fixtures/glm-5.2/conformance.json",
    };
    let mut json = serde_json::to_string_pretty(&report).expect("serialize RSS probe");
    json.push('\n');
    if let Some(path) = &args.output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create RSS output directory");
        }
        std::fs::write(path, &json).expect("write RSS probe");
    }
    print!("{json}");
}

fn measure(
    implementation: &'static str,
    scenario: &'static str,
    threads: usize,
    iterations: usize,
    mut prepare: impl FnMut(),
    mut operation: impl FnMut() -> usize,
) -> Measurement {
    let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
    let bytes_before = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let mut samples_ns = Vec::with_capacity(iterations);
    let mut tokens = 0;
    for _ in 0..iterations {
        prepare();
        let started = Instant::now();
        tokens = black_box(operation());
        samples_ns.push(started.elapsed().as_nanos() as u64);
    }
    samples_ns.sort_unstable();
    Measurement {
        implementation,
        scenario,
        threads,
        tokens,
        p50_ms: percentile_ms(&samples_ns, 50),
        p95_ms: percentile_ms(&samples_ns, 95),
        p99_ms: percentile_ms(&samples_ns, 99),
        allocated_bytes: ALLOCATED_BYTES
            .load(Ordering::Relaxed)
            .saturating_sub(bytes_before),
        allocations: ALLOCATIONS
            .load(Ordering::Relaxed)
            .saturating_sub(allocations_before),
        peak_rss_bytes: peak_rss_bytes(),
        speedup_vs_hf_p50: None,
    }
}

fn percentile_ms(samples: &[u64], percentile: usize) -> f64 {
    let index = (samples.len() * percentile)
        .div_ceil(100)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[index] as f64 / 1_000_000.0
}

fn render_coding_history(target_bytes: usize, turns: usize) -> String {
    let turns = turns.max(1);
    let mut rendered = String::with_capacity(target_bytes + turns * 256);
    rendered.push_str("<|system|>\nYou are a coding agent. Inspect files before editing, preserve behavior, and run focused tests.\n");
    rendered.push_str("<|user|>\nImprove the parser incrementally and keep every previous requirement in context.\n");
    let bytes_per_turn = target_bytes.div_ceil(turns).max(1);
    for turn in 0..turns {
        rendered.push_str("<|assistant|>\n<think>Inspect parser module and test output.</think>\n<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"src/parser_");
        rendered.push_str(&turn.to_string());
        rendered.push_str(".rs\"}}</tool_call>\n<|observation|>\n");
        let line = format!(
            "pub fn parse_{turn}(input: &str) -> Result<Node, ParseError> {{ /* preserve byte offsets and diagnostics */ todo!() }}\n\
             test parser::handles_nested_generics ... ok\n\
             test parser::keeps_error_message ... ok\n"
        );
        let start = rendered.len();
        while rendered.len() - start < bytes_per_turn {
            rendered.push_str(&line);
        }
        rendered.truncate(start + bytes_per_turn);
        rendered.push_str("\n<|assistant|>\nIteration complete: preserve diagnostics and add a regression test.\n<|user|>\nContinue and rerun the focused test.\n");
    }
    rendered.push_str("<|assistant|>\n");
    rendered
}

fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    let rss = unsafe { usage.assume_init().ru_maxrss as u64 };
    #[cfg(target_os = "linux")]
    {
        rss.saturating_mul(1024)
    }
    #[cfg(not(target_os = "linux"))]
    {
        rss
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("benchmark crate has a workspace parent")
        .to_path_buf()
}

fn path_string(path: &Path) -> &str {
    path.to_str().expect("workspace path is valid UTF-8")
}

struct Args {
    targets: Vec<usize>,
    iterations: usize,
    turns: usize,
    threads: Vec<usize>,
    cache_bytes: u64,
    segment_cache_bytes: u64,
    output: Option<PathBuf>,
    probe: Option<String>,
}

impl Args {
    fn parse() -> Self {
        let mut full = false;
        let mut targets = None;
        let mut iterations = None;
        let mut turns = 128;
        let mut threads = vec![1, 2, 4, 8];
        let mut cache_bytes = 64 * 1024 * 1024;
        let mut segment_cache_bytes = 128 * 1024 * 1024;
        let mut output = None;
        let mut probe = None;
        let arguments: Vec<String> = std::env::args().skip(1).collect();
        let mut index = 0;
        while index < arguments.len() {
            match arguments[index].as_str() {
                "--full" => full = true,
                "--targets" => {
                    index += 1;
                    targets = Some(parse_list(&arguments[index]));
                }
                "--iterations" => {
                    index += 1;
                    iterations = Some(arguments[index].parse().expect("valid iterations"));
                }
                "--turns" => {
                    index += 1;
                    turns = arguments[index].parse().expect("valid turns");
                }
                "--threads" => {
                    index += 1;
                    threads = parse_list(&arguments[index]);
                }
                "--cache-bytes" => {
                    index += 1;
                    cache_bytes = arguments[index].parse().expect("valid cache bytes");
                }
                "--segment-cache-bytes" => {
                    index += 1;
                    segment_cache_bytes =
                        arguments[index].parse().expect("valid segment cache bytes");
                }
                "--output" => {
                    index += 1;
                    output = Some(PathBuf::from(&arguments[index]));
                }
                "--probe" => {
                    index += 1;
                    probe = Some(arguments[index].clone());
                }
                unknown => panic!("unknown benchmark argument: {unknown}"),
            }
            index += 1;
        }
        let targets = targets.unwrap_or_else(|| {
            if full {
                vec![1_500_000, 5_000_000, 10_000_000, 20_000_000]
            } else {
                vec![100_000]
            }
        });
        let iterations = iterations.unwrap_or(if full { 1 } else { 5 });
        Self {
            targets,
            iterations: iterations.max(1),
            turns,
            threads,
            cache_bytes,
            segment_cache_bytes,
            output,
            probe,
        }
    }
}

fn parse_list(value: &str) -> Vec<usize> {
    value
        .split(',')
        .map(|entry| entry.parse().expect("comma-separated integer"))
        .collect()
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    profile: &'static str,
    oracle: &'static str,
    hardware: Hardware,
    cache_max_bytes: u64,
    segment_cache_max_bytes: u64,
    iterations: usize,
    methodology: Vec<&'static str>,
    workloads: Vec<WorkloadReport>,
}

#[derive(Serialize)]
struct Hardware {
    architecture: &'static str,
    operating_system: &'static str,
    logical_cpus: usize,
}

#[derive(Serialize)]
struct WorkloadReport {
    requested_tokens: usize,
    actual_tokens: usize,
    rendered_bytes: usize,
    ids_verified: bool,
    measurements: Vec<Measurement>,
}

#[derive(Serialize)]
struct Measurement {
    implementation: &'static str,
    scenario: &'static str,
    threads: usize,
    tokens: usize,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    allocated_bytes: u64,
    allocations: u64,
    peak_rss_bytes: u64,
    speedup_vs_hf_p50: Option<f64>,
}

#[derive(Serialize)]
struct RssProbeReport<'a> {
    schema_version: u32,
    implementation: &'a str,
    requested_tokens: usize,
    actual_tokens: usize,
    rendered_bytes: usize,
    elapsed_ms: f64,
    peak_rss_bytes: u64,
    cache_max_bytes: u64,
    segment_cache_max_bytes: u64,
    parity_manifest: &'static str,
}
