#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    dstok_fuzz::fuzz_glm52_against_hf(data);
});
