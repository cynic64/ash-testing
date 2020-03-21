use std::path::PathBuf;

pub fn relative_path(local_path: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), local_path].iter().collect()
}

pub fn get_elapsed(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs() as f64 + start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0
}
