use std::path::PathBuf;

pub fn relative_path(local_path: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), local_path].iter().collect()
}

pub fn get_elapsed(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs() as f64 + start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0
}

pub fn size_of_slice<T>(slice: &[T]) -> u64 {
    (std::mem::size_of::<T>() * slice.len()) as u64
}

// a timer that can be started and stopped many times and can print the average
// timed duration, useful for averaging durations across loop iterations
pub struct LoopTimer {
    name: String,
    total_time: f64,
    last_start: Option<std::time::Instant>,
    stop_count: u32,
}

impl LoopTimer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            total_time: 0.0,
            last_start: None,
            stop_count: 0,
        }
    }

    pub fn start(&mut self) {
        match self.last_start {
            Some(_) => panic!(
                "Start called on already started LoopTimer (name {})",
                self.name
            ),
            None => self.last_start = Some(std::time::Instant::now()),
        }
    }

    pub fn stop(&mut self) {
        match self.last_start {
            Some(time) => {
                self.total_time += get_elapsed(time);
                self.stop_count += 1;
                self.last_start = None;
            }
            None => eprintln!("Stop called on unstarted LoopTimer"),
        }
    }

    pub fn print(&mut self) {
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!(
            "{} average delta: {:.3} ms",
            self.name,
            self.total_time / self.stop_count as f64 * 1_000.0
        );
        println!(
            "--------------------------------------------------------------------------------"
        );
    }
}
