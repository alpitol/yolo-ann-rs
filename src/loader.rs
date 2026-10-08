//! Background image decoding with a priority queue that is replaced whenever
//! the current image changes, so stale prefetch work is dropped.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

use egui::ColorImage;

/// GL implementations commonly support 8192+; downscale anything larger for display.
const MAX_TEXTURE_SIDE: u32 = 8192;

pub struct Decoded {
    pub path: PathBuf,
    /// Original image size in pixels and decoded pixels (possibly downscaled).
    pub result: Result<([u32; 2], ColorImage), String>,
    pub file_size: u64,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<PathBuf>,
    in_flight: HashSet<PathBuf>,
    shutdown: bool,
}

pub struct Loader {
    shared: Arc<(Mutex<Queue>, Condvar)>,
    rx: Receiver<Decoded>,
}

impl Loader {
    pub fn new(ctx: egui::Context, workers: usize) -> Self {
        let shared = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let (tx, rx) = channel();
        for i in 0..workers.max(1) {
            let shared = shared.clone();
            let tx: Sender<Decoded> = tx.clone();
            let ctx = ctx.clone();
            std::thread::Builder::new()
                .name(format!("decode-{i}"))
                .spawn(move || worker(shared, tx, ctx))
                .expect("spawn decoder thread");
        }
        Loader { shared, rx }
    }

    /// Replace pending work with `paths` (in priority order). Paths already
    /// being decoded are skipped.
    pub fn set_wanted(&self, paths: Vec<PathBuf>) {
        let (lock, cv) = &*self.shared;
        let mut q = lock.lock().unwrap();
        q.jobs.clear();
        for p in paths {
            if !q.in_flight.contains(&p) && !q.jobs.contains(&p) {
                q.jobs.push_back(p);
            }
        }
        cv.notify_all();
    }

    pub fn is_busy(&self, path: &Path) -> bool {
        let q = self.shared.0.lock().unwrap();
        q.in_flight.contains(path) || q.jobs.iter().any(|p| p == path)
    }

    pub fn try_recv(&self) -> Option<Decoded> {
        self.rx.try_recv().ok()
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        let (lock, cv) = &*self.shared;
        lock.lock().unwrap().shutdown = true;
        cv.notify_all();
    }
}

fn worker(shared: Arc<(Mutex<Queue>, Condvar)>, tx: Sender<Decoded>, ctx: egui::Context) {
    let (lock, cv) = &*shared;
    loop {
        let path = {
            let mut q = lock.lock().unwrap();
            loop {
                if q.shutdown {
                    return;
                }
                if let Some(p) = q.jobs.pop_front() {
                    q.in_flight.insert(p.clone());
                    break p;
                }
                q = cv.wait(q).unwrap();
            }
        };
        let decoded = decode(&path);
        lock.lock().unwrap().in_flight.remove(&path);
        if tx.send(decoded).is_err() {
            return;
        }
        ctx.request_repaint();
    }
}

pub fn decode(path: &Path) -> Decoded {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let result = (|| -> Result<([u32; 2], ColorImage), String> {
        let img = image::ImageReader::open(path)
            .map_err(|e| e.to_string())?
            .with_guessed_format()
            .map_err(|e| e.to_string())?
            .decode()
            .map_err(|e| e.to_string())?;
        let orig = [img.width(), img.height()];
        let img = if orig[0] > MAX_TEXTURE_SIDE || orig[1] > MAX_TEXTURE_SIDE {
            img.resize(
                MAX_TEXTURE_SIDE,
                MAX_TEXTURE_SIDE,
                image::imageops::FilterType::Triangle,
            )
        } else {
            img
        };
        let size = [img.width() as usize, img.height() as usize];
        let color = match img {
            image::DynamicImage::ImageRgb8(rgb) => ColorImage::from_rgb(size, rgb.as_raw()),
            other => ColorImage::from_rgba_unmultiplied(size, other.to_rgba8().as_raw()),
        };
        Ok((orig, color))
    })();
    Decoded {
        path: path.to_path_buf(),
        result,
        file_size,
    }
}

#[cfg(test)]
mod tests {
    /// `YOLO_ANN_BENCH_DIR=dir cargo test --release decode_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn decode_speed() {
        let dir = std::path::PathBuf::from(
            std::env::var("YOLO_ANN_BENCH_DIR").expect("set YOLO_ANN_BENCH_DIR"),
        );
        let names = crate::model::scan_images(&dir).unwrap();
        let t = std::time::Instant::now();
        for n in &names {
            let d = super::decode(&dir.join(n));
            assert!(d.result.is_ok());
        }
        let per = t.elapsed().as_secs_f64() * 1000.0 / names.len() as f64;
        println!("{} images, {per:.1} ms/image (single thread)", names.len());
    }
}
