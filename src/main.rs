mod app;
mod backup;
mod loader;
mod model;

use std::path::PathBuf;

use clap::Parser;

/// Fast local YOLO bounding box annotation tool.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Folder of images to annotate (defaults to the last opened folder).
    image_dir: Option<PathBuf>,
    /// Import YOLO .txt labels from this folder; they are also saved back here.
    /// Default: the image folder, if it contains label files.
    #[arg(short, long)]
    labels: Option<PathBuf>,
    /// Class names file: one per line, or a data.yaml. Default: classes.txt / *.names / data.yaml in the image or labels folder.
    #[arg(short, long)]
    classes: Option<PathBuf>,
    /// Import a COCO json on startup.
    #[arg(long)]
    coco: Option<PathBuf>,
    /// Number of images to prefetch ahead.
    #[arg(long, default_value_t = 6)]
    ahead: usize,
    /// Number of images to keep/prefetch behind.
    #[arg(long, default_value_t = 2)]
    behind: usize,
    /// Decoder threads (default: half the CPUs, 2..=6).
    #[arg(long)]
    workers: Option<usize>,
    /// Write label files automatically when leaving an image.
    #[arg(long)]
    autosave: Option<bool>,
}

fn icon() -> egui::IconData {
    let img = image::load_from_memory(include_bytes!("../logo.png"))
        .expect("embedded logo.png")
        .into_rgba8();
    let (width, height) = img.dimensions();
    egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    }
}

fn main() -> eframe::Result {
    let args = Args::parse();
    let workers = args.workers.unwrap_or_else(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        (n / 2).clamp(2, 6)
    });
    let cfg = app::Config {
        image_dir: args.image_dir,
        labels_dir: args.labels,
        classes: args.classes,
        coco: args.coco,
        ahead: args.ahead,
        behind: args.behind,
        workers,
        autosave: args.autosave,
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("yolo-ann")
            .with_app_id("yolo-ann")
            .with_icon(std::sync::Arc::new(icon())),
        ..Default::default()
    };
    eframe::run_native(
        "yolo-ann",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc, cfg)))),
    )
}
