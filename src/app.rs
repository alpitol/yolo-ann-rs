use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use egui::{
    Align2, Color32, CursorIcon, FontId, Key, Modifiers, PointerButton, Pos2, Rect, RichText,
    Sense, Shape, Stroke, StrokeKind, TextureHandle, TextureOptions, Vec2, pos2, vec2,
};

use crate::backup::{self, Backup, Settings};
use crate::loader::{Decoded, Loader};
use crate::model::{self, BBox, ImageAnn};

pub struct Config {
    pub image_dir: Option<PathBuf>,
    pub labels_dir: Option<PathBuf>,
    pub classes: Option<PathBuf>,
    pub coco: Option<PathBuf>,
    pub ahead: usize,
    pub behind: usize,
    pub workers: usize,
    pub autosave: Option<bool>,
}

const PALETTE: [Color32; 10] = [
    Color32::from_rgb(230, 25, 75),
    Color32::from_rgb(60, 180, 75),
    Color32::from_rgb(255, 225, 25),
    Color32::from_rgb(0, 130, 200),
    Color32::from_rgb(245, 130, 48),
    Color32::from_rgb(145, 30, 180),
    Color32::from_rgb(70, 240, 240),
    Color32::from_rgb(240, 50, 230),
    Color32::from_rgb(210, 245, 60),
    Color32::from_rgb(250, 190, 212),
];

fn class_color(c: usize) -> Color32 {
    PALETTE[c % PALETTE.len()]
}

const MIN_BOX_SCREEN_PX: f32 = 4.0;
const BACKUP_DEBOUNCE: Duration = Duration::from_secs(2);
const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 40.0;

enum Slot {
    Ready {
        tex: TextureHandle,
        size: [u32; 2],
        file_size: u64,
        last_used: u64,
    },
    Failed(String),
}

// Handle bitmask for resizing.
const H_LEFT: u8 = 1;
const H_RIGHT: u8 = 2;
const H_TOP: u8 = 4;
const H_BOTTOM: u8 = 8;

#[derive(Clone, Copy)]
enum Drag {
    None,
    Draw {
        start: (f64, f64),
        cur: (f64, f64),
    },
    Move {
        idx: usize,
        start: (f64, f64),
        orig: BBox,
        cur: BBox,
    },
    Resize {
        idx: usize,
        handle: u8,
        cur: BBox,
    },
}

struct CropJob {
    done: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
    finished: Arc<Mutex<Option<String>>>,
}

pub struct App {
    /// Screen rect of the image in the last frame (for tests).
    pub(crate) last_img_rect: Option<Rect>,
    ahead: usize,
    behind: usize,
    settings: Settings,

    image_dir: Option<PathBuf>,
    /// YOLO labels folder: where Ctrl+S / autosave write label files.
    labels_dir: Option<PathBuf>,
    /// `labels_dir` was picked automatically (label files next to the images).
    labels_auto: bool,
    classes_file: Option<PathBuf>,
    images: Vec<String>,
    visible: Vec<usize>,
    search: String,
    /// Show only images that have annotation data (an entry in `anns`).
    filter_data: bool,
    cur: Option<usize>,
    classes: Vec<String>,
    cur_class: usize,
    anns: HashMap<String, ImageAnn>,
    selected: Option<usize>,

    loader: Loader,
    cache: HashMap<PathBuf, Slot>,
    tick: u64,

    zoom: f32,
    pan: Vec2,
    drag: Drag,
    hover_px: Option<(f64, f64)>,

    status: String,
    scroll_to_current: bool,
    title: String,

    backup_path: Option<PathBuf>,
    backup_gen: u64,
    backup_written_gen: u64,
    backup_last_write: Instant,
    backup_first_write: bool,
    backup_saved_at: Option<u64>,
    pending_restore: Option<Backup>,
    show_quit_dialog: bool,
    allow_close: bool,
    crop_job: Option<CropJob>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, cfg: Config) -> Self {
        let mut settings = Settings::load();
        if let Some(a) = cfg.autosave {
            settings.autosave_labels = a;
        }
        let mut app = App {
            last_img_rect: None,
            ahead: cfg.ahead,
            behind: cfg.behind,
            settings,
            image_dir: None,
            labels_dir: None,
            labels_auto: false,
            classes_file: None,
            images: Vec::new(),
            visible: Vec::new(),
            search: String::new(),
            filter_data: false,
            cur: None,
            classes: Vec::new(),
            cur_class: 0,
            anns: HashMap::new(),
            selected: None,
            loader: Loader::new(cc.egui_ctx.clone(), cfg.workers),
            cache: HashMap::new(),
            tick: 0,
            zoom: 1.0,
            pan: Vec2::ZERO,
            drag: Drag::None,
            hover_px: None,
            status: String::new(),
            scroll_to_current: false,
            title: String::new(),
            backup_path: None,
            backup_gen: 0,
            backup_written_gen: 0,
            backup_last_write: Instant::now() - BACKUP_DEBOUNCE,
            backup_first_write: true,
            backup_saved_at: None,
            pending_restore: None,
            show_quit_dialog: false,
            allow_close: false,
            crop_job: None,
        };

        let dir = cfg
            .image_dir
            .clone()
            .or_else(|| app.settings.last_image_dir.clone().filter(|d| d.is_dir()));
        if let Some(dir) = dir {
            let from_settings = cfg.image_dir.is_none();
            let labels = cfg.labels_dir.clone().or_else(|| {
                if from_settings {
                    app.settings.last_labels_dir.clone()
                } else {
                    None
                }
            });
            let classes = cfg.classes.clone().or_else(|| {
                if from_settings {
                    app.settings.last_classes_file.clone()
                } else {
                    None
                }
            });
            app.open_dir(&dir, labels, classes);
            if from_settings
                && let Some(name) = app.settings.last_image.clone()
                && let Some(i) = app.images.iter().position(|n| *n == name)
            {
                app.go_to(i, false);
            }
        } else if let Some(c) = &cfg.classes {
            app.load_classes(c);
        }
        if let Some(coco) = &cfg.coco {
            app.import_coco(coco);
        }
        app
    }

    // ---------------------------------------------------------------- state

    fn set_status(&mut self, s: impl Into<String>) {
        self.status = s.into();
    }

    fn mark_changed(&mut self) {
        self.backup_gen += 1;
    }

    fn cur_name(&self) -> Option<&str> {
        self.cur.map(|i| self.images[i].as_str())
    }

    fn image_path(&self, idx: usize) -> Option<PathBuf> {
        Some(self.image_dir.as_ref()?.join(&self.images[idx]))
    }

    /// An image has annotation data once it was imported, restored, edited or visited.
    /// New entries are unsaved: they don't exist in the labels folder yet.
    fn ensure_ann(&mut self, name: &str) -> &mut ImageAnn {
        self.anns
            .entry(name.to_string())
            .or_insert_with(|| ImageAnn {
                dirty: true,
                ..Default::default()
            })
    }

    /// Images whose boxes were edited since the last import/export.
    fn unsaved_count(&self) -> usize {
        self.anns.values().filter(|a| a.modified()).count()
    }

    fn has_unsaved(&self) -> bool {
        self.anns.values().any(ImageAnn::modified)
    }

    fn data_count(&self) -> usize {
        self.anns.len()
    }

    fn open_dir(&mut self, dir: &Path, labels_dir: Option<PathBuf>, classes_file: Option<PathBuf>) {
        self.leave_folder();
        let dir = &dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let labels_dir = labels_dir.map(|d| d.canonicalize().unwrap_or(d));
        let images = match model::scan_images(dir) {
            Ok(v) => v,
            Err(e) => {
                self.set_status(format!("{e:#}"));
                return;
            }
        };
        self.image_dir = Some(dir.to_path_buf());
        self.labels_dir = None;
        self.labels_auto = false;
        self.images = images;
        self.anns.clear();
        self.cache.clear();
        self.search.clear();
        self.visible = (0..self.images.len()).collect();
        self.cur = None;
        self.selected = None;
        self.drag = Drag::None;
        self.backup_gen = 0;
        self.backup_written_gen = 0;
        self.backup_first_write = true;

        let classes_file = classes_file
            .filter(|p| p.is_file())
            .or_else(|| model::find_classes_file(dir))
            .or_else(|| self.classes_file.clone());
        if let Some(c) = classes_file {
            self.load_classes(&c);
        }

        let bp = backup::backup_path(dir);
        self.pending_restore = backup::load_backup(&bp).filter(|b| !b.images.is_empty());
        self.backup_saved_at = self.pending_restore.as_ref().map(|b| b.saved_at);
        self.backup_path = Some(bp);

        self.settings.last_image_dir = Some(dir.to_path_buf());
        self.settings.last_labels_dir = None;
        self.settings.last_image = None;
        self.settings.save();

        let mut status = format!("Opened {} images from {}", self.images.len(), dir.display());
        match labels_dir {
            Some(ld) => {
                self.import_yolo(&ld, false);
                status = format!("{status}; {}", self.status);
            }
            None => {
                if let Ok((found, _)) = model::read_yolo_dir(dir, &self.images)
                    && !found.is_empty()
                {
                    self.import_yolo(dir, true);
                    status = format!("{status}; autoselected labels: {}", self.status);
                }
            }
        }
        self.set_status(status);
        self.compute_visible();
        if !self.images.is_empty() {
            self.go_to(0, false);
        }
    }

    /// Flush state for the currently open folder.
    fn leave_folder(&mut self) {
        if self.image_dir.is_none() {
            return;
        }
        if self.settings.autosave_labels && self.labels_dir.is_some() {
            self.save_yolo();
        }
        self.write_backup(true);
    }

    fn load_classes(&mut self, path: &Path) {
        match model::read_classes(path) {
            Ok(c) => {
                self.set_status(format!(
                    "Loaded {} classes from {}",
                    c.len(),
                    path.display()
                ));
                self.classes = c;
                self.cur_class = 0;
                self.classes_file = Some(path.to_path_buf());
                self.settings.last_classes_file = Some(path.to_path_buf());
                self.settings.save();
            }
            Err(e) => self.set_status(format!("{e:#}")),
        }
    }

    /// `user`: the user moved away from the current image (arrow keys, list click).
    /// Only then does a never-annotated image become one with (empty) annotation data,
    /// so programmatic jumps (opening a folder, filtering) don't create data.
    fn go_to(&mut self, idx: usize, user: bool) {
        if idx >= self.images.len() {
            return;
        }
        if let Some(prev) = self.cur
            && prev != idx
        {
            let name = self.images[prev].clone();
            if user && !self.anns.contains_key(&name) {
                self.ensure_ann(&name);
                self.mark_changed();
                self.compute_visible();
            }
            if self.settings.autosave_labels {
                self.save_one(&name);
            }
        }
        self.cur = Some(idx);
        self.selected = None;
        self.drag = Drag::None;
        self.settings.last_image = Some(self.images[idx].clone());
        self.update_prefetch();
        self.scroll_to_current = true;
        self.write_backup(false);
    }

    fn visible_pos(&self) -> Option<usize> {
        self.visible.binary_search(&self.cur?).ok()
    }

    fn navigate(&mut self, delta: isize) {
        let n = self.visible.len();
        if n == 0 {
            return;
        }
        let pos = match self.visible_pos() {
            Some(p) => (p as isize + delta).rem_euclid(n as isize) as usize,
            None => 0,
        };
        self.go_to(self.visible[pos], true);
    }

    fn cycle_class(&mut self, delta: isize) {
        let n = self.classes.len();
        if n == 0 {
            return;
        }
        self.cur_class = (self.cur_class as isize + delta).rem_euclid(n as isize) as usize;
        // Like the original tool: changing class deselects the box.
        self.selected = None;
    }

    /// Recompute the visible list (search + data filter) without moving.
    fn compute_visible(&mut self) {
        let q = self.search.to_lowercase();
        self.visible = (0..self.images.len())
            .filter(|&i| {
                let name = &self.images[i];
                (q.is_empty() || name.to_lowercase().contains(&q))
                    && (!self.filter_data || self.anns.contains_key(name))
            })
            .collect();
    }

    /// After the search text or the data filter changed: jump to the first
    /// visible image if the current one got hidden.
    fn apply_filters(&mut self) {
        self.compute_visible();
        if self.visible_pos().is_none()
            && let Some(&first) = self.visible.first()
        {
            self.go_to(first, false);
        } else {
            self.update_prefetch();
        }
        self.scroll_to_current = true;
    }

    // -------------------------------------------------------------- caching

    fn prefetch_window(&self) -> Vec<usize> {
        let n = self.visible.len();
        let Some(cur) = self.cur else {
            return Vec::new();
        };
        let Some(pos) = self.visible_pos() else {
            return vec![cur];
        };
        let mut out = vec![cur];
        for d in 1..=self.ahead.min(n) {
            out.push(self.visible[(pos + d) % n]);
        }
        for d in 1..=self.behind.min(n) {
            out.push(self.visible[(pos + n - d % n) % n]);
        }
        let mut seen = HashSet::new();
        out.retain(|i| seen.insert(*i));
        out
    }

    fn update_prefetch(&mut self) {
        let Some(dir) = self.image_dir.clone() else {
            return;
        };
        let window: Vec<PathBuf> = self
            .prefetch_window()
            .into_iter()
            .map(|i| dir.join(&self.images[i]))
            .collect();
        let wanted: Vec<PathBuf> = window
            .iter()
            .filter(|p| !self.cache.contains_key(*p))
            .cloned()
            .collect();
        self.loader.set_wanted(wanted);
        self.evict(&window);
    }

    fn evict(&mut self, window: &[PathBuf]) {
        let cap = self.ahead + self.behind + 5;
        if self.cache.len() <= cap {
            return;
        }
        let keep: HashSet<&PathBuf> = window.iter().collect();
        let mut old: Vec<(u64, PathBuf)> = self
            .cache
            .iter()
            .filter(|(p, _)| !keep.contains(p))
            .map(|(p, s)| {
                (
                    if let Slot::Ready { last_used, .. } = s {
                        *last_used
                    } else {
                        0
                    },
                    p.clone(),
                )
            })
            .collect();
        old.sort();
        for (_, p) in old {
            if self.cache.len() <= cap {
                break;
            }
            self.cache.remove(&p);
        }
    }

    fn poll_loader(&mut self, ctx: &egui::Context) {
        let mut got_any = false;
        while let Some(Decoded {
            path,
            result,
            file_size,
        }) = self.loader.try_recv()
        {
            got_any = true;
            if self.image_dir.as_deref() != path.parent() {
                continue;
            }
            let slot = match result {
                Ok((size, color)) => {
                    let opts = TextureOptions {
                        magnification: egui::TextureFilter::Nearest,
                        minification: egui::TextureFilter::Linear,
                        wrap_mode: egui::TextureWrapMode::ClampToEdge,
                        mipmap_mode: Some(egui::TextureFilter::Linear),
                    };
                    let tex = ctx.load_texture(path.to_string_lossy(), color, opts);
                    self.tick += 1;
                    Slot::Ready {
                        tex,
                        size,
                        file_size,
                        last_used: self.tick,
                    }
                }
                Err(e) => Slot::Failed(e),
            };
            self.cache.insert(path, slot);
        }
        if got_any {
            let dir = self.image_dir.clone().unwrap_or_default();
            let window: Vec<PathBuf> = self
                .prefetch_window()
                .into_iter()
                .map(|i| dir.join(&self.images[i]))
                .collect();
            self.evict(&window);
        }
    }

    fn image_size(&self, name: &str) -> Option<[u32; 2]> {
        let p = self.image_dir.as_ref()?.join(name);
        if let Some(Slot::Ready { size, .. }) = self.cache.get(&p) {
            return Some(*size);
        }
        imagesize::size(&p)
            .ok()
            .map(|s| [s.width as u32, s.height as u32])
    }

    // --------------------------------------------------------------- saving

    fn save_one(&mut self, name: &str) -> bool {
        let Some(ld) = self.labels_dir.clone() else {
            return false;
        };
        let Some(ann) = self.anns.get_mut(name) else {
            return true;
        };
        if !ann.dirty {
            return true;
        }
        match model::write_atomic(
            &model::label_path(&ld, name),
            model::format_yolo(&ann.boxes).as_bytes(),
        ) {
            Ok(()) => {
                ann.dirty = false;
                self.backup_gen += 1;
                true
            }
            Err(e) => {
                self.status = format!("Save failed: {e:#}");
                false
            }
        }
    }

    /// Ctrl+S: write changed label files to the labels folder (asks for one if unset).
    fn save_yolo(&mut self) {
        if self.labels_dir.is_none() {
            if let Some(d) = self.pick_labels_dir("Folder for YOLO label files") {
                self.export_yolo(&d);
            }
            return;
        }
        let dirty: Vec<String> = self
            .anns
            .iter()
            .filter(|(_, a)| a.dirty)
            .map(|(n, _)| n.clone())
            .collect();
        let mut ok = 0;
        for n in &dirty {
            if self.save_one(n) {
                ok += 1;
            }
        }
        // Everything that matches the labels folder is safely stored.
        for a in self.anns.values_mut().filter(|a| !a.dirty) {
            a.mark_exported();
        }
        if ok == dirty.len() {
            let ld = self
                .labels_dir
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            self.set_status(format!("Saved {ok} label files to {ld}"));
            self.write_classes_if_missing();
        }
        self.write_backup(true);
    }

    /// YOLO label files are useless without the class list; put one next to them.
    fn write_classes_if_missing(&mut self) {
        let Some(ld) = &self.labels_dir else { return };
        if self.classes.is_empty() || model::find_classes_file(ld).is_some() {
            return;
        }
        let p = ld.join("classes.txt");
        let _ = model::write_atomic(&p, (self.classes.join("\n") + "\n").as_bytes());
    }

    fn write_backup(&mut self, force: bool) {
        if self.backup_gen == self.backup_written_gen {
            return;
        }
        if !force && self.backup_last_write.elapsed() < BACKUP_DEBOUNCE / 4 {
            return;
        }
        let (Some(path), Some(image_dir)) = (self.backup_path.clone(), self.image_dir.clone())
        else {
            return;
        };
        let labels_dir = self.labels_dir.clone();
        // Don't overwrite a not-yet-restored backup with an empty one.
        let images: std::collections::BTreeMap<String, Vec<BBox>> = self
            .anns
            .iter()
            .filter(|(_, a)| a.modified())
            .map(|(n, a)| (n.clone(), a.boxes.clone()))
            .collect();
        if images.is_empty() && self.pending_restore.is_some() {
            return;
        }
        let b = Backup {
            image_dir,
            labels_dir,
            classes: self.classes.clone(),
            saved_at: backup::now_secs(),
            images,
        };
        match backup::write_backup(&path, &b, self.backup_first_write) {
            Ok(()) => {
                self.backup_first_write = false;
                self.backup_written_gen = self.backup_gen;
                self.backup_saved_at = Some(b.saved_at);
            }
            Err(e) => self.status = format!("Backup failed: {e:#}"),
        }
        self.backup_last_write = Instant::now();
    }

    fn restore_backup(&mut self, b: Backup) {
        if self.classes.is_empty() {
            self.classes = b.classes.clone();
        } else if self.classes != b.classes {
            self.set_status(
                "Warning: backup class list differs from loaded classes; class indices kept as-is",
            );
        }
        let known: HashSet<String> = self.images.iter().cloned().collect();
        let mut n = 0;
        for (name, boxes) in b.images {
            if !known.contains(&name) {
                continue;
            }
            let ann = self.ensure_ann(&name);
            ann.snapshot();
            ann.boxes = boxes;
            n += 1;
        }
        self.selected = None;
        self.mark_changed();
        self.compute_visible();
        self.update_prefetch();
        if !self.status.starts_with("Warning") {
            self.set_status(format!("Restored annotations of {n} images from backup"));
        }
    }

    fn import_coco(&mut self, path: &Path) {
        let Some(dir) = self.image_dir.clone() else {
            self.set_status("Open an image folder before importing COCO");
            return;
        };
        let known: HashSet<&str> = self.images.iter().map(String::as_str).collect();
        match model::import_coco(path, &self.classes, &dir, &known) {
            Ok(imp) => {
                if self.classes.is_empty() {
                    self.classes = imp.categories.clone();
                    self.cur_class = 0;
                }
                let n = imp.images.len();
                let nb: usize = imp.images.values().map(Vec::len).sum();
                for (name, boxes) in imp.images {
                    let ann = self.ensure_ann(&name);
                    ann.snapshot();
                    ann.boxes = boxes;
                    ann.mark_exported();
                }
                self.selected = None;
                self.mark_changed();
                self.compute_visible();
                self.update_prefetch();
                let mut s = format!("Imported {nb} boxes for {n} images from {}", path.display());
                if imp.skipped_unknown_class > 0 {
                    s += &format!(
                        "; skipped {} boxes with unknown class",
                        imp.skipped_unknown_class
                    );
                }
                if imp.skipped_unknown_image > 0 {
                    s += &format!("; {} images not in folder", imp.skipped_unknown_image);
                }
                self.set_status(s);
            }
            Err(e) => self.set_status(format!("COCO import failed: {e:#}")),
        }
    }

    /// Exports every image with annotation data (including empty ones).
    fn export_coco(&mut self, path: &Path) {
        let mut entries = Vec::with_capacity(self.anns.len());
        for name in self.images.iter().filter(|n| self.anns.contains_key(*n)) {
            let Some([w, h]) = self.image_size(name) else {
                self.set_status(format!("Could not read size of {name}"));
                return;
            };
            entries.push((name.clone(), w, h, self.anns[name].boxes.clone()));
        }
        let json = model::export_coco(&self.classes, &entries);
        match model::write_atomic(path, json.to_string().as_bytes()) {
            Ok(()) => {
                self.anns.values_mut().for_each(ImageAnn::mark_exported);
                self.backup_gen += 1;
                self.set_status(format!(
                    "Exported {} images to {}",
                    self.anns.len(),
                    path.display()
                ));
            }
            Err(e) => self.set_status(format!("COCO export failed: {e:#}")),
        }
    }

    /// Load `<stem>.txt` files from `dir` into memory. The first imported folder
    /// becomes the labels folder (saved back there); a folder imported on top of
    /// another one is treated like any other unsaved edit.
    fn import_yolo(&mut self, dir: &Path, auto: bool) {
        let (found, errors) = match model::read_yolo_dir(dir, &self.images) {
            Ok(v) => v,
            Err(e) => {
                self.set_status(format!("YOLO import failed: {e:#}"));
                return;
            }
        };
        let is_target = self.labels_dir.is_none() || self.labels_dir.as_deref() == Some(dir);
        if is_target {
            self.labels_dir = Some(dir.to_path_buf());
            self.labels_auto = auto;
            self.settings.last_labels_dir = (!auto).then(|| dir.to_path_buf());
            self.settings.save();
        }
        if self.classes.is_empty()
            && let Some(c) = model::find_classes_file(dir)
        {
            self.load_classes(&c);
        }
        let n = found.len();
        let nb: usize = found.values().map(Vec::len).sum();
        for (name, boxes) in found {
            match self.anns.get_mut(&name) {
                Some(ann) => {
                    ann.snapshot();
                    ann.boxes = boxes;
                    ann.mark_exported();
                    ann.dirty = !is_target;
                }
                None => {
                    let mut ann = ImageAnn::clean(boxes);
                    ann.dirty = !is_target;
                    self.anns.insert(name, ann);
                }
            }
        }
        if !is_target && n > 0 {
            self.mark_changed();
        }
        self.selected = None;
        self.compute_visible();
        self.update_prefetch();
        let mut s = format!("Imported {nb} boxes for {n} images from {}", dir.display());
        if errors > 0 {
            s += &format!(" ({errors} unreadable files)");
        }
        self.set_status(s);
    }

    /// Write every image with annotation data to `dir`, which becomes the labels folder.
    fn export_yolo(&mut self, dir: &Path) {
        let names: Vec<String> = self.anns.keys().cloned().collect();
        let mut failed = 0;
        for name in &names {
            let ann = &self.anns[name];
            match model::write_atomic(
                &model::label_path(dir, name),
                model::format_yolo(&ann.boxes).as_bytes(),
            ) {
                Ok(()) => {
                    let a = self.anns.get_mut(name).unwrap();
                    a.dirty = false;
                    a.mark_exported();
                }
                Err(e) => {
                    failed += 1;
                    self.status = format!("Save failed: {e:#}");
                }
            }
        }
        self.labels_dir = Some(dir.to_path_buf());
        self.labels_auto = false;
        self.settings.last_labels_dir = Some(dir.to_path_buf());
        self.settings.save();
        self.backup_gen += 1;
        if failed == 0 {
            self.write_classes_if_missing();
            self.set_status(format!(
                "Exported {} label files to {}",
                names.len(),
                dir.display()
            ));
        }
        self.write_backup(true);
    }

    fn pick_labels_dir(&self, title: &str) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title(title)
            .set_directory(self.labels_dir.clone().unwrap_or_else(|| self.start_dir()))
            .pick_folder()
    }

    fn start_crop(&mut self, out_dir: PathBuf) {
        let Some(dir) = self.image_dir.clone() else {
            return;
        };
        let jobs: Vec<(PathBuf, String, Vec<BBox>)> = self
            .images
            .iter()
            .filter_map(|n| Some((dir.join(n), n.clone(), self.anns.get(n)?.boxes.clone())))
            .filter(|(_, _, b)| !b.is_empty())
            .collect();
        let classes = self.classes.clone();
        let done = Arc::new(AtomicUsize::new(0));
        let total = Arc::new(AtomicUsize::new(jobs.len()));
        let finished = Arc::new(Mutex::new(None));
        let (d, f) = (done.clone(), finished.clone());
        std::thread::spawn(move || {
            let mut crops = 0;
            let mut errors = 0;
            for (path, name, boxes) in jobs {
                match image::open(&path) {
                    Ok(img) => {
                        let (w, h) = (img.width(), img.height());
                        let stem = Path::new(&name)
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        let ext = Path::new(&name)
                            .extension()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        for (i, b) in boxes.iter().enumerate() {
                            let b = b.clamped();
                            let x = (b.x0 * w as f64).round() as u32;
                            let y = (b.y0 * h as f64).round() as u32;
                            let cw = ((b.x1 * w as f64).round() as u32)
                                .saturating_sub(x)
                                .max(1)
                                .min(w - x.min(w - 1));
                            let ch = ((b.y1 * h as f64).round() as u32)
                                .saturating_sub(y)
                                .max(1)
                                .min(h - y.min(h - 1));
                            let cls = classes
                                .get(b.class)
                                .cloned()
                                .unwrap_or_else(|| b.class.to_string());
                            let out = out_dir.join(format!("{stem}-{cls}-{i}.{ext}"));
                            if img
                                .crop_imm(x.min(w - 1), y.min(h - 1), cw, ch)
                                .save(&out)
                                .is_ok()
                            {
                                crops += 1;
                            } else {
                                errors += 1;
                            }
                        }
                    }
                    Err(_) => errors += 1,
                }
                d.fetch_add(1, Ordering::Relaxed);
            }
            *f.lock().unwrap() = Some(format!(
                "Saved {crops} crops to {} ({errors} errors)",
                out_dir.display()
            ));
        });
        self.crop_job = Some(CropJob {
            done,
            total,
            finished,
        });
    }

    // ------------------------------------------------------------ keyboard

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.text_edit_focused() || self.show_quit_dialog {
            return;
        }
        let (right, left, up, down, del, esc, undo, redo, save, open, fit, digit) =
            ctx.input_mut(|i| {
                let mut del = i.count_and_consume_key(Modifiers::NONE, Key::Delete)
                    + i.count_and_consume_key(Modifiers::NONE, Key::Backspace)
                    + i.count_and_consume_key(Modifiers::COMMAND, Key::Backspace);
                del += i.count_and_consume_key(Modifiers::NONE, Key::X);
                let redo = i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
                    || i.consume_key(Modifiers::COMMAND, Key::Y);
                let digit = (1..=9).find(|d| {
                    let key = Key::from_name(&d.to_string()).unwrap();
                    i.consume_key(Modifiers::NONE, key)
                });
                (
                    i.count_and_consume_key(Modifiers::NONE, Key::ArrowRight)
                        + i.count_and_consume_key(Modifiers::NONE, Key::D),
                    i.count_and_consume_key(Modifiers::NONE, Key::ArrowLeft)
                        + i.count_and_consume_key(Modifiers::NONE, Key::A),
                    i.count_and_consume_key(Modifiers::NONE, Key::ArrowUp)
                        + i.count_and_consume_key(Modifiers::NONE, Key::W),
                    i.count_and_consume_key(Modifiers::NONE, Key::ArrowDown)
                        + i.count_and_consume_key(Modifiers::NONE, Key::S),
                    del,
                    i.consume_key(Modifiers::NONE, Key::Escape),
                    i.consume_key(Modifiers::COMMAND, Key::Z),
                    redo,
                    i.consume_key(Modifiers::COMMAND, Key::S),
                    i.consume_key(Modifiers::COMMAND, Key::O),
                    i.consume_key(Modifiers::NONE, Key::F)
                        || i.consume_key(Modifiers::NONE, Key::Home),
                    digit,
                )
            });
        let nav = right as isize - left as isize;
        if nav != 0 {
            self.navigate(nav);
        }
        let cls = down as isize - up as isize;
        if cls != 0 {
            self.cycle_class(cls);
        }
        if let Some(d) = digit
            && d <= self.classes.len()
        {
            self.cur_class = d - 1;
            self.selected = None;
        }
        if del > 0 {
            self.delete_selected();
        }
        if esc {
            self.drag = Drag::None;
            self.selected = None;
        }
        if (undo || redo)
            && let Some(name) = self.cur_name().map(String::from)
        {
            let Some(ann) = self.anns.get_mut(&name) else {
                return;
            };
            let changed = if redo { ann.redo() } else { ann.undo() };
            if changed {
                self.selected = None;
                self.drag = Drag::None;
                self.mark_changed();
            }
        }
        if save {
            self.save_yolo();
        }
        if open {
            self.dialog_open_dir();
        }
        if fit {
            self.zoom = 1.0;
            self.pan = Vec2::ZERO;
        }
    }

    fn delete_selected(&mut self) {
        let (Some(sel), Some(name)) = (self.selected, self.cur_name().map(String::from)) else {
            return;
        };
        let Some(ann) = self.anns.get_mut(&name) else {
            return;
        };
        if sel < ann.boxes.len() {
            ann.snapshot();
            ann.boxes.remove(sel);
            self.mark_changed();
        }
        self.selected = None;
        self.drag = Drag::None;
    }

    // -------------------------------------------------------------- dialogs

    fn dialog_open_dir(&mut self) {
        let mut d = rfd::FileDialog::new().set_title("Open image folder");
        if let Some(p) = &self.image_dir {
            d = d.set_directory(p);
        }
        if let Some(dir) = d.pick_folder() {
            self.open_dir(&dir, None, None);
        }
    }

    fn start_dir(&self) -> PathBuf {
        self.image_dir
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
    }

    // ------------------------------------------------------------------- UI

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📂 Open image folder")
                .on_hover_text("Ctrl+O")
                .clicked()
            {
                self.dialog_open_dir();
            }
            ui.separator();
            if ui
                .button("Import YOLO")
                .on_hover_text(
                    "Load .txt label files from a folder. The first imported folder is \
                     also where Ctrl+S saves.",
                )
                .clicked()
                && self.image_dir.is_some()
                && let Some(p) = self.pick_labels_dir("Import YOLO labels from folder")
            {
                self.import_yolo(&p, false);
            }
            if ui
                .button("YOLO classes")
                .on_hover_text("Load classes.txt / *.names")
                .clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .set_directory(self.labels_dir.clone().unwrap_or_else(|| self.start_dir()))
                    .add_filter("classes", &["txt", "names", "csv"])
                    .pick_file()
            {
                self.load_classes(&p);
            }
            let export_label = if self.has_unsaved() {
                "💾 Export YOLO*"
            } else {
                "💾 Export YOLO"
            };
            ui.menu_button(export_label, |ui| {
                let update = match &self.labels_dir {
                    Some(d) => format!("Update {} (Ctrl+S)", d.display()),
                    None => "Update labels folder (Ctrl+S)".into(),
                };
                if ui
                    .add_enabled(self.labels_dir.is_some(), egui::Button::new(update))
                    .on_hover_text("Write only the changed label files")
                    .on_disabled_hover_text("No labels folder yet: import or export one first")
                    .clicked()
                {
                    ui.close();
                    self.save_yolo();
                }
                if ui
                    .button("Export to another folder")
                    .on_hover_text(
                        "Write label files of all images with annotation data to a new or \
                         existing folder; it becomes the labels folder",
                    )
                    .clicked()
                {
                    ui.close();
                    if let Some(d) = self.pick_labels_dir("Export YOLO labels to folder") {
                        self.export_yolo(&d);
                    }
                }
            });
            if ui
                .checkbox(&mut self.settings.autosave_labels, "Autosave")
                .on_hover_text(
                    "Write the label file to the labels folder whenever you leave an image",
                )
                .changed()
            {
                self.settings.save();
            }
            ui.separator();
            if ui.button("Import COCO").clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .set_directory(
                        self.image_dir
                            .as_ref()
                            .and_then(|d| d.parent())
                            .unwrap_or(Path::new(".")),
                    )
                    .add_filter("COCO json", &["json"])
                    .pick_file()
            {
                self.import_coco(&p);
            }
            if ui
                .button("Export COCO")
                .on_hover_text("Export all images with annotation data")
                .clicked()
            {
                let default_name = self
                    .image_dir
                    .as_ref()
                    .and_then(|d| d.file_name())
                    .map(|n| format!("{}_HUMAN_FIXED.json", n.to_string_lossy()))
                    .unwrap_or_else(|| "bboxes_coco.json".into());
                if let Some(p) = rfd::FileDialog::new()
                    .set_directory(
                        self.image_dir
                            .as_ref()
                            .and_then(|d| d.parent())
                            .unwrap_or(Path::new(".")),
                    )
                    .set_file_name(default_name)
                    .add_filter("COCO json", &["json"])
                    .save_file()
                {
                    self.export_coco(&p);
                }
            }
            ui.separator();
            let cropping = self.crop_job.is_some();
            if ui
                .add_enabled(!cropping, egui::Button::new("Crop & Save"))
                .clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .set_title("Folder for crops")
                    .set_directory(self.start_dir())
                    .pick_folder()
            {
                self.start_crop(p);
            }
            ui.separator();
            if ui
                .button("Restore backup")
                .on_hover_text("Re-load the automatic backup of this folder")
                .clicked()
                && let Some(bp) = &self.backup_path
            {
                let b = backup::load_backup(bp)
                    .filter(|b| !b.images.is_empty())
                    .or_else(|| backup::load_backup(&bp.with_extension("prev.json")));
                match b {
                    Some(b) => self.pending_restore = Some(b),
                    None => self.set_status("No backup found for this folder"),
                }
            }
        });
    }

    fn restore_banner(&mut self, ui: &mut egui::Ui) {
        let Some(b) = &self.pending_restore else {
            return;
        };
        let n = b.images.len();
        let when = backup::ago(b.saved_at);
        let mut action = 0;
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "⚠ Backup from {when} has unsaved annotations for {n} images."
                ))
                .color(Color32::from_rgb(255, 200, 80)),
            );
            if ui.button("Restore").clicked() {
                action = 1;
            }
            if ui
                .button("Ignore")
                .on_hover_text("Keep the backup file, don't load it")
                .clicked()
            {
                action = 2;
            }
            if ui
                .button("Delete backup")
                .on_hover_text("Moves it to *.prev.json")
                .clicked()
            {
                action = 3;
            }
        });
        match action {
            1 => {
                let b = self.pending_restore.take().unwrap();
                self.restore_backup(b);
                self.write_backup(true);
            }
            2 => self.pending_restore = None,
            3 => {
                self.pending_restore = None;
                if let Some(bp) = &self.backup_path {
                    let _ = std::fs::rename(bp, bp.with_extension("prev.json"));
                }
            }
            _ => {}
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Search:");
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("filter file names")
                    .desired_width(f32::INFINITY),
            );
            if r.changed() {
                self.apply_filters();
            }
            if r.lost_focus() || ui.input(|i| i.key_pressed(Key::Escape)) && r.has_focus() {
                r.surrender_focus();
            }
        });
        if let Some(dir) = &self.image_dir {
            ui.label(RichText::new(dir.display().to_string()).small().weak());
        }
        let pos = self.visible_pos().map(|p| p + 1).unwrap_or(0);
        ui.label(format!(
            "{pos} / {} shown ({} total)",
            self.visible.len(),
            self.images.len()
        ));

        // Virtualized image list: only visible rows are laid out.
        let row_h = ui.text_style_height(&egui::TextStyle::Body);
        let list_h = (ui.available_height() * 0.45).max(120.0);
        let spacing = ui.spacing().item_spacing.y;
        let mut area = egui::ScrollArea::vertical()
            .id_salt("image_list")
            .max_height(list_h)
            .auto_shrink([false, false]);
        if self.scroll_to_current {
            self.scroll_to_current = false;
            if let Some(p) = self.visible_pos() {
                let id = ui.make_persistent_id("image_list_view");
                let (off, h) = ui
                    .data(|d| d.get_temp::<(f32, f32)>(id))
                    .unwrap_or((0.0, list_h));
                let y = p as f32 * (row_h + spacing);
                if y < off || y + row_h > off + h {
                    area = area.vertical_scroll_offset((y - h / 2.0).max(0.0));
                }
            }
        }
        let mut clicked = None;
        let out = area.show_rows(ui, row_h, self.visible.len(), |ui, range| {
            for vi in range {
                let i = self.visible[vi];
                let name = &self.images[i];
                let mut text = String::new();
                match self.anns.get(name) {
                    Some(a) if a.modified() => text.push_str(&format!("*{} ", a.boxes.len())),
                    Some(a) => text.push_str(&format!("[{}] ", a.boxes.len())),
                    None => {}
                }
                text.push_str(name);
                let mut rt = RichText::new(text);
                if self.anns.get(name).is_some_and(ImageAnn::modified) {
                    rt = rt.color(Color32::from_rgb(255, 200, 80));
                }
                if ui
                    .add(egui::Button::selectable(Some(i) == self.cur, rt).truncate())
                    .clicked()
                {
                    clicked = Some(i);
                }
            }
        });
        let id = ui.make_persistent_id("image_list_view");
        ui.data_mut(|d| d.insert_temp(id, (out.state.offset.y, out.inner_rect.height())));
        let filter_text = format!("Filter to annotation data ({})", self.data_count());
        if ui
            .checkbox(&mut self.filter_data, filter_text)
            .on_hover_text(
                "Show only images that have annotations (also empty ones) from YOLO/COCO \
                 import, editing or visiting",
            )
            .changed()
        {
            self.apply_filters();
        }
        if let Some(i) = clicked {
            self.go_to(i, true);
            self.scroll_to_current = false;
        }

        // Current image information.
        if let Some(name) = self.cur_name().map(String::from) {
            let path = self.image_dir.as_ref().unwrap().join(&name);
            let nboxes = self.anns.get(&name).map(|a| a.boxes.len()).unwrap_or(0);
            let info = match self.cache.get(&path) {
                Some(Slot::Ready {
                    size, file_size, ..
                }) => {
                    format!(
                        "{}x{}, {}, {nboxes} boxes",
                        size[0],
                        size[1],
                        format_bytes(*file_size)
                    )
                }
                Some(Slot::Failed(e)) => format!("Error: {e}"),
                None => "loading…".into(),
            };
            ui.label(RichText::new(info).strong());
            let labels = match &self.labels_dir {
                Some(ld) if self.labels_auto => {
                    format!("labels: {} (autoselected)", ld.display())
                }
                Some(ld) => format!("labels: {}", ld.display()),
                None => "labels: none (Import YOLO or Export YOLO)".into(),
            };
            ui.label(RichText::new(labels).small().weak());
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label(RichText::new("Classes").strong());
            ui.label(RichText::new("(Up/Down, 1-9)").weak());
        });
        if self.classes.is_empty() {
            ui.label(
                RichText::new("No classes loaded — use \"YOLO classes\"").color(Color32::LIGHT_RED),
            );
        }
        let class_h = (ui.available_height() * 0.4).max(80.0);
        egui::ScrollArea::vertical()
            .id_salt("class_list")
            .max_height(class_h)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for (i, c) in self.classes.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let (r, _) = ui.allocate_exact_size(vec2(12.0, 12.0), Sense::hover());
                        ui.painter().rect_filled(r, 2.0, class_color(i));
                        let label = if i < 9 {
                            format!("{} {c}", i + 1)
                        } else {
                            format!("  {c}")
                        };
                        if ui.selectable_label(i == self.cur_class, label).clicked() {
                            self.cur_class = i;
                            self.selected = None;
                        }
                    });
                }
            });

        // Selected box information.
        ui.separator();
        self.bbox_info(ui);

        ui.separator();
        egui::CollapsingHeader::new("Shortcuts")
            .default_open(false)
            .show(ui, |ui| {
                for (k, v) in [
                    ("Left / Right (A / D)", "previous / next image"),
                    ("Up / Down (W / S)", "previous / next class"),
                    ("1-9", "select class"),
                    ("Delete / Backspace / X", "remove selected box"),
                    ("Left drag", "draw box (Shift: always draw)"),
                    ("Left click", "select box (click again to cycle overlaps)"),
                    ("Drag selected", "move; drag edges/corners to resize"),
                    ("Mouse wheel", "zoom"),
                    ("Middle / right drag", "pan"),
                    ("F / Home", "reset zoom"),
                    ("Esc", "deselect / cancel"),
                    ("Ctrl+Z / Ctrl+Y", "undo / redo (per image)"),
                    ("Ctrl+S", "update YOLO labels folder"),
                    ("Ctrl+O", "open image folder"),
                ] {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(k).monospace().strong());
                        ui.label(v);
                    });
                }
            });
    }

    fn bbox_info(&mut self, ui: &mut egui::Ui) {
        let Some(name) = self.cur_name().map(String::from) else {
            return;
        };
        let size = self
            .image_dir
            .as_ref()
            .and_then(|d| match self.cache.get(&d.join(&name)) {
                Some(Slot::Ready { size, .. }) => Some(*size),
                _ => None,
            });
        let preview = self.drag_preview().map(|(_, b)| b);
        let sel = self
            .selected
            .and_then(|s| self.anns.get(&name)?.boxes.get(s).copied());
        let Some(b) = preview.or(sel) else {
            ui.label(RichText::new("No box selected").weak());
            return;
        };
        if let Some([w, h]) = size {
            let (x, y, bw, bh) = b.normalized().to_pixels(w, h);
            ui.label(
                RichText::new(format!(
                    "{}x{} ({}, {}) ({}, {})",
                    bw.floor(),
                    bh.floor(),
                    x.floor(),
                    y.floor(),
                    (x + bw).floor(),
                    (y + bh).floor()
                ))
                .monospace(),
            );
        }
        let Some(sel_idx) = self.selected else { return };
        let mut class = b.class;
        let label = |c: usize| {
            self.classes
                .get(c)
                .cloned()
                .unwrap_or_else(|| format!("#{c}"))
        };
        let selected_text = label(class);
        let mut changed = false;
        egui::ComboBox::from_label("class of box")
            .selected_text(selected_text)
            .show_ui(ui, |ui| {
                for i in 0..self.classes.len() {
                    changed |= ui.selectable_value(&mut class, i, label(i)).changed();
                }
            });
        if changed {
            let ann = self.ensure_ann(&name);
            ann.snapshot();
            ann.boxes[sel_idx].class = class;
            self.mark_changed();
        }
    }

    fn drag_preview(&self) -> Option<(Option<usize>, BBox)> {
        match self.drag {
            Drag::None => None,
            Drag::Draw { start, cur } => Some((
                None,
                BBox {
                    class: self.cur_class,
                    x0: start.0,
                    y0: start.1,
                    x1: cur.0,
                    y1: cur.1,
                }
                .normalized(),
            )),
            Drag::Move { idx, cur, .. } => Some((Some(idx), cur)),
            Drag::Resize { idx, cur, .. } => Some((Some(idx), cur.normalized())),
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(24));

        let Some(cur) = self.cur else {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Open an image folder (Ctrl+O)",
                FontId::proportional(20.0),
                Color32::GRAY,
            );
            return;
        };
        let name = self.images[cur].clone();
        let path = self.image_path(cur).unwrap();
        self.tick += 1;
        let tick = self.tick;
        let (tex_id, size) = match self.cache.get_mut(&path) {
            Some(Slot::Ready {
                tex,
                size,
                last_used,
                ..
            }) => {
                *last_used = tick;
                (tex.id(), *size)
            }
            Some(Slot::Failed(e)) => {
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    format!("Failed to load {name}:\n{e}"),
                    FontId::proportional(16.0),
                    Color32::LIGHT_RED,
                );
                return;
            }
            None => {
                if !self.loader.is_busy(&path) {
                    self.update_prefetch();
                }
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    format!("Loading {name}…"),
                    FontId::proportional(18.0),
                    Color32::GRAY,
                );
                return;
            }
        };

        let img_px = vec2(size[0] as f32, size[1] as f32);
        let fit = (rect.width() / img_px.x).min(rect.height() / img_px.y);

        // Zoom around the cursor.
        if let Some(p) = resp.hover_pos() {
            let (scroll, zd) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = if zd != 1.0 {
                zd
            } else {
                1.002f32.powf(scroll * 6.0)
            };
            if factor != 1.0 {
                let s = fit * self.zoom;
                let min = rect.center() + self.pan - img_px * s / 2.0;
                let u = (p - min) / s;
                self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
                let s2 = fit * self.zoom;
                self.pan = p.to_vec2() - u * s2 - rect.center().to_vec2() + img_px * s2 / 2.0;
            }
        }
        if resp.dragged_by(PointerButton::Middle) || resp.dragged_by(PointerButton::Secondary) {
            self.pan += resp.drag_delta();
        }

        let scale = fit * self.zoom;
        let img_rect = Rect::from_min_size(
            rect.center() + self.pan - img_px * scale / 2.0,
            img_px * scale,
        );
        self.last_img_rect = Some(img_rect);
        painter.image(
            tex_id,
            img_rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );

        let to_norm = |p: Pos2| -> (f64, f64) {
            (
                ((p.x - img_rect.min.x) / img_rect.width()) as f64,
                ((p.y - img_rect.min.y) / img_rect.height()) as f64,
            )
        };
        let to_screen = |b: &BBox| -> Rect {
            Rect::from_min_max(
                pos2(
                    img_rect.min.x + b.x0 as f32 * img_rect.width(),
                    img_rect.min.y + b.y0 as f32 * img_rect.height(),
                ),
                pos2(
                    img_rect.min.x + b.x1 as f32 * img_rect.width(),
                    img_rect.min.y + b.y1 as f32 * img_rect.height(),
                ),
            )
        };
        let clamp01 = |(x, y): (f64, f64)| (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));

        let pointer = resp.hover_pos().or(resp.interact_pointer_pos());
        self.hover_px = pointer.map(|p| {
            let (x, y) = to_norm(p);
            (x * size[0] as f64, y * size[1] as f64)
        });

        let boxes = self
            .anns
            .get(&name)
            .map(|a| a.boxes.clone())
            .unwrap_or_default();
        if self.selected.is_some_and(|s| s >= boxes.len()) {
            self.selected = None;
        }

        // ---- left-button interaction
        let shift = ui.input(|i| i.modifiers.shift);
        if resp.drag_started_by(PointerButton::Primary) {
            let origin = ui
                .input(|i| i.pointer.press_origin())
                .unwrap_or(rect.center());
            let n = to_norm(origin);
            let sel = self.selected.map(|s| (s, boxes[s]));
            let handle = sel
                .filter(|_| !shift)
                .map(|(_, b)| hit_handle(to_screen(&b), origin))
                .unwrap_or(0);
            self.drag = if let (Some((idx, b)), true) = (sel, handle != 0) {
                Drag::Resize {
                    idx,
                    handle,
                    cur: b,
                }
            } else if let Some((idx, b)) =
                sel.filter(|(_, b)| !shift && to_screen(b).expand(2.0).contains(origin))
            {
                Drag::Move {
                    idx,
                    start: n,
                    orig: b,
                    cur: b,
                }
            } else if !self.classes.is_empty() {
                self.selected = None;
                let s = clamp01(n);
                Drag::Draw { start: s, cur: s }
            } else {
                self.set_status("Load classes before drawing boxes");
                Drag::None
            };
        }
        if let (true, Some(p)) = (
            resp.dragged_by(PointerButton::Primary),
            resp.interact_pointer_pos(),
        ) {
            let m = to_norm(p);
            match &mut self.drag {
                Drag::Draw { cur, .. } => *cur = clamp01(m),
                Drag::Move {
                    start, orig, cur, ..
                } => {
                    let dx = (m.0 - start.0).clamp(-orig.x0, 1.0 - orig.x1);
                    let dy = (m.1 - start.1).clamp(-orig.y0, 1.0 - orig.y1);
                    *cur = BBox {
                        x0: orig.x0 + dx,
                        x1: orig.x1 + dx,
                        y0: orig.y0 + dy,
                        y1: orig.y1 + dy,
                        ..*orig
                    };
                }
                Drag::Resize { handle, cur, .. } => {
                    let (mx, my) = clamp01(m);
                    if *handle & H_LEFT != 0 {
                        cur.x0 = mx;
                    }
                    if *handle & H_RIGHT != 0 {
                        cur.x1 = mx;
                    }
                    if *handle & H_TOP != 0 {
                        cur.y0 = my;
                    }
                    if *handle & H_BOTTOM != 0 {
                        cur.y1 = my;
                    }
                }
                Drag::None => {}
            }
        }
        if resp.drag_stopped_by(PointerButton::Primary) {
            if let Some((idx, b)) = self.drag_preview() {
                let r = to_screen(&b);
                let new_box_ok = r.width() >= MIN_BOX_SCREEN_PX && r.height() >= MIN_BOX_SCREEN_PX;
                if idx.is_none() && new_box_ok && !self.anns.contains_key(&name) {
                    self.ensure_ann(&name);
                    self.compute_visible();
                }
                let ann = self.anns.get_mut(&name);
                match (idx, ann) {
                    (None, Some(ann)) if new_box_ok => {
                        ann.snapshot();
                        ann.boxes.push(b);
                        self.selected = Some(ann.boxes.len() - 1);
                        self.mark_changed();
                    }
                    (Some(i), Some(ann))
                        if ann.boxes[i] != b && r.width() >= 1.0 && r.height() >= 1.0 =>
                    {
                        ann.snapshot();
                        ann.boxes[i] = b;
                        self.mark_changed();
                    }
                    _ => {}
                }
            }
            self.drag = Drag::None;
        }
        if resp.clicked_by(PointerButton::Primary)
            && let Some(p) = resp.interact_pointer_pos()
        {
            let (x, y) = to_norm(p);
            // Smallest box first; clicking again cycles to the next overlapping one.
            let mut hits: Vec<usize> = (0..boxes.len())
                .filter(|&i| boxes[i].normalized().contains(x, y))
                .collect();
            hits.sort_by(|a, b| boxes[*a].area().total_cmp(&boxes[*b].area()));
            self.selected = match self
                .selected
                .and_then(|s| hits.iter().position(|h| *h == s))
            {
                Some(pos) => Some(hits[(pos + 1) % hits.len()]),
                None => hits.first().copied(),
            };
        }

        // Interactions above may have changed the boxes.
        let boxes = self
            .anns
            .get(&name)
            .map(|a| a.boxes.clone())
            .unwrap_or_default();

        // ---- cursor
        if let Some(p) = resp.hover_pos() {
            let icon = match (self.drag, self.selected) {
                (Drag::Move { .. }, _) => CursorIcon::Grabbing,
                (Drag::Resize { handle, .. }, _) => handle_cursor(handle),
                (Drag::None, Some(s)) if !shift => {
                    let r = to_screen(&boxes[s]);
                    match hit_handle(r, p) {
                        0 if r.contains(p) => CursorIcon::Grab,
                        0 => CursorIcon::Crosshair,
                        h => handle_cursor(h),
                    }
                }
                _ => CursorIcon::Crosshair,
            };
            ui.ctx().set_cursor_icon(icon);
        }

        // ---- draw boxes
        let preview = self.drag_preview();
        let font = FontId::proportional(13.0);
        for (i, b) in boxes.iter().enumerate() {
            let (b, selected) = match preview {
                Some((Some(pi), pb)) if pi == i => (pb, true),
                _ => (*b, self.selected == Some(i)),
            };
            self.paint_box(
                &painter,
                to_screen(&b.normalized()),
                b.class,
                selected,
                &font,
            );
        }
        if let Some((None, b)) = preview {
            self.paint_box(&painter, to_screen(&b), b.class, true, &font);
        }

        // ---- cursor guidelines
        if let Some(p) = resp.hover_pos() {
            let stroke = Stroke::new(1.0, Color32::from_white_alpha(170));
            painter.extend(Shape::dashed_line(
                &[pos2(rect.left(), p.y), pos2(rect.right(), p.y)],
                stroke,
                5.0,
                5.0,
            ));
            painter.extend(Shape::dashed_line(
                &[pos2(p.x, rect.top()), pos2(p.x, rect.bottom())],
                stroke,
                5.0,
                5.0,
            ));
            if let Some(cls) = self.classes.get(self.cur_class)
                && matches!(self.drag, Drag::None)
            {
                painter.text(
                    p + vec2(12.0, 12.0),
                    Align2::LEFT_TOP,
                    cls,
                    font.clone(),
                    class_color(self.cur_class),
                );
            }
        }
    }

    fn paint_box(
        &self,
        painter: &egui::Painter,
        r: Rect,
        class: usize,
        selected: bool,
        font: &FontId,
    ) {
        let color = class_color(class);
        let (fill_a, width) = if selected { (70, 2.0) } else { (28, 1.5) };
        painter.rect_filled(r, 0.0, color.gamma_multiply_u8(fill_a));
        painter.rect_stroke(
            r,
            0.0,
            Stroke::new(width, if selected { Color32::WHITE } else { color }),
            StrokeKind::Middle,
        );
        if selected {
            painter.rect_stroke(
                r.expand(width),
                0.0,
                Stroke::new(1.0, Color32::BLACK),
                StrokeKind::Middle,
            );
        }
        // Center cross with 10% padding on each side, like the original tool.
        let c = r.center();
        let cross = Stroke::new(1.0, color.gamma_multiply(0.6));
        painter.line_segment(
            [
                pos2(c.x, r.top() + r.height() * 0.1),
                pos2(c.x, r.bottom() - r.height() * 0.1),
            ],
            cross,
        );
        painter.line_segment(
            [
                pos2(r.left() + r.width() * 0.1, c.y),
                pos2(r.right() - r.width() * 0.1, c.y),
            ],
            cross,
        );

        let label = self
            .classes
            .get(class)
            .cloned()
            .unwrap_or_else(|| format!("#{class}"));
        let galley = painter.layout_no_wrap(label, font.clone(), Color32::BLACK);
        let tr = Rect::from_min_size(
            pos2(r.left(), r.top() - galley.size().y - 2.0),
            galley.size() + vec2(6.0, 2.0),
        );
        painter.rect_filled(tr, 2.0, if selected { Color32::WHITE } else { color });
        painter.galley(tr.min + vec2(3.0, 1.0), galley, Color32::BLACK);

        if selected {
            for p in [
                r.left_top(),
                r.right_top(),
                r.left_bottom(),
                r.right_bottom(),
                r.center_top(),
                r.center_bottom(),
                r.left_center(),
                r.right_center(),
            ] {
                let h = Rect::from_center_size(p, Vec2::splat(7.0));
                painter.rect_filled(h, 0.0, Color32::WHITE);
                painter.rect_stroke(h, 0.0, Stroke::new(1.0, Color32::BLACK), StrokeKind::Inside);
            }
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(job) = &self.crop_job {
                let (d, t) = (
                    job.done.load(Ordering::Relaxed),
                    job.total.load(Ordering::Relaxed),
                );
                ui.add(
                    egui::ProgressBar::new(d as f32 / t.max(1) as f32)
                        .desired_width(160.0)
                        .text(format!("crop {d}/{t}")),
                );
                let finished = job.finished.lock().unwrap().take();
                match finished {
                    Some(msg) => {
                        self.status = msg;
                        self.crop_job = None;
                    }
                    None => ui.ctx().request_repaint_after(Duration::from_millis(100)),
                }
            }
            ui.label(&self.status);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(t) = self.backup_saved_at {
                    let tip = self
                        .backup_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default();
                    ui.label(RichText::new(format!("backup {}", backup::ago(t))).weak())
                        .on_hover_text(tip);
                }
                let ready = self
                    .cache
                    .values()
                    .filter(|s| matches!(s, Slot::Ready { .. }))
                    .count();
                ui.label(RichText::new(format!("cache {ready}")).weak());
                ui.label(RichText::new(format!("zoom {:.0}%", self.zoom * 100.0)).weak());
                if let Some((x, y)) = self.hover_px {
                    ui.label(
                        RichText::new(format!("{:.0}, {:.0}", x, y))
                            .monospace()
                            .weak(),
                    );
                }
            });
        });
    }

    fn quit_dialog(&mut self, ctx: &egui::Context) {
        if !self.show_quit_dialog {
            return;
        }
        let n = self.unsaved_count();
        let mut action = 0;
        egui::Modal::new(egui::Id::new("quit_dialog")).show(ctx, |ui| {
            ui.heading("Unsaved annotations");
            ui.label(format!(
                "{n} images have box edits that are not saved to YOLO label files or exported to COCO."
            ));
            if let Some(bp) = &self.backup_path {
                ui.label(
                    RichText::new(format!("They are kept in the backup:\n{}", bp.display()))
                        .small(),
                );
            }
            ui.horizontal(|ui| {
                if ui.button("Export YOLO & quit").clicked() {
                    action = 1;
                }
                if ui.button("Quit").clicked() {
                    action = 2;
                }
                if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(Key::Escape)) {
                    action = 3;
                }
            });
        });
        match action {
            1 => {
                self.save_yolo();
                if !self.has_unsaved() {
                    self.allow_close = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                self.show_quit_dialog = false;
            }
            2 => {
                self.allow_close = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            3 => self.show_quit_dialog = false,
            _ => {}
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_loader(&ctx);

        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close {
            if self.settings.autosave_labels && self.labels_dir.is_some() {
                self.save_yolo();
            }
            self.write_backup(true);
            if self.has_unsaved() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.show_quit_dialog = true;
            }
        }

        self.handle_keys(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(2.0);
            self.toolbar(ui);
            self.restore_banner(ui);
            ui.add_space(2.0);
        });
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        egui::Panel::left("side")
            .resizable(true)
            .default_size(320.0)
            .show(ui, |ui| self.side_panel(ui));
        egui::CentralPanel::no_frame().show(ui, |ui| self.canvas(ui));
        self.quit_dialog(&ctx);

        // Debounced automatic backup.
        if self.backup_gen != self.backup_written_gen {
            let elapsed = self.backup_last_write.elapsed();
            if elapsed >= BACKUP_DEBOUNCE {
                self.write_backup(true);
            } else {
                ctx.request_repaint_after(BACKUP_DEBOUNCE - elapsed);
            }
        }

        let dirty_mark = if self.has_unsaved() { " *" } else { "" };
        let title = match (self.cur_name(), &self.image_dir) {
            (Some(n), Some(d)) => format!(
                "{n}{dirty_mark} — {} [{}/{}] — yolo-ann",
                d.file_name().unwrap_or_default().to_string_lossy(),
                self.cur.unwrap() + 1,
                self.images.len()
            ),
            _ => "yolo-ann".into(),
        };
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.write_backup(true);
        self.settings.save();
    }
}

fn hit_handle(r: Rect, p: Pos2) -> u8 {
    const T: f32 = 6.0;
    if !r.expand(T).contains(p) {
        return 0;
    }
    let mut h = 0;
    if (p.x - r.left()).abs() <= T {
        h |= H_LEFT;
    } else if (p.x - r.right()).abs() <= T {
        h |= H_RIGHT;
    }
    if (p.y - r.top()).abs() <= T {
        h |= H_TOP;
    } else if (p.y - r.bottom()).abs() <= T {
        h |= H_BOTTOM;
    }
    h
}

fn handle_cursor(h: u8) -> CursorIcon {
    match h {
        x if x == H_LEFT | H_TOP || x == H_RIGHT | H_BOTTOM => CursorIcon::ResizeNwSe,
        x if x == H_RIGHT | H_TOP || x == H_LEFT | H_BOTTOM => CursorIcon::ResizeNeSw,
        H_LEFT | H_RIGHT => CursorIcon::ResizeHorizontal,
        _ => CursorIcon::ResizeVertical,
    }
}

fn format_bytes(b: u64) -> String {
    match b {
        0..1024 => format!("{b} B"),
        1024..1_048_576 => format!("{:.1} KB", b as f64 / 1024.0),
        _ => format!("{:.2} MB", b as f64 / 1_048_576.0),
    }
}

#[cfg(test)]
mod tests;
