//! Headless UI tests driving the real app with simulated input.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use egui::{Event, Key, Modifiers, PointerButton, Pos2, pos2};
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};

use super::{App, Config, Slot};

const NAMES: [&str; 5] = ["img1.png", "img2.png", "img3.png", "img10.png", "img11.png"];

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("yolo_ann_ui_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    // All tests in this process share the env var; each test uses its own image dir,
    // so per-directory backups never collide.
    unsafe {
        std::env::set_var(
            "YOLO_ANN_HOME",
            std::env::temp_dir().join(format!("yolo_ann_home_{}", std::process::id())),
        )
    };
    // The app canonicalizes folders (/var -> /private/var on macOS, \\?\ on Windows).
    root.canonicalize().unwrap()
}

fn make_dataset(root: &Path) -> PathBuf {
    let dir = root.join("images");
    std::fs::create_dir_all(&dir).unwrap();
    for (i, n) in NAMES.iter().enumerate() {
        let img = image::RgbImage::from_pixel(400, 200, image::Rgb([40 * i as u8, 100, 200]));
        img.save(dir.join(n)).unwrap();
    }
    std::fs::write(dir.join("classes.txt"), "ship\nboat\nbuoy\n").unwrap();
    std::fs::write(dir.join("img2.txt"), "1 0.5 0.5 0.2 0.4\n").unwrap();
    dir
}

fn harness(dir: &Path) -> Harness<'static, App> {
    let cfg = Config {
        image_dir: Some(dir.to_path_buf()),
        labels_dir: None,
        classes: None,
        coco: None,
        ahead: 2,
        behind: 1,
        workers: 2,
        autosave: Some(false),
    };
    Harness::builder()
        .with_size([1200.0, 800.0])
        .build_eframe(move |cc| App::new(cc, cfg))
}

fn wait_until(h: &mut Harness<'static, App>, what: &str, cond: impl Fn(&App) -> bool) {
    let start = Instant::now();
    loop {
        h.step();
        if cond(h.state()) {
            h.step();
            return;
        }
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "timed out waiting for {what}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn wait_loaded(h: &mut Harness<'static, App>) {
    wait_until(h, "current image", |a| {
        let p = a.image_path(a.cur.unwrap()).unwrap();
        matches!(a.cache.get(&p), Some(Slot::Ready { .. })) && a.last_img_rect.is_some()
    });
    h.run_ok();
}

fn norm_to_screen(h: &Harness<'static, App>, x: f32, y: f32) -> Pos2 {
    let r = h.state().last_img_rect.unwrap();
    pos2(r.min.x + x * r.width(), r.min.y + y * r.height())
}

fn drag(h: &mut Harness<'static, App>, from: Pos2, to: Pos2) {
    h.hover_at(from);
    h.step();
    h.drag_at(from);
    h.step();
    for t in [0.25, 0.5, 0.75, 1.0] {
        h.hover_at(from + (to - from) * t);
        h.step();
    }
    h.event(Event::PointerButton {
        pos: to,
        button: PointerButton::Primary,
        pressed: false,
        modifiers: Modifiers::NONE,
    });
    h.step();
    h.step();
}

fn click(h: &mut Harness<'static, App>, at: Pos2) {
    h.hover_at(at);
    h.step();
    h.drag_at(at);
    h.step();
    h.event(Event::PointerButton {
        pos: at,
        button: PointerButton::Primary,
        pressed: false,
        modifiers: Modifiers::NONE,
    });
    h.step();
    h.step();
}

fn cur_name(h: &Harness<'static, App>) -> String {
    h.state().cur_name().unwrap().to_string()
}

#[test]
fn natural_order_navigation_and_class_hotkeys() {
    let dir = make_dataset(&temp_root("nav"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    assert_eq!(h.state().images, NAMES);
    assert_eq!(h.state().classes, ["ship", "boat", "buoy"]);
    assert_eq!(cur_name(&h), "img1.png");

    h.key_press(Key::ArrowRight);
    h.step();
    assert_eq!(cur_name(&h), "img2.png");
    // Label files next to the images are autoselected.
    assert!(h.state().labels_auto);
    assert_eq!(h.state().labels_dir.as_deref(), Some(dir.as_path()));
    assert_eq!(h.state().anns["img2.png"].boxes.len(), 1);
    assert!(!h.state().anns["img2.png"].dirty);
    assert_eq!(h.state().anns["img2.png"].boxes[0].class, 1);

    h.key_press(Key::ArrowLeft);
    h.step();
    h.key_press(Key::ArrowLeft);
    h.step();
    assert_eq!(cur_name(&h), "img11.png", "left wraps around");
    h.key_press(Key::ArrowRight);
    h.step();
    assert_eq!(cur_name(&h), "img1.png", "right wraps around");

    h.key_press(Key::ArrowDown);
    h.step();
    assert_eq!(h.state().cur_class, 1);
    h.key_press(Key::ArrowUp);
    h.step();
    h.key_press(Key::ArrowUp);
    h.step();
    assert_eq!(h.state().cur_class, 2, "up wraps around");
    h.key_press(Key::Num1);
    h.step();
    assert_eq!(h.state().cur_class, 0);
}

#[test]
fn arrows_change_class_of_selected_box() {
    let dir = make_dataset(&temp_root("sel_class"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    let (a, b) = (norm_to_screen(&h, 0.2, 0.2), norm_to_screen(&h, 0.6, 0.6));
    drag(&mut h, a, b);
    assert_eq!(h.state().selected, Some(0));
    assert_eq!(h.state().anns["img1.png"].boxes[0].class, 0);

    h.key_press(Key::ArrowDown);
    h.step();
    assert_eq!(h.state().anns["img1.png"].boxes[0].class, 1);
    assert_eq!(h.state().selected, Some(0), "box stays selected");
    assert_eq!(h.state().cur_class, 0, "drawing class unchanged");
    h.key_press(Key::ArrowUp);
    h.step();
    h.key_press(Key::W);
    h.step();
    assert_eq!(h.state().anns["img1.png"].boxes[0].class, 2, "up wraps");

    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.step();
    assert_eq!(h.state().anns["img1.png"].boxes[0].class, 0, "undoable");

    // Without a selection the arrows change the drawing class again.
    h.key_press(Key::Escape);
    h.step();
    h.key_press(Key::ArrowDown);
    h.step();
    assert_eq!(h.state().cur_class, 1);
    assert_eq!(h.state().anns["img1.png"].boxes[0].class, 0);
}

#[test]
fn ctrl_c_copies_image_file() {
    let dir = make_dataset(&temp_root("copy_file"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    h.key_press(Key::ArrowRight);
    h.step();
    h.key_press_modifiers(Modifiers::COMMAND, Key::C);
    h.step();
    let expected = std::path::absolute(dir.join("img2.png")).unwrap();
    assert_eq!(h.state().clipboard.copied, [expected]);
    assert!(!h.state().status_warn, "{}", h.state().status);

    // The real backend sends Event::Copy instead of a key press.
    h.state_mut().clipboard.copied.clear();
    h.event(egui::Event::Copy);
    h.step();
    assert_eq!(h.state().clipboard.copied.len(), 1);
}

#[test]
fn prefetches_neighbours() {
    let dir = make_dataset(&temp_root("prefetch"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    // ahead=2, behind=1 => img1, img2, img3 and (wrapping) img11.
    wait_until(&mut h, "prefetch window", |a| {
        ["img1.png", "img2.png", "img3.png", "img11.png"]
            .iter()
            .all(|n| matches!(a.cache.get(&dir.join(n)), Some(Slot::Ready { .. })))
    });
    assert!(!h.state().cache.contains_key(&dir.join("img10.png")));
}

#[test]
fn draw_select_move_delete_undo_save() {
    let dir = make_dataset(&temp_root("draw"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    h.key_press(Key::Num3); // buoy
    h.step();

    let (a, b) = (
        norm_to_screen(&h, 0.25, 0.25),
        norm_to_screen(&h, 0.5, 0.75),
    );
    drag(&mut h, a, b);
    let boxes = h.state().anns["img1.png"].boxes.clone();
    assert_eq!(boxes.len(), 1, "box drawn");
    let bx = boxes[0];
    assert_eq!(bx.class, 2);
    assert!(
        (bx.x0 - 0.25).abs() < 0.01 && (bx.y0 - 0.25).abs() < 0.01,
        "{bx:?}"
    );
    assert!(
        (bx.x1 - 0.5).abs() < 0.01 && (bx.y1 - 0.75).abs() < 0.01,
        "{bx:?}"
    );
    assert_eq!(h.state().selected, Some(0), "new box is selected");
    assert!(h.state().has_unsaved());

    // Tiny drags are discarded.
    h.key_press(Key::Escape);
    h.step();
    let p = norm_to_screen(&h, 0.8, 0.1);
    drag(&mut h, p, p + egui::vec2(2.0, 2.0));
    assert_eq!(h.state().anns["img1.png"].boxes.len(), 1);

    // Click to select, drag to move.
    let p = norm_to_screen(&h, 0.4, 0.5);
    click(&mut h, p);
    assert_eq!(h.state().selected, Some(0));
    let (p0, p1) = (norm_to_screen(&h, 0.4, 0.5), norm_to_screen(&h, 0.5, 0.5));
    drag(&mut h, p0, p1);
    let moved = h.state().anns["img1.png"].boxes[0];
    assert!(
        (moved.x0 - 0.35).abs() < 0.01 && (moved.x1 - 0.6).abs() < 0.01,
        "{moved:?}"
    );

    // Resize from the right edge.
    let (p0, p1) = (norm_to_screen(&h, 0.6, 0.5), norm_to_screen(&h, 0.9, 0.5));
    drag(&mut h, p0, p1);
    let resized = h.state().anns["img1.png"].boxes[0];
    assert!(
        (resized.x1 - 0.9).abs() < 0.01 && (resized.x0 - 0.35).abs() < 0.01,
        "{resized:?}"
    );

    h.key_press(Key::Delete);
    h.step();
    assert!(h.state().anns["img1.png"].boxes.is_empty(), "deleted");

    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.step();
    assert_eq!(h.state().anns["img1.png"].boxes.len(), 1, "undo restores");

    h.key_press_modifiers(Modifiers::COMMAND, Key::S);
    h.step();
    assert!(!h.state().has_unsaved());
    let text = std::fs::read_to_string(dir.join("img1.txt")).unwrap();
    let parsed = crate::model::parse_yolo(&text, &Default::default());
    assert_eq!(parsed.len(), 1);
    assert!((parsed[0].x1 - 0.9).abs() < 0.01);
    // Untouched images aren't rewritten.
    assert!(!dir.join("img3.txt").exists());
}

#[test]
fn backup_survives_restart_and_restores() {
    let dir = make_dataset(&temp_root("backup"));
    {
        let mut h = harness(&dir);
        wait_loaded(&mut h);
        let (p0, p1) = (norm_to_screen(&h, 0.1, 0.1), norm_to_screen(&h, 0.3, 0.3));
        drag(&mut h, p0, p1);
        assert_eq!(h.state().anns["img1.png"].boxes.len(), 1);
        // Navigating away flushes the backup.
        h.key_press(Key::ArrowRight);
        h.step();
        let bp = h.state().backup_path.clone().unwrap();
        let b = crate::backup::load_backup(&bp).expect("backup written");
        assert_eq!(b.images["img1.png"].len(), 1);
        // Simulated crash: harness dropped without saving.
    }
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    assert!(h.state().pending_restore.is_some(), "restore offered");
    assert!(!dir.join("img1.txt").exists());
    let b = h.state_mut().pending_restore.take().unwrap();
    h.state_mut().restore_backup(b);
    h.step();
    assert_eq!(h.state().anns["img1.png"].boxes.len(), 1);
    assert!(h.state().anns["img1.png"].dirty);
}

#[test]
fn coco_import_export_roundtrip() {
    let root = temp_root("coco");
    let dir = make_dataset(&root);
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    let out = root.join("out.json");
    h.state_mut().export_coco(&out);
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(
        json["images"].as_array().unwrap().len(),
        1,
        "only images with annotation data"
    );
    assert_eq!(json["annotations"].as_array().unwrap().len(), 1);
    let bbox = &json["annotations"][0]["bbox"];
    assert!(
        (bbox[0].as_f64().unwrap() - 160.0).abs() < 0.01
            && (bbox[3].as_f64().unwrap() - 80.0).abs() < 0.01
    );

    // Import into a fresh session: replaces boxes of listed images.
    let dir2 = make_dataset(&root.join("second"));
    std::fs::remove_file(dir2.join("img2.txt")).unwrap();
    let mut h2 = harness(&dir2);
    wait_loaded(&mut h2);
    h2.state_mut().import_coco(&out);
    let ann = &h2.state().anns["img2.png"];
    assert_eq!(ann.boxes.len(), 1);
    assert_eq!(ann.boxes[0].class, 1);
    assert!((ann.boxes[0].x0 - 0.4).abs() < 1e-6);
}

#[test]
fn coco_import_replaces_everything() {
    let root = temp_root("coco_replace");
    let dir = make_dataset(&root);
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    assert!(h.state().labels_dir.is_some());
    let (p0, p1) = (norm_to_screen(&h, 0.1, 0.1), norm_to_screen(&h, 0.3, 0.3));
    drag(&mut h, p0, p1);
    assert!(h.state().has_unsaved());
    let coco = root.join("c.json");
    std::fs::write(
        &coco,
        r#"{"images":[{"id":7,"file_name":"img3.png","width":400,"height":200}],
            "annotations":[{"id":1,"image_id":7,"category_id":0,"bbox":[40,20,80,40],"iscrowd":0,
                "segmentation":{"size":[200,400],"counts":"abc"}}],
            "categories":[{"id":0,"name":"powerboat"},{"id":1,"name":"ship"}]}"#,
    )
    .unwrap();

    // Unsaved edits: asks first, Cancel keeps everything.
    h.state_mut()
        .request_discard(super::Discard::ImportCoco(coco.clone()));
    h.step();
    h.step();
    assert!(h.state().confirm_discard.is_some());
    h.get_by_label("Cancel").click();
    h.step();
    h.step();
    assert!(h.state().confirm_discard.is_none());
    assert_eq!(h.state().anns["img1.png"].boxes.len(), 1);
    assert_eq!(h.state().classes, ["ship", "boat", "buoy"]);

    h.state_mut()
        .request_discard(super::Discard::ImportCoco(coco.clone()));
    h.step();
    h.step();
    h.get_by_label("Discard & import").click();
    h.step();
    h.step();
    let a = h.state();
    assert!(a.confirm_discard.is_none());
    assert_eq!(a.classes, ["powerboat", "ship"]);
    assert_eq!(a.anns.keys().collect::<Vec<_>>(), ["img3.png"]);
    assert_eq!(a.anns["img3.png"].boxes[0].class, 0);
    assert!(a.labels_dir.is_none() && a.classes_file.is_none());
    assert!(a.settings.last_classes_file.is_none());
    assert!(!a.has_unsaved());
}

#[test]
fn opening_folder_clears_classes() {
    let root = temp_root("open_clears");
    let dir = make_dataset(&root);
    let other = root.join("other");
    std::fs::create_dir_all(&other).unwrap();
    image::RgbImage::from_pixel(40, 20, image::Rgb([9, 9, 9]))
        .save(other.join("a.png"))
        .unwrap();
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    assert_eq!(h.state().classes.len(), 3);
    h.state_mut()
        .request_discard(super::Discard::OpenDir(other.clone()));
    h.step();
    let a = h.state();
    assert_eq!(a.images, ["a.png"]);
    assert!(a.classes.is_empty(), "classes not carried over");
    assert!(a.classes_file.is_none() && a.settings.last_classes_file.is_none());
    assert!(a.anns.is_empty() && a.labels_dir.is_none());
}

fn visible_names(h: &Harness<'static, App>) -> Vec<String> {
    let a = h.state();
    a.visible.iter().map(|&i| a.images[i].clone()).collect()
}

fn toggle_filter(h: &mut Harness<'static, App>) {
    h.get_by_label_contains("Filter to annotation data").click();
    h.step();
    h.step();
}

#[test]
fn data_filter_and_visit_rules() {
    let dir = make_dataset(&temp_root("filter"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    assert_eq!(cur_name(&h), "img1.png");
    assert!(
        !h.state().anns.contains_key("img1.png"),
        "the first image is not data just by being shown"
    );

    // Filtering away from img1 must not turn it into data.
    toggle_filter(&mut h);
    assert!(h.state().filter_data);
    assert_eq!(visible_names(&h), ["img2.png"]);
    assert_eq!(cur_name(&h), "img2.png");
    assert!(!h.state().anns.contains_key("img1.png"));
    assert!(!h.state().has_unsaved());

    // Unfiltered: leaving images with arrow keys makes them (empty) data.
    toggle_filter(&mut h);
    assert_eq!(visible_names(&h).len(), 5);
    h.key_press(Key::ArrowRight); // img2 -> img3
    h.step();
    h.key_press(Key::ArrowRight); // img3 -> img10, img3 becomes empty data
    h.step();
    assert_eq!(cur_name(&h), "img10.png");
    assert!(h.state().anns["img3.png"].boxes.is_empty());
    assert!(h.state().anns["img3.png"].dirty);
    assert!(
        !h.state().anns.contains_key("img10.png"),
        "current not yet left"
    );
    assert!(
        !h.state().has_unsaved(),
        "visiting is not an edit: no quit prompt"
    );

    toggle_filter(&mut h);
    assert_eq!(visible_names(&h), ["img2.png", "img3.png"]);
    assert_eq!(cur_name(&h), "img2.png");
    assert!(
        !h.state().anns.contains_key("img10.png"),
        "filter jump doesn't mark"
    );
    h.key_press(Key::ArrowRight);
    h.step();
    assert_eq!(cur_name(&h), "img3.png");

    // COCO import replaces all annotation data with its images.
    let coco = dir.parent().unwrap().join("c.json");
    std::fs::write(
        &coco,
        r#"{"images":[{"id":1,"file_name":"img11.png","width":400,"height":200},
                      {"id":2,"file_name":"img3.png","width":400,"height":200}],
            "annotations":[],"categories":[{"id":1,"name":"ship"}]}"#,
    )
    .unwrap();
    h.state_mut().import_coco(&coco);
    h.step();
    assert_eq!(visible_names(&h), ["img3.png", "img11.png"]);
    assert!(!h.state().has_unsaved(), "imported data is not an edit");

    // Drawing is an edit; undoing it is not.
    wait_loaded(&mut h);
    let (p0, p1) = (norm_to_screen(&h, 0.1, 0.1), norm_to_screen(&h, 0.3, 0.3));
    drag(&mut h, p0, p1);
    assert!(h.state().has_unsaved());
    let bp = h.state().backup_path.clone().unwrap();
    h.state_mut().write_backup(true);
    let b = crate::backup::load_backup(&bp).unwrap();
    assert_eq!(
        b.images.keys().collect::<Vec<_>>(),
        ["img3.png"],
        "backup holds only edited images"
    );
    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.step();
    assert!(!h.state().has_unsaved());
}

#[test]
fn yolo_import_and_export_modes() {
    let root = temp_root("yolo_io");
    let dir = make_dataset(&root);
    std::fs::remove_file(dir.join("img2.txt")).unwrap();
    let labels = root.join("my_labels");
    std::fs::create_dir_all(&labels).unwrap();
    std::fs::write(labels.join("img3.txt"), "0 0.5 0.5 0.2 0.2\n").unwrap();
    std::fs::write(labels.join("img10.txt"), "").unwrap();

    let mut h = harness(&dir);
    wait_loaded(&mut h);
    assert!(h.state().labels_dir.is_none(), "no label folder guessing");
    assert!(h.state().anns.is_empty());

    h.state_mut().import_yolo(&labels, false);
    h.step();
    assert_eq!(h.state().labels_dir.as_deref(), Some(labels.as_path()));
    assert_eq!(h.state().anns.len(), 2, "empty label file is data too");
    assert!(h.state().anns.values().all(|a| !a.dirty));

    // Update existing folder: only changed files are written.
    h.key_press(Key::ArrowRight); // img1 left -> empty data
    h.step();
    h.key_press_modifiers(Modifiers::COMMAND, Key::S);
    h.step();
    assert_eq!(
        std::fs::read_to_string(labels.join("img1.txt")).unwrap(),
        ""
    );
    assert!(!h.state().has_unsaved());

    // Export to a new folder: everything with data is written there.
    let out = root.join("new_labels");
    h.state_mut().export_yolo(&out);
    let mut written: Vec<_> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    written.sort();
    assert_eq!(
        written,
        ["classes.txt", "img1.txt", "img10.txt", "img3.txt"]
    );
    assert_eq!(h.state().labels_dir.as_deref(), Some(out.as_path()));
}

/// `root/{images,labels,metadata.txt}` with 1-based class ids in the label files.
#[test]
fn metadata_class_ids_roundtrip() {
    let root = temp_root("metadata");
    let dir = make_dataset(&root);
    std::fs::remove_file(dir.join("classes.txt")).unwrap();
    std::fs::remove_file(dir.join("img2.txt")).unwrap();
    std::fs::write(
        root.join("metadata.txt"),
        "1 motor_boat\n2 sailing_boat\n3 ship\n",
    )
    .unwrap();
    let labels = root.join("labels");
    std::fs::create_dir_all(&labels).unwrap();
    std::fs::write(
        labels.join("img1.txt"),
        "3 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n",
    )
    .unwrap();

    let mut h = harness(&dir);
    wait_loaded(&mut h);
    assert_eq!(h.state().classes, ["motor_boat", "sailing_boat", "ship"]);
    assert_eq!(
        h.state().labels_dir.as_deref(),
        Some(labels.canonicalize().unwrap().as_path()),
        "sibling labels folder autoselected"
    );
    assert!(h.state().labels_auto);
    let classes: Vec<_> = h.state().anns["img1.png"]
        .boxes
        .iter()
        .map(|b| b.class)
        .collect();
    assert_eq!(classes, [2, 0], "file ids 3 and 1 are ship and motor_boat");

    h.state_mut().anns.get_mut("img1.png").unwrap().boxes[0].class = 1;
    h.state_mut().anns.get_mut("img1.png").unwrap().dirty = true;
    h.state_mut().save_yolo();
    let text = std::fs::read_to_string(labels.join("img1.txt")).unwrap();
    let ids: Vec<_> = text.lines().map(|l| l.split(' ').next().unwrap()).collect();
    assert_eq!(ids, ["2", "1"], "saved with metadata ids");
    assert!(
        !labels.join("classes.txt").exists(),
        "metadata.txt already covers it"
    );

    let out = root.join("export");
    h.state_mut().export_yolo(&out);
    assert_eq!(
        std::fs::read_to_string(out.join("metadata.txt")).unwrap(),
        "1 motor_boat\n2 sailing_boat\n3 ship\n"
    );
    assert!(!out.join("classes.txt").exists());
}

/// Loading a `<id> <name>` class file after labels were read (as plain 0-based ids)
/// remaps the loaded boxes so the file ids keep their meaning.
#[test]
fn loading_class_ids_remaps_loaded_labels() {
    let root = temp_root("metadata_late");
    let dir = make_dataset(&root);
    std::fs::remove_file(dir.join("classes.txt")).unwrap();
    std::fs::write(
        dir.join("img1.txt"),
        "3 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n",
    )
    .unwrap();
    let meta = root.join("meta.txt");
    std::fs::write(&meta, "1 motor_boat\n2 sailing_boat\n3 ship\n").unwrap();

    let mut h = harness(&dir);
    wait_loaded(&mut h);
    let classes = |h: &Harness<'static, App>, name: &str| -> Vec<usize> {
        h.state().anns[name].boxes.iter().map(|b| b.class).collect()
    };
    assert_eq!(classes(&h, "img1.png"), [3, 1]);
    {
        let ann = h.state_mut().anns.get_mut("img1.png").unwrap();
        ann.snapshot();
        ann.boxes.pop();
    }

    h.state_mut().load_classes(&meta);
    assert_eq!(classes(&h, "img1.png"), [2], "file id 3 is ship");
    assert_eq!(classes(&h, "img2.png"), [0], "file id 1 is motor_boat");
    let ann = &h.state().anns["img1.png"];
    assert!(ann.modified(), "exported state remapped along with boxes");
    assert_eq!(
        ann.undo[0].iter().map(|b| b.class).collect::<Vec<_>>(),
        [2, 0]
    );

    h.state_mut().anns.get_mut("img2.png").unwrap().dirty = true;
    h.state_mut().save_yolo();
    assert_eq!(
        std::fs::read_to_string(dir.join("img1.txt")).unwrap(),
        "3 0.500000 0.500000 0.200000 0.200000\n"
    );
    assert!(
        std::fs::read_to_string(dir.join("img2.txt"))
            .unwrap()
            .starts_with("1 "),
        "ids unchanged on disk"
    );
}

fn untrack_via_menu(h: &mut Harness<'static, App>, name: &str) -> bool {
    h.get_by_label_contains(&format!("] {name}"))
        .click_secondary();
    h.step();
    h.step();
    let item = h.get_by_label("Untrack empty image");
    let enabled = !item.accesskit_node().is_disabled();
    if enabled {
        item.click();
    } else {
        h.key_press(Key::Escape);
    }
    h.step();
    h.step();
    assert!(
        h.query_by_label("Untrack empty image").is_none(),
        "menu closed for {name} (enabled={enabled})"
    );
    enabled
}

#[test]
fn untrack_empty_image() {
    let dir = make_dataset(&temp_root("untrack"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    h.key_press(Key::ArrowRight); // img1 becomes empty data
    h.step();
    h.key_press_modifiers(Modifiers::COMMAND, Key::S);
    h.step();
    assert!(dir.join("img1.txt").is_file());

    assert!(untrack_via_menu(&mut h, "img1.png"));
    assert!(!h.state().anns.contains_key("img1.png"));
    assert!(
        !dir.join("img1.txt").exists(),
        "empty label file removed too"
    );
    assert!(
        !untrack_via_menu(&mut h, "img2.png"),
        "images with boxes can't be untracked"
    );
    assert!(h.state().anns.contains_key("img2.png"));

    // Untracking the current image: leaving it must not re-create data.
    h.key_press(Key::ArrowRight); // img2 -> img3
    h.step();
    h.key_press(Key::ArrowRight); // img3 -> img10, img3 becomes data
    h.step();
    h.key_press(Key::ArrowLeft); // img10 -> img3, img10 becomes data
    h.step();
    assert_eq!(cur_name(&h), "img3.png");
    assert!(untrack_via_menu(&mut h, "img3.png"));
    h.key_press(Key::ArrowRight);
    h.step();
    assert_eq!(cur_name(&h), "img10.png");
    assert!(!h.state().anns.contains_key("img3.png"));

    // With the filter on, untracking the current image moves to the next one.
    toggle_filter(&mut h);
    assert_eq!(visible_names(&h), ["img2.png", "img10.png"]);
    assert!(untrack_via_menu(&mut h, "img10.png"));
    assert_eq!(visible_names(&h), ["img2.png"]);
    assert_eq!(cur_name(&h), "img2.png");
    assert!(!h.state().has_unsaved());
}

#[test]
fn copy_file_name_from_list_menu() {
    let dir = make_dataset(&temp_root("copy_name"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    h.get_by_label("img10.png").click_secondary();
    h.step();
    h.step();
    h.get_by_label("Copy file name").click();
    h.step();
    let copied = h
        .output()
        .platform_output
        .commands
        .iter()
        .any(|c| matches!(c, egui::OutputCommand::CopyText(t) if t == "img10.png"));
    assert!(copied, "{:?}", h.output().platform_output.commands);
    h.step();
    assert!(h.query_by_label("Copy file name").is_none(), "menu closed");
    assert_eq!(cur_name(&h), "img1.png", "copying doesn't navigate");
}

#[test]
fn delete_image_file() {
    let dir = make_dataset(&temp_root("delete_file"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    h.key_press(Key::ArrowRight); // img1 becomes data, img2 has a label file
    h.step();
    assert_eq!(cur_name(&h), "img2.png");

    // Shift+Delete asks first; Escape cancels and keeps the box and the file.
    h.key_press_modifiers(Modifiers::SHIFT, Key::Delete);
    h.step();
    h.step();
    assert_eq!(h.state().confirm_delete.as_deref(), Some("img2.png"));
    assert_eq!(h.state().anns["img2.png"].boxes.len(), 1);
    h.key_press(Key::Escape);
    h.step();
    h.step();
    assert!(h.state().confirm_delete.is_none());
    assert!(dir.join("img2.png").is_file());

    h.key_press_modifiers(Modifiers::SHIFT, Key::Delete);
    h.step();
    h.step();
    h.get_by_label("Delete").click();
    h.step();
    h.step();
    assert!(h.state().confirm_delete.is_none());
    assert!(!dir.join("img2.png").exists());
    assert!(!dir.join("img2.txt").exists(), "label file deleted too");
    assert_eq!(
        h.state().images,
        ["img1.png", "img3.png", "img10.png", "img11.png"]
    );
    assert!(!h.state().anns.contains_key("img2.png"));
    assert_eq!(cur_name(&h), "img3.png", "moved to the next image");

    // From the list menu, for an image before the current one.
    h.get_by_label_contains("] img1.png").click_secondary();
    h.step();
    h.step();
    h.get_by_label("Delete file…").click();
    h.step();
    h.step();
    assert_eq!(h.state().confirm_delete.as_deref(), Some("img1.png"));
    h.get_by_label("Delete").click();
    h.step();
    h.step();
    assert!(!dir.join("img1.png").exists());
    assert_eq!(h.state().images, ["img3.png", "img10.png", "img11.png"]);
    assert_eq!(cur_name(&h), "img3.png", "current image kept");
    assert_eq!(h.state().visible, [0, 1, 2]);
    assert!(!h.state().anns.contains_key("img1.png"));
}

#[test]
fn wheel_zoom_is_instant() {
    let dir = make_dataset(&temp_root("zoom"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    let p = norm_to_screen(&h, 0.5, 0.5);
    h.hover_at(p);
    h.step();
    let z0 = h.state().zoom;
    h.event(Event::MouseWheel {
        unit: egui::MouseWheelUnit::Line,
        delta: egui::vec2(0.0, 1.0),
        phase: egui::TouchPhase::Move,
        modifiers: Modifiers::NONE,
    });
    h.step();
    let z1 = h.state().zoom;
    assert!(
        (z1 / z0 - 1.002f32.powf(40.0 * 6.0)).abs() < 1e-3,
        "{z0} -> {z1}"
    );
    for _ in 0..5 {
        h.step();
    }
    assert_eq!(h.state().zoom, z1, "no trailing animation");
}

#[test]
fn f11_toggles_fullscreen() {
    let dir = make_dataset(&temp_root("fullscreen"));
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    h.event(Event::Key {
        key: Key::F11,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    h.step();
    let full = h.output().viewport_output.values().any(|v| {
        v.commands
            .iter()
            .any(|c| matches!(c, egui::ViewportCommand::Fullscreen(true)))
    });
    assert!(full, "F11 should request full screen");
}

#[test]
fn failures_are_warnings() {
    let root = temp_root("warnings");
    let dir = make_dataset(&root);
    let mut h = harness(&dir);
    wait_loaded(&mut h);
    assert!(!h.state().status_warn);
    h.state_mut().import_coco(&root.join("missing.json"));
    assert!(h.state().status_warn, "{}", h.state().status);
    h.state_mut().set_status("ok");
    assert!(!h.state().status_warn);
}
