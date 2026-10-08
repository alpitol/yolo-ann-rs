# yolo-ann (local)

A local Rust/egui rewrite of the web tool <https://lddl.github.io/yolo-ann/> for drawing YOLO bounding boxes.

- Images are read lazily from disk. Only the current image and its neighbours are decoded, so folders with thousands of images open instantly.
- The next 6 and previous 2 images are decoded in background threads, so switching images is instant. A 4K jpg takes about 60 ms per thread.
- Annotations are backed up to a local file automatically, so nothing is lost on a crash, kill or power-off.
- YOLO txt labels are read and written directly. COCO json import and export is supported (the export has the same format as the original tool). VOC is not supported.

## Build & run

```sh
cargo build --release
./target/release/yolo-ann path/to/images            # or no argument: reopen the last folder
./target/release/yolo-ann real4/images --coco real4/annotations.json
./target/release/yolo-ann --help
```

| Option | Meaning |
|---|---|
| `-l, --labels DIR` | YOLO labels folder. Default: `../labels` if the image folder is named `images` and that exists, otherwise the image folder itself |
| `-c, --classes FILE` | Class names, one per line. Auto-detected: `classes.txt`, `obj.names`, `*.names` in the image dir, the labels dir or their parent |
| `--coco FILE` | Import a COCO json on startup. Categories are matched to classes by name. If no classes are loaded, the COCO categories are used |
| `--ahead N` / `--behind N` | Prefetch window (default 6 / 2) |
| `--workers N` | Decoder threads |
| `--autosave true` | Write the label file automatically when leaving an image |

## Hotkeys

| Key | Action |
|---|---|
| ← / → (A / D) | previous / next image (wraps) |
| ↑ / ↓ (W / S) | previous / next class (wraps) |
| 1–9 | select class |
| Delete / Backspace / X | remove selected box |
| Left drag | draw box (Shift: always draw, even on top of a box) |
| Left click | select box; click again to cycle overlapping boxes |
| Drag selected box | move; drag its edges or corners to resize |
| Mouse wheel | zoom at cursor |
| Middle / right drag | pan |
| F / Home | reset zoom |
| Esc | deselect / cancel drag |
| Ctrl+Z, Ctrl+Y / Ctrl+Shift+Z | undo / redo (per image) |
| Ctrl+S | save YOLO labels |
| Ctrl+O | open folder |

In the image list, `*N` marks an image with unsaved changes and N boxes. `[N]` marks a visited image with N boxes.

## Saving & backup

- **Save YOLO** (Ctrl+S) writes `<labels>/<image>.txt` for every changed image, plus `classes.txt` if the labels dir has none. Writes are atomic.
- **Export COCO** writes a json in the same format as the original web tool (category id = class + 1). The default file name is `<folder>_HUMAN_FIXED.json`.
- **Crop & Save** writes one crop per box to a chosen folder.
- **Automatic backup:** all unsaved annotations are written about 2 s after every change, and on every image switch and on exit, to
  `~/.local/share/yolo-ann/backups/<folder>-<hash>.json` (one file per image folder). The previous session's file is kept as `*.prev.json`.
  When a folder whose backup contains unsaved annotations is opened, a banner offers **Restore / Ignore / Delete backup**. **Restore backup…** loads any backup file manually.
- Settings (last folder and image) are stored in `~/.config/yolo-ann/settings.json`.
- Closing the window with unsaved changes asks whether to save first.

## Tests

```sh
cargo test                     # unit tests and headless UI tests (egui_kittest)
YOLO_ANN_BENCH_DIR=dir cargo test --release decode_speed -- --ignored --nocapture
```

`YOLO_ANN_HOME=dir` redirects the backup and settings directories, which the tests use.
