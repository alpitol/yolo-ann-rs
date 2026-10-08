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
| `-l, --labels DIR` | Import YOLO labels from this folder (also the save target). Default: the image folder, if it contains label files (shown as *autoselected*) |
| `-c, --classes FILE` | Class names, one per line. Auto-detected: `classes.txt`, `obj.names`, `*.names` in the image folder or the imported labels folder. The last used file is remembered |
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
| Ctrl+S | update the YOLO labels folder |
| Ctrl+O | open image folder |

In the image list, `*N` marks an image with unsaved changes and N boxes. `[N]` marks a visited image with N boxes.

## Saving & backup

No label folder layout is assumed: labels come only from what you import (or from `.txt` files next to the images).

**Annotation data.** An image has annotation data, possibly an empty box list, if it was imported from YOLO or COCO, restored from a backup, edited, or left with the arrow keys or a list click after being shown. Opening a folder or jumping because of a filter does not count. Exports include exactly these images. **Filter to annotation data**, below the image list, shows only them, which is useful for inspecting and editing a subset inside a folder that holds the whole dataset.

- **Import YOLO** loads `<stem>.txt` files from a folder. The first imported folder becomes the *labels folder*. Folders imported on top of it are unsaved edits.
- **Export YOLO** has two modes:
  - *Update labels folder* (Ctrl+S) writes only the changed files.
  - *Export to another folder* writes every image with data to a new or existing folder, which then becomes the labels folder.

  Ctrl+S without a labels folder asks for one. `classes.txt` is added if the folder has none. Writes are atomic.
- **Import COCO** matches categories to classes by name. **Export COCO** writes a json in the same format as the original web tool (category id = class + 1). The default file name is `<folder>_HUMAN_FIXED.json`.
- **Crop & Save** writes one crop per box to a chosen folder.
- **Automatic backup:** all unsaved annotations are written about 2 s after every change, and on every image switch and on exit, to
  `~/.local/share/yolo-ann/backups/<folder>-<hash>.json` (one file per image folder). The previous session's file is kept as `*.prev.json`.
  When a folder whose backup contains unsaved annotations is opened, a banner offers **Restore / Ignore / Delete backup**. **Restore backup** loads any backup file manually.
- Settings (last folder and image) are stored in `~/.config/yolo-ann/settings.json`.
- Closing the window with unsaved changes asks whether to save first.

## Tests

```sh
cargo test                     # unit tests and headless UI tests (egui_kittest)
YOLO_ANN_BENCH_DIR=dir cargo test --release decode_speed -- --ignored --nocapture
```

`YOLO_ANN_HOME=dir` redirects the backup and settings directories, which the tests use.
