# Copilot instructions for yolo-ann

A local Rust/egui (eframe 0.36, glow backend) desktop tool for drawing YOLO bounding boxes, a rewrite of <https://lddl.github.io/yolo-ann/>. `README.md` is the user-facing spec of behaviour (CLI options, hotkeys, import/export, backup and "annotation data" rules); keep it in sync when behaviour changes.

## Build, test, lint

```sh
cargo build --release                      # binary: target/release/yolo-ann
cargo test                                 # unit tests + headless egui_kittest UI tests (no display needed)
cargo test data_filter_and_visit_rules     # single test by name substring
cargo test app::tests::                    # only the UI tests
cargo fmt --check && cargo clippy --all-targets   # both are clean; keep them that way
YOLO_ANN_BENCH_DIR=dir cargo test --release decode_speed -- --ignored --nocapture   # decode benchmark
```

CI (`.github/workflows/ci.yml`) runs fmt, clippy `-D warnings` and tests on Linux/macOS/Windows. Releases: bump `version` in `Cargo.toml`, commit, then push a matching tag (`git tag v0.2.0 && git push origin v0.2.0`); `.github/workflows/release.yml` builds bare per-platform binaries and publishes the GitHub release. `THIRD-PARTY-LICENSES.txt` (embedded for `--license`) must be regenerated after dependency changes with `cargo about generate about.hbs -o THIRD-PARTY-LICENSES.txt`; CI fails if it is stale.

Dev builds compile all dependencies with `opt-level = 3` (`[profile.dev.package."*"]`) because image decoding is unusably slow otherwise. Edition 2024 (let-chains `if let ... && let ...` are used throughout).

## Architecture

- `main.rs` – clap `Args` → `app::Config`, then `eframe::run_native`. New CLI flags must be threaded through `Config` and documented in the README options table.
- `app.rs` – one large `App` struct holding all UI and session state. `eframe::App::ui` polls the loader, handles keys, draws toolbar/side panel/canvas/status bar, then runs the debounced backup. Sections are separated by `// ---- state / caching / saving` comment banners.
- `model.rs` – pure data + file formats: `BBox`, `ImageAnn` (per-image boxes with undo/redo), YOLO txt parse/format, class files (`classes.txt`, `<id> <name>` `metadata.txt`, Ultralytics-style `data.yaml`), COCO import/export, `write_atomic`, natural sort, label-dir discovery. No egui here.
- `loader.rs` – decoder thread pool. `set_wanted` **replaces** the whole job queue (priority order) whenever the current image changes, so stale prefetches are dropped. Results come back over an mpsc channel and become textures in `App::poll_loader`; `App.cache` is an LRU of `Slot`s evicted down to `ahead + behind + 5`. Images larger than 8192 px are downscaled for display, but the original size is kept for coordinates.
- `backup.rs` – crash backup (`Backup`, one JSON per image folder, rotated to `*.prev.json` on the first write of a session) and persistent `Settings`. `$YOLO_ANN_HOME` overrides the data/config dirs.
- `app/tests.rs` – child module of `app`, so tests read private `App` fields and call private methods (e.g. `import_yolo`, `export_coco`) directly instead of going through `rfd` file dialogs, which can't run headless.

## Key conventions and invariants

- **Box coordinates**: `BBox` stores normalized (0..1) corner coords `x0,y0,x1,y1` and an internal class *index*. Conversion to YOLO center/size happens only in `parse_yolo`/`format_yolo`; conversion between class index and on-disk label id goes through `ClassIds` (`to_index`/`to_id`). Empty `ClassIds` means identity (plain YOLO). Unknown ids map past the class list so they round-trip unchanged. COCO always uses category id = class index + 1.
- **Three distinct "changed" notions** for an image – don't conflate them:
  - `ImageAnn.dirty`: differs from the labels folder on disk; drives Ctrl+S / autosave writes.
  - `ImageAnn::modified()` (`boxes != exported`): real edits since the last YOLO/COCO import or export; drives the quit prompt, the `*N` list marker, the window-title `*`, and what goes into the backup. Call `mark_exported()` after a successful import/export.
  - Presence in `App.anns` = the image "has annotation data" (what exports include and the data filter shows). Created via `ensure_ann`, which only happens on import/restore/edit or when the *user* leaves an image (`go_to(idx, user = true)`). Programmatic jumps must pass `user = false`.
- **Every box mutation** follows: `ann.snapshot()` (pushes undo, clears redo, sets `dirty`) → mutate `ann.boxes` → `self.mark_changed()` (bumps `backup_gen` so the debounced backup fires) → `compute_visible()` if it can affect the list filter.
- **All file writes** (labels, classes, backups, settings) go through `model::write_atomic` (temp file + rename).
- Never overwrite a not-yet-restored backup with an empty one (`pending_restore` guard in `App::write_backup`).
- Labels are only read from explicit imports, `.txt` files next to the images, or the sibling `images`→`labels` folder (`sibling_labels_dir`); don't add other guessing. Auto-picked label folders set `labels_auto` and are not persisted to settings.
- `Settings` / `Backup` fields added later must be `#[serde(default)]` so older JSON files still load (see `settings_from_older_version_still_load`).
- UI tests: build a dataset with `make_dataset(&temp_root("unique_name"))`, create the app with `harness(dir)`, `wait_loaded`, then drive input with `drag` / `click` / `h.key_press` / `h.key_press_modifiers`, using `norm_to_screen` (based on `App.last_img_rect`) to target image coordinates. Each test needs its own `temp_root` name because all tests share one `YOLO_ANN_HOME` per process and backups are keyed by image folder.
