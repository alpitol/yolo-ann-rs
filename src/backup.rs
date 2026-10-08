//! Crash/close backup of unsaved annotations and small persistent settings.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::model::{BBox, write_atomic};

#[derive(Serialize, Deserialize, Default)]
pub struct Backup {
    pub image_dir: PathBuf,
    pub labels_dir: PathBuf,
    pub classes: Vec<String>,
    /// Unix seconds.
    pub saved_at: u64,
    /// Image file name -> boxes, for every image with unsaved changes.
    pub images: BTreeMap<String, Vec<BBox>>,
}

/// `$YOLO_ANN_HOME` (used by tests) overrides the platform data/config dirs.
fn app_dir(kind: &str, platform: Option<PathBuf>) -> PathBuf {
    match std::env::var_os("YOLO_ANN_HOME") {
        Some(home) => PathBuf::from(home).join(kind),
        None => platform.unwrap_or_else(std::env::temp_dir).join("yolo-ann"),
    }
}

fn data_dir() -> PathBuf {
    app_dir("data", dirs::data_local_dir())
}

fn fnv1a(s: &[u8]) -> u64 {
    s.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x100000001b3)
    })
}

/// One backup file per image directory.
pub fn backup_path(image_dir: &Path) -> PathBuf {
    let canon = image_dir
        .canonicalize()
        .unwrap_or_else(|_| image_dir.to_path_buf());
    let tail: Vec<String> = canon
        .components()
        .rev()
        .take(2)
        .map(|c| {
            c.as_os_str().to_string_lossy().replace(
                |c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_',
                "_",
            )
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let name = format!(
        "{}-{:016x}.json",
        tail.join("_"),
        fnv1a(canon.as_os_str().as_encoded_bytes())
    );
    data_dir().join("backups").join(name)
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn load_backup(path: &Path) -> Option<Backup> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// `first_write`: rotate the existing file to `*.prev.json` so that starting a
/// new session never destroys the previous session's backup.
pub fn write_backup(path: &Path, backup: &Backup, first_write: bool) -> Result<()> {
    if first_write && path.is_file() {
        let prev = path.with_extension("prev.json");
        std::fs::copy(path, &prev)
            .with_context(|| format!("rotating backup to {}", prev.display()))?;
    }
    write_atomic(path, serde_json::to_string(backup)?.as_bytes())
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Settings {
    pub last_image_dir: Option<PathBuf>,
    pub last_labels_dir: Option<PathBuf>,
    pub last_classes_file: Option<PathBuf>,
    pub last_image: Option<String>,
    pub autosave_labels: bool,
}

fn settings_path() -> PathBuf {
    app_dir("config", dirs::config_dir()).join("settings.json")
}

impl Settings {
    pub fn load() -> Self {
        std::fs::read_to_string(settings_path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let _ = write_atomic(&settings_path(), s.as_bytes());
        }
    }
}

/// "5 min ago" style formatting.
pub fn ago(secs: u64) -> String {
    let d = now_secs().saturating_sub(secs);
    match d {
        0..=59 => format!("{d} s ago"),
        60..=3599 => format!("{} min ago", d / 60),
        3600..=86399 => format!("{} h ago", d / 3600),
        _ => format!("{} days ago", d / 86400),
    }
}
