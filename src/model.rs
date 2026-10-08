//! Annotation data model plus YOLO / COCO (de)serialization.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp"];

/// Bounding box in normalized image coordinates (0..1), corners.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub class: usize,
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl BBox {
    pub fn normalized(mut self) -> Self {
        if self.x0 > self.x1 {
            std::mem::swap(&mut self.x0, &mut self.x1);
        }
        if self.y0 > self.y1 {
            std::mem::swap(&mut self.y0, &mut self.y1);
        }
        self
    }

    pub fn clamped(self) -> Self {
        let b = self.normalized();
        BBox {
            x0: b.x0.clamp(0.0, 1.0),
            y0: b.y0.clamp(0.0, 1.0),
            x1: b.x1.clamp(0.0, 1.0),
            y1: b.y1.clamp(0.0, 1.0),
            ..b
        }
    }

    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }

    pub fn area(&self) -> f64 {
        (self.x1 - self.x0).abs() * (self.y1 - self.y0).abs()
    }

    /// Pixel rect (x, y, w, h).
    pub fn to_pixels(self, w: u32, h: u32) -> (f64, f64, f64, f64) {
        let (w, h) = (w as f64, h as f64);
        (
            self.x0 * w,
            self.y0 * h,
            (self.x1 - self.x0) * w,
            (self.y1 - self.y0) * h,
        )
    }
}

/// Annotations of a single image held in memory.
#[derive(Clone, Debug, Default)]
pub struct ImageAnn {
    pub boxes: Vec<BBox>,
    /// Not yet written to the YOLO labels folder.
    pub dirty: bool,
    /// Boxes as of the last import or export (YOLO or COCO). Differing boxes are
    /// real edits that would be lost on quit.
    pub exported: Vec<BBox>,
    pub undo: Vec<Vec<BBox>>,
    pub redo: Vec<Vec<BBox>>,
}

impl ImageAnn {
    pub fn clean(boxes: Vec<BBox>) -> Self {
        ImageAnn {
            exported: boxes.clone(),
            boxes,
            ..Default::default()
        }
    }

    pub fn modified(&self) -> bool {
        self.boxes != self.exported
    }

    pub fn mark_exported(&mut self) {
        self.exported = self.boxes.clone();
    }

    /// Call before every mutation.
    pub fn snapshot(&mut self) {
        self.undo.push(self.boxes.clone());
        if self.undo.len() > 200 {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.dirty = true;
    }

    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(std::mem::replace(&mut self.boxes, prev));
                self.dirty = true;
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(std::mem::replace(&mut self.boxes, next));
                self.dirty = true;
                true
            }
            None => false,
        }
    }
}

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Lists image file names of a directory (non-recursive), naturally sorted.
pub fn scan_images(dir: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| !t.is_dir()).unwrap_or(false))
        .filter(|e| is_image(&e.path()))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort_by(|a, b| natural_cmp(a, b));
    Ok(names)
}

/// "img2" < "img10".
pub fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    natural_cmp_ci(a.as_bytes(), b.as_bytes()).then_with(|| a.cmp(b))
}

fn natural_cmp_ci(mut a: &[u8], mut b: &[u8]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    loop {
        match (a.first(), b.first()) {
            (None, None) => return Ordering::Equal,
            (None, _) => return Ordering::Less,
            (_, None) => return Ordering::Greater,
            (Some(ca), Some(cb)) if ca.is_ascii_digit() && cb.is_ascii_digit() => {
                let la = a.iter().take_while(|c| c.is_ascii_digit()).count();
                let lb = b.iter().take_while(|c| c.is_ascii_digit()).count();
                let (na, nb) = (&a[..la], &b[..lb]);
                let ta = trim_zeros(na);
                let tb = trim_zeros(nb);
                let ord = ta
                    .len()
                    .cmp(&tb.len())
                    .then_with(|| ta.cmp(tb))
                    .then_with(|| la.cmp(&lb));
                if ord != Ordering::Equal {
                    return ord;
                }
                a = &a[la..];
                b = &b[lb..];
            }
            (Some(ca), Some(cb)) => {
                let ord = ca.to_ascii_lowercase().cmp(&cb.to_ascii_lowercase());
                if ord != Ordering::Equal {
                    return ord;
                }
                a = &a[1..];
                b = &b[1..];
            }
        }
    }
}

fn trim_zeros(s: &[u8]) -> &[u8] {
    let n = s.iter().take_while(|c| **c == b'0').count();
    &s[n..]
}

pub fn label_path(labels_dir: &Path, image_name: &str) -> PathBuf {
    let stem = Path::new(image_name)
        .file_stem()
        .map(|s| s.to_os_string())
        .unwrap_or_default();
    let mut p = labels_dir.join(stem);
    p.set_extension("txt");
    p
}

/// Reads `<stem>.txt` for every image that has one in `dir`.
/// Returns image name -> boxes, and the number of unreadable files.
pub fn read_yolo_dir(dir: &Path, images: &[String]) -> Result<(HashMap<String, Vec<BBox>>, usize)> {
    let present: std::collections::HashSet<std::ffi::OsString> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .flatten()
        .map(|e| e.file_name())
        .collect();
    let mut out = HashMap::new();
    let mut errors = 0;
    for name in images {
        let p = label_path(dir, name);
        if !p.file_name().is_some_and(|f| present.contains(f)) {
            continue;
        }
        match std::fs::read_to_string(&p) {
            Ok(t) => {
                out.insert(name.clone(), parse_yolo(&t));
            }
            Err(_) => errors += 1,
        }
    }
    Ok((out, errors))
}

pub fn parse_yolo(text: &str) -> Vec<BBox> {
    text.lines()
        .filter_map(|line| {
            let mut it = line.split_whitespace();
            let class: usize = it.next()?.parse::<f64>().ok().map(|c| c as usize)?;
            let v: Vec<f64> = it.take(4).filter_map(|s| s.parse().ok()).collect();
            if v.len() != 4 {
                return None;
            }
            let (cx, cy, w, h) = (v[0], v[1], v[2], v[3]);
            Some(BBox {
                class,
                x0: cx - w / 2.0,
                y0: cy - h / 2.0,
                x1: cx + w / 2.0,
                y1: cy + h / 2.0,
            })
        })
        .collect()
}

pub fn format_yolo(boxes: &[BBox]) -> String {
    let mut s = String::new();
    for b in boxes {
        let b = b.normalized();
        s.push_str(&format!(
            "{} {:.6} {:.6} {:.6} {:.6}\n",
            b.class,
            (b.x0 + b.x1) / 2.0,
            (b.y0 + b.y1) / 2.0,
            b.x1 - b.x0,
            b.y1 - b.y0
        ));
    }
    s
}

/// Write via temp file + rename so a crash never leaves a half-written file.
pub fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, data).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))?;
    Ok(())
}

pub fn read_classes(path: &Path) -> Result<Vec<String>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("yaml") || ext.eq_ignore_ascii_case("yml") {
        return parse_data_yaml(&text).with_context(|| format!("parsing {}", path.display()));
    }
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect())
}

/// Class names from an Ultralytics-style `data.yaml`: `names` is either a list
/// or a `{index: name}` map.
pub fn parse_data_yaml(text: &str) -> Result<Vec<String>> {
    use serde_yaml_ng::Value;
    let doc: Value = serde_yaml_ng::from_str(text)?;
    let names = doc.get("names").context("no `names` in yaml")?;
    let classes: Vec<String> = match names {
        Value::Sequence(seq) => seq.iter().map(yaml_scalar).collect::<Result<_>>()?,
        Value::Mapping(map) => {
            let mut by_idx = std::collections::BTreeMap::new();
            for (k, v) in map {
                let i = match k {
                    Value::Number(n) => n.as_u64(),
                    Value::String(s) => s.trim().parse().ok(),
                    _ => None,
                }
                .with_context(|| format!("class index {k:?} is not a number"))?;
                by_idx.insert(i as usize, yaml_scalar(v)?);
            }
            let n = by_idx.keys().next_back().map_or(0, |&i| i + 1);
            (0..n)
                .map(|i| by_idx.remove(&i).unwrap_or_else(|| format!("class{i}")))
                .collect()
        }
        _ => anyhow::bail!("`names` must be a list or a map"),
    };
    if let Some(nc) = doc.get("nc").and_then(Value::as_u64)
        && nc as usize != classes.len()
    {
        anyhow::bail!("nc is {nc} but {} names are given", classes.len());
    }
    Ok(classes)
}

fn yaml_scalar(v: &serde_yaml_ng::Value) -> Result<String> {
    use serde_yaml_ng::Value;
    Ok(match v {
        Value::String(s) => s.trim().to_string(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => anyhow::bail!("class name {v:?} is not a string"),
    })
}

/// Look for a classes file (classes.txt, obj.names, *.names, data.yaml) directly in `dir`.
pub fn find_classes_file(dir: &Path) -> Option<PathBuf> {
    for name in ["classes.txt", "obj.names", "classes.names"] {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|e| e.to_str()) == Some("names"))
        .or_else(|| {
            ["data.yaml", "dataset.yaml"]
                .iter()
                .map(|n| dir.join(n))
                .find(|p| p.is_file())
        })
}

pub struct CocoImport {
    /// image file name -> boxes
    pub images: HashMap<String, Vec<BBox>>,
    /// Classes from categories (sorted by id); used when no classes are loaded.
    pub categories: Vec<String>,
    pub skipped_unknown_class: usize,
    pub skipped_unknown_image: usize,
}

/// Parses COCO json. `classes` maps category names to class indices; if empty,
/// categories (sorted by id) become the classes.
pub fn import_coco(
    path: &Path,
    classes: &[String],
    image_dir: &Path,
    known_images: &std::collections::HashSet<&str>,
) -> Result<CocoImport> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).context("parsing COCO json")?;

    let mut cats: Vec<(i64, String)> = json["categories"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|c| Some((c["id"].as_i64()?, c["name"].as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default();
    cats.sort();
    let categories: Vec<String> = cats.iter().map(|(_, n)| n.clone()).collect();
    let class_list: &[String] = if classes.is_empty() {
        &categories
    } else {
        classes
    };
    let cat_to_class: HashMap<i64, usize> = cats
        .iter()
        .filter_map(|(id, name)| Some((*id, class_list.iter().position(|c| c == name)?)))
        .collect();

    struct Img {
        name: String,
        w: f64,
        h: f64,
    }
    let mut images: HashMap<i64, Img> = HashMap::new();
    let mut out: HashMap<String, Vec<BBox>> = HashMap::new();
    let mut skipped_unknown_image = 0;
    for im in json["images"].as_array().into_iter().flatten() {
        let (Some(id), Some(name)) = (im["id"].as_i64(), im["file_name"].as_str()) else {
            continue;
        };
        let name = Path::new(name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(name)
            .to_string();
        if !known_images.contains(name.as_str()) {
            skipped_unknown_image += 1;
            continue;
        }
        let (mut w, mut h) = (
            im["width"].as_f64().unwrap_or(0.0),
            im["height"].as_f64().unwrap_or(0.0),
        );
        if (w <= 0.0 || h <= 0.0)
            && let Ok(sz) = imagesize::size(image_dir.join(&name))
        {
            (w, h) = (sz.width as f64, sz.height as f64);
        }
        out.insert(name.clone(), Vec::new());
        images.insert(id, Img { name, w, h });
    }

    let mut skipped_unknown_class = 0;
    for a in json["annotations"].as_array().into_iter().flatten() {
        let Some(img) = a["image_id"].as_i64().and_then(|id| images.get(&id)) else {
            continue;
        };
        let Some(class) = a["category_id"].as_i64().and_then(|c| cat_to_class.get(&c)) else {
            skipped_unknown_class += 1;
            continue;
        };
        let Some(bb) = a["bbox"].as_array() else {
            continue;
        };
        let v: Vec<f64> = bb.iter().filter_map(|x| x.as_f64()).collect();
        if v.len() != 4 || img.w <= 0.0 || img.h <= 0.0 {
            continue;
        }
        out.get_mut(&img.name).unwrap().push(BBox {
            class: *class,
            x0: v[0] / img.w,
            y0: v[1] / img.h,
            x1: (v[0] + v[2]) / img.w,
            y1: (v[1] + v[3]) / img.h,
        });
    }
    Ok(CocoImport {
        images: out,
        categories,
        skipped_unknown_class,
        skipped_unknown_image,
    })
}

/// COCO export in the same layout as the original yolo-ann tool
/// (category id = class index + 1, image id = list index + 1).
pub fn export_coco(
    classes: &[String],
    images: &[(String, u32, u32, Vec<BBox>)],
) -> serde_json::Value {
    use serde_json::json;
    let categories: Vec<_> = classes
        .iter()
        .enumerate()
        .map(|(i, n)| json!({"supercategory": "none", "id": i + 1, "name": n}))
        .collect();
    let mut imgs = Vec::new();
    let mut anns = Vec::new();
    let mut id = 0;
    for (idx, (name, w, h, boxes)) in images.iter().enumerate() {
        imgs.push(json!({"id": idx + 1, "file_name": name, "width": w, "height": h}));
        for b in boxes {
            let (x, y, bw, bh) = b.normalized().to_pixels(*w, *h);
            id += 1;
            anns.push(json!({
                "segmentation": [[x, y, x, y + bh, x + bw, y + bh, x + bw, y]],
                "area": bw * bh,
                "iscrowd": 0,
                "ignore": 0,
                "image_id": idx + 1,
                "bbox": [x, y, bw, bh],
                "category_id": b.class + 1,
                "id": id,
            }));
        }
    }
    json!({"images": imgs, "type": "instances", "annotations": anns, "categories": categories})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yolo_roundtrip() {
        let b = vec![BBox {
            class: 2,
            x0: 0.1,
            y0: 0.2,
            x1: 0.5,
            y1: 0.6,
        }];
        let back = parse_yolo(&format_yolo(&b));
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].class, 2);
        assert!((back[0].x0 - 0.1).abs() < 1e-6 && (back[0].y1 - 0.6).abs() < 1e-6);
    }

    #[test]
    fn yolo_ignores_garbage_and_extra_columns() {
        let v = parse_yolo("0 0.5 0.5 0.2 0.2 0.93\n\nfoo\n1 0.1 0.1\n");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn natural_sort() {
        let mut v = vec!["img10.jpg", "img2.jpg", "img1.jpg", "Img3.jpg"];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, vec!["img1.jpg", "img2.jpg", "Img3.jpg", "img10.jpg"]);
    }

    #[test]
    fn coco_roundtrip() {
        let classes = vec!["a".to_string(), "b".to_string()];
        let b = BBox {
            class: 1,
            x0: 0.25,
            y0: 0.5,
            x1: 0.5,
            y1: 1.0,
        };
        let json = export_coco(&classes, &[("x.jpg".into(), 400, 200, vec![b])]);
        let dir = std::env::temp_dir().join(format!("yolo_ann_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("c.json");
        std::fs::write(&p, json.to_string()).unwrap();
        let known: std::collections::HashSet<&str> = ["x.jpg"].into_iter().collect();
        let imp = import_coco(&p, &classes, &dir, &known).unwrap();
        let got = &imp.images["x.jpg"][0];
        assert_eq!(got.class, 1);
        assert!((got.x0 - 0.25).abs() < 1e-9 && (got.y1 - 1.0).abs() < 1e-9);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn data_yaml_classes() {
        let map = "path: /x\ntrain: images\nnc: 3\nnames:\n  0: seamark\n  2: ship\n  1: boat\n";
        assert_eq!(parse_data_yaml(map).unwrap(), ["seamark", "boat", "ship"]);
        let flow = "names: ['a', b]\n";
        assert_eq!(parse_data_yaml(flow).unwrap(), ["a", "b"]);
        let block = "nc: 2\nnames:\n  - x\n  - y\n";
        assert_eq!(parse_data_yaml(block).unwrap(), ["x", "y"]);
        assert!(parse_data_yaml("nc: 3\nnames: [a, b]\n").is_err());
        assert!(parse_data_yaml("train: images\n").is_err());
    }
}
