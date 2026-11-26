//! What a check is given and what it returns: the same JSON shapes as Vision
//! Station, so recipes, result files and the screen agree.

use crate::geom::{self, Affine, Region};
use crate::img::{Blob, Img};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize)]
pub struct Graphic {
    pub kind: String, // rect | poly | line | point
    pub points: Vec<[f64; 2]>,
    pub status: String,
    pub label: String,
    pub role: String,
}

impl Graphic {
    pub fn poly(points: Vec<[f64; 2]>, status: &str, label: &str) -> Graphic {
        Graphic { kind: "poly".into(), points, status: status.into(), label: label.into(), role: String::new() }
    }
    pub fn line(a: [f64; 2], b: [f64; 2], status: &str, label: &str) -> Graphic {
        Graphic { kind: "line".into(), points: vec![a, b], status: status.into(), label: label.into(), role: String::new() }
    }
    pub fn point(p: [f64; 2], status: &str, label: &str) -> Graphic {
        Graphic { kind: "point".into(), points: vec![p], status: status.into(), label: label.into(), role: String::new() }
    }
    /// A box in patch coordinates, mapped to the image.
    pub fn patch_box(r: &Region, x0: f64, y0: f64, x1: f64, y1: f64, status: &str, label: &str) -> Graphic {
        Graphic::poly(vec![r.map(x0, y0), r.map(x1, y0), r.map(x1, y1), r.map(x0, y1)], status, label)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolResult {
    pub tool_id: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub status: String, // pass | fail | warn | error | skip
    pub score: f64,
    pub values: Map<String, Value>,
    pub graphics: Vec<Graphic>,
    pub message: String,
    pub ms: f64,
    pub caption: Vec<String>,
}

impl ToolResult {
    pub fn new(t: &ToolSpec, status: &str) -> ToolResult {
        ToolResult {
            tool_id: t.id.clone(),
            ty: t.ty.clone(),
            status: status.into(),
            score: 0.0,
            values: Map::new(),
            graphics: vec![],
            message: String::new(),
            ms: 0.0,
            caption: vec![],
        }
    }
    pub fn error(t: &ToolSpec, msg: impl Into<String>) -> ToolResult {
        let mut r = ToolResult::new(t, "error");
        r.message = msg.into();
        r
    }
    pub fn set(&mut self, k: &str, v: impl Into<Value>) -> &mut Self {
        self.values.insert(k.to_string(), v.into());
        self
    }
    pub fn pass_if(&mut self, ok: bool) {
        self.status = if ok { "pass".into() } else { "fail".into() };
    }
    pub fn num(&self, k: &str) -> Option<f64> {
        self.values.get(k).and_then(|v| v.as_f64())
    }
}

/// One check of a recipe.
#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub id: String,
    pub ty: String,
    pub p: Map<String, Value>,
}

impl ToolSpec {
    pub fn from_json(v: &Value) -> ToolSpec {
        ToolSpec {
            id: v.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            ty: v.get("type").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            p: v.get("params").and_then(|x| x.as_object()).cloned().unwrap_or_default(),
        }
    }
    pub fn f(&self, k: &str, d: f64) -> f64 {
        match self.p.get(k) {
            Some(Value::Number(n)) => n.as_f64().unwrap_or(d),
            Some(Value::String(s)) => s.trim().parse().unwrap_or(d),
            Some(Value::Bool(b)) => if *b { 1.0 } else { 0.0 },
            _ => d,
        }
    }
    pub fn i(&self, k: &str, d: i64) -> i64 {
        self.f(k, d as f64).round() as i64
    }
    pub fn b(&self, k: &str, d: bool) -> bool {
        match self.p.get(k) {
            Some(Value::Bool(b)) => *b,
            Some(Value::Number(n)) => n.as_f64().map(|v| v != 0.0).unwrap_or(d),
            Some(Value::String(s)) => matches!(s.as_str(), "true" | "1" | "yes"),
            _ => d,
        }
    }
    pub fn s(&self, k: &str, d: &str) -> String {
        match self.p.get(k) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => d.to_string(),
            Some(v) => v.to_string(),
        }
    }
    pub fn rect(&self, k: &str) -> Option<[f64; 4]> {
        let a = self.p.get(k)?.as_array()?;
        if a.len() < 4 {
            return None;
        }
        let mut r = [0.0; 4];
        for i in 0..4 {
            r[i] = a[i].as_f64()?;
        }
        if r[2] < 1.0 || r[3] < 1.0 {
            return None;
        }
        Some(r)
    }
    pub fn roi(&self) -> Option<[f64; 4]> {
        if self.s("shape", "rect") == "polygon" {
            let pts = self.points();
            if pts.len() >= 3 {
                let x0 = pts.iter().map(|p| p[0]).fold(f64::MAX, f64::min);
                let y0 = pts.iter().map(|p| p[1]).fold(f64::MAX, f64::min);
                let x1 = pts.iter().map(|p| p[0]).fold(f64::MIN, f64::max);
                let y1 = pts.iter().map(|p| p[1]).fold(f64::MIN, f64::max);
                return Some([x0, y0, (x1 - x0).max(1.0), (y1 - y0).max(1.0)]);
            }
        }
        self.rect("roi")
    }
    pub fn points(&self) -> Vec<[f64; 2]> {
        self.p.get("points").and_then(|v| v.as_array()).map(|a| {
            a.iter().filter_map(|p| {
                let q = p.as_array()?;
                Some([q.first()?.as_f64()?, q.get(1)?.as_f64()?])
            }).collect()
        }).unwrap_or_default()
    }
    pub fn angle(&self) -> f64 {
        let a = self.f("angle", 0.0);
        // stored as 357.9 as often as -2.1
        if a > 180.0 { a - 360.0 } else { a }
    }
    pub fn strlist(&self, k: &str) -> Vec<String> {
        match self.p.get(k) {
            Some(Value::Array(a)) => a.iter().map(|v| v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string())).collect(),
            Some(Value::String(s)) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
            _ => vec![],
        }
    }
    pub fn enabled(&self) -> bool {
        self.b("enabled", true)
    }
    pub fn critical(&self) -> bool {
        self.b("critical", true)
    }
}

/// Everything a check can read besides its own picture.
pub struct Ctx {
    pub reference: Option<Arc<Img>>,
    /// reference -> current image, set by Locate part
    pub fixture: Option<Affine>,
    /// test only (VS_TEST_FIXTURES): the place another engine found, used instead of Locate's
    pub fixture_override: Option<Affine>,
    pub mm_per_px: f64,
    pub folder: std::path::PathBuf,
    pub blobs: HashMap<String, (Vec<Blob>, Region)>,
    pub results: HashMap<String, ToolResult>,
    pub images: HashMap<String, Arc<Img>>,
    pub barcode: Option<String>,
    /// the reference turned the same way as a picture a check produced
    pub ref_images: HashMap<String, Arc<Img>>,
    /// every check of the recipe, by id (a fixture from points re-runs some)
    pub specs: Arc<HashMap<String, ToolSpec>>,
    /// points a check found (blob and match centres), image coordinates
    pub points: HashMap<String, Vec<[f64; 2]>>,
}

impl Ctx {
    pub fn new(reference: Option<Arc<Img>>, folder: std::path::PathBuf, mm_per_px: f64) -> Ctx {
        Ctx {
            reference, fixture: None, fixture_override: None, mm_per_px, folder,
            blobs: HashMap::new(), results: HashMap::new(), images: HashMap::new(), barcode: None,
            ref_images: HashMap::new(), specs: Arc::new(HashMap::new()), points: HashMap::new(),
        }
    }

    /// A copy for one check run on its own thread: what it reads, not what others wrote.
    pub fn light(&self) -> Ctx {
        Ctx {
            reference: self.reference.clone(), fixture: self.fixture, fixture_override: None, mm_per_px: self.mm_per_px,
            folder: self.folder.clone(), blobs: HashMap::new(), results: HashMap::new(),
            images: HashMap::new(), barcode: self.barcode.clone(), ref_images: HashMap::new(),
            specs: self.specs.clone(), points: HashMap::new(),
        }
    }

    /// The check's region in the current picture (fixture applied).
    pub fn region(&self, t: &ToolSpec, img: &Img) -> Option<Region> {
        let roi = t.roi()?;
        let mut r = geom::extract(img, roi, t.angle(), self.fixture.as_ref());
        self.shape_mask(t, roi, &mut r);
        Some(r)
    }
    /// The same region in the reference picture (no fixture).
    pub fn ref_region(&self, t: &ToolSpec) -> Option<Region> {
        let roi = t.roi()?;
        let refi = self.reference.as_ref()?;
        let mut r = geom::extract(refi, roi, t.angle(), None);
        self.shape_mask(t, roi, &mut r);
        Some(r)
    }
    fn shape_mask(&self, t: &ToolSpec, roi: [f64; 4], r: &mut Region) {
        match t.s("shape", "rect").as_str() {
            "ellipse" => r.mask = Some(geom::ellipse_mask(r.patch.w, r.patch.h)),
            "polygon" => {
                let pts: Vec<[f64; 2]> = t.points().iter().map(|p| [p[0] - roi[0], p[1] - roi[1]]).collect();
                r.mask = Some(geom::polygon_mask(r.patch.w, r.patch.h, &pts));
            }
            _ => {}
        }
    }
    /// The drawn outline of a check, in the current picture.
    pub fn outline(&self, t: &ToolSpec) -> Option<Vec<[f64; 2]>> {
        let roi = t.roi()?;
        let fx = self.fixture.unwrap_or(Affine::IDENTITY);
        if t.s("shape", "rect") == "polygon" {
            let pts = t.points();
            if pts.len() >= 3 {
                return Some(pts.iter().map(|p| { let (x, y) = fx.apply(p[0], p[1]); [x, y] }).collect());
            }
        }
        let m = fx.then_after(&geom::patch_to_ref(roi, t.angle()));
        let (w, h) = (roi[2], roi[3]);
        if t.s("shape", "rect") == "ellipse" {
            return Some((0..40).map(|i| {
                let a = i as f64 / 40.0 * std::f64::consts::TAU;
                let (x, y) = m.apply(w / 2.0 + w / 2.0 * a.cos(), h / 2.0 + h / 2.0 * a.sin());
                [x, y]
            }).collect());
        }
        Some([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].iter().map(|&(x, y)| { let (a, b) = m.apply(x, y); [a, b] }).collect())
    }
    pub fn mm(&self, t: &ToolSpec) -> f64 {
        let own = t.f("mm_per_px", 0.0);
        if own > 0.0 { own } else { self.mm_per_px }
    }
}

pub fn round(v: f64, n: i32) -> f64 {
    let f = 10f64.powi(n);
    (v * f).round() / f
}

pub fn jnum(v: f64, n: i32) -> Value {
    json!(round(v, n))
}
