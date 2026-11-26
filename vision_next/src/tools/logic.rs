//! Results analysis (an expression), data analysis (statistics over runs) and
//! image processing (a filtered picture later checks can read).

use crate::img::{self, Img, Plane};
use crate::model::{jnum, Ctx, ToolResult, ToolSpec};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

// ------------------------------------------------------------ expressions
#[derive(Clone, Debug, PartialEq)]
pub enum V {
    N(f64),
    S(String),
    B(bool),
}

impl V {
    fn num(&self) -> Result<f64, String> {
        match self {
            V::N(n) => Ok(*n),
            V::B(b) => Ok(if *b { 1.0 } else { 0.0 }),
            V::S(s) => s.parse().map_err(|_| format!("'{}' is not a number", s)),
        }
    }
    fn truth(&self) -> bool {
        match self {
            V::N(n) => *n != 0.0,
            V::B(b) => *b,
            V::S(s) => !s.is_empty() && s != "fail" && s != "error",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Str(String),
    Id(String),
    Op(String),
    LP,
    RP,
    Comma,
}

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = vec![];
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_ascii_digit() || (ch == '.' && i + 1 < c.len() && c[i + 1].is_ascii_digit()) {
            let st = i;
            while i < c.len() && (c[i].is_ascii_digit() || c[i] == '.' || c[i] == 'e') {
                i += 1;
            }
            let t: String = c[st..i].iter().collect();
            out.push(Tok::Num(t.parse().map_err(|_| format!("bad number {}", t))?));
        } else if ch == '\'' || ch == '"' {
            let q = ch;
            i += 1;
            let st = i;
            while i < c.len() && c[i] != q {
                i += 1;
            }
            out.push(Tok::Str(c[st..i.min(c.len())].iter().collect()));
            i += 1;
        } else if ch.is_alphanumeric() || ch == '_' {
            let st = i;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_' || c[i] == '.') {
                i += 1;
            }
            out.push(Tok::Id(c[st..i].iter().collect()));
        } else if ch == '(' {
            out.push(Tok::LP);
            i += 1;
        } else if ch == ')' {
            out.push(Tok::RP);
            i += 1;
        } else if ch == ',' {
            out.push(Tok::Comma);
            i += 1;
        } else {
            let two: String = c[i..(i + 2).min(c.len())].iter().collect();
            if ["==", "!=", "<=", ">=", "&&", "||"].contains(&two.as_str()) {
                out.push(Tok::Op(two));
                i += 2;
            } else if "+-*/<>!=".contains(ch) {
                out.push(Tok::Op(ch.to_string()));
                i += 1;
            } else {
                return Err(format!("unexpected '{}'", ch));
            }
        }
    }
    Ok(out)
}

struct Parser<'a> {
    t: Vec<Tok>,
    i: usize,
    look: &'a dyn Fn(&str) -> Option<V>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn is_op(&self, ops: &[&str]) -> Option<String> {
        match self.peek() {
            Some(Tok::Op(o)) if ops.contains(&o.as_str()) => Some(o.clone()),
            Some(Tok::Id(o)) if ops.contains(&o.as_str()) => Some(o.clone()),
            _ => None,
        }
    }
    fn or(&mut self) -> Result<V, String> {
        let mut l = self.and()?;
        while self.is_op(&["or", "||"]).is_some() {
            self.i += 1;
            let r = self.and()?;
            l = V::B(l.truth() || r.truth());
        }
        Ok(l)
    }
    fn and(&mut self) -> Result<V, String> {
        let mut l = self.not()?;
        while self.is_op(&["and", "&&"]).is_some() {
            self.i += 1;
            let r = self.not()?;
            l = V::B(l.truth() && r.truth());
        }
        Ok(l)
    }
    fn not(&mut self) -> Result<V, String> {
        if self.is_op(&["not", "!"]).is_some() {
            self.i += 1;
            return Ok(V::B(!self.not()?.truth()));
        }
        self.cmp()
    }
    fn cmp(&mut self) -> Result<V, String> {
        let l = self.add()?;
        if let Some(op) = self.is_op(&["==", "!=", "<", "<=", ">", ">=", "="]) {
            self.i += 1;
            let r = self.add()?;
            let res = match (&l, &r) {
                (V::S(a), V::S(b)) => match op.as_str() {
                    "==" | "=" => a == b,
                    "!=" => a != b,
                    _ => return Err("strings can only be compared with == or !=".into()),
                },
                _ => {
                    let (a, b) = (l.num()?, r.num()?);
                    match op.as_str() {
                        "==" | "=" => (a - b).abs() < 1e-9,
                        "!=" => (a - b).abs() >= 1e-9,
                        "<" => a < b,
                        "<=" => a <= b,
                        ">" => a > b,
                        _ => a >= b,
                    }
                }
            };
            return Ok(V::B(res));
        }
        Ok(l)
    }
    fn add(&mut self) -> Result<V, String> {
        let mut l = self.mul()?;
        while let Some(op) = self.is_op(&["+", "-"]) {
            self.i += 1;
            let r = self.mul()?;
            l = V::N(if op == "+" { l.num()? + r.num()? } else { l.num()? - r.num()? });
        }
        Ok(l)
    }
    fn mul(&mut self) -> Result<V, String> {
        let mut l = self.unary()?;
        while let Some(op) = self.is_op(&["*", "/"]) {
            self.i += 1;
            let r = self.unary()?;
            l = V::N(if op == "*" { l.num()? * r.num()? } else { l.num()? / r.num()? });
        }
        Ok(l)
    }
    fn unary(&mut self) -> Result<V, String> {
        if self.is_op(&["-"]).is_some() {
            self.i += 1;
            return Ok(V::N(-self.unary()?.num()?));
        }
        self.primary()
    }
    fn primary(&mut self) -> Result<V, String> {
        let t = self.peek().cloned().ok_or("expression ends too soon")?;
        self.i += 1;
        match t {
            Tok::Num(n) => Ok(V::N(n)),
            Tok::Str(s) => Ok(V::S(s)),
            Tok::LP => {
                let v = self.or()?;
                if self.peek() != Some(&Tok::RP) {
                    return Err("missing )".into());
                }
                self.i += 1;
                Ok(v)
            }
            Tok::Id(id) => {
                match id.as_str() {
                    "true" | "True" => return Ok(V::B(true)),
                    "false" | "False" => return Ok(V::B(false)),
                    _ => {}
                }
                if self.peek() == Some(&Tok::LP) {
                    self.i += 1;
                    let mut args = vec![];
                    while self.peek() != Some(&Tok::RP) {
                        args.push(self.or()?.num()?);
                        if self.peek() == Some(&Tok::Comma) {
                            self.i += 1;
                        } else if self.peek() != Some(&Tok::RP) {
                            return Err("missing ) after arguments".into());
                        }
                    }
                    self.i += 1;
                    let f = |a: &[f64]| a.first().copied().unwrap_or(0.0);
                    return Ok(V::N(match id.as_str() {
                        "abs" => f(&args).abs(),
                        "min" => args.iter().copied().fold(f64::MAX, f64::min),
                        "max" => args.iter().copied().fold(f64::MIN, f64::max),
                        "round" => f(&args).round(),
                        _ => return Err(format!("unknown function {}", id)),
                    }));
                }
                (self.look)(&id).ok_or(format!("'{}' has no value (has that check run?)", id))
            }
            other => Err(format!("unexpected {:?}", other)),
        }
    }
}

pub fn eval(expr: &str, look: &dyn Fn(&str) -> Option<V>) -> Result<V, String> {
    let t = lex(expr)?;
    let mut p = Parser { t, i: 0, look };
    let v = p.or()?;
    if p.i < p.t.len() {
        return Err("extra text after the expression".into());
    }
    Ok(v)
}

/// `tool.key` from earlier results, as Vision Station resolves links.
pub fn lookup(ctx: &Ctx, name: &str) -> Option<V> {
    let (tool, key) = match name.split_once('.') {
        Some((a, b)) => (a, b),
        None => (name, "pass"),
    };
    let r = ctx.results.get(tool)?;
    match key {
        "pass" => Some(V::B(r.status == "pass")),
        "status" => Some(V::S(r.status.clone())),
        "score" => Some(V::N(r.score)),
        k => match r.values.get(k)? {
            Value::Number(n) => n.as_f64().map(V::N),
            Value::String(s) => Some(V::S(s.clone())),
            Value::Bool(b) => Some(V::B(*b)),
            Value::Array(a) => Some(V::N(a.len() as f64)),
            _ => None,
        },
    }
}

pub fn results_analysis(t: &ToolSpec, _f: &Img, ctx: &mut Ctx) -> ToolResult {
    let e = t.s("expression", "");
    if e.trim().is_empty() {
        return ToolResult::error(t, "no expression");
    }
    match eval(&e, &|n| lookup(ctx, n)) {
        Ok(v) => {
            let mut r = ToolResult::new(t, "pass");
            r.pass_if(v.truth());
            if !v.truth() {
                r.message = format!("{} is false", e);
            }
            r.set("result", match v { V::N(n) => Value::from(n), V::S(s) => Value::from(s), V::B(b) => Value::from(b) });
            r
        }
        Err(m) => ToolResult::error(t, m),
    }
}

fn series() -> &'static Mutex<HashMap<String, VecDeque<f64>>> {
    static S: OnceLock<Mutex<HashMap<String, VecDeque<f64>>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn data_analysis(t: &ToolSpec, _f: &Img, ctx: &mut Ctx) -> ToolResult {
    let src = t.s("source", "");
    let Some(v) = lookup(ctx, &src).and_then(|v| v.num().ok()) else {
        return ToolResult::error(t, format!("'{}' has no number", src));
    };
    let key = format!("{}|{}", ctx.folder.display(), t.id);
    let window = t.i("window", 100).max(2) as usize;
    let vals: Vec<f64> = {
        let mut m = series().lock().unwrap();
        let q = m.entry(key).or_default();
        q.push_back(v);
        while q.len() > window {
            q.pop_front();
        }
        q.iter().copied().collect()
    };
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let std = if n > 1.0 { (vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { 0.0 };
    let lo = t.s("lower_limit", "").trim().parse::<f64>().ok();
    let hi = t.s("upper_limit", "").trim().parse::<f64>().ok();
    let mut r = ToolResult::new(t, "pass");
    let mut why = vec![];
    if let Some(l) = lo { if v < l { why.push(format!("{:.4} below {}", v, l)); } }
    if let Some(h) = hi { if v > h { why.push(format!("{:.4} above {}", v, h)); } }
    let cpk = match (lo, hi) {
        _ if std <= 0.0 => None,
        (Some(l), Some(h)) => Some(((h - mean) / (3.0 * std)).min((mean - l) / (3.0 * std))),
        (Some(l), None) => Some((mean - l) / (3.0 * std)),
        (None, Some(h)) => Some((h - mean) / (3.0 * std)),
        _ => None,
    };
    let min_cpk = t.f("min_cpk", 0.0);
    if let Some(c) = cpk {
        r.set("cpk", jnum(c, 3));
        if min_cpk > 0.0 && vals.len() as i64 >= t.i("min_samples", 25) && c < min_cpk {
            why.push(format!("Cpk {:.2} below {:.2}", c, min_cpk));
        }
    }
    r.pass_if(why.is_empty());
    r.message = why.join("; ");
    r.set("value", jnum(v, 4)).set("mean", jnum(mean, 4)).set("std", jnum(std, 4)).set("samples", vals.len() as i64);
    r
}

// --------------------------------------------------------- image process
pub fn image_process(t: &ToolSpec, frame: &Img, ctx: &mut Ctx) -> ToolResult {
    let op = t.s("operator", "smooth");
    let k = (t.i("kernel", 5).max(1) as usize) | 1;
    let amount = t.f("amount", 1.0) as f32;
    let level = t.f("level", 128.0) as f32;
    let iters = t.i("iterations", 1).max(1) as usize;
    // the region (or the whole picture), processed, pasted back into a copy
    let (patch, x0, y0) = match t.roi() {
        Some(r) if ctx.fixture.is_none() && t.angle() == 0.0 => {
            (frame.crop(r[0] as i64, r[1] as i64, r[2] as usize, r[3] as usize), r[0] as i64, r[1] as i64)
        }
        Some(_) => match ctx.region(t, frame) {
            Some(reg) => {
                let (x, y) = reg.to_img.apply(0.0, 0.0);
                (reg.patch, x.round() as i64, y.round() as i64)
            }
            None => (frame.clone(), 0, 0),
        },
        None => (frame.clone(), 0, 0),
    };
    let g = patch.gray();
    let (w, h) = (g.w, g.h);
    let to_img = |p: &Plane| Img { w: p.w, h: p.h, c: 1, data: p.d.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect() };
    let out: Img = match op.as_str() {
        "smooth" => to_img(&Plane::from_gray(&g).box_blur(k / 2)),
        "median" => {
            let r = (k / 2).min(4);
            let mut o = g.clone();
            let mut buf = Vec::with_capacity((2 * r + 1).pow(2));
            for y in 0..h {
                for x in 0..w {
                    buf.clear();
                    for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                        for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                            buf.push(g.data[yy * w + xx]);
                        }
                    }
                    buf.sort_unstable();
                    o.data[y * w + x] = buf[buf.len() / 2];
                }
            }
            o
        }
        "sharpen" => {
            let p = Plane::from_gray(&g);
            let b = p.box_blur(k / 2);
            to_img(&Plane { w, h, d: p.d.iter().zip(&b.d).map(|(a, bl)| a + amount * (a - bl)).collect() })
        }
        "equalize" => {
            let mut hist = [0usize; 256];
            for &v in &g.data { hist[v as usize] += 1; }
            let mut cdf = [0usize; 256];
            let mut run = 0;
            for i in 0..256 { run += hist[i]; cdf[i] = run; }
            let min = cdf.iter().copied().find(|&c| c > 0).unwrap_or(0);
            let n = (w * h).max(1);
            let lut: Vec<u8> = (0..256).map(|i| (((cdf[i] - min) as f64 / (n - min).max(1) as f64) * 255.0).round() as u8).collect();
            Img { w, h, c: 1, data: g.data.iter().map(|&v| lut[v as usize]).collect() }
        }
        "threshold" => {
            let mode = t.s("threshold_mode", "otsu");
            let data: Vec<u8> = if mode == "adaptive" {
                let b = Plane::from_gray(&g).box_blur(k.max(3) * 2);
                g.data.iter().zip(&b.d).map(|(&v, &m)| if v as f32 > m - (level - 128.0) / 10.0 { 255 } else { 0 }).collect()
            } else {
                let thr = if mode == "fixed" { level.clamp(0.0, 255.0) as u8 } else { img::otsu(&g.data) };
                g.data.iter().map(|&v| if v > thr { 255 } else { 0 }).collect()
            };
            Img { w, h, c: 1, data }
        }
        "erode" | "dilate" | "open" | "close" => {
            let mut d = g.data.clone();
            for _ in 0..iters {
                d = match op.as_str() {
                    "erode" => img::morph(&d, w, h, k, false),
                    "dilate" => img::morph(&d, w, h, k, true),
                    "open" => img::open(&d, w, h, k),
                    _ => img::close(&d, w, h, k),
                };
            }
            Img { w, h, c: 1, data: d }
        }
        "edges" => {
            let mut o = Img::new(w, h, 1);
            for y in 1..h.saturating_sub(1) {
                for x in 1..w.saturating_sub(1) {
                    let p = |xx: usize, yy: usize| g.data[yy * w + xx] as f32;
                    let gx = p(x + 1, y - 1) + 2.0 * p(x + 1, y) + p(x + 1, y + 1) - p(x - 1, y - 1) - 2.0 * p(x - 1, y) - p(x - 1, y + 1);
                    let gy = p(x - 1, y + 1) + 2.0 * p(x, y + 1) + p(x + 1, y + 1) - p(x - 1, y - 1) - 2.0 * p(x, y - 1) - p(x + 1, y - 1);
                    o.data[y * w + x] = ((gx * gx + gy * gy).sqrt() / 4.0).min(255.0) as u8;
                }
            }
            o
        }
        "invert" => {
            let mut o = patch.clone();
            for v in o.data.iter_mut() { *v = 255 - *v; }
            o
        }
        "channel" => patch.channel(match t.s("channel", "green").as_str() { "red" => 0, "blue" => 2, _ => 1 }),
        "gain" => {
            let mut o = patch.clone();
            for v in o.data.iter_mut() { *v = ((*v as f32 - 128.0) * amount + level).round().clamp(0.0, 255.0) as u8; }
            o
        }
        other => return ToolResult::error(t, format!("unknown operator {}", other)),
    };
    let mut full = if out.c == 1 && frame.c == 3 { frame.gray() } else { frame.clone() };
    full.paste(&out, x0, y0);
    ctx.images.insert(t.id.clone(), Arc::new(full));
    let mut r = ToolResult::new(t, "pass");
    r.set("operator", op);
    r
}
