//! NumPy .npz files (a zip of .npy arrays): read Vision Station's trained
//! models, write ours in the same form so either program opens them. Also
//! the plain zip reading and writing used by packages.

use std::collections::HashMap;
use std::path::Path;

pub struct Array {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Array {
    fn each<T>(&self, f: impl Fn(&[u8]) -> T) -> Vec<T> {
        let n = match self.dtype.as_str() {
            "<f4" | "<i4" | "<u4" => 4,
            "<f8" | "<i8" | "<u8" => 8,
            "<i2" | "<u2" => 2,
            _ => 1,
        };
        self.data.chunks_exact(n).map(f).collect()
    }
    pub fn as_f64(&self) -> Vec<f64> {
        match self.dtype.as_str() {
            "<f4" => self.each(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64),
            "<f8" => self.each(|b| f64::from_le_bytes(b.try_into().unwrap())),
            "<i8" => self.each(|b| i64::from_le_bytes(b.try_into().unwrap()) as f64),
            "<u8" => self.each(|b| u64::from_le_bytes(b.try_into().unwrap()) as f64),
            "<i4" => self.each(|b| i32::from_le_bytes(b.try_into().unwrap()) as f64),
            "<u4" => self.each(|b| u32::from_le_bytes(b.try_into().unwrap()) as f64),
            "<i2" => self.each(|b| i16::from_le_bytes(b.try_into().unwrap()) as f64),
            "<u2" => self.each(|b| u16::from_le_bytes(b.try_into().unwrap()) as f64),
            "|b1" | "|u1" => self.data.iter().map(|&v| v as f64).collect(),
            "|i1" => self.data.iter().map(|&v| v as i8 as f64).collect(),
            _ => vec![],
        }
    }
    pub fn as_f32(&self) -> Vec<f32> {
        if self.dtype == "<f4" {
            return self.each(|b| f32::from_le_bytes(b.try_into().unwrap()));
        }
        self.as_f64().into_iter().map(|v| v as f32).collect()
    }
}

fn parse_npy(b: &[u8]) -> Result<Array, String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("not a .npy array".into());
    }
    let (hlen, start) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let header = String::from_utf8_lossy(&b[start..start + hlen]).to_string();
    let field = |k: &str| -> String {
        let i = header.find(&format!("'{}'", k)).map(|i| i + k.len() + 3).unwrap_or(0);
        header[i..].trim_start_matches([':', ' ']).to_string()
    };
    let descr = field("descr");
    let descr = descr.trim_start_matches('\'').split('\'').next().unwrap_or("").to_string();
    if field("fortran_order").starts_with("True") {
        return Err("Fortran-ordered arrays are not read".into());
    }
    let sh = field("shape");
    let sh = sh.trim_start_matches('(').split(')').next().unwrap_or("");
    let shape: Vec<usize> = sh.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    Ok(Array { dtype: descr, shape, data: b[start + hlen..].to_vec() })
}

fn u16le(b: &[u8], i: usize) -> usize { u16::from_le_bytes([b[i], b[i + 1]]) as usize }
fn u32le(b: &[u8], i: usize) -> usize { u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize }

/// Every file in a zip archive: name -> bytes (stored or deflated).
pub fn unzip(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    // the central directory names every entry with its true sizes
    let eocd = (0..bytes.len().saturating_sub(21)).rev().find(|&i| bytes[i..].starts_with(b"PK\x05\x06")).ok_or("not a zip file")?;
    let count = u16le(bytes, eocd + 10);
    let mut at = u32le(bytes, eocd + 16);
    let mut out = vec![];
    for _ in 0..count {
        if !bytes[at..].starts_with(b"PK\x01\x02") {
            return Err("damaged zip directory".into());
        }
        let method = u16le(bytes, at + 10);
        let csize = u32le(bytes, at + 20);
        let nlen = u16le(bytes, at + 28);
        let elen = u16le(bytes, at + 30);
        let clen = u16le(bytes, at + 32);
        let local = u32le(bytes, at + 42);
        let name = String::from_utf8_lossy(&bytes[at + 46..at + 46 + nlen]).to_string();
        at += 46 + nlen + elen + clen;
        let lstart = local + 30 + u16le(bytes, local + 26) + u16le(bytes, local + 28);
        let raw = &bytes[lstart..lstart + csize];
        let data = match method {
            0 => raw.to_vec(),
            8 => miniz_oxide::inflate::decompress_to_vec(raw).map_err(|e| format!("{}: {:?}", name, e))?,
            m => return Err(format!("{}: compression {} not read", name, m)),
        };
        out.push((name, data));
    }
    Ok(out)
}

pub fn read(path: &Path) -> Result<HashMap<String, Array>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut out = HashMap::new();
    for (name, data) in unzip(&bytes)? {
        let key = name.trim_end_matches(".npy").to_string();
        out.insert(key, parse_npy(&data)?);
    }
    Ok(out)
}

fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
            }
            *v = c;
        }
        t
    });
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// A zip archive, deflated, built in memory.
#[derive(Default)]
pub struct Zip {
    files: Vec<(String, Vec<u8>)>,
}

impl Zip {
    pub fn add(&mut self, name: &str, data: Vec<u8>) {
        self.files.push((name.replace('\\', "/"), data));
    }
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = vec![];
        let mut dir = vec![];
        for (name, data) in &self.files {
            let comp = miniz_oxide::deflate::compress_to_vec(data, 6);
            let (method, body) = if comp.len() < data.len() { (8u16, comp) } else { (0u16, data.clone()) };
            let crc = crc32(data);
            let off = out.len() as u32;
            let n = name.as_bytes();
            let mut h = vec![];
            h.extend_from_slice(b"PK\x03\x04");
            h.extend_from_slice(&20u16.to_le_bytes());
            h.extend_from_slice(&0x0800u16.to_le_bytes()); // UTF-8 names
            h.extend_from_slice(&method.to_le_bytes());
            h.extend_from_slice(&[0, 0, 0x21, 0]); // time, date
            h.extend_from_slice(&crc.to_le_bytes());
            h.extend_from_slice(&(body.len() as u32).to_le_bytes());
            h.extend_from_slice(&(data.len() as u32).to_le_bytes());
            h.extend_from_slice(&(n.len() as u16).to_le_bytes());
            h.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&h);
            out.extend_from_slice(n);
            out.extend_from_slice(&body);
            let mut c = vec![];
            c.extend_from_slice(b"PK\x01\x02");
            c.extend_from_slice(&20u16.to_le_bytes());
            c.extend_from_slice(&h[4..]); // version needed .. extra length
            c.extend_from_slice(&[0u8; 10]); // comment length, disk, internal and external attributes
            c.extend_from_slice(&off.to_le_bytes());
            c.extend_from_slice(n);
            dir.extend_from_slice(&c);
        }
        let dstart = out.len() as u32;
        out.extend_from_slice(&dir);
        out.extend_from_slice(b"PK\x05\x06\0\0\0\0");
        out.extend_from_slice(&(self.files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(self.files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(dir.len() as u32).to_le_bytes());
        out.extend_from_slice(&dstart.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }
}

/// Arrays to write as .npz.
#[derive(Default)]
pub struct Npz {
    zip: Zip,
}

fn npy(descr: &str, shape: &[usize], data: Vec<u8>) -> Vec<u8> {
    let shape_s = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", descr, shape_s);
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out.extend_from_slice(&data);
    out
}

impl Npz {
    pub fn new() -> Npz {
        Npz::default()
    }
    pub fn f32(&mut self, name: &str, v: &[f32], shape: &[usize]) {
        self.zip.add(&format!("{}.npy", name), npy("<f4", shape, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }
    pub fn f64(&mut self, name: &str, v: &[f64], shape: &[usize]) {
        self.zip.add(&format!("{}.npy", name), npy("<f8", shape, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }
    pub fn i64(&mut self, name: &str, v: &[i64], shape: &[usize]) {
        self.zip.add(&format!("{}.npy", name), npy("<i8", shape, v.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }
    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.zip.bytes()).map_err(|e| format!("{}: {}", path.display(), e))
    }
}
