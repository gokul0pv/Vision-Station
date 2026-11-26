//! A read-only SQLite file reader -- enough to read Vision Station's run log
//! (results.db) without a C compiler or the sqlite library: the file header,
//! table b-trees in rowid order (either way), overflow pages, records, and
//! the write-ahead log, so runs Vision Station committed but has not yet
//! copied into the main file are read too. Never writes, never locks.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Val {
    pub fn f64(&self) -> Option<f64> {
        match self {
            Val::Int(i) => Some(*i as f64),
            Val::Real(r) => Some(*r),
            _ => None,
        }
    }
    pub fn i64(&self) -> Option<i64> {
        match self {
            Val::Int(i) => Some(*i),
            Val::Real(r) => Some(*r as i64),
            _ => None,
        }
    }
    pub fn str(&self) -> &str {
        match self {
            Val::Text(s) => s,
            _ => "",
        }
    }
}

/// One row: its values by column name.
pub struct Row<'a> {
    pub rowid: i64,
    cols: &'a HashMap<String, usize>,
    vals: Vec<Val>,
}

static NULL: Val = Val::Null;

impl<'a> Row<'a> {
    pub fn get(&self, name: &str) -> &Val {
        self.cols.get(name).and_then(|&i| self.vals.get(i)).unwrap_or(&NULL)
    }
}

pub struct Db {
    file: File,
    page_size: usize,
    usable: usize,
    pages: u32,
    wal: HashMap<u32, Vec<u8>>,
    tables: HashMap<String, (u32, HashMap<String, usize>, Option<usize>)>,
}

fn be16(b: &[u8], i: usize) -> usize {
    ((b[i] as usize) << 8) | b[i + 1] as usize
}

fn be32(b: &[u8], i: usize) -> u32 {
    u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// A varint at `i`: (value, the index after it).
fn varint(b: &[u8], mut i: usize) -> Option<(u64, usize)> {
    let mut v: u64 = 0;
    for k in 0..9 {
        let c = *b.get(i)? as u64;
        i += 1;
        if k == 8 {
            return Some(((v << 8) | c, i));
        }
        v = (v << 7) | (c & 0x7f);
        if c & 0x80 == 0 {
            return Some((v, i));
        }
    }
    None
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(f, buf, off)
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(f, buf, off)
}

fn read_exact_at(f: &File, buf: &mut [u8], mut off: u64) -> Result<(), String> {
    let mut done = 0;
    while done < buf.len() {
        let n = read_at(f, &mut buf[done..], off).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("the file ended early".into());
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

/// The WAL's running checksum over `d`, from `s`.
fn wal_sum(big: bool, d: &[u8], s: (u32, u32)) -> (u32, u32) {
    let (mut s0, mut s1) = s;
    for c in d.chunks_exact(8) {
        let (x0, x1) = if big {
            (u32::from_be_bytes([c[0], c[1], c[2], c[3]]), u32::from_be_bytes([c[4], c[5], c[6], c[7]]))
        } else {
            (u32::from_le_bytes([c[0], c[1], c[2], c[3]]), u32::from_le_bytes([c[4], c[5], c[6], c[7]]))
        };
        s0 = s0.wrapping_add(x0).wrapping_add(s1);
        s1 = s1.wrapping_add(x1).wrapping_add(s0);
    }
    (s0, s1)
}

/// The committed pages of a write-ahead log, and the database's page count
/// after its last commit. Frames after the last good commit are ignored, as
/// SQLite does.
fn read_wal(path: &Path, page_size: usize) -> (HashMap<u32, Vec<u8>>, Option<u32>) {
    let mut done = HashMap::new();
    let Ok(w) = std::fs::read(path) else { return (done, None) };
    if w.len() < 32 {
        return (done, None);
    }
    let magic = be32(&w, 0);
    if magic != 0x377f0682 && magic != 0x377f0683 {
        return (done, None);
    }
    let big = magic == 0x377f0683;
    let ps = be32(&w, 8) as usize;
    if ps != page_size {
        return (done, None);
    }
    let mut s = wal_sum(big, &w[..24], (0, 0));
    if s != (be32(&w, 24), be32(&w, 28)) {
        return (done, None);
    }
    let mut pending: HashMap<u32, Vec<u8>> = HashMap::new();
    let mut size = None;
    let mut off = 32;
    while off + 24 + ps <= w.len() {
        let fh = &w[off..off + 24];
        if fh[8..16] != w[16..24] {
            break; // a frame of an older log generation
        }
        let data = &w[off + 24..off + 24 + ps];
        s = wal_sum(big, &fh[..8], s);
        s = wal_sum(big, data, s);
        if s != (be32(fh, 16), be32(fh, 20)) {
            break;
        }
        pending.insert(be32(fh, 0), data.to_vec());
        let commit = be32(fh, 4);
        if commit != 0 {
            done.extend(pending.drain());
            size = Some(commit);
        }
        off += 24 + ps;
    }
    (done, size)
}

/// Column names from a CREATE TABLE statement, and which one (if any) is the
/// INTEGER PRIMARY KEY -- stored as the rowid, NULL in the record.
fn columns(sql: &str) -> (HashMap<String, usize>, Option<usize>) {
    let mut cols = HashMap::new();
    let mut ipk = None;
    let (Some(a), Some(b)) = (sql.find('('), sql.rfind(')')) else { return (cols, ipk) };
    if b <= a {
        return (cols, ipk);
    }
    let body = &sql[a + 1..b];
    let mut parts = vec![];
    let (mut depth, mut start) = (0i32, 0usize);
    for (i, ch) in body.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&body[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&body[start..]);
    let mut n = 0;
    for p in parts {
        let p = p.trim();
        let first = p.split_whitespace().next().unwrap_or("");
        let up = first.to_uppercase();
        if p.is_empty() || ["PRIMARY", "UNIQUE", "CHECK", "FOREIGN", "CONSTRAINT"].contains(&up.as_str()) {
            continue;
        }
        let name = first.trim_matches(|c| c == '"' || c == '`' || c == '[' || c == ']' || c == '\'').to_string();
        let rest = p.to_uppercase();
        let words: Vec<&str> = rest.split_whitespace().collect();
        if words.get(1) == Some(&"INTEGER") && rest.contains("PRIMARY KEY") {
            ipk = Some(n);
        }
        cols.insert(name, n);
        n += 1;
    }
    (cols, ipk)
}

fn record(p: &[u8]) -> Vec<Val> {
    let mut out = vec![];
    let Some((hlen, mut i)) = varint(p, 0) else { return out };
    let hlen = hlen as usize;
    let mut types = vec![];
    while i < hlen.min(p.len()) {
        let Some((t, j)) = varint(p, i) else { break };
        types.push(t);
        i = j;
    }
    let mut at = hlen;
    for t in types {
        let int = |n: usize, at: usize| -> Option<i64> {
            let b = p.get(at..at + n)?;
            let mut v: i64 = if b[0] & 0x80 != 0 { -1 } else { 0 };
            for &x in b {
                v = (v << 8) | x as i64;
            }
            Some(v)
        };
        let (v, n) = match t {
            0 => (Val::Null, 0),
            1..=4 => (int(t as usize, at).map(Val::Int).unwrap_or(Val::Null), t as usize),
            5 => (int(6, at).map(Val::Int).unwrap_or(Val::Null), 6),
            6 => (int(8, at).map(Val::Int).unwrap_or(Val::Null), 8),
            7 => (p.get(at..at + 8).map(|b| Val::Real(f64::from_be_bytes(b.try_into().unwrap()))).unwrap_or(Val::Null), 8),
            8 => (Val::Int(0), 0),
            9 => (Val::Int(1), 0),
            t if t >= 12 && t % 2 == 0 => {
                let n = ((t - 12) / 2) as usize;
                (p.get(at..at + n).map(|b| Val::Blob(b.to_vec())).unwrap_or(Val::Null), n)
            }
            t if t >= 13 => {
                let n = ((t - 13) / 2) as usize;
                (p.get(at..at + n).map(|b| Val::Text(String::from_utf8_lossy(b).into_owned())).unwrap_or(Val::Null), n)
            }
            _ => (Val::Null, 0),
        };
        out.push(v);
        at += n;
    }
    out
}

impl Db {
    pub fn open(path: &Path) -> Result<Db, String> {
        let file = File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let mut h = [0u8; 100];
        read_exact_at(&file, &mut h, 0)?;
        if &h[..16] != b"SQLite format 3\0" {
            return Err(format!("{} is not an SQLite database", path.display()));
        }
        let page_size = match be16(&h, 16) { 1 => 65536, n => n };
        if page_size < 512 || !page_size.is_power_of_two() {
            return Err("bad page size".into());
        }
        let usable = page_size - h[20] as usize;
        let len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut pages = (len / page_size as u64) as u32;
        let mut wal_path = path.as_os_str().to_owned();
        wal_path.push("-wal");
        let (wal, size) = read_wal(Path::new(&wal_path), page_size);
        if let Some(s) = size {
            pages = s;
        }
        let mut db = Db { file, page_size, usable, pages, wal, tables: HashMap::new() };
        // the schema: sqlite_master on page 1 (type, name, tbl_name, rootpage, sql)
        let mut found = vec![];
        db.walk(1, true, 0, &mut |_, v| {
            if v.first().map(|x| x.str()) == Some("table") {
                found.push((v.get(1).map(|x| x.str().to_string()).unwrap_or_default(),
                    v.get(3).and_then(|x| x.i64()).unwrap_or(0) as u32,
                    v.get(4).map(|x| x.str().to_string()).unwrap_or_default()));
            }
            true
        })?;
        for (name, root, sql) in found {
            let (cols, ipk) = columns(&sql);
            db.tables.insert(name, (root, cols, ipk));
        }
        Ok(db)
    }

    pub fn has_table(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }

    fn page(&self, n: u32) -> Result<Vec<u8>, String> {
        if n == 0 || n > self.pages {
            return Err(format!("page {} is outside the file", n));
        }
        if let Some(p) = self.wal.get(&n) {
            return Ok(p.clone());
        }
        let mut b = vec![0u8; self.page_size];
        read_exact_at(&self.file, &mut b, (n as u64 - 1) * self.page_size as u64)?;
        Ok(b)
    }

    /// A table leaf cell's whole payload, following overflow pages.
    fn payload(&self, page: &[u8], at: usize) -> Result<(i64, Vec<u8>), String> {
        let bad = || "a damaged cell".to_string();
        let (plen, i) = varint(page, at).ok_or_else(bad)?;
        let (rowid, i) = varint(page, i).ok_or_else(bad)?;
        let p = plen as usize;
        let u = self.usable;
        let x = u - 35;
        let local = if p <= x {
            p
        } else {
            let m = ((u - 12) * 32 / 255) - 23;
            let k = m + ((p - m) % (u - 4));
            if k <= x { k } else { m }
        };
        let mut out = page.get(i..i + local).ok_or_else(bad)?.to_vec();
        if local < p {
            let mut next = be32(page.get(i + local..i + local + 4).ok_or_else(bad)?, 0);
            let mut hops = 0;
            while out.len() < p && next != 0 {
                hops += 1;
                if hops > 1_000_000 {
                    return Err(bad());
                }
                let o = self.page(next)?;
                let take = (p - out.len()).min(u - 4);
                out.extend_from_slice(&o[4..4 + take]);
                next = be32(&o, 0);
            }
        }
        Ok((rowid as i64, out))
    }

    /// Every row of a table b-tree in rowid order (`newest_first`: descending)
    /// until `f` returns false. Returns false when stopped.
    fn walk(&self, pg: u32, newest_first: bool, depth: usize, f: &mut dyn FnMut(i64, Vec<Val>) -> bool) -> Result<bool, String> {
        if depth > 40 {
            return Err("the table is damaged (too deep)".into());
        }
        let page = self.page(pg)?;
        let h = if pg == 1 { 100 } else { 0 };
        let n = be16(&page, h + 3);
        match page[h] {
            0x0d => {
                let cell = |k: usize| be16(&page, h + 8 + 2 * k);
                let order: Box<dyn Iterator<Item = usize>> = if newest_first { Box::new((0..n).rev()) } else { Box::new(0..n) };
                for k in order {
                    let (rowid, p) = self.payload(&page, cell(k))?;
                    if !f(rowid, record(&p)) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            0x05 => {
                let mut kids: Vec<u32> = (0..n).map(|k| be32(&page, be16(&page, h + 12 + 2 * k))).collect();
                kids.push(be32(&page, h + 8));
                if newest_first {
                    kids.reverse();
                }
                for c in kids {
                    if !self.walk(c, newest_first, depth + 1, f)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            t => Err(format!("page {} is not a table page ({})", pg, t)),
        }
    }

    /// The rows of `table`, newest (highest rowid) first, until `f` says stop.
    pub fn rows_newest_first(&self, table: &str, f: &mut dyn FnMut(&Row) -> bool) -> Result<(), String> {
        let (root, cols, ipk) = self.tables.get(table).ok_or_else(|| format!("no table {}", table))?;
        self.walk(*root, true, 0, &mut |rowid, mut vals| {
            if let Some(i) = ipk {
                if let Some(v) = vals.get_mut(*i) {
                    if *v == Val::Null {
                        *v = Val::Int(rowid);
                    }
                }
            }
            f(&Row { rowid, cols, vals })
        })?;
        Ok(())
    }
}
