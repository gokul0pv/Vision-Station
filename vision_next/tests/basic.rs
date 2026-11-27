use vision_next::geom::{self, Affine};
use vision_next::img::{Img, Plane};
use vision_next::ncc;
use vision_next::tools::logic::{eval, V};

fn pattern(w: usize, h: usize) -> Img {
    // pseudo-random, so the cut-out matches in one place only
    let mut im = Img::new(w, h, 1);
    let mut s: u32 = 12345;
    for v in im.data.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (s >> 16) as u8;
    }
    im
}

#[test]
fn template_found_where_cut() {
    let im = pattern(200, 150);
    let tpl = im.crop(63, 41, 40, 30);
    let m = ncc::match_template(&Plane::from_gray(&im), &Plane::from_gray(&tpl));
    let (x, y, s) = m.best();
    assert_eq!((x, y), (63, 41));
    assert!(s > 0.999);
}

#[test]
fn affine_inverse_round_trip() {
    let a = Affine::rotate_about(12.5, 40.0, 30.0).then_after(&Affine::translate(5.0, -3.0));
    let (x, y) = a.apply(17.0, 9.0);
    let (bx, by) = a.invert().apply(x, y);
    assert!((bx - 17.0).abs() < 1e-9 && (by - 9.0).abs() < 1e-9);
}

#[test]
fn region_extract_is_a_crop_when_upright() {
    let im = pattern(100, 80);
    let r = geom::extract(&im, [10.0, 20.0, 30.0, 15.0], 0.0, None);
    assert_eq!(r.patch.data, im.crop(10, 20, 30, 15).data);
}

#[test]
fn expressions() {
    let look = |n: &str| match n {
        "a.count" => Some(V::N(4.0)),
        "b" => Some(V::B(true)),
        "c.status" => Some(V::S("pass".into())),
        _ => None,
    };
    assert_eq!(eval("a.count == 4 and b", &look), Ok(V::B(true)));
    assert_eq!(eval("(a.count + 1) * 2 > 9 && c.status == 'pass'", &look), Ok(V::B(true)));
    assert_eq!(eval("not b or a.count < 3", &look), Ok(V::B(false)));
    assert!(eval("missing.x > 1", &look).is_err());
}

#[test]
fn result_file_day() {
    let off = vision_next::engine::utc_offset();
    assert_eq!(vision_next::engine::day(0.0 - off), "1970-01-01");
    assert_eq!(vision_next::engine::day(1_795_000_000.0 - off), "2026-11-18");
}
