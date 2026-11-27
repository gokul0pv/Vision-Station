//! The automatic light level, against a simulated light and camera: grey
//! rises with the level, a white patch burns out above a level.
use std::cell::Cell;
use vision_next::img::Img;
use vision_next::station::{light_auto_level, light_measure, light_next_level};

fn picture(level: u32) -> Img {
    // 64 x 64 grey at level * 0.8 / 255; a bright quarter that clips above level 200
    let (w, h) = (64usize, 64usize);
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let base = level as f64 * 0.8;
            let v = if x < 16 && y < 16 { (level as f64 * 1.3).min(255.0) } else { base };
            for c in 0..3 {
                data[(y * w + x) * 3 + c] = v.round() as u8;
            }
        }
    }
    Img { w, h, c: 3, data }
}

#[test]
fn adjust_now_finds_the_target_grey() {
    let lv = Cell::new(0u32);
    let target = 0.45;
    let (level, mean, clipped, tries) = light_auto_level(|v| { lv.set(v); Ok(()) }, || Ok(picture(lv.get())), target, 0.5).unwrap();
    assert!((mean - target).abs() < 0.03, "mean {} level {} tries {:?}", mean, level, tries);
    assert!(clipped <= 0.5);
    assert_eq!(lv.get(), level, "the light is left at the level chosen");
    assert!(tries.len() <= 11);
}

#[test]
fn adjust_now_never_burns_out_more_than_allowed() {
    let lv = Cell::new(0u32);
    // aiming very bright with almost no white allowed: stays below the clipping level
    let (level, _, clipped, _) = light_auto_level(|v| { lv.set(v); Ok(()) }, || Ok(picture(lv.get())), 0.9, 0.01).unwrap();
    assert!(clipped <= 0.01, "level {} clipped {}", level, clipped);
}

#[test]
fn keep_adjusted_steps() {
    // too dark: up, by at most 35 %
    assert_eq!(light_next_level(100, 0.2, 0.0, 0.45, 0.005), 135);
    // within the dead band: unchanged
    assert_eq!(light_next_level(100, 0.45, 0.0, 0.45, 0.005), 100);
    // burnt out: down 15 %
    assert_eq!(light_next_level(100, 0.45, 0.2, 0.45, 0.005), 85);
    // never below 1 or above 255
    assert_eq!(light_next_level(250, 0.01, 0.0, 0.45, 0.005), 255);
    assert_eq!(light_next_level(1, 0.9, 0.5, 0.45, 0.005), 1);
    let m = light_measure(&picture(100)).unwrap();
    assert!((m.0 - 0.33).abs() < 0.02);
}
