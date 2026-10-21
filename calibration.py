"""Lens distortion calibration, from photographs of a checkerboard.

A wide or cheap lens bends straight lines, most towards the edges of the frame
(barrel distortion). Every measurement taken away from the centre is then
wrong, and the error grows with distance from the centre -- so a part measured
in the corner of the frame reads differently from the same part in the middle.
Calibration measures the lens once, and every frame is then corrected before
any tool sees it.

    1. Print a checkerboard and note its square size in millimetres.
    2. Photograph it 8-15 times with the inspection camera, tilted and moved
       around so the corners of the frame are covered -- the distortion is
       strongest there and the calibration can only learn what it sees.
    3. Calibrate. The reprojection error says how well the model fits: under
       0.5 px is good, over 1 px means blurred or mis-detected images.

The reference image is corrected too. Tools compare the live frame with the
reference, and correcting one but not the other would mismatch everywhere --
most of all at the edges, exactly where the correction is largest.
"""
import json
import os

import cv2
import numpy as np

FILE = "calibration.json"
_MAPS = {}


def find_corners(img, pattern):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(g, tuple(pattern), flags)
    if not ok:
        return None
    return cv2.cornerSubPix(g, corners, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3))


def calibrate(images, pattern, square_mm):
    """images: list of BGR images of the board. pattern: inner corners
    (columns, rows). Returns a calibration dict, including which images the
    board could not be found in."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * float(square_mm)
    obj, img_pts, used, rejected, size = [], [], [], [], None
    for i, im in enumerate(images):
        size = (im.shape[1], im.shape[0])
        c = find_corners(im, pattern)
        if c is None:
            rejected.append(i)
            continue
        obj.append(objp)
        img_pts.append(c)
        used.append(i)
    if len(used) < 3:
        raise ValueError("the board was found in only %d image(s); at least 3 are "
                         "needed and 8-15 is better" % len(used))
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img_pts, size, None, None)
    # Scale at the board plane, measured on the corrected corners: average
    # distance between neighbouring corners, against the known square size.
    spans = []
    for c in img_pts:
        u = cv2.undistortPoints(c, K, dist, P=K).reshape(pattern[1], pattern[0], 2)
        spans.append(np.linalg.norm(np.diff(u, axis=1), axis=2).mean())
        spans.append(np.linalg.norm(np.diff(u, axis=0), axis=2).mean())
    mm_per_px = float(square_mm) / float(np.mean(spans))
    return {"camera_matrix": K.tolist(), "dist_coeffs": dist.ravel().tolist(),
            "image_size": list(size), "rms_px": float(rms), "mm_per_px": mm_per_px,
            "pattern": list(pattern), "square_mm": float(square_mm),
            "images_used": used, "images_rejected": rejected}


def save(folder, cal):
    with open(os.path.join(folder, FILE), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)


def load(folder):
    p = os.path.join(folder, FILE)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def undistort(img, cal):
    """Correct one frame. The correction maps are computed once per image size
    and cached -- building them costs far more than applying them."""
    if cal is None or img is None:
        return img
    h, w = img.shape[:2]
    K = np.array(cal["camera_matrix"], np.float64)
    d = np.array(cal["dist_coeffs"], np.float64)
    cw, ch = cal["image_size"]
    if (w, h) != (cw, ch):
        # a different camera resolution than calibrated: scale the intrinsics
        s = np.diag([w / float(cw), h / float(ch), 1.0])
        K = s @ K
    key = (w, h, tuple(np.round(K.ravel(), 6)), tuple(np.round(d, 9)))
    if key not in _MAPS:
        _MAPS[key] = cv2.initUndistortRectifyMap(K, d, None, K, (w, h), cv2.CV_16SC2)
    m1, m2 = _MAPS[key]
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
