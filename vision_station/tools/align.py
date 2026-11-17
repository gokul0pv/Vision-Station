"""Fixturing. Everything downstream depends on this being right — if the part
shifts and the fixture is wrong, every ROI lands in the wrong place and you get
false rejects that destroy operator trust. Two strategies:

  method="template"  fast, translation + coarse rotation, needs a distinct fiducial
  method="features"  SIFT + RANSAC, survives bigger pose change and lighting
  method="auto"      features first, template when the region has too little
                     detail for features -- the default for a new check
"""
import time
import cv2
import numpy as np
from .base import Tool, ToolResult, Graphic, register


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


@register("align")
class Align(Tool):
    def run(self, image, ctx):
        t0 = time.perf_counter()
        ref = ctx.reference
        if ref is None:
            return ToolResult(self.id, self.type_name, "error",
                              message="no reference image loaded for this project",
                              ms=(time.perf_counter() - t0) * 1000)

        method = self.p.get("method", "template")
        min_score = float(self.p.get("min_score", 0.6))
        self._last_margin = None
        self._last_inliers = 0
        try:
            if method == "auto":
                # Template matching alone, the old default, could not follow a
                # part that moved further than the region's own margin or
                # turned past the angle range: on a real board it reported
                # "no match above 0.60 (best 0.34)" for a part plainly in
                # view. Feature matching follows any shift and rotation;
                # template matching remains for a region too plain for it.
                rg, ig = _gray(ref), _gray(image)
                M, score = self._features(rg, ig)
                method = "features"
                if M is None or score < min_score:
                    inl = self._last_inliers
                    M2, score2 = self._template(rg, ig)
                    if M2 is not None and score2 > (score if M is not None else -1.0):
                        M, score, method = M2, score2, "template"
                    else:
                        self._last_margin = None
                        self._last_inliers = inl
            elif method == "features":
                M, score = self._features(_gray(ref), _gray(image))
            else:
                M, score = self._template(_gray(ref), _gray(image))
        except Exception as e:
            return ToolResult(self.id, self.type_name, "error", message=str(e),
                              ms=(time.perf_counter() - t0) * 1000)

        ok = M is not None and score >= min_score
        if M is None and method == "features" and getattr(self, "_last_inliers", 0) < \
                int(self.p.get("min_inliers", 12)):
            why = ("part not found - only %d matching points agree on a position (needs %d). "
                   "Move or enlarge the region onto a part of the product with more detail."
                   % (getattr(self, "_last_inliers", 0), int(self.p.get("min_inliers", 12))))
        else:
            why = "" if ok else "no match above %.2f (best %.2f)" % (min_score, score)

        # Refuse an ambiguous match even when it scores well.
        #
        # The score says how well the feature matched; it says nothing about
        # whether the same feature appears elsewhere. On a real chassis a small
        # fixture region returned a confidently wrong pose at three of five
        # offsets, scoring 0.64 to 0.68 against a 0.45 threshold, because the
        # region matched a different part of the panel. Publishing that pose
        # silently moves every downstream region onto the wrong place, which is
        # far worse than failing: the operator sees plausible results that are
        # about something else.
        margin = getattr(self, "_last_margin", None)
        min_margin = float(self.p.get("min_uniqueness", 0.06))
        if ok and margin is not None and margin < min_margin:
            ok = False
            why = ("match is not unique - the fixture feature appears more than "
                   "once (margin %.3f, needs %.3f). Enlarge the region or move "
                   "it onto something that appears only once on the part."
                   % (margin, min_margin))

        if ok:
            ctx.fixture = M                      # <- every later tool uses this

        dx = float(M[0, 2]) if M is not None else 0.0
        dy = float(M[1, 2]) if M is not None else 0.0
        ang = float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))) if M is not None else 0.0

        g = []
        if M is not None:
            h, w = ref.shape[:2]
            box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            box = cv2.transform(box, M[:2])
            g = [Graphic("poly", [(float(p[0][0]), float(p[0][1])) for p in box],
                         "pass" if ok else "fail", "fixture")]

        return ToolResult(
            self.id, self.type_name, "pass" if ok else "fail", score,
            {"shift_x_px": round(dx, 2), "shift_y_px": round(dy, 2),
             "angle_deg": round(ang, 3),
             "uniqueness": (None if margin is None else round(margin, 3)),
             "inliers": (getattr(self, "_last_inliers", None) if method == "features" else None),
             "method": method,
             # the full transform, reference -> this image: displays use it to
             # read the pointer in the part's own coordinates
             "matrix": ([[round(float(v), 6) for v in row] for row in M[:2]]
                        if ok and M is not None else None)},
            g, why, (time.perf_counter() - t0) * 1000)

    # -- strategies ------------------------------------------------------
    def _template(self, ref_g, img_g):
        """Rotated template matching, coarse to fine, with subpixel refinement.

        The previous version built its rotation matrix about the image origin
        rather than about the template centre. Translation-only cases looked
        perfect, so it passed casual testing, but any rotation placed every
        downstream ROI badly wrong while still reporting a high match score --
        measured at 147 px of corner error at 9 degrees with a score of 0.967.
        A confidently wrong fixture is worse than an honest failure, because
        nothing downstream can tell it happened.

        The correct composition is: rotate about the template centre in
        reference coordinates, then translate that centre to where the match
        was found.
        """
        H, W = ref_g.shape[:2]
        x, y, w, h = self.roi or [0, 0, W, H]
        x, y = max(0, int(x)), max(0, int(y))
        w, h = max(8, int(min(w, W - x))), max(8, int(min(h, H - y)))
        tpl = ref_g[y:y + h, x:x + w]

        rng = self.p.get("angle_range", [0, 0]) or [0, 0]
        lo, hi = float(rng[0]), float(rng[1])
        fine_step = float(self.p.get("angle_step", 1.0)) or 1.0

        if abs(hi - lo) < 1e-9:
            cands = [0.0] if lo == 0 else [lo]
            best = self._search(tpl, img_g, cands)
        elif self.p.get("pyramid", True):
            # Coarse angle search on a half-size pair. Correlation cost scales
            # with area, so this is roughly four times cheaper per angle, and a
            # coarse search does not need full resolution to pick the right
            # neighbourhood.
            sc = 0.5
            tpl_s = cv2.resize(tpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            img_s = cv2.resize(img_g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            coarse_step = max(fine_step, (hi - lo) / 16.0)
            coarse = list(np.arange(lo, hi + 1e-9, coarse_step))
            if min(tpl_s.shape[:2]) >= 8:
                c = self._search(tpl_s, img_s, coarse)
                a0 = c[2] if c[0] is not None else 0.0
            else:
                a0 = 0.0
            fine = [a for a in np.arange(a0 - coarse_step, a0 + coarse_step + 1e-9,
                                         fine_step) if lo - 1e-9 <= a <= hi + 1e-9]
            best = self._search(tpl, img_g, fine or [a0])
        else:
            # Coarse pass over the range, then refine around the winner. A flat
            # sweep at the fine step costs one full correlation per angle and
            # was the reason this method ran ten times slower than features.
            coarse_step = max(fine_step, (hi - lo) / 16.0)
            coarse = list(np.arange(lo, hi + 1e-9, coarse_step))
            best = self._search(tpl, img_g, coarse)
            if best[0] is not None:
                a0 = best[2]
                fine = [a for a in np.arange(a0 - coarse_step, a0 + coarse_step + 1e-9,
                                             fine_step) if lo - 1e-9 <= a <= hi + 1e-9]
                cand = self._search(tpl, img_g, fine)
                if cand[1] > best[1]:
                    best = cand

        loc, score, angle = best
        if loc is None:
            return None, 0.0
        self._last_margin = getattr(self, "_margin", None)

        cx_ref, cy_ref = x + w / 2.0, y + h / 2.0
        cx_cur, cy_cur = loc[0] + w / 2.0, loc[1] + h / 2.0
        # Sign matters and is easy to get wrong. cv2.getRotationMatrix2D, used
        # to rotate the template, yields [[cos, sin], [-sin, cos]] -- the
        # transpose of the textbook counter-clockwise matrix. Writing the
        # textbook form here inverts the recovered rotation, which produced an
        # error of exactly twice the true angle: about 9.7 px of corner error
        # per degree. Refinement hid it completely, so it only became visible
        # once the unrefined estimate was tested on its own.
        rad = np.radians(angle)
        ca, sa = np.cos(rad), np.sin(rad)
        # T(c_cur) . R_cv(angle) . T(-c_ref)
        M = np.float32([
            [ca,  sa, cx_cur - (ca * cx_ref + sa * cy_ref)],
            [-sa, ca, cy_cur - (-sa * cx_ref + ca * cy_ref)],
            [0.0, 0.0, 1.0]])

        if self.p.get("refine", True):
            refined = self._refine_ecc(ref_g, img_g, M)
            if refined is not None:          # `refined or M` raises on arrays
                # Keep whichever transform actually explains the image better.
                # A distance-based gate was tried first and was wrong: the raw
                # template seed can be tens of pixels out on a rotated part, so
                # capping how far refinement may move it rejected exactly the
                # corrections that were working. Measured quality is the only
                # honest test.
                if self._fit(ref_g, img_g, refined) >= self._fit(ref_g, img_g, M):
                    M = refined
        return M, float(score)

    @staticmethod
    def _fit(ref_g, img_g, M, size=240):
        """Normalised correlation between the reference warped by M and the
        live image. Downscaled, because this is a comparison, not a measurement."""
        try:
            h, w = img_g.shape[:2]
            sc = min(1.0, size / float(max(h, w)))
            A = np.array(M[:2], dtype=np.float32).copy()
            A[0, 2] *= sc
            A[1, 2] *= sc
            rs = cv2.resize(ref_g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            isg = cv2.resize(img_g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            warped = cv2.warpAffine(rs, A, (isg.shape[1], isg.shape[0]),
                                    flags=cv2.INTER_LINEAR)
            m = (warped > 0).astype(np.uint8)
            if m.sum() < 50:
                return -1.0
            a = warped[m > 0].astype(np.float32)
            b = isg[m > 0].astype(np.float32)
            a -= a.mean(); b -= b.mean()
            d = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / d) if d > 1e-9 else -1.0
        except cv2.error:
            return -1.0

    def _search(self, tpl, img_g, angles):
        """Best (location, score, angle) over a list of candidate angles.

        Also records how unique that best match was: `self._margin` is the gap
        between the best correlation peak and the strongest rival peak
        elsewhere in the image. A fixture feature that appears twice produces a
        small margin, and that is the failure worth catching -- on a real
        chassis, aligning on a 190 px brass insert returned a confidently wrong
        pose at three of five offsets, scoring 0.64 to 0.68 against a 0.45
        threshold. Nothing in the score revealed it; the margin does.
        """
        h, w = tpl.shape[:2]
        best_loc, best_score, best_ang = None, -1.0, 0.0
        self._margin = None
        for a in angles:
            if abs(a) < 1e-9:
                t = tpl
            else:
                R = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(a), 1.0)
                t = cv2.warpAffine(tpl, R, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            if img_g.shape[0] < h or img_g.shape[1] < w:
                continue
            res = cv2.matchTemplate(img_g, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > best_score:
                best_score, best_ang = float(mx), float(a)
                best_loc = self._subpixel(res, loc)
                # Strongest rival peak, ignoring a neighbourhood around the
                # winner, so a broad peak is not mistaken for a second match.
                masked = res.copy()
                x0 = max(0, loc[0] - w // 2)
                y0 = max(0, loc[1] - h // 2)
                masked[y0:loc[1] + h // 2, x0:loc[0] + w // 2] = -1.0
                rival = float(masked.max()) if masked.size else -1.0
                self._margin = float(mx) - rival
        return best_loc, best_score, best_ang

    @staticmethod
    def _subpixel(res, loc):
        """Parabolic fit around the correlation peak. Integer-pixel fixturing
        quantises every downstream ROI to whole pixels, which shows up as
        measurement jitter of up to half a pixel for free."""
        px, py = loc
        h, w = res.shape[:2]
        fx, fy = float(px), float(py)
        if 0 < px < w - 1:
            l, c, r = res[py, px - 1], res[py, px], res[py, px + 1]
            d = l - 2 * c + r
            if abs(d) > 1e-12:
                fx = px + float(np.clip(0.5 * (l - r) / d, -1.0, 1.0))
        if 0 < py < h - 1:
            u, c, dn = res[py - 1, px], res[py, px], res[py + 1, px]
            d = u - 2 * c + dn
            if abs(d) > 1e-12:
                fy = py + float(np.clip(0.5 * (u - dn) / d, -1.0, 1.0))
        return (fx, fy)

    def _refine_ecc(self, ref_g, img_g, M):
        """Polish the estimate with intensity-based alignment.

        Runs on a downscaled pair: ECC at full resolution dominated the cycle
        time (measured 841 ms against 231 ms for the unrefined version) and buys
        nothing, because the starting estimate is already within a pixel or two
        and ECC converges on shape, not on fine detail.

        The result is only accepted if it does not disagree wildly with the
        estimate it started from. ECC can converge into a neighbouring local
        optimum, and silently swapping in a worse answer is exactly the failure
        this whole tool is supposed to prevent."""
        try:
            target = float(self.p.get("refine_size", 420))
            scale = min(1.0, target / max(ref_g.shape[:2]))
            warp = np.ascontiguousarray(M[:2].astype(np.float32)).copy()
            if scale < 1.0:
                r = cv2.resize(ref_g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                i = cv2.resize(img_g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                warp[0, 2] *= scale
                warp[1, 2] *= scale
            else:
                r, i = ref_g, img_g
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(self.p.get("refine_iters", 30)), 1e-5)
            cv2.findTransformECC(r, i, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
            if scale < 1.0:
                warp[0, 2] /= scale
                warp[1, 2] /= scale
            out = np.vstack([warp, [0, 0, 1]]).astype(np.float32)
            if not np.all(np.isfinite(out)):
                return None
            return out
        except cv2.error:
            return None

    _ref_cache = {}

    def _feature_windows(self, shape):
        """(reference window, search window) as (x0, y0, x1, y1).

        Reference features come from the fixture region only; the live image
        is searched in that region plus a margin for how far the part may move.
        Matching whole images let the static background -- robot, conveyor,
        cell -- outvote a small part: on a real chassis with the region a third
        of the frame, every movement came back as 'nothing moved', and every
        tool after the fixture inspected the wrong place.

        Windows are cropped rather than masked: a mask only discards keypoints
        after the detector has built its pyramid over the whole frame."""
        H, W = shape[:2]
        if not self.roi:
            return (0, 0, W, H), (0, 0, W, H)
        x, y, w, h = [int(v) for v in self.roi]
        pad = self.p.get("search_pad_px")
        pad = int(pad) if pad else int(0.35 * max(w, h))
        ref_w = (max(0, x), max(0, y), min(W, x + w), min(H, y + h))
        img_w = (max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad))
        return ref_w, img_w

    @staticmethod
    def _detect(orb, gray, win):
        x0, y0, x1, y1 = win
        kps, des = orb.detectAndCompute(np.ascontiguousarray(gray[y0:y1, x0:x1]), None)
        pts = [(k.pt[0] + x0, k.pt[1] + y0) for k in kps]      # back to image coordinates
        return pts, des

    @staticmethod
    def _detector(n):
        """SIFT when the OpenCV build has it (every opencv-python since 4.4),
        ORB otherwise. SIFT matched a softer, differently exposed photograph of
        the same chassis with 800 consistent points where ORB found 60."""
        try:
            return cv2.SIFT_create(n), cv2.NORM_L2
        except AttributeError:
            return cv2.ORB_create(n), cv2.NORM_HAMMING

    def _fit_roi(self, ref_g, img_g, M, size=640):
        """How well the located part matches the reference: normalised
        correlation over the fixture region, with the live image warped back
        into reference coordinates by M. Downscaled -- a comparison, not a
        measurement. The static background outside the region plays no part,
        which is what made a whole-frame correlation useless for this."""
        H, W = ref_g.shape[:2]
        x, y, w, h = [int(v) for v in (self.roi or [0, 0, W, H])]
        x, y = max(0, x), max(0, y)
        w, h = max(8, min(w, W - x)), max(8, min(h, H - y))
        sc = min(1.0, size / float(max(w, h)))
        try:
            Minv = cv2.invertAffineTransform(np.asarray(M[:2], np.float64))
            # reference ROI -> image, then shrink: compose with a scale about the ROI origin
            S = np.array([[sc, 0, -x * sc], [0, sc, -y * sc], [0, 0, 1]], np.float64)
            A = (S @ np.vstack([Minv, [0, 0, 1]]))[:2]
            back = cv2.warpAffine(img_g, A, (int(w * sc), int(h * sc)), flags=cv2.INTER_AREA)
            ref = cv2.resize(ref_g[y:y + h, x:x + w], (int(w * sc), int(h * sc)),
                             interpolation=cv2.INTER_AREA)
            a = ref.astype(np.float32) - float(ref.mean())
            b = back.astype(np.float32) - float(back.mean())
            d = float(np.linalg.norm(a) * np.linalg.norm(b))
            return max(0.0, float((a * b).sum() / d)) if d > 1e-9 else 0.0
        except cv2.error:
            return 0.0

    def _features(self, ref_g, img_g):
        """Keypoint matching, verified.

        Detection runs on the fixture window downscaled to `feature_px` on its
        long side (1400 by default): on a 12-megapixel photograph that is
        five times faster and loses nothing a pose estimate needs. Matches
        pass Lowe's ratio test, then RANSAC fits a similarity transform.

        The score is the correlation of the located part with the reference
        (`_fit_roi`), not the inlier ratio. The ratio measured how many
        keypoints agreed, which collapses when a photograph is softer or the
        labels on the part differ -- on a real chassis the faulty unit scored
        0.07 with the part found exactly, and was rejected by the fixture
        rather than by the check that should have named the defect. A pose
        with `min_inliers` consistent points (12) and a region that correlates
        is a located part; the inlier count and ratio are still reported.
        ECC then polishes the estimate, as the template method does."""
        n = int(self.p.get("n_features", 2000))
        det, norm = self._detector(max(n, 1000))
        ref_w, img_w = self._feature_windows(ref_g.shape)
        long_side = max(img_w[2] - img_w[0], img_w[3] - img_w[1])
        sc = min(1.0, float(self.p.get("feature_px", 1400)) / max(1, long_side))

        def detect(gray, win):
            x0, y0, x1, y1 = win
            crop = np.ascontiguousarray(gray[y0:y1, x0:x1])
            if sc < 1.0:
                crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            kps, des = det.detectAndCompute(crop, None)
            pts = [(k.pt[0] / sc + x0, k.pt[1] / sc + y0) for k in kps]
            return pts, des
        # The reference never changes during a run of the line, so its features
        # are computed once -- they were recomputed on every inspection.
        key = (ref_g.shape, ref_w, n, sc, norm, float(ref_g[::97, ::89].sum()))
        cached = Align._ref_cache.get(key)
        if cached is None:
            cached = detect(ref_g, ref_w)
            if len(Align._ref_cache) > 32:
                Align._ref_cache.clear()
            Align._ref_cache[key] = cached
        k1, d1 = cached
        k2, d2 = detect(img_g, img_w)
        self._last_inliers = 0
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return None, 0.0
        bf = cv2.BFMatcher(norm)
        pairs = bf.knnMatch(d1, d2, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 8:
            return None, 0.0
        src = np.float32([k1[x.queryIdx] for x in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[x.trainIdx] for x in good]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=4.0)
        if M is None or mask is None:
            return None, 0.0
        inliers = int(mask.sum())
        self._last_inliers = inliers
        self._last_inlier_ratio = inliers / float(len(good))
        if inliers < int(self.p.get("min_inliers", 12)):
            return None, 0.0
        M = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        if self.p.get("refine", True):
            # Polished within the fixture window, and judged there: the
            # whole-frame version let a static background pull a small part
            # back to "nothing moved".
            refined = self._refine_ecc_window(ref_g, img_g, M, ref_w, img_w)
            if refined is not None and \
                    self._fit_roi(ref_g, img_g, refined) >= self._fit_roi(ref_g, img_g, M):
                M = refined
        return M, self._fit_roi(ref_g, img_g, M)

    def _refine_ecc_window(self, ref_g, img_g, M, ref_w, img_w):
        """ECC on the fixture window alone. The warp is moved into the crops'
        coordinates, refined there at a reduced size, and moved back."""
        try:
            rx0, ry0, rx1, ry1 = ref_w
            ix0, iy0, ix1, iy1 = img_w
            r = ref_g[ry0:ry1, rx0:rx1]
            i = img_g[iy0:iy1, ix0:ix1]
            target = float(self.p.get("refine_size", 420))
            scale = min(1.0, target / max(max(r.shape[:2]), max(i.shape[:2])))
            T_r = np.array([[1, 0, rx0], [0, 1, ry0], [0, 0, 1]], np.float64)   # ref crop -> ref
            T_i = np.array([[1, 0, -ix0], [0, 1, -iy0], [0, 0, 1]], np.float64)  # img -> img crop
            S = np.diag([scale, scale, 1.0])
            Sinv = np.diag([1 / scale, 1 / scale, 1.0])
            Mc = S @ T_i @ np.asarray(M, np.float64) @ T_r @ Sinv
            warp = np.ascontiguousarray(Mc[:2].astype(np.float32))
            if scale < 1.0:
                r = cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                i = cv2.resize(i, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(self.p.get("refine_iters", 30)), 1e-5)
            cv2.findTransformECC(r, i, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
            Mc = np.vstack([warp.astype(np.float64), [0, 0, 1]])
            out = np.linalg.inv(T_i) @ Sinv @ Mc @ S @ np.linalg.inv(T_r)
            out = out.astype(np.float32)
            return out if np.all(np.isfinite(out)) else None
        except (cv2.error, np.linalg.LinAlgError):
            return None
