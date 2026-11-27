"""Parameter schema. Drives the property sheet so every tool gets real
spinboxes, dropdowns and checkboxes instead of a JSON blob.

Each entry: (key, label, kind, default, extra)
  kind: int | float | bool | text | choice | intlist | strlist | boollist
  extra: (min, max, step) for numbers, [options] for choice
"""

# Rows several tools share. Written out in each tool before, so a change to
# one -- a new runtime, a new channel -- had to be repeated in every copy.
INFERENCE_ROWS = [
    ("backend", "Inference runtime", "choice", "auto", ["auto", "openvino", "onnxruntime"]),
    ("precision", "Precision", "choice", "accuracy", ["accuracy", "speed"]),
]
# What every text tool is tuned with. The reader finds the print itself, at
# any angle and on any background, so the old hand controls -- upscale,
# threshold window, denoise, invert, orientation -- are gone.
TEXT_ROWS = [
    ("min_confidence", "Ignore lines read with confidence below", "float", 0.50,
     (0.0, 1.0, 0.05)),
    ("charset", "Allowed characters (empty = any)", "text", "", None),
    ("min_chars", "Ignore lines shorter than (characters)", "int", 1, (1, 40, 1)),
    ("detail", "Print size", "choice", "auto", ["auto", "fine", "fast"]),
    ("script", "Script", "choice", "latin", ["latin", "any"]),
    ("ignore_stray_marks", "Ignore stray marks at the edges", "bool", True, None),
]
CHANNEL_ROW = ("channel", "Colour channel", "choice", "auto",
               ["auto", "grey", "red", "green", "blue"])
# Read text and Read code both compare what they read with the scanned barcode.
BARCODE_ROW = ("compare_barcode", "Compare barcode with barcode scanned", "bool", False, None)
# Graphics card or not is one choice for the whole application (Settings >
# General shows what this PC uses; VS_DEVICE overrides it). A "Use GPU if
# available" box in each AI check said the same thing again, differently for
# each check -- and off by default, so a check could quietly differ.

SCHEMA = {
    "align": [
        ("method", "Search method", "choice", "auto", ["auto", "features", "template"]),
        ("min_score", "Accept threshold", "float", 0.50, (0.0, 1.0, 0.01)),
        ("angle_range", "Angle range (deg)", "intlist", [-5, 5], None),
        ("angle_step", "Angle step (deg)", "float", 1.0, (0.1, 10.0, 0.1)),
        ("n_features", "Feature count", "int", 2000, (100, 20000, 100)),
        ("min_inliers", "Points that must agree (features)", "int", 12, (8, 500, 1)),
        ("feature_px", "Detect features at (px, long side)", "int", 1400, (400, 4000, 100)),
    ],
    "calibrate_scale": [
        ("known_mm", "Known distance (mm)", "float", 10.0, (0.001, 10000.0, 0.1)),
        ("axis", "Measure along", "choice", "x", ["x", "y"]),
        ("polarity", "Edge polarity", "choice", "any", ["any", "dark_to_light", "light_to_dark"]),
        ("min_contrast", "Minimum contrast", "float", 2.0, (0.1, 100.0, 0.1)),
    ],
    "presence": [
        ("min_score", "Accept threshold", "float", 0.65, (0.0, 1.0, 0.01)),
        ("search_pad", "Search margin (px)", "int", 15, (0, 500, 1)),
        ("check_seated", "Also check it sits right (lifted, tilted parts fail)", "bool", False,
         None),
        ("max_shift", "Largest shift from its place (px, 0 = off)", "float", 3.0,
         (0.0, 200.0, 0.5)),
        ("max_change", "Largest changed patch (% of the part, 0 = off)", "float", 4.0,
         (0.0, 100.0, 0.5)),
        ("change_level", "Counts as changed from (grey levels)", "float", 30.0,
         (5.0, 255.0, 1.0)),
    ],
    "count_template": [
        ("template_roi", "Sample region", "intlist", None, None),
        ("expect_count", "Expected count", "int", 4, (0, 9999, 1)),
        ("min_score", "Accept threshold", "float", 0.70, (0.0, 1.0, 0.01)),
        ("overlap", "Overlap reject", "float", 0.30, (0.0, 1.0, 0.05)),
    ],
    "count_blob": [
        ("space", "Channel", "choice", "hsv_v", ["gray", "hsv_v", "hsv_s"]),
        ("threshold", "Threshold", "text", "otsu", None),
        ("invert", "Dark features", "bool", False, None),
        ("open_px", "Noise filter (px)", "int", 3, (0, 51, 1)),
        ("min_area", "Minimum area (px)", "int", 40, (1, 1000000, 5)),
        ("max_area", "Maximum area (px)", "int", 1000000, (1, 100000000, 100)),
        ("expect_count", "Expected count", "int", 4, (0, 9999, 1)),
        ("row_tol", "Row grouping (px)", "int", 20, (1, 500, 1)),
    ],
    "color_classify": [
        ("source", "Read blobs from", "text", "", None),
        ("clip_level", "Treat as blown out above", "int", 250, (200, 255, 1)),
        ("halo", "Halo sample width", "float", 1.8, (1.1, 4.0, 0.1)),
        ("expect", "Expected colours", "strlist", [], None),
        ("deltaE_max", "Colour tolerance", "float", 28.0, (1.0, 200.0, 1.0)),
    ],
    "led_glow": [
        ("source", "Read blobs from", "text", "", None),
        ("v_on", "Brightness for ON", "int", 140, (0, 255, 1)),
        ("s_on", "Saturation for ON", "int", 60, (0, 255, 1)),
        ("expect_state", "Expected states", "boollist", [], None),
    ],
    "defect_diff": [
        ("threshold", "Difference threshold", "int", 22, (1, 255, 1)),
        ("illum_blur", "Lighting correction", "int", 51, (3, 301, 2)),
        ("open_px", "Noise filter (px)", "int", 2, (0, 51, 1)),
        ("min_area", "Ignore below (px)", "int", 25, (1, 100000, 5)),
        ("scratch_elongation", "Scratch aspect ratio", "float", 3.5, (1.0, 50.0, 0.1)),
        ("max_defects", "Allowed defects", "int", 0, (0, 9999, 1)),
        ("max_area_px", "Allowed defect size", "int", 0, (0, 1000000, 10)),
    ],
    "anomaly": [
        *INFERENCE_ROWS,
        ("model", "ONNX model file", "text", "models/anomaly.onnx", None),
        ("threshold", "Anomaly threshold", "float", 0.50, (0.0, 10.0, 0.01)),
        ("input_size", "Model input size", "int", 256, (64, 1024, 32)),
        ("min_area", "Ignore below (px)", "int", 20, (1, 100000, 5)),
    ],
    "ocr": [
        ("expect_contains", "Must contain", "text", "", None),
        BARCODE_ROW,
        *TEXT_ROWS,
    ],
    "image_process": [
        ("operator", "Operator", "choice", "smooth",
         ["smooth", "median", "sharpen", "equalize", "threshold", "erode", "dilate",
          "open", "close", "edges", "invert", "channel", "gain"]),
        ("kernel", "Kernel size (px)", "int", 5, (1, 99, 2)),
        ("amount", "Amount / gain / clip limit", "float", 1.0, (0.0, 20.0, 0.1)),
        ("level", "Level / offset", "float", 128.0, (-255.0, 255.0, 1.0)),
        ("threshold_mode", "Threshold mode", "choice", "otsu", ["otsu", "fixed", "adaptive"]),
        ("iterations", "Morphology iterations", "int", 1, (1, 20, 1)),
        ("channel", "Channel", "choice", "green", ["red", "green", "blue"]),
    ],
    "find_line": [
        CHANNEL_ROW,
        ("polarity", "Edge polarity", "choice", "any",
         ["any", "dark_to_light", "light_to_dark"]),
        ("num_calipers", "Calipers", "int", 12, (3, 100, 1)),
        ("caliper_width", "Caliper width (px)", "int", 6, (1, 60, 1)),
        ("min_contrast", "Minimum edge contrast", "float", 8.0, (0.5, 200.0, 0.5)),
        ("min_found", "Calipers that must agree", "int", 6, (3, 100, 1)),
        ("max_rms", "Maximum fit error (px)", "float", 1.5, (0.05, 20.0, 0.05)),
    ],
    "find_circle": [
        CHANNEL_ROW,
        ("polarity", "Edge polarity (outward)", "choice", "any",
         ["any", "dark_to_light", "light_to_dark"]),
        ("num_calipers", "Calipers", "int", 24, (6, 180, 1)),
        ("search_inner", "Search from (fraction of radius)", "float", 0.45, (0.0, 0.95, 0.05)),
        ("search_outer", "Search to (fraction of radius)", "float", 1.0, (0.2, 1.0, 0.05)),
        ("min_contrast", "Minimum edge contrast", "float", 8.0, (0.5, 200.0, 0.5)),
        ("min_found", "Calipers that must agree", "int", 10, (4, 180, 1)),
        ("min_radius", "Minimum radius (px, 0 = off)", "float", 0.0, (0.0, 5000.0, 0.5)),
        ("max_radius", "Maximum radius (px, 0 = off)", "float", 0.0, (0.0, 5000.0, 0.5)),
        ("max_rms", "Maximum fit error (px)", "float", 1.5, (0.05, 20.0, 0.05)),
    ],
    "caliper": [
        ("pair_polarity", "Pairs are", "choice", "dark", ["dark", "light"]),
        CHANNEL_ROW,
        ("min_contrast", "Minimum edge contrast", "float", 8.0, (0.5, 200.0, 0.5)),
        ("expect_count", "Expected pairs (0 = any)", "int", 0, (0, 500, 1)),
        # one width: it had two -- "expected width" (px, dropped pairs of
        # another width) and "nominal width" (mm once calibrated, failed them)
        ("nominal_width", "Width (mm once calibrated, else px; 0 = any)", "float", 0.0,
         (0.0, 100000.0, 0.01)),
        ("tolerance", "Width tolerance (same unit)", "float", 0.0, (0.0, 100000.0, 0.01)),
        ("max_results", "Most pairs to report", "int", 20, (1, 500, 1)),
    ],
    "histogram": [
        ("mean_min", "Mean at least (0 = off)", "float", 0.0, (0.0, 255.0, 1.0)),
        ("mean_max", "Mean at most (0 = off)", "float", 0.0, (0.0, 255.0, 1.0)),
        ("std_min", "Spread at least (0 = off)", "float", 0.0, (0.0, 128.0, 0.5)),
        ("std_max", "Spread at most (0 = off)", "float", 0.0, (0.0, 128.0, 0.5)),
        ("band_low", "Grey band from", "int", 0, (0, 255, 1)),
        ("band_high", "Grey band to", "int", 255, (0, 255, 1)),
        ("band_pct_min", "% in band at least (0 = off)", "float", 0.0, (0.0, 100.0, 0.5)),
        ("band_pct_max", "% in band at most (0 = off)", "float", 0.0, (0.0, 100.0, 0.5)),
    ],
    "distance": [
        ("first", "First feature (tool id)", "toolref", "", None),
        ("second", "Second feature (tool id, empty = diameter)", "toolref", "", None),
        ("min_value", "At least (0 = off)", "float", 0.0, (0.0, 100000.0, 0.01)),
        ("max_value", "At most (0 = off)", "float", 0.0, (0.0, 100000.0, 0.01)),
    ],
    "find_ellipse": [
        CHANNEL_ROW,
        ("polarity", "Edge polarity (outward)", "choice", "any",
         ["any", "dark_to_light", "light_to_dark"]),
        ("num_calipers", "Calipers", "int", 36, (8, 180, 1)),
        ("search_inner", "Search from (fraction of size)", "float", 0.45, (0.0, 0.95, 0.05)),
        ("search_outer", "Search to (fraction of size)", "float", 1.0, (0.2, 1.0, 0.05)),
        ("min_contrast", "Minimum edge contrast", "float", 8.0, (0.5, 200.0, 0.5)),
        ("num_to_ignore", "Outliers to ignore", "int", 4, (0, 90, 1)),
        ("min_found", "Minimum edge points", "int", 12, (6, 180, 1)),
        ("max_rms", "Maximum fit error (px)", "float", 1.5, (0.05, 20.0, 0.05)),
    ],
    "sharpness": [
        ("relative", "Compare with the reference image", "bool", True, None),
        ("min_relative", "Focus at least (fraction of reference)", "float", 0.6, (0.05, 2.0, 0.05)),
        ("min_sharpness", "Or sharpness at least (absolute)", "float", 50.0, (0.0, 100000.0, 1.0)),
    ],
    "color_segment": [
        ("hue_min", "Hue from (0-179)", "int", 0, (0, 179, 1)),
        ("hue_max", "Hue to (0-179; below 'from' wraps past red)", "int", 179, (0, 179, 1)),
        ("sat_min", "Saturation at least", "int", 60, (0, 255, 1)),
        ("val_min", "Brightness at least", "int", 40, (0, 255, 1)),
        ("min_pct", "At least % of the region (0 = off)", "float", 0.0, (0.0, 100.0, 0.5)),
        ("max_pct", "At most % of the region (0 = off)", "float", 0.0, (0.0, 100.0, 0.5)),
    ],
    "classify": [
        ("model", "Classifier model (.onnx)", "text", "", None),
        ("classes", "Class names, in order (comma-separated)", "text", "", None),
        ("accept", "Accepted classes (comma-separated)", "text", "", None),
        ("min_confidence", "Confidence at least", "float", 0.6, (0.0, 1.0, 0.01)),
        ("normalize", "Input normalisation", "choice", "imagenet", ["imagenet", "none"]),
        *INFERENCE_ROWS,
    ],
    "affine_transform": [
        ("scale", "Output scale", "float", 1.0, (0.1, 8.0, 0.1)),
    ],
    "polar_unwrap": [
        ("inner_radius", "Inner radius (fraction of outer)", "float", 0.5, (0.0, 0.95, 0.05)),
        ("start_angle", "Start angle (deg)", "float", 0.0, (-360.0, 360.0, 1.0)),
        ("sweep", "Sweep (deg)", "float", 360.0, (10.0, 360.0, 5.0)),
        ("direction", "Direction", "choice", "clockwise", ["clockwise", "counterclockwise"]),
        ("width", "Output width (px, 0 = circumference)", "int", 0, (0, 20000, 10)),
    ],
    "image_combine": [
        ("operation", "Operation", "choice", "difference",
         ["difference", "subtract", "subtract_reference_from", "minimum", "maximum", "average"]),
    ],
    "fixture_npoint": [
        ("points", "Feature tools (comma-separated ids)", "text", "", None),
        ("max_rms", "Largest point disagreement (px)", "float", 2.0, (0.1, 50.0, 0.1)),
        ("max_scale_change", "Largest scale change (fraction)", "float", 0.05, (0.0, 1.0, 0.01)),
    ],
    "calibrate_npoint": [
        ("pairs", "Point pairs 'px,py = wx,wy; ...'", "text", "", None),
        ("max_rms", "Largest fit error (world units)", "float", 0.05, (0.0001, 100.0, 0.001)),
    ],
    "construct": [
        ("operation", "Make", "choice", "line_through_points",
         ["line_through_points", "midpoint", "perpendicular", "parallel", "circle_through_points"]),
        ("first", "First feature (tool id)", "toolref", "", None),
        ("second", "Second feature (tool id)", "toolref", "", None),
        ("third", "Third feature (tool id)", "toolref", "", None),
    ],
    "bead": [
        ("polarity", "Bead is", "choice", "dark", ["dark", "light"]),
        ("width", "Expected width (px)", "float", 12.0, (1.0, 500.0, 0.5)),
        ("width_tolerance", "Width tolerance (px)", "float", 4.0, (0.1, 200.0, 0.5)),
        ("max_offset", "Largest offset from the path (px)", "float", 6.0, (0.0, 200.0, 0.5)),
        ("stations", "Stations along the path", "int", 30, (4, 1000, 1)),
        ("search_half_width", "Search either side (px)", "float", 25.0, (2.0, 500.0, 1.0)),
        ("min_contrast", "Minimum edge contrast", "float", 15.0, (1.0, 255.0, 1.0)),
        ("max_bad_stations", "Bad stations allowed", "int", 0, (0, 1000, 1)),
    ],
    "data_analysis": [
        ("source", "Value to follow (tool.value)", "text", "", None),
        ("lower_limit", "Lower limit (empty = none)", "text", "", None),
        ("upper_limit", "Upper limit (empty = none)", "text", "", None),
        ("min_cpk", "Cpk at least (0 = off)", "float", 0.0, (0.0, 10.0, 0.05)),
        ("min_samples", "Samples before judging Cpk", "int", 25, (2, 100000, 1)),
        ("window", "Values kept", "int", 100, (2, 100000, 1)),
    ],
    "results_analysis": [
        ("expression", "Accept when", "text", "", None),
    ],
    "intersect": [
        ("first", "First line or circle (tool id)", "toolref", "", None),
        ("second", "Second line or circle (tool id)", "toolref", "", None),
        ("expect_points", "Expected intersections (circles; 0 = any)", "int", 0, (0, 2, 1)),
        ("min_angle", "Angle at least (deg, 0 = off)", "float", 0.0, (0.0, 180.0, 0.1)),
        ("max_angle", "Angle at most (deg, 0 = off)", "float", 0.0, (0.0, 180.0, 0.1)),
    ],
    "fit_shape": [
        ("fit", "Fit", "choice", "line", ["line", "circle", "ellipse"]),
        ("source", "Centres from (tool id)", "text", "", None),
        ("max_deviation", "Largest deviation allowed (0 = off)", "float", 0.0,
         (0.0, 10000.0, 0.01)),
    ],
    "color_match": [
        ("max_delta_e", "Largest colour difference (delta E)", "float", 10.0, (0.5, 100.0, 0.5)),
    ],
    "trained_surface": [
        ("method", "Model", "choice", "auto", ["auto", "deep", "statistical"]),
        ("deep_sensitivity", "Deep model: threshold scale (1 = as trained)", "float", 1.0,
         (0.3, 3.0, 0.05)),
        ("max_clipped", "Overexposed above (fraction clipped at white)", "float", 0.15,
         (0.01, 1.0, 0.01)),
        ("resolution", "Model resolution (px, long side)", "int", 640, (160, 2400, 32)),
        ("flatten_lighting", "Ignore slow lighting changes", "bool", False, None),
        ("model", "Trained model (.npz)", "text", "", None),
        ("sensitivity_sigma", "Sensitivity (sigma, 0 = as trained)", "float", 0.0,
         (0.0, 12.0, 0.5)),
        ("max_defect_area", "Allowed defect area (0 = as trained)", "float", 0.0,
         (0.0, 100000.0, 1.0)),
    ],
    "ocv": [
        ("expect_text", "Exact text", "text", "", None),
        ("expect_regex", "Or a pattern", "text", "", None),
        ("strip_spaces", "Ignore spaces", "bool", True, None),
        ("look_alike", "Look-alikes match", "bool", True, None),
        *TEXT_ROWS,
    ],
    # Earlier recipes only: both now run Read text. Not offered for new checks.
    "text_detect": [
        ("expect_contains", "Must contain", "text", "", None),
        *TEXT_ROWS,
    ],
    "ocr_auto": [
        ("expect_contains", "Must contain", "text", "", None),
        *TEXT_ROWS,
    ],
    "code_read": [
        BARCODE_ROW,
        ("datamatrix", "Try DataMatrix", "bool", True, None),
        ("timeout_ms", "Decode timeout (ms)", "int", 800, (50, 10000, 50)),
        ("expect_regex", "Expected pattern", "text", "", None),
    ],
    "yolo_detect": [
        *INFERENCE_ROWS,
        ("model", "Model file (.onnx)", "text", "models/best.onnx", None),
        ("confidence", "Confidence threshold", "float", 0.35, (0.05, 0.99, 0.01)),
        ("iou", "Overlap merge (IoU)", "float", 0.45, (0.1, 0.9, 0.05)),
        ("expect_counts", "Expected per class", "text", {}, None),
        ("expect_total", "Expected total (0 = any)", "int", 0, (0, 9999, 1)),
        ("reject_when_found", "Reject the part when anything is found (defect model)",
         "bool", False, None),
        ("max_defects", "Detections allowed before rejecting", "int", 0, (0, 999, 1)),
        ("only_classes", "Only these classes", "strlist", [], None),
        ("min_area", "Ignore boxes below (px)", "int", 0, (0, 1000000, 10)),
    ],
    "yolo_defect": [
        *INFERENCE_ROWS,
        ("model", "Model file (.onnx)", "text", "models/best.onnx", None),
        ("confidence", "Confidence threshold", "float", 0.35, (0.05, 0.99, 0.01)),
        ("iou", "Overlap merge (IoU)", "float", 0.45, (0.1, 0.9, 0.05)),
        ("max_defects", "Allowed defects", "int", 0, (0, 999, 1)),
        ("only_classes", "Only these classes", "strlist", [], None),
        ("min_area", "Ignore boxes below (px)", "int", 0, (0, 1000000, 10)),
    ],
    "measure_length": [
        ("axis", "Measure along", "choice", "x", ["x", "y"]),
        ("polarity", "Edge polarity", "choice", "any", ["any", "dark_to_light", "light_to_dark"]),
        ("min_contrast", "Minimum contrast", "float", 2.0, (0.1, 100.0, 0.1)),
        ("max_edges", "Edges to find", "int", 2, (2, 20, 1)),
        ("mm_per_px", "Scale (mm/px, 0=auto)", "float", 0.0, (0.0, 100.0, 0.0001)),
        ("nominal_mm", "Nominal (mm)", "float", 0.0, (0.0, 100000.0, 0.001)),
        ("tolerance_mm", "Tolerance (mm)", "float", 0.05, (0.0, 1000.0, 0.001)),
    ],
}

LABELS = {
    "align": ("Locate part", "Finds the part and shifts every other region to follow it."),
    "calibrate_scale": ("Calibrate scale", "Turns pixels into millimetres from one known dimension."),
    "presence": ("Part present", "Checks a feature is there by comparing to the reference."),
    "count_template": ("Count by shape", "Counts screws, bolts or any repeated feature."),
    "count_blob": ("Count by brightness", "Counts LEDs, holes or pads by thresholding."),
    "color_classify": ("LED colour", "Names each LED colour and checks the order."),
    "led_glow": ("LED lit state", "Decides whether each LED is on or off."),
    "defect_diff": ("Dents and scratches", "Compares against the reference and flags changes."),
    "anomaly": ("External anomaly model (earlier recipe)",
                "Runs an anomalib model exported elsewhere. Use Trained surface check."),
    "ocr": ("Read text",
            "Finds every line of text in the region, at any angle, and reads it."),
    "image_process": ("Image processing",
                      "Filters the image for later tools: smooth, sharpen, threshold, morphology..."),
    "find_line": ("Find line",
                  "Fits a line through many calipers laid across an edge."),
    "find_circle": ("Find circle",
                    "Fits a circle through calipers radiating from the region's centre."),
    "caliper": ("Caliper (edge pairs)",
                "Finds bars or gaps between edge pairs: counts them, measures each width and the pitch."),
    "histogram": ("Histogram",
                  "Grey-level statistics over the region, judged against limits."),
    "distance": ("Distance / angle",
                 "Measures between features found by other tools."),
    "find_ellipse": ("Find ellipse", "Fits an ellipse through calipers radiating from the centre."),
    "sharpness": ("Image sharpness", "Is the image in focus - against the reference."),
    "color_segment": ("Color segmenter", "How much of the region is within a colour range."),
    "classify": ("Classify", "A trained classifier sorts the region into classes."),
    "affine_transform": ("Affine transform", "The region resampled upright into an image of its own."),
    "polar_unwrap": ("Polar unwrap", "A ring unrolled into a straight strip."),
    "image_combine": ("Two-image operation", "The image combined with the reference: difference and more."),
    "fixture_npoint": ("Fixture from points", "The part's pose from features other tools found."),
    "calibrate_npoint": ("N-point calibration", "Pixels to real-world units from known points."),
    "construct": ("Create geometry", "Lines, midpoints and circles made from found features."),
    "bead": ("Bead inspect", "A glue bead along a path: present, the right width, in place."),
    "data_analysis": ("Data analysis", "A value's statistics over runs: mean, spread, Cpk."),
    "results_analysis": ("Results analysis",
                         "Accept or reject from an expression over earlier results."),
    "intersect": ("Intersect lines",
                  "The corner where two fitted lines meet, and the angle between them."),
    "fit_shape": ("Fit line / circle",
                  "A line or circle through the centres another tool found."),
    "color_match": ("Color match",
                    "How far the region's colour is from the reference, as delta E."),
    "trained_surface": ("Trained surface check",
                        "Learns the region from OK images; flags anything that deviates."),
    "text_detect": ("Read text (earlier recipe)", "Now the same as Read text."),
    "ocr_auto": ("Read text (earlier recipe)", "Now the same as Read text."),
    "ocv": ("Verify text", "Checks a marking against the exact text or a pattern."),
    "code_read": ("Read code", "Decodes DataMatrix, QR and 1D barcodes."),
    "yolo_detect": ("Detect with YOLO",
                    "Counts objects, or rejects on defects, using a model you trained."),
    "yolo_defect": ("Find defects with YOLO (earlier recipe)",
                    "Now Detect with YOLO, with 'Reject the part when anything is found'."),
    "measure_length": ("Measure length", "Subpixel distance between two edges."),
}

GROUPS = [
    ("Image", ["image_process", "affine_transform", "polar_unwrap", "image_combine", "sharpness"]),
    ("Locate", ["fixture_npoint", "calibrate_npoint", "align", "calibrate_scale"]),
    ("Presence and count", ["presence", "count_template", "count_blob"]),
    ("Colour", ["color_match", "color_segment", "color_classify", "led_glow"]),
    ("Geometry", ["find_line", "find_circle", "find_ellipse", "caliper", "fit_shape", "intersect",
                  "construct", "distance"]),
    ("Surface", ["trained_surface", "defect_diff", "bead", "histogram"]),
    ("Trained models", ["classify", "yolo_detect"]),
    ("Text and codes", ["ocr", "ocv", "code_read"]),
    ("Logic", ["results_analysis", "data_analysis"]),
    ("Measurement", ["measure_length"]),
]


# Checks that duplicated another and were folded into it. They are no longer
# offered, but a recipe that has one still loads, runs and can be edited.
LEGACY_TYPES = {"text_detect": "ocr", "ocr_auto": "ocr",
                "yolo_defect": "yolo_detect", "anomaly": "trained_surface"}


def defaults(tool_type):
    d = {"enabled": True, "critical": True, "angle": 0.0, "shape": "rect",
         "capture": "default"}
    for key, _, kind, dv, _x in SCHEMA.get(tool_type, []):
        d[key] = dv
    return d


# ---------------------------------------------------------------- plain language
# The simple view asks what the person wants to know about the part, not
# which algorithm they want. Each question lists its checks, the most usual
# first. GROUPS above stays the engineering catalogue (Advanced mode).
QUESTIONS = [
    ("Is the part where it should be?",
     "Find the part first, so every other check follows it when a part arrives shifted or turned.",
     ["align", "calibrate_scale", "fixture_npoint", "calibrate_npoint"]),
    ("Is something there, and how many?",
     "A component that must be fitted, screws that must all be present, LEDs or holes to count.",
     ["presence", "count_template", "count_blob", "yolo_detect"]),
    ("Is the text or code right?",
     "Lot codes, serial numbers, date stamps, barcodes and DataMatrix.",
     ["ocr", "ocv", "code_read"]),
    ("Are the colours right? Are the LEDs on?",
     "Colour of a part or a label, the colour and state of each LED.",
     ["color_classify", "led_glow", "color_match", "color_segment"]),
    ("Is the surface clean?",
     "Scratches, dents, marks, missing glue, anything that differs from a good part.",
     ["defect_diff", "trained_surface", "yolo_detect", "bead", "histogram"]),
    ("Is a size or position within tolerance?",
     "Lengths, widths, diameters, gaps and angles, in pixels or millimetres.",
     ["measure_length", "caliper", "find_line", "find_circle", "distance",
      "find_ellipse", "intersect", "fit_shape", "construct"]),
    ("More",
     "Image preparation, trained classifiers and logic over the other checks' results.",
     ["image_process", "sharpness", "classify", "results_analysis", "data_analysis",
      "affine_transform", "polar_unwrap", "image_combine"]),
]

# Hidden in the picker until "Show advanced checks" is ticked: each needs
# another check's result, a trained model, or a notion from the manual.
ADVANCED_TYPES = {
    "fixture_npoint", "calibrate_npoint", "color_segment",
    "histogram", "find_ellipse", "intersect", "fit_shape", "construct", "classify",
    "results_analysis", "data_analysis", "affine_transform", "polar_unwrap",
    "image_combine", "sharpness", "bead",
}

# The few settings a check is usually tuned with. Everything else is behind
# "Show all settings". A check not listed here shows all its settings.
ESSENTIAL = {
    "align": ["min_score", "angle_range"],
    "calibrate_scale": ["known_mm", "axis"],
    "presence": ["min_score", "check_seated"],
    "count_template": ["expect_count", "template_roi", "min_score"],
    "count_blob": ["expect_count", "threshold", "invert", "min_area"],
    "color_classify": ["expect", "deltaE_max", "source"],
    "led_glow": ["expect_state", "source"],
    "defect_diff": ["threshold", "min_area", "max_defects"],
    "anomaly": ["model", "threshold"],
    "ocr": ["expect_contains", "compare_barcode", "min_confidence"],
    "ocv": ["expect_text", "expect_regex", "min_confidence"],
    "ocr_auto": ["expect_contains", "min_confidence"],
    "text_detect": ["expect_contains", "min_confidence"],
    "code_read": ["compare_barcode", "expect_regex", "datamatrix"],
    "measure_length": ["nominal_mm", "tolerance_mm", "axis"],
    "caliper": ["expect_count", "nominal_width", "tolerance"],
    "find_line": ["polarity", "min_contrast"],
    "find_circle": ["polarity", "min_contrast"],
    "find_ellipse": ["polarity", "min_contrast"],
    "distance": ["first", "second", "min_value", "max_value"],
    "histogram": ["mean_min", "mean_max"],
    "color_match": ["max_delta_e"],
    "color_segment": ["hue_min", "hue_max", "min_pct", "max_pct"],
    "classify": ["model", "accept", "min_confidence"],
    "yolo_detect": ["model", "confidence", "expect_total", "reject_when_found"],
    "yolo_defect": ["model", "confidence", "max_defects"],
    "trained_surface": ["method", "resolution", "deep_sensitivity", "model"],
    "sharpness": ["min_relative"],
    "bead": ["width", "width_tolerance", "max_offset"],
    "image_process": ["operator", "kernel"],
    "results_analysis": ["expression"],
    "data_analysis": ["source", "lower_limit", "upper_limit"],
    "intersect": ["first", "second"],
    "fit_shape": ["fit", "source"],
    "construct": ["operation", "first", "second"],
    "fixture_npoint": ["points"],
    "calibrate_npoint": ["pairs"],
    "affine_transform": ["scale"],
    "polar_unwrap": ["inner_radius", "sweep"],
    "image_combine": ["operation"],
}

# One plain sentence under a setting. Keyed by setting name, or by
# (check type, setting name) when the same name means something else.
HELP = {
    ("ocv", "look_alike"): "O and 0, I and 1, S and 5, B and 8, Z and 2 (and i, s, z) count "
                           "as the same character: the reader confuses them, a marking "
                           "does not.",
    "compare_barcode": "With the barcode check on in Settings: what this check reads on the "
                       "part must be the scanned code, or the part is NG - Barcode not "
                       "good - and its other checks are skipped. Checking it moves this "
                       "check to just after the fixture.",
    "ignore_stray_marks": "A pad, a solder joint or the next part at the edge of the box is "
                          "sometimes read as a letter, differently on each picture. On, those "
                          "are left out so one part gives one result every run.",
    "check_seated": "Part present on its own asks only whether the part is there, and a "
                    "lifted, tilted or half-soldered part is. Ticked, the part must also be "
                    "where it belongs (after Locate part) and look like the reference part all "
                    "over: a raised end, a missing solder fillet or a shadow under it fails.",
    "max_shift": "How far from its place the part may be found, in pixels, once the board has "
                 "been located. A part lifted at one end or pushed off its pads moves.",
    "max_change": "How large a patch of the part may look different from the reference, as a "
                  "share of the box. Good pictures in the Train tab raise it to what good parts "
                  "need on their own.",
    "change_level": "How much darker or brighter a pixel must be, after matching overall "
                    "brightness, to count as changed. Raise it for shiny parts.",
    "min_score": "How closely it must match the reference, 0 to 1. Lower it if good parts are "
                 "missed; raise it if wrong parts get through.",
    ("align", "min_score"): "How closely the part must match the reference, 0 to 1. Lower it if "
                            "the part is not found; raise it if the wrong place is found.",
    "angle_range": "How far the part may be turned, in degrees - for example [-5, 5].",
    "known_mm": "The real length, in millimetres, of what is inside the box.",
    "axis": "Measure left to right (x) or top to bottom (y).",
    "expect_count": "How many should be found. Any other number fails the check.",
    "template_roi": "One example of the thing to count: its x, y, width and height on the "
                    "reference image, like [20, 20, 40, 40].",
    ("count_blob", "threshold"): "Brightness that counts as 'lit', 0 to 255 - or 'otsu' to choose it "
                                 "automatically.",
    ("count_blob", "invert"): "Tick when the things to count are darker than their background.",
    "min_area": "Ignore anything smaller than this many pixels. Raise it to remove noise.",
    "expect": "The colours in order, left to right, like [\"red\", \"green\", \"blue\"].",
    "deltaE_max": "How different a colour may be and still count as the same. 30 is forgiving, "
                  "10 is strict.",
    "source": "The counting check whose LEDs or blobs this check reads.",
    "expect_state": "Which LEDs should be on, in order, like [true, true, false].",
    ("defect_diff", "threshold"): "How much a spot must differ from the reference to count as a mark. "
                                  "Lower it to find fainter marks.",
    "max_defects": "Marks allowed before the part is rejected.",
    "expect_regex": "A pattern the text must match, when it changes from part to part. "
                    "[A-Z]{2}[0-9]{6} means two capital letters then six digits.",
    "expect_text": "The exact text expected. Leave it and the pattern empty to accept "
                   "any text that is read.",
    "expect_contains": "Text that must appear somewhere in what is read. Leave empty to accept "
                       "whatever is read.",
    "min_confidence": "How sure the reader must be of a line, 0 to 1. Lines read with less "
                      "are left out. Lower it for faint or damaged print.",
    "charset": "Only these characters are kept, like ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 for "
               "a serial number. Leave empty to keep everything that is read.",
    "script": "latin keeps letters, digits and symbols of Latin-script languages. any also "
              "keeps Chinese, Japanese and other characters the reader knows.",
    "detail": "auto suits most print. fine looks twice as closely, for print under about "
              "10 pixels tall - slower. fast makes one pass over the whole region.",
    "reject_when_found": "For a model trained on defects: finding anything rejects the part, "
                         "instead of counting what is found.",
    "nominal_mm": "The dimension on the drawing, in millimetres.",
    "tolerance_mm": "How far from nominal is still acceptable, in millimetres.",
    "polarity": "Which way the edge goes: dark to light, light to dark, or either.",
    "min_contrast": "How strong an edge must be to count. Raise it to ignore faint texture.",
    "model": "The trained model file, kept in the project's models folder.",
    "confidence": "Detections scoring below this are ignored, 0 to 1.",
    "expect_total": "How many objects should be found. 0 accepts any number.",
    "max_delta_e": "The largest colour difference from the reference still accepted.",
    "expression": "Accept when this is true, like count_blob_1.count == 4.",
    ("caliper", "nominal_width"): "The width of each bar or gap: in millimetres once a Calibrate "
                                  "scale check has run, in pixels before. 0 accepts any width.",
    ("caliper", "tolerance"): "How far from that width a pair may be, in the same unit.",
    "width_tolerance": "How far from the expected width is still acceptable.",
    "min_value": "Accept only when at least this. 0 means no lower limit.",
    "max_value": "Accept only when at most this. 0 means no upper limit.",
    "datamatrix": "Also try DataMatrix codes. Slower, so untick it for plain barcodes.",
    "min_chars": "Ignore anything shorter than this many characters.",
    "first": "A line or circle found by an earlier check.",
    "second": "Another line or circle found by an earlier check.",
    "mean_min": "The average brightness must be at least this. 0 switches the limit off.",
    "mean_max": "The average brightness must be at most this. 0 switches the limit off.",
    "accept": "The class names that mean a good part, separated by commas.",
    "sensitivity_sigma": "How far from the learned surface counts as a defect. 0 keeps what was "
                         "learned.",
    "flatten_lighting": "Tick when the lamp or the daylight changes between parts: gradual "
                        "brightness differences are ignored and only local marks count. "
                        "Retrain after changing it.",
    "resolution": "How finely the surface is modelled. 640 for a region up to about 1300 px; "
                  "1600 or more for a whole panel on a 12-megapixel photo, or thin scratches "
                  "are averaged away. Retrain after changing it.",
    ("trained_surface", "method"): "deep learns what good parts look like with a pretrained "
                                   "network: it copes with texture and lighting that differ "
                                   "from part to part. statistical compares pixel by pixel: "
                                   "for parts that repeat exactly. auto uses deep when there "
                                   "are at least 4 OK images, statistical with fewer. "
                                   "Retrain after changing it.",
    "deep_sensitivity": "Below 1 rejects smaller differences, above 1 only larger ones. "
                        "No retraining needed.",
    "min_relative": "How sharp the image must be, as a fraction of the reference image.",
    "width": "The expected width of the bead in pixels.",
    "max_offset": "How far the bead may wander from its path, in pixels.",
    "operator": "What to do to the image before the next checks read it.",
    "kernel": "The size of the filter in pixels. Larger smooths or thickens more.",
    "lower_limit": "The value must stay above this. Leave empty for no lower limit.",
    "upper_limit": "The value must stay below this. Leave empty for no upper limit.",
    "hue_min": "The start of the colour range, 0 to 179 (red is near 0 and 179).",
    "hue_max": "The end of the colour range, 0 to 179.",
    "min_pct": "At least this much of the box must be that colour. 0 switches it off.",
    "max_pct": "At most this much of the box may be that colour. 0 switches it off.",
    "fit": "The shape to fit through the points.",
    "operation": "What to make or do.",
    "points": "The checks that found the features, separated by commas.",
    "pairs": "Image points and their real-world positions.",
    "scale": "How much larger the output image is.",
    "inner_radius": "Where the ring starts, as a fraction of its outer edge.",
    "sweep": "How much of the ring to unroll, in degrees.",
    "threshold": "The score above which something is reported.",
}


def help_for(tool_type, key):
    return HELP.get((tool_type, key)) or HELP.get(key) or ""
