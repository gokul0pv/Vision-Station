from .base import REGISTRY, Tool, ToolResult, Graphic, make_tool, register  # noqa: F401  (re-exported)
from . import align, count, color, defect, text, measure, yolo, ocr_auto, text_detect, learned, geometry, imageproc, analysis, more_tools, transform, construct_tools  # noqa: F401,E402  (imported to register each tool)


def catalog():
    """Everything the editor UI offers in its tool palette."""
    return [
        {"type": "align", "group": "Locate", "label": "Align part",
         "help": "Finds the part and shifts every other ROI to follow it. Add this first."},
        {"type": "calibrate_scale", "group": "Locate", "label": "Calibrate scale",
         "help": "Turns pixels into millimetres using one known dimension."},
        {"type": "presence", "group": "Presence", "label": "Part present",
         "help": "Is this feature there? Compares the region against the reference."},
        {"type": "count_template", "group": "Presence", "label": "Count by shape",
         "help": "Counts screws, bolts, or any repeated part matching one sample."},
        {"type": "count_blob", "group": "Presence", "label": "Count by brightness",
         "help": "Counts LEDs, holes, or pads by thresholding the region."},
        {"type": "color_classify", "group": "Colour", "label": "Check LED colour",
         "help": "Names the colour of each LED and compares to the expected order."},
        {"type": "led_glow", "group": "Colour", "label": "Check LED lit",
         "help": "Decides whether each LED is on or off."},
        {"type": "defect_diff", "group": "Surface", "label": "Find dents and scratches",
         "help": "Compares against the reference and flags what changed."},
        {"type": "anomaly", "group": "Surface", "label": "AI surface check",
         "help": "Trained on good parts only. Catches defects you never sampled."},
        {"type": "yolo_detect", "group": "Trained models", "label": "Detect with YOLO",
         "help": "Counts and identifies objects using a model you trained."},
        {"type": "yolo_defect", "group": "Trained models", "label": "Find defects with YOLO",
         "help": "Any detection above the threshold rejects the part."},
        {"type": "ocr", "group": "Text", "label": "Read text",
         "help": "Reads whatever text is in the region."},
        {"type": "ocv", "group": "Text", "label": "Verify text",
         "help": "Checks the marking against an expected pattern. Use this on the line."},
        {"type": "code_read", "group": "Text", "label": "Read code",
         "help": "Decodes DataMatrix, QR, and 1D barcodes."},
        {"type": "measure_length", "group": "Measure", "label": "Measure length",
         "help": "Subpixel distance between two edges, in millimetres."},
    ]
