"""Image files on any path.

OpenCV's imread and imwrite take the path as narrow bytes, so on Windows a
path with a character outside the system code page -- a user folder named
"Gökul", a project under "Документы" -- fails: imread returns None and imwrite
returns False, and neither says why. A job saved there lost its reference image
without a word, and every grab of a part image was an acquisition error.

These go through Python's own file handling instead, which takes any path, and
let OpenCV do only the encoding.
"""
import os

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread for any path: the image, or None when it cannot be read."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError, TypeError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, image, params=None):
    """cv2.imwrite for any path. True when the file was written."""
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, image, list(params or []))
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True
