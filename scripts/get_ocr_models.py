"""Fetch the text reader's networks into models/ocr.

    python scripts/get_ocr_models.py            the full-size models (about 145 MB)
    python scripts/get_ocr_models.py --small    the small ones (about 38 MB), for a slow PC

Read text and Verify text run PP-OCRv6 through onnxruntime: a detector that
finds every line of print, an orientation network, and a recogniser. The files
come from PaddlePaddle's own repositories on Hugging Face, already in ONNX
format. Nothing else is downloaded and nothing is installed.

A package exported from QuickBuild carries these files with it, so a line PC
that only runs packages never needs this script.

If the download is refused -- some company networks inspect traffic to the
file host and the certificate check fails -- fetch the files on another
network and copy the models/ocr folder across; the names are listed below.
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(HERE, "models", "ocr")
HOST = "https://huggingface.co/PaddlePaddle/%s/resolve/main/%s"

# local name -> (repository, file)
FULL = {
    "det_medium.onnx": ("PP-OCRv6_medium_det_onnx", "inference.onnx"),
    "rec_medium.onnx": ("PP-OCRv6_medium_rec_onnx", "inference.onnx"),
    "rec_medium.yml": ("PP-OCRv6_medium_rec_onnx", "inference.yml"),
    "textline_ori.onnx": ("PP-LCNet_x1_0_textline_ori_onnx", "inference.onnx"),
}
SMALL = {
    "det_small.onnx": ("PP-OCRv6_small_det_onnx", "inference.onnx"),
    "rec_small.onnx": ("PP-OCRv6_small_rec_onnx", "inference.onnx"),
    "rec_small.yml": ("PP-OCRv6_small_rec_onnx", "inference.yml"),
    "textline_ori.onnx": ("PP-LCNet_x1_0_textline_ori_onnx", "inference.onnx"),
}


def dictionary_from_yml(yml_path, txt_path):
    """The recogniser's character list, one per line, from the inference.yml
    that ships beside it. Parsed by hand: PyYAML is not a dependency."""
    chars, inside = [], False
    with open(yml_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip(" \t\r\n")     # ASCII only: U+3000 is a character
            if stripped.startswith("character_dict:"):
                inside = True
                continue
            if not inside:
                continue
            if not stripped.startswith("- ") and stripped != "-":
                if stripped:
                    break
                continue
            v = stripped[2:] if stripped.startswith("- ") else ""
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                q = v[0]
                v = v[1:-1]
                v = v.replace("''", "'") if q == "'" else \
                    v.encode("utf-8").decode("unicode_escape") if "\\" in v else v
            chars.append(v)
    if not chars:
        raise ValueError("no character_dict in %s" % yml_path)
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(chars) + "\n")
    return len(chars)


def fetch(url, path, say=print):
    """Download with resume, to a .part file renamed when complete."""
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "vision-station"})
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:
            have = 0                              # the server restarted from zero
        total = have + int(r.headers.get("Content-Length") or 0)
        with open(part, "ab" if have else "wb") as f:
            done, shown = have, -1
            while True:
                chunk = r.read(1 << 18)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                pct = int(100 * done / total) if total else 0
                if pct // 10 != shown:
                    shown = pct // 10
                    say("    %3d%%  %.1f MB" % (pct, done / 1e6))
    os.replace(part, path)


def main(argv=None, say=print):
    argv = sys.argv[1:] if argv is None else argv
    wanted = SMALL if "--small" in argv else FULL
    os.makedirs(DEST, exist_ok=True)
    failed = []
    for name, (repo, remote) in wanted.items():
        path = os.path.join(DEST, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            say("  have  %s" % name)
            continue
        say("  get   %s" % name)
        for attempt in range(1, 6):
            try:
                fetch(HOST % (repo, remote), path, say)
                break
            except Exception as e:
                say("    attempt %d failed: %s: %s" % (attempt, type(e).__name__,
                                                      str(e)[:140]))
        else:
            failed.append(name)
    for name in wanted:
        if name.endswith(".yml") and os.path.exists(os.path.join(DEST, name)):
            txt = os.path.join(DEST, name[:-4] + ".txt")
            n = dictionary_from_yml(os.path.join(DEST, name), txt)
            say("  wrote %s (%d characters)" % (os.path.basename(txt), n))
    if failed:
        say("")
        say("  Not fetched: %s" % ", ".join(failed))
        say("  Copy them into %s from a PC where this script works." % DEST)
        return 1
    say("  The text reader's models are in %s" % DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
