"""One part, one result: stray marks around small print do not change what
a text check says from frame to frame. Pictures in a folder per day, found
by barcode as it is typed. The camera's firewall rule.

    python -m pytest tests/test_repeatability_and_finder.py -q
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from vision_station.tools import ocr_engine as E          # noqa: E402
from vision_station.tools import text as T                 # noqa: E402


def _line(text, confs, box=(0, 0, 40, 30), conf=None):
    return {"text": text, "char_conf": list(confs), "box": list(box),
            "confidence": conf if conf is not None else float(np.mean(confs))}


# ------------------------------------------------------------- stray marks
def test_a_stray_letter_beside_sure_print_gives_a_second_reading():
    # measured on a real board: the edge of the next part read as 'P'
    assert E.core_text(_line("PR15", [0.79, 0.97, 0.998, 0.999])) == "R15"
    # a pad's corner after a space
    assert E.core_text(_line("R15 D", [0.999, 0.999, 0.999, 0.52, 0.42])) == "R15"
    # a glint read as an apostrophe inside the marking
    assert E.core_text(_line("Ri'5", [0.99, 0.98, 0.40, 0.99])) == "Ri5"


def test_print_read_unsurely_throughout_is_left_alone():
    assert E.core_text(_line("R15", [0.99, 0.99, 0.99])) is None, "nothing to trim"
    # the rest is not sure enough to say the end is the stray one
    assert E.core_text(_line("PR15", [0.70, 0.80, 0.90, 0.92])) is None
    # never trimmed below three characters
    assert E.core_text(_line("AB", [0.3, 0.99])) is None


def test_a_thin_one_letter_fragment_is_dropped_but_not_real_lines():
    main = _line("451", [1, 1, 1], box=(5, 7, 51, 30))
    pad = _line("A", [0.97], box=(20, 41, 19, 8))          # a sliver of a round pad
    small_word = _line("LOT 7", [0.99] * 5, box=(5, 40, 50, 9))
    big_one = _line("7", [0.9], box=(60, 7, 18, 28))
    out = [l["text"] for l in E.clean_lines([main, pad, small_word, big_one])]
    assert out == ["451", "LOT 7", "7"]
    # with no main print in the region nothing is judged a fragment
    assert [l["text"] for l in E.clean_lines([pad])] == ["A"]


def test_verify_text_accepts_the_marking_and_rejects_a_wrong_one():
    from vision_station.core.engine import Context
    tool = T.Ocv("t", {"expect_text": "R15", "min_confidence": 0.3, "roi": [0, 0, 10, 10]})

    def run(lines):
        import vision_station.tools.text as mod
        keep = mod.read_region
        mod.read_region = lambda *_a: (lines, [], (0, 0), "test")
        try:
            return tool.run(np.zeros((10, 10, 3), np.uint8), Context()).status
        finally:
            mod.read_region = keep

    stray = dict(_line("PR15", [0.79, 0.97, 0.998, 0.999]), core="R15", poly=None)
    assert run([stray]) == "pass", "a stray edge letter failed a good part"
    wrong = dict(_line("PR16", [0.79, 0.97, 0.998, 0.999]), core="R16", poly=None)
    assert run([wrong]) == "fail", "a wrong marking passed"
    assert run([dict(_line("R10", [0.99] * 3), poly=None)]) == "fail"


def test_a_glint_on_a_one_is_a_look_alike():
    assert T.look_alike("RiS") == T.look_alike("R15")
    assert T.look_alike("R1s") == T.look_alike("R15")
    assert T.look_alike("R16") != T.look_alike("R15")


# ------------------------------------------------------------- day folders
def test_every_picture_goes_in_the_day_folder(tmp_path):
    from vision_station.core import barcode, result_images as ri
    when = time.mktime((2026, 11, 19, 14, 35, 2, 0, 0, -1))
    plain = {"folder": str(tmp_path)}
    ann, _orig = ri.paths_for(plain, "Top", "PASS", when)
    assert os.path.dirname(ann) == os.path.join(str(tmp_path), "2026-11-19")
    cfg = barcode.result_cfg(plain, str(tmp_path), "FIC123/4")
    ann, _orig = ri.paths_for(cfg, "Top", "FAIL", when)
    assert os.path.dirname(ann) == os.path.join(str(tmp_path), "2026-11-19", "FIC123_4")
    assert cfg["which"] == "all", "a scanned part keeps every picture"
    assert barcode.result_folder(str(tmp_path), "FIC1", when) == \
        os.path.join(str(tmp_path), "2026-11-19", "FIC1")


# ------------------------------------------------------------- the finder
def _touch(path, mtime):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff")
    os.utime(path, (mtime, mtime))


def test_the_index_finds_pictures_by_barcode_newest_and_exact_first(tmp_path):
    from vision_station.core.picture_index import PictureIndex
    r = str(tmp_path)
    _touch(os.path.join(r, "2026-11-18", "FIC1234567", "101500_001_Top_ACCEPT.jpg"), 1000)
    _touch(os.path.join(r, "2026-11-19", "FIC12345678", "143502_118_Top_REJECT.jpg"), 3000)
    _touch(os.path.join(r, "2026-11-19", "FIC12345678",
                        "143502_118_Top_REJECT_original.jpg"), 3000)
    _touch(os.path.join(r, "2026-11-19", "FIC1234567", "090000_002_Top_ACCEPT.jpg"), 2000)
    _touch(os.path.join(r, "2026-11-19", "120000_003_Side_ACCEPT.jpg"), 2500)
    _touch(os.path.join(r, "2026-11-19", "notes.txt"), 2500)
    ix = PictureIndex()
    ix.set_roots([r, r + os.sep])                 # the same folder twice is one root
    assert len(ix.roots) == 1
    ix.scan()
    assert len(ix) == 5, "the text file was indexed, or a picture missed"

    found, total = ix.search("fic1234567")
    assert total == 4
    # the exact code first (newest first within it), then codes starting with it
    assert [p.code for p in found] == ["FIC1234567", "FIC1234567", "FIC12345678",
                                       "FIC12345678"]
    assert found[0].day == "2026-11-19" and found[0].time == "09:00:00"
    p = next(p for p in found if not p.original and p.code == "FIC12345678")
    assert (p.verdict, p.job) == ("REJECT", "Top")

    found, _ = ix.search("12345678", originals=False)
    assert [p.code for p in found] == ["FIC12345678"]
    found, _ = ix.search("2026-11-19 side")
    assert [(p.code, p.job) for p in found] == [("", "Side")], "a part without a code"
    assert ix.search("", verdict="REJECT")[1] == 2
    assert ix.search("nothing-like-this")[1] == 0

    new = os.path.join(r, "2026-11-19", "FIC9", "150000_004_Top_ACCEPT.jpg")
    _touch(new, 4000)
    v = ix.version
    ix.add([new, os.path.join(str(tmp_path.parent), "elsewhere.jpg")])
    assert ix.version > v and ix.search("fic9")[0][0].path == new
    assert len(ix) == 6, "a picture outside the folders was indexed"


def test_searching_a_large_index_is_instant(tmp_path):
    from vision_station.core.picture_index import Picture, PictureIndex
    ix = PictureIndex()
    root = str(tmp_path)
    ix.roots = [root]
    items = {}
    for i in range(100_000):
        path = os.path.join(root, "2026-11-%02d" % (1 + i % 28), "FIC%08d" % i,
                            "101500_001_Top_ACCEPT.jpg")
        items[path] = Picture(path, root, float(i))
    ix._items, ix._dirty = items, True
    ix.search("x")                                  # sorted once, as after a scan
    t0 = time.perf_counter()
    for q in ("f", "fic0001", "fic00012345", "2026-11-05 fic"):
        ix.search(q)
    per = (time.perf_counter() - t0) / 4
    assert ix.search("fic00012345")[0][0].code == "FIC00012345"
    assert per < 0.25, "a search took %.0f ms" % (per * 1000)


# ------------------------------------------------------------- firewall
def test_the_firewall_rule_is_offered_only_for_the_packaged_program(monkeypatch):
    from vision_station.hardware import firewall
    assert firewall.rule_name(r"C:\x\VisionStation.exe") == \
        "Vision Station camera (VisionStation.exe)"
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert firewall.should_offer() is False, "offered when run from Python"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(firewall, "has_rule", lambda prog=None: False)
    assert firewall.should_offer() is (sys.platform == "win32")
    monkeypatch.setattr(firewall, "has_rule", lambda prog=None: True)
    assert firewall.should_offer() is False
