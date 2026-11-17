"""Packages (.vsp) and the GUI-free runtime.

    python -m pytest tests_package.py -q
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
DEMO = os.path.join(HERE, "projects", "demo_led_panel")


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    from vision_station.core import package
    monkeypatch.setenv("VISION_STATION_CACHE", str(tmp_path / "cache"))
    path = str(tmp_path / "demo.vsp")
    m = package.export_package(DEMO, path)
    return path, m


def test_package_contents(pkg):
    path, m = pkg
    names = zipfile.ZipFile(path).namelist()
    assert "manifest.json" in names and "vsruntime.py" in names
    assert "app/project.json" in names and "app/reference.png" in names
    assert "runtime/vision_station/core/engine.py" in names
    assert "runtime/vision_station/tools/base.py" in names
    assert not any(n.startswith("runtime/vision_station/ui/") for n in names), "the user interface was packed into the runtime"
    assert not any("/results/" in n or "__pycache__" in n for n in names)
    assert m["jobs"][0]["name"] == "LED panel rev C"
    assert any(r.startswith("onnxruntime") for r in m["requires"]), m["requires"]


def test_training_images_are_left_out_unless_asked(tmp_path):
    from vision_station.core import package
    app = tmp_path / "app"
    shutil.copytree(DEMO, str(app))
    t = app / "training" / "lot_code" / "ok"
    t.mkdir(parents=True)
    shutil.copy(os.path.join(DEMO, "good.png"), str(t / "a.png"))
    package.export_package(str(app), str(tmp_path / "a.vsp"))
    package.export_package(str(app), str(tmp_path / "b.vsp"), include_training=True)
    assert not any("training" in n for n in zipfile.ZipFile(str(tmp_path / "a.vsp")).namelist())
    assert any("training" in n for n in zipfile.ZipFile(str(tmp_path / "b.vsp")).namelist())


def test_a_program_with_only_the_package_gets_the_applications_result(pkg, tmp_path):
    """A clean Python process, started elsewhere, with nothing but the .vsp on
    its path: it must import the runtime from the package and reproduce the
    application's verdict tool for tool."""
    from vision_station.core.engine import Project
    path, _m = pkg
    img = os.path.join(DEMO, "bad.png")
    ref = Project(DEMO).inspect(cv2.imread(img))
    work = tmp_path / "elsewhere"
    work.mkdir()
    shutil.copy(path, str(work / "demo.vsp"))
    shutil.copy(img, str(work / "part.png"))
    code = (
        "import sys, json, cv2\n"
        "sys.path.insert(0, 'demo.vsp')\n"
        "import vsruntime\n"
        "app = vsruntime.load('demo.vsp')\n"
        "from vision_station.core import engine          # importable only once a package is loaded\n"
        "r = app.run('part.png')\n"
        "cv2.imwrite('out.png', r.annotated())\n"
        "print(json.dumps({'verdict': r.verdict, 'job': r.job_id,\n"
        "  'status': {k: v['status'] for k, v in r.tools.items()},\n"
        "  'runtime': vsruntime.__file__, 'engine': engine.__file__}))\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path),
           "VISION_STATION_CACHE": str(tmp_path / "cache2")}
    out = subprocess.run([sys.executable, "-c", code], cwd=str(work), env=env,
                         capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["runtime"].startswith("demo.vsp"), "the runtime did not come from the package"
    assert HERE not in got["engine"], "the engine came from the application folder"
    assert got["verdict"] == ref["verdict"]
    assert got["status"] == {t["tool_id"]: t["status"] for t in ref["tools"]}
    assert got["job"] == "demo_led_panel", "the job id is the unpack folder, not the model's"
    annotated = cv2.imread(str(work / "out.png"))
    assert annotated.shape[0] > 420 and annotated.shape[1] == 640


def test_banner_sits_above_the_image_not_over_it(pkg):
    from vision_station.core import vsruntime
    path, _m = pkg
    app = vsruntime.load(path)
    img = cv2.imread(os.path.join(DEMO, "good.png"))
    r = app.run(img)
    out = r.annotated()
    bh = out.shape[0] - img.shape[0]
    assert bh > 0, "the banner was painted over the image"
    plain = r.annotated(banner=False)
    assert plain.shape == img.shape


def test_annotated_can_leave_out_labels_and_passing_tools(pkg):
    from vision_station.core import vsruntime
    path, _m = pkg
    app = vsruntime.load(path)
    img = cv2.imread(os.path.join(DEMO, "bad.png"))
    r = app.run(img)
    full = r.annotated(banner=False)
    bare = r.annotated(banner=False, labels=False)
    fails = r.annotated(banner=False, only_failures=True)
    changed = lambda a: int((np.abs(a.astype(int) - img.astype(int)).sum(axis=2) > 0).sum())
    assert changed(bare) < changed(full), "labels=False still drew the labels"
    assert changed(fails) < changed(full), "only_failures still drew passing tools"
    assert changed(fails) > 0


def test_runtime_values_failed_tools_and_errors(pkg):
    from vision_station.core import vsruntime
    path, _m = pkg
    app = vsruntime.load(path)
    r = app.run(os.path.join(DEMO, "bad.png"))
    assert not r.passed and "led_row" in r.failed_tools
    assert r.value("led_row", "count") == 3
    assert r.value("nowhere", "x", "default") == "default"
    with pytest.raises(KeyError):
        app.run(os.path.join(DEMO, "good.png"), job="no such job")
    with pytest.raises(IOError):
        app.run("missing.png")


def test_multi_job_package_runs_every_job_and_names_them(tmp_path, monkeypatch):
    from vision_station.core import job as jobmod
    from vision_station.core import package
    from vision_station.core import vsruntime
    from vision_station.core.schema import defaults
    monkeypatch.setenv("VISION_STATION_CACHE", str(tmp_path / "cache"))
    img = cv2.imread(os.path.join(DEMO, "good.png"))
    j = jobmod.create_job(str(tmp_path / "chassis"), "Chassis")
    for sid, name in (("top", "Top"), ("side", "Side")):
        j.add_side(sid, name, img, [{"id": "h", "type": "histogram",
                                     "params": dict(defaults("histogram"), roi=[10, 10, 50, 50])}])
    package.export_package(str(tmp_path / "chassis"), str(tmp_path / "c.vsp"))
    app = vsruntime.load(str(tmp_path / "c.vsp"))
    assert app.jobs == [("top", "Top"), ("side", "Side")]
    with pytest.raises(ValueError):
        app.run(img)                              # several jobs: must say which
    res = app.run_all({"Top": img, "side": img})
    assert app.verdict(res) == "PASS"
    assert app.verdict({"top": res["top"]}) == "INCOMPLETE"


def test_import_puts_the_application_back_without_overwriting(pkg, tmp_path):
    from vision_station.core import package
    path, _m = pkg
    root = tmp_path / "projects"
    root.mkdir()
    a = package.import_package(path, str(root))
    b = package.import_package(path, str(root))
    assert a != b and os.path.exists(str(root / a / "project.json"))


def test_a_package_from_a_newer_version_is_refused(tmp_path):
    from vision_station.core import vsruntime
    p = tmp_path / "future.vsp"
    with zipfile.ZipFile(str(p), "w") as z:
        z.writestr("manifest.json", json.dumps({"format": "vision-station-package",
                                                "version": 99, "name": "x"}))
    with pytest.raises(ValueError) as e:
        vsruntime.load(str(p))
    assert "newer" in str(e.value)


def test_a_package_cannot_write_outside_its_folder(tmp_path, monkeypatch):
    """A crafted zip with ../ paths must not write outside the cache."""
    from vision_station.core import vsruntime
    monkeypatch.setenv("VISION_STATION_CACHE", str(tmp_path / "cache"))
    p = tmp_path / "evil.vsp"
    with zipfile.ZipFile(str(p), "w") as z:
        z.writestr("manifest.json", json.dumps({"format": "vision-station-package",
                                                "version": 1, "name": "x"}))
        z.writestr("../../escaped.txt", "x")
    with pytest.raises(ValueError):
        vsruntime.load(str(p))
    assert not (tmp_path / "escaped.txt").exists()


# ------------------------------------------------------------- taught sources and job files
def _taught_app(tmp_path, sources):
    """A two-job application whose jobs were taught image sources."""
    from vision_station.core import job as jobmod
    from vision_station.core.schema import defaults
    img = cv2.imread(os.path.join(DEMO, "good.png"))
    j = jobmod.create_job(str(tmp_path / "app"), "Panel")
    tool = [{"id": "leds", "type": "count_blob",
             "params": dict(defaults("count_blob"), roi=[120, 110, 420, 80],
                            channel="saturation", min_area=200, expect_count=4)}]
    for sid, name in (("top", "Top"), ("side", "Side")):
        s = j.add_side(sid, name, img, tool)
        s.spec["image_source"] = sources[sid]
        s.save(s.spec)
    return str(tmp_path / "app")


def test_run_taught_uses_each_jobs_taught_images(tmp_path, monkeypatch):
    from vision_station.core import vsruntime
    monkeypatch.setenv("VISION_STATION_CACHE", str(tmp_path / "cache"))
    folder = tmp_path / "top_parts"
    folder.mkdir()
    for n in ("1.png", "2.png"):
        shutil.copy(os.path.join(DEMO, "good.png"), str(folder / n))
    shutil.copy(os.path.join(DEMO, "bad.png"), str(folder / "3.png"))
    app = vsruntime.load(_taught_app(tmp_path, {
        "top": {"mode": "folder", "folder": str(folder)},
        "side": {"mode": "files", "paths": [os.path.join(DEMO, "good.png")]}}))
    seen = []
    results = app.run_taught(save_to=str(tmp_path / "out"), on_result=seen.append)
    assert [(r.job_id, os.path.basename(r.source), r.verdict) for r in results] == [
        ("top", "1.png", "PASS"), ("top", "2.png", "PASS"), ("top", "3.png", "FAIL"),
        ("side", "good.png", "PASS")]
    assert len(seen) == 4, "the callback did not receive every result"
    saved = sorted(os.listdir(str(tmp_path / "out")))
    assert len(saved) == 4 and any(n.endswith("FAIL.jpg") for n in saved)
    assert app.verdict(results) == "FAIL"


def test_an_unreadable_image_is_reported_without_stopping_other_jobs(tmp_path, monkeypatch):
    from vision_station.core import vsruntime
    monkeypatch.setenv("VISION_STATION_CACHE", str(tmp_path / "cache"))
    app = vsruntime.load(_taught_app(tmp_path, {
        "top": {"mode": "files", "paths": [str(tmp_path / "missing.png")]},
        "side": {"mode": "files", "paths": [os.path.join(DEMO, "good.png")]}}))
    results = app.run_taught()
    assert [r.verdict for r in results] == ["ERROR", "PASS"]
    assert "missing.png" in results[0].error
    assert app.verdict(results) == "INCOMPLETE"


def test_only_failures_saves_just_the_rejects(tmp_path, monkeypatch):
    from vision_station.core import vsruntime
    monkeypatch.setenv("VISION_STATION_CACHE", str(tmp_path / "cache"))
    app = vsruntime.load(_taught_app(tmp_path, {
        "top": {"mode": "files", "paths": [os.path.join(DEMO, "bad.png")]},
        "side": {"mode": "files", "paths": [os.path.join(DEMO, "good.png")]}}))
    app.run_taught(save_to=str(tmp_path / "out"), only_failures=True)
    assert [n.split("__")[0] for n in os.listdir(str(tmp_path / "out"))] == ["Top"]


def test_generated_job_file_runs_every_job_as_taught(tmp_path):
    """python <job>.py with no arguments: every job on its taught images, the
    results written where taught -- in a clean process, from another folder."""
    from vision_station.core import package
    from vision_station.core import job as jobmod
    app_folder = _taught_app(tmp_path, {
        "top": {"mode": "files", "paths": [os.path.join(DEMO, "bad.png")]},
        "side": {"mode": "files", "paths": [os.path.join(DEMO, "good.png")]}})
    j = jobmod.Job(app_folder)
    j.spec["output"] = {"folder": str(tmp_path / "results"), "show": False,
                        "only_failures": False}
    j.save()
    py, vsp, _m = package.export_job_file(app_folder, str(tmp_path / "out" / "panel job.py"))
    assert os.path.basename(py) == "panel_job.py" and os.path.exists(vsp)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path),
           "VISION_STATION_CACHE": str(tmp_path / "cache")}
    run = subprocess.run([sys.executable, py], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=240)
    assert run.returncode == 1, run.stdout + run.stderr[-1500:]       # one job fails
    assert "Top" in run.stdout and "FAIL" in run.stdout and "Panel: FAIL" in run.stdout
    assert sorted(n.split("__")[0] for n in os.listdir(str(tmp_path / "results"))) == \
        ["Side", "Top"]
    # a different source from the command line
    run2 = subprocess.run([sys.executable, py, "--source", "Top=" + os.path.join(DEMO, "good.png"),
                           "--output", str(tmp_path / "r2")], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=240)
    assert run2.returncode == 0 and "Panel: PASS" in run2.stdout, run2.stdout + run2.stderr[-800:]


def test_job_file_can_be_imported_by_other_code(tmp_path):
    from vision_station.core import package
    app_folder = _taught_app(tmp_path, {
        "top": {"mode": "files", "paths": [os.path.join(DEMO, "good.png")]},
        "side": {"mode": "files", "paths": [os.path.join(DEMO, "good.png")]}})
    py, _v, _m = package.export_job_file(app_folder, str(tmp_path / "panel.py"))
    code = ("import sys; sys.path.insert(0, %r)\n"
            "import panel\n"
            "rs = panel.run(show=False, output_folder='')\n"
            "print([(r.job_name, r.verdict) for r in rs])\n") % str(tmp_path)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path),
           "VISION_STATION_CACHE": str(tmp_path / "cache")}
    out = subprocess.run([sys.executable, "-c", code], cwd="/", env=env, capture_output=True,
                         text=True, timeout=240)
    assert "[('Top', 'PASS'), ('Side', 'PASS')]" in out.stdout, out.stdout + out.stderr[-800:]


def test_a_job_file_and_its_images_can_move_together(tmp_path):
    """Relative sources and output are taken from the job file's own folder,
    so copying the folder to another PC and running it from anywhere works."""
    from vision_station.core import package
    app_folder = _taught_app(tmp_path, {
        "top": {"mode": "files", "paths": ["imgs/bad.png"]},
        "side": {"mode": "folder", "folder": "imgs"}})
    from vision_station.core import job as jobmod
    j = jobmod.Job(app_folder)
    j.spec["output"] = {"folder": "out", "show": False}
    j.save()
    first = tmp_path / "made_here"
    py, _v, _m = package.export_job_file(app_folder, str(first / "panel.py"))
    (first / "imgs").mkdir()
    shutil.copy(os.path.join(DEMO, "bad.png"), str(first / "imgs" / "bad.png"))
    moved = tmp_path / "another_pc" / "panel"
    shutil.copytree(str(first), str(moved))
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path),
           "VISION_STATION_CACHE": str(tmp_path / "cache")}
    run = subprocess.run([sys.executable, str(moved / "panel.py")], cwd="/", env=env,
                         capture_output=True, text=True, timeout=240)
    assert "Top" in run.stdout and "Side" in run.stdout, run.stdout + run.stderr[-800:]
    assert "ERROR" not in run.stdout, "a relative path was resolved against the wrong folder"
    assert len(os.listdir(str(moved / "out"))) == 2, "results were not written beside the job file"


def test_the_result_gives_your_code_pose_shapes_values_and_plain_data(pkg):
    """For code that takes it from here: where the part is, what each tool
    found as geometry, every measurement, and the whole result as JSON."""
    from vision_station.core import vsruntime
    path, _m = pkg
    app = vsruntime.load(path)
    ref = cv2.imread(os.path.join(DEMO, "reference.png"))
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
    M[0, 2] += 12
    M[1, 2] -= 7
    moved = cv2.warpAffine(ref, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    r = app.run(moved)
    p = r.pose()
    assert p is not None and p["tool"] == "locate_part"
    assert abs(abs(p["angle"]) - 2.0) < 0.3, p
    x, y = M @ np.array([170.0, 150.0, 1.0])            # the red LED, moved
    rx, ry = r.to_reference(x, y)
    assert abs(rx - 170.0) < 1.5 and abs(ry - 150.0) < 1.5, (rx, ry)
    leds = r.shapes("led_row")
    assert len(leds) == 4 and all(s["kind"] in ("rect", "poly") for s in leds), leds
    assert not any(s["tool"] == "led_row" and "led_row" == s["label"] for s in leds), \
        "region outlines should be left out by default"
    assert r.values("led_row")["count"] == 4 and r.values("nope") == {}
    d = r.to_dict()
    text = json.dumps(d)                                # must be plain data
    assert '"verdict"' in text and d["pose"]["tool"] == "locate_part"
    assert set(d["tools"]) == set(r.tools)


def test_gallery_application_shows_every_tool_working(tmp_path):
    """The gallery photographs must show tools working: every tool on the good
    panel, chassis and label passes; the rejected panel fails for exactly its
    planted defects -- a missing LED and screw, a misprint, a scratch."""
    import make_gallery as G
    from vision_station.core import job as jobmod
    j = jobmod.Job(G.build_app(str(tmp_path)))
    from vision_station.tools import REGISTRY
    used = set()
    for side in j.sides:
        src = side.spec.get("image_source")
        img = cv2.imread(src["paths"][0]) if src else side.reference
        res = side.inspect(img)
        used |= {t["type"] for t in res["tools"]}
        failed = {t["tool_id"] for t in res["tools"] if t["status"] != "pass"}
        if side.side_id == "reject":
            assert failed == {"led_row", "led_colours", "corner_screws", "lot_code", "surface",
                              "panel_surface"}, failed
        elif side.side_id == "bead":
            assert failed == {"glue"}, failed          # the planted gap
            r = {t["tool_id"]: t for t in res["tools"]}["glue"]
            assert r["values"]["gaps"] >= 1
        else:
            assert not failed, "%s: %s" % (side.name, failed)
    needs_own_model = {"yolo_detect", "yolo_defect", "anomaly"}
    if not any(s.side_id == "chassis" for s in j.sides):
        # The chassis side is built from production photographs that are not
        # published; without them its tools are not shown, so every tool cannot
        # be expected to appear.
        return
    assert used >= set(REGISTRY) - needs_own_model, set(REGISTRY) - needs_own_model - used


def test_one_model_called_from_many_threads_gives_each_its_own_result(tmp_path):
    """A cached model is shared by every job using it. Called from several
    threads at once -- jobs run in parallel -- each call must get the result
    for its own input, exactly as if the calls had been made one by one."""
    from concurrent.futures import ThreadPoolExecutor
    from vision_station.core import inference
    from make_test_onnx import build_color_classifier
    path = build_color_classifier(str(tmp_path / "m.onnx"), ["r", "g", "b"])
    inference.clear_cache()
    rng = np.random.default_rng(3)
    xs = [rng.random((1, 3, 32, 32)).astype(np.float32) for _ in range(48)]
    model = inference.load(path)
    one_by_one = [np.asarray(model(x)[0]) for x in xs]
    with ThreadPoolExecutor(max_workers=8) as ex:
        together = list(ex.map(lambda x: np.asarray(inference.load(path)(x)[0]), xs))
    for a, b in zip(one_by_one, together):
        assert np.allclose(a, b, atol=1e-6), "a parallel call got another input's result"
    assert len({id(inference.load(path)) for _ in range(4)}) == 1, "the model was built twice"
