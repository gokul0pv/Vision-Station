# ADR-001: Vision Core — a C++ inspection application

**Status:** Accepted
**Date:** 2026-11-26
**Deciders:** Gokul PV

## Context

Two applications exist today:

- **Vision Station** (`vision_station/`, Python 3.12 + PySide6 + OpenCV 4.10 +
  onnxruntime, ~38,400 lines). Production; its results are the reference.
  Uses ~471 MB, needs Python and its packages installed.
- **Vision Next** (`vision_next/`, Rust + a web UI in WebView2, ~15,000 lines).
  ~220 MB, one exe, and since the OpenCV bridge 100 % of 61,424 checks agree
  with Vision Station — but its interface still has bugs.

The goal is a third, separate application in C/C++, in the spirit of Cognex
VisionPro: a native, robust inspection engine and editor that must not fail on
the line. Neither existing application may be changed until the new one is
proven.

Constraints and requirements:

- **Same results as Vision Station**, check for check (status, score, values).
- **Never stop the line**: no crash, hang or lost result takes the station
  down. A failure is reported as a failed or errored check, never a dead app.
- CPU-only on customer PCs (GPU optional), low memory, Windows 10/11 x64.
- Hardware: Basler GigE camera (pylon), serial light controller, barcode
  scanner, TCP line interface.
- Existing jobs (project.json, reference.png, models, training pictures) must
  open unchanged.
- Toolchain available on this PC: MSVC 14.44 + Windows SDK (portable),
  official OpenCV 4.10.0 build, pylon SDK (installed, C++ headers present).

## Decision

Build **Vision Core** in `vision_core/`: C++17, MSVC, CMake + Ninja.

| Area | Choice |
|---|---|
| Image processing | Official **OpenCV 4.10.0** (opencv_world, IPP) — same library as Vision Station's cv2, results bit-identical (proven: 3x cubic 0/10M values, SIFT, align) |
| Neural networks | **ONNX Runtime 1.24** C++ API, CPU default, DirectML optional — same version Vision Station runs |
| Camera | **pylon C++ SDK** (Basler), opened by IP as today |
| Barcode | **zbar** (the library pyzbar wraps) — same decoder, same results |
| Light, scanner | Win32 serial (overlapped I/O, timeouts on every call) |
| Results | **SQLite** (amalgamation, WAL) in Vision Station's `results.db` layout |
| JSON | nlohmann/json (header only) |
| UI | **Slint** (declarative, native GPU rendering, C++ API) — chosen 2026-11-26 over Qt (not wanted) and WebView2/WinUI; fully modern design: rounded corners on every surface and button, soft shadows, Lucide icons, Inter font, light/dark |
| Tests | GoogleTest + a parity harness against Vision Station |

### Process architecture (fault isolation)

```
 vc-supervisor.exe  (tiny; starts, watches, restarts; owns nothing else)
     |-- vc-engine.exe   headless: camera, light, IO, inspection, results
     |                   named pipe API; survives the UI closing or crashing
     '-- vc-studio.exe   Slint UI: run view, job editor, history, settings
```

- The engine never runs UI code; the UI never touches hardware. Either can
  crash or be restarted without taking down the other.
- Inside the engine every check runs behind a guard: C++ exceptions and
  Windows structured exceptions (access violation, divide by zero) become an
  `error` result for that check; a per-inspection time limit turns a hang
  into a reported timeout; the supervisor restarts the engine if it stops
  answering its heartbeat, and the line sees "engine restarted" — never
  silence.
- Every result is written to the journal before it is reported; a crash
  dump (MiniDumpWriteDump) is kept for each failure.
- Job files are validated on load (schema, files present, models open)
  before a job can go live; a bad job is refused, not run.

## Options Considered

### Option A: C++ (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | High (manual memory, but contained with RAII and guards) |
| Result parity | Exact: same OpenCV/ORT/zbar binaries as Vision Station |
| Robustness | Proven model (VisionPro, Halcon are C/C++); fault isolation by process |
| Memory | Lowest |
| Team familiarity | New code base; matches the request |

**Pros:** exact parity by construction; native speed; every vision library is
native C/C++; Qt lets Vision Station's UI be ported rather than redesigned.
**Cons:** C++ allows memory bugs Rust would block — mitigated by sanitizers
(ASan in test builds), RAII-only ownership, process isolation, fuzzing.

### Option B: Fix Vision Next (Rust + web UI)
| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Result parity | Exact via the C++ bridge |
| Robustness | Good engine; UI bugs remain to be found |

**Pros:** most work done. **Cons:** not what was asked; the web UI is the
source of the remaining bugs; two languages.

### Option C: Keep Python, harden it
**Pros:** reference already. **Cons:** memory, start-up, packaging size,
Python + Qt in one process (a UI fault stops inspection).

## Trade-off Analysis

Exact results are only cheap when the same libraries do the work; C++ gets
that for free, where Rust needed a bridge and hand ports. The main risk of C++
— memory-safety bugs — is addressed by the process split (a fault costs one
inspection, not the line), the per-check guard, and a test regime (parity,
fuzzing, sanitizers, soak) that is stricter than either existing app has.
"Bug free" cannot be promised by any language; it is approached by testing
every check against Vision Station on real pictures and by designing so that
the bugs that remain cannot stop the station.

## Consequences

- Easier: parity (same binaries), packaging (a folder of DLLs, no Python),
  memory, start-up time, crash containment.
- Harder: the picture canvas (zoom, region handles, overlays) is custom work in
  Slint. Slint's royalty-free licence requires visible attribution ("Made with
  Slint" in About); a paid licence removes that.
- Revisit: training stays in Python on the GPU PC (exports ONNX); the AGPL
  licence of Ultralytics models must be settled before public release.

## Phases

| Phase | Content | Exit test |
|---|---|---|
| 0 | Skeleton: CMake/Ninja, deps, core types, job loader (Vision Station format), engine loop, check guard, `vc-run` CLI | Loads every job in `projects/`; runs with no checks |
| 1 | Checks with OpenCV only: align, presence, count, colour, defect, geometry, measure, logic | Parity harness 100 % on the 182-picture job and the demo jobs |
| 2 | ONNX checks: OCR/OCV (PP-OCR pipeline), YOLO, classify, AI part finder, advanced training apply | Parity 100 % incl. text, GPU and CPU |
| 3 | Engine service: camera, light, scanner, TCP, results journal, supervisor, crash dumps | Fault-injection suite (unplug camera, kill light port, corrupt picture, kill engine) passes; 24 h soak, no leak |
| 4 | Vision Core Studio (Slint): run view, job editor, canvas, history, settings — Vision Station's features, new modern design | UI test suite; operator walkthrough |
| 5 | Packaging, installer, upgrade of existing jobs | Clean-PC install, CPU-only |

## Action Items

1. [x] Name "Vision Core"; UI = Slint, modern rounded design (decided 2026-11-26).
2. [ ] Phase 0: fetch CMake, Ninja, ONNX Runtime 1.24 C++ package, nlohmann/json, SQLite, GoogleTest.
3. [ ] Phase 0: engine core + `vc-run`; parity harness reusing Vision Station's cached results.
4. [ ] Phase 1 onward as above; nothing in `vision_station/` or `vision_next/` changes.
