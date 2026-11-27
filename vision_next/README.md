# Vision Next

Vision Station's engine and interface rebuilt in Rust: one process, no Python,
no Qt. It sits beside Vision Station and does not change it.

| | |
|---|---|
| Engine | Rust: checks, template matching, edges, blobs, OCR pipeline, YOLO |
| Window | Windows' own WebView2 (tao + wry): no browser engine shipped |
| Interface | `ui/` – plain HTML, CSS and JavaScript, built into the program |
| Inference | ONNX Runtime (DirectML) loaded from `onnxruntime.dll` at start |
| Camera | Basler pylon C library (`PylonC_v*.dll`) loaded at start; GigE by address |
| Recipes | the same `job.json` / `project.json` folders as Vision Station |

## Build and run

```
run.bat                 build if needed, then start the window
cargo build --release   target\release\vision-next.exe and vs-run.exe
```

Rust's GNU toolchain is enough (no Visual Studio), plus GNU `as` for its
`dlltool`: MSYS2's binutils (with zlib, zstd, libwinpthread, gettext-runtime,
libiconv) unpacked to `%LOCALAPPDATA%\mingw-binutils\bin`, which `run.bat`
puts on PATH. During development the
program finds `runtime\onnxruntime.dll` and `..\models` by searching upward
from its own folder; `VS_ORT_DLL`, `VS_MODELS` and `VS_PROJECTS` override.

## Measured (this PC, 28 cores)

| | Vision Station (Python) | Vision Next |
|---|---|---|
| real board job, 172 checks, 20 MP picture | ~2.8 s per inspection | ~0.65 s (CPU), ~1.2 s (DirectML) |
| demo LED panel, 9 checks | ~70–90 ms | ~36–42 ms |
| memory, that job open and run | engine alone 471 MB (peak 615) | whole app 209–243 MB (peak ~480) |
| program size | 701 MB folder | 7.4 MB exe + ONNX Runtime 40 MB + models |

Independent checks in a job run side by side on every core; models run on
the CPU by default because there they serve those checks at once
(`VS_DEVICE=gpu` for DirectML).

## Without a window

```
vs-run serve --port 8765 [--open APP]     engine + pages at http://127.0.0.1:8765
vs-run inspect projects\myjob pictures\     run a recipe on pictures, print verdicts and times
vs-run list
```

## Projects

Vision Next keeps its own `projects\` folder. The start page offers Vision
Station's applications to **import**: they are copied (without results and
training sets) and Vision Station's files stay as they were.

## Checks and station features

All 40 of Vision Station's check types run, with every setting (Settings,
Inputs: fixture / picture / linked settings, Graphics, Train, Result).
Trained surface models are .npz files either program opens.

Station: cameras (exposure, gain, auto modes, white balance, pixel format,
gamma, black level, trigger, frame rate, packets, rotation, mirror, live view),
result pictures (annotated, per day, per barcode), line TCP interface
(TRIGGER / GET / STATUS / CODE / PING / VERSION, broadcast), light controller
(HIFLY / OPT serial protocol), barcode-gated inspection (USB or serial
scanner), operator mode with supervisor password, history with thumbnails and
search, packages (.vnp export / import), undo / redo, command palette (Ctrl+K).

Not here: Vision Station's deep-network surface models (retrain as
statistical), advanced-training confirmation (model teaching run), the AI part
finder, lens calibration and automatic light levelling.
