# Vision Station

Region-based inspection builder with an XD-style canvas and a built-in YOLO
labelling workspace. Label here, train on a GPU machine, run on the line PC
with no GPU.

## Install

Double-click **install.bat**. It checks the Python version, installs the core
packages, then attempts each optional one separately so a single failure does
not stop the rest, and finishes by reporting what actually ended up installed.

The batch file is only a launcher; the work is in `install.py`. That split is
deliberate. Batch cannot be executed from a Linux development machine, so the
version check, the tiered installs and the failure handling were all shipped
having never run once. In Python they are tested against a fake pip: optional
packages failing, a core package failing, pip missing entirely, and an
unsupported Python version. A test also asserts the batch file stays a
launcher, because logic that creeps back into it becomes untestable again.

```
python install.py --dry-run    show what would happen, change nothing
python install.py --core       core packages only
python install.py --all        include the large OCR download
```

By hand:

```
pip install PySide6 opencv-python numpy onnxruntime
python preflight.py
python app.py
```

`preflight.py` lists every package, what each one is for, and the exact command
to install anything missing. Nothing optional is fatal: the application starts
without a camera SDK, without an OCR engine and without ONNX, and says which
tools are unavailable as a result. It is also in the app under **Help → Check
installation**, and a thin environment is flagged in the status bar at startup.

That check exists to prevent a specific and expensive failure. A recipe tuned on
a bench machine with PaddleOCR, deployed to a line PC that only has Tesseract,
reads differently -- and nothing in the inspection result would tell you why. The
engine that actually read a marking is now recorded in the result and in the
results log.

**A missing library no longer looks like a bad part.** `Read code` used to
swallow the ImportError from a missing decoder and report "no code decoded",
which is indistinguishable from a code the camera genuinely cannot read. That
sends an operator to check lighting and focus for what is really a missing pip
package. It now reports the missing package by name. OCR is optional — add `pip install paddleocr paddlepaddle`
when you need the text tools.

Note: on the line PC install `onnxruntime`, **not** `onnxruntime-gpu`.

---

## Two workspaces

The toolbar's top-left pair switches between them.

**Inspect** — build the recipe, draw tool regions, run against camera or images.

**Label** — draw training boxes on your test images and export a YOLO dataset.

---

## The model workflow

**1 · Add test images.** Left dock, *Add images*. This is the step that decides
your model's quality. Get the real variation in: every lighting condition, every
part orientation, good parts and bad. 100+ images per class.

**2 · Switch to Label, create your classes.** *New class* for each thing you want
found — `screw`, `led`, `scratch`, `chip_body`. Keep the list minimal; every class
you add needs its own 100 examples.

**3 · Draw boxes.** Click a test image, pick a class, drag on the picture. Box each
object tightly. Label *every* instance in every image you touch — one missed screw
teaches the model that screws are sometimes background, and that single mistake
costs more accuracy than fifty extra images will win back.

Same canvas controls as the Inspect workspace: drag to move, handles to resize,
arrow keys to nudge, Del to remove, wheel to zoom, space-drag to pan.

**4 · Check dataset.** Flags thin classes and imbalance *before* you spend GPU
hours discovering them.

**5 · Export for training.** Writes a standard Ultralytics folder:

```
<model>_dataset\
  images\train\  images\val\
  labels\train\  labels\val\
  data.yaml
  train.py          <- tuned for inspection, read the comments
  README.txt
```

**6 · Train on a GPU machine.**

```
pip install ultralytics
python train.py
```

Read the mAP50 it prints at the end:

| mAP50 | Meaning |
|---|---|
| above 0.95 | ready for the line |
| 0.85 – 0.95 | usable, expect occasional misses |
| below 0.85 | label more images, not more epochs |

The last line is the path to `best.onnx`.

**7 · Deploy.** Copy `best.onnx` into `projects\<model>\models\`. Switch to
Inspect, add **Tools → Trained models → Detect with YOLO**, draw its region, set
*Model file* to `models/best.onnx`. No GPU needed — inference is one forward pass
and runs 15–40 ms on CPU with yolov8n.

Class names travel inside the ONNX file, so the tool discovers them itself.

---

## The two YOLO tools

**Detect with YOLO** — counts and identifies. Set *Expected per class* to
`{"screw": 4, "led": 4}` and it fails the part when the count is wrong.

**Find defects with YOLO** — reject-on-detection. Any box above the confidence
threshold fails. Use with a model trained on defect classes.

Both take a region, so you can run different models on different parts of the
same board.

---

## When to use YOLO and when not to

YOLO is the better choice for counting screws, bolts, LEDs, or components under
variable lighting or slight rotation — it learned from examples instead of one
reference crop.

It is the *worse* choice for measurement (boxes are pixel-accurate at best,
`Measure length` gives you 1/10 pixel), for alignment (`Locate part` gives you a
pose transform, YOLO gives you boxes), and often for fine surface defects where
golden-image differencing is more sensitive and needs no training data at all.

Mix them freely. A realistic recipe: `Locate part` → `Detect with YOLO` for the
components → `Measure length` for the critical dimension → `Verify text` for the
lot code.

---

## All tools

| Group | Tool |
|---|---|
| Locate | Locate part, Calibrate scale |
| Presence and count | Part present, Count by shape, Count by brightness |
| Colour | LED colour, LED lit state |
| Surface | Dents and scratches, AI surface check |
| Trained models | Detect with YOLO, Find defects with YOLO |
| Text and codes | Read text, Verify text, Read code |
| Measurement | Measure length |

Put `Locate part` first. If it fails, the engine stops rather than reporting a
page of meaningless failures from regions that landed in the wrong place.

---

## Canvas controls

| Action | How |
|---|---|
| Select region | Click |
| Add to selection | Shift-click |
| Select several | Marquee-drag on empty canvas (Inspect workspace) |
| New label box | Drag on empty canvas (Label workspace) |
| Move | Drag, or arrows (Shift = 10 px) |
| Resize | Eight handles; Shift keeps aspect ratio |
| Rotate | Round grip above a selected region; Shift snaps to 15 deg |
| Undo / redo | Ctrl+Z / Ctrl+Shift+Z (Ctrl+Y also redoes) |
| Copy / paste | Ctrl+C / Ctrl+V |
| Duplicate | Ctrl+D, or Alt-drag |
| Delete | Del |
| Pan | Space+drag, H for hand tool, or middle-drag |
| Zoom | Wheel, Ctrl+0 fit, Ctrl+1 actual, Ctrl+2 to selection |
| Run | F5 once, F6 continuous |
| Diagnostics | F8 |
| Production history | F9 |

Help → Shortcuts has the full list. Dataset → How to train repeats step 1–8.

---

## Files

```
app.py       main window, both workspaces
canvas.py    image view, selection, snapping, tool and label layers
schema.py    which widget each tool parameter gets
dataset.py   label storage, quality checks, YOLO export
engine.py    loads project.json, runs the tool chain
vstools\     the inspection tools, including yolo.py
projects\    one folder per model:
               project.json  the recipe
               reference.png the golden part
               images\       test images
               dataset\      your labels
               models\       trained .onnx files
```


---

## Basler Pylon cameras

GigE and USB3 Basler cameras are supported through the Pylon SDK. If pypylon is
missing or no Basler camera is found, the app falls back to an OpenCV device so
it still runs.

**Install**

1. Pylon SDK from baslerweb.com (the SDK must come first)
2. `pip install pypylon`

**Use**

*Start camera* lists the Basler cameras it can actually see, with vendor, model
and serial, so you pick from a list instead of guessing an index. The status bar
confirms which camera connected and over which backend.

*File -> Camera settings* controls exposure, gain and white balance. The dialog
reads the real limits off your camera and shows them, so you cannot dial in an
exposure the sensor will silently clamp. Changes apply live while you watch the
image.

Once your lighting is fixed, turn every auto setting off. Auto exposure drifting
between parts is one of the most common causes of intermittent false rejects.

**Camera family note**

Exposure and gain live under different GenICam node names depending on the
camera: older GigE "classic" ace cameras use `ExposureTimeAbs` and `GainRaw`,
while USB3, ace2 and dart cameras use `ExposureTime` and `Gain`. This is handled
automatically -- the code checks which nodes your camera exposes. You do not
need to know which family you own.

## Rotated regions

Select a region and drag the round grip above it, or type an exact value into
Angle on the Region tab. Hold Shift while dragging to snap to 15 degrees.

This matters for angled components. A plain rectangle around a rotated chip has
to be oversized to contain it, so the region drags in background, neighbouring
parts and solder that the tool then has to cope with. A rotated region bounds the
part tightly instead.

Underneath, a rotated region is deskewed before the tool sees it, so every tool
receives an upright patch and none of them needed changing. Result overlays are
mapped back through the same transform and drawn as true quadrilaterals rather
than misleading axis-aligned boxes.

Angle is stored per tool in `project.json` and travels with the recipe.

## Anomaly tool: shipped, never executed

The AI surface-check tool had never been run once -- not in testing, not
anywhere -- because exercising it looked like it needed a trained anomalib
model. It does not: `make_test_onnx.py` now also emits a fake heatmap model,
which covers preprocessing, the threshold comparison, blob extraction and the
region offset.

It had a real defect. The onnxruntime session was cached in a single unkeyed
class attribute, so the first model loaded was reused for every anomaly tool in
the recipe. A second tool pointed at a different model silently returned the
first model's heatmap -- on a two-surface inspection, one surface would never
actually be checked, and nothing in the result would say so. Sessions are now
keyed by model path.

It also asked for the CUDA execution provider unconditionally, which on a line
PC with no GPU prints a warning on every inspection. CPU is now the default,
with a **Use GPU if available** setting, matching the YOLO tools.

## Canvas interaction: three bugs, one of them serious

Coverage put `canvas.py` at 45%, the lowest of any module you touch directly.
The gap was not accidental: every test drove selection programmatically through
`select_ids()`, so not one of a hundred passing tests had ever generated a mouse
event. Driving the canvas with real Qt events found three defects immediately.

**Clicking a region was broken outright.** `RegionItem` stored the region shape
in an attribute named `shape`, which shadows `QGraphicsItem.shape()` -- the
method Qt calls to hit-test a click. Clicking a region, marquee selection and
`contains()` all raised `TypeError`. The primary interaction of the application
had been broken since elliptical regions were added, and the test suite was
silent about it. The attribute is now `region_shape`, and a test asserts
`shape` is still a method.

**Shift-click replaced the selection instead of extending it.** Qt's own
multi-select modifier is Ctrl; Shift is what every design tool uses and what the
shortcut list here promises. Both now work. The fix needed two parts, because Qt
finishes a plain click's selection on *release* -- suppressing the press alone
left the extra selection undone a moment later.

**Regions snapped to an invisible margin.** `boundingRect` is deliberately
inflated to leave room for the grab handles and the label above the box, and
snapping used `sceneBoundingRect()`, so edges aligned to a boundary about 15 px
outside the region. Guides pointed at nothing and edges landed short. Snapping
now uses the region's own rectangle.

There are now 25 canvas tests driven by real mouse events -- click, shift-click,
marquee, drag, group drag, handle resize, arrow nudge, alt-drag, hand-tool pan,
snapping, and label drawing -- plus app-level tests that a click reaches the job
tree and the property sheet.

## Basler: a units bug that would have hit real hardware

The camera code was written from the real pypylon source but has never run
against a camera, and cannot be run against one here. Coverage put it at 21%,
the least tested module in the app and the next thing that gets plugged in.

`fake_pylon.py` simulates four cameras -- a USB3/ace2, an older GigE "classic"
ace, one exposing almost no nodes, and one whose nodes are read-only -- plus
grab failures and mid-run disconnects. The real `PylonCamera` class runs against
it unmodified. Coverage went from 21% to 84%.

It found a units bug. Gain is not one quantity across Basler's range: USB3, ace2
and dart cameras expose `Gain` in decibels, while older GigE ace cameras expose
`GainRaw`, an integer in sensor-specific units typically spanning 0-512. The code
wrote a decibel figure straight into whichever node it found, so on a classic
GigE camera a request for 3 dB became a raw value of 3 -- essentially no gain --
while the dialog reported decibels. Some models also reject a float on an integer
node outright.

Gain now discovers its unit, converts, and hands integer nodes integers. The
settings dialog labels the control **Gain (raw)** or **Gain (dB)** to match the
camera in front of it, and says so plainly when the unit is raw.

Nothing here proves the code works against real hardware -- only that it does
the right thing against a faithful model of the API. The first real camera is
still the real test.

## Responsiveness

Three things were making the application feel slow. None of them was the
rendering, which measures well under a millisecond per operation.

**Inspection ran on the interface thread.** A 280 ms recipe therefore froze the
whole window for 280 ms of every cycle. Measured with a 50 ms probe timer during
continuous running, the event loop was serviced only every ~300 ms, so clicks,
repaints and menu input all stalled. Inspection now runs on a worker thread.

```
                          before      after
  UI servicing interval   304 ms      50 ms
  probe ticks in 4 s      8           76
  inspections in 4 s      12          12
```

Throughput is unchanged; the interface is simply no longer blocked while it
happens.

**Continuous mode used a repeating 120 ms timer** against a cycle that takes
longer than 120 ms, so Qt queued timeouts faster than they could be serviced.
The next cycle is now started when the previous one finishes.

**Text reading tried four orientations on every part.** The auto-rotate search
runs the recogniser four times, which made OCR 77% of the inspection cycle --
504 ms of 652 ms. Text is upright on the overwhelming majority of parts, so the
search is a recovery path, not the normal one: a confident upright read is
accepted immediately and the other three turns are never attempted. Rotated text
still reads at every angle, it just costs more when it happens.

```
  cycle time   652 ms  ->  284 ms
  OCR          504 ms  ->  139 ms
```

## Undo while drawing

<kbd>Ctrl</kbd>+<kbd>Z</kbd> during a polygon trace takes back the last point
placed. It previously went straight to the recipe-snapshot history, which knows
nothing about a trace in progress, so the keystroke did nothing visible and the
only way out of a misplaced corner was to abandon the whole polygon with
<kbd>Esc</kbd>.

## The interface

**A tool palette down the left**, the way a drawing application works: Select,
Rectangle, Ellipse, Polygon, Pan, with V / R / E / P / H as shortcuts. Pick what
the region should check from **New region** in the toolbar, then draw it.

Elliptical regions existed for a long time before this and went unfound, because
the only way to reach them was a dropdown on a tab. That is a discoverability
failure, not a user error: a capability nobody can see may as well not exist.

**Polygon regions.** Click each corner, then Enter or right-click to close.
Double-click an edge to add a vertex, double-click a vertex to remove it, drag
any vertex to adjust. Use these for angled or irregular features -- a component
sitting at an angle cannot be tightly bounded by a rectangle, so the box drags in
whatever sits next to it. Measured on a rotated part: the rectangle caught two
neighbouring features, the polygon caught none.

**Two toolbar rows, grouped by purpose.** The first is the workspace and what
you are looking at: Inspect/Label, camera, load image, set reference, and what a
drawn region should check. The second is running it and reading the result: run,
save, Diagnose, Exposure, History, and how many checks the recipe holds.

The single row this replaces put eleven unrelated items in one uninterrupted
line, with Select and Hand appearing twice over once the palette existed --
two controls for one state, and two places to look for it. Zoom moved to the
View menu and the wheel, where it costs nothing to reach and stops competing
with the controls that matter.

**A verdict banner across the top.** PASS or FAIL large enough to read across a
workshop, with the first failing check and its reason beside it, and running
yield, cycle time and count on the right. This used to be twelve-point text at
the bottom of the screen.

**Region names only where they help.** Naming every region at once turned a
dense recipe into unreadable soup -- nine overlapping boxes each with a tag, over
the part the operator is trying to look at. Names now appear on the selection and
on anything that failed. Passing regions are drawn thin and faint so the eye goes
to what went wrong. **View -> Region names** switches to all or selection-only.

## Offset: what happens when the part moves

A fixture that reports success while having found the wrong feature is the most
dangerous failure this application can produce, because every later region then
sits on the wrong part of the image and the results look plausible.

Measured on the real chassis, sliding the part around the frame:

```
  fixture region        behaviour across five offsets
  brass insert, 190 px  3 of 5 WRONG, scoring 0.64 to 0.68 against a 0.45 pass
  wide area, template   all correct, within 0.3 px
  wide area, features   all correct, within 0.1 px
```

The algorithm was never the problem -- the region was. A small feature that
repeats on the part cannot fix it, and the match score does not reveal this: a
wrong match on a repeated feature still scores well.

`Locate part` now measures how *unique* its match is, not just how good it is:
the gap between the best correlation peak and the strongest rival peak
elsewhere. An ambiguous match is refused with an explanation rather than
published. On the repeated brass insert the margin was 0.006 and 0.04 for the
wrong matches against 0.25 for the right one, so the two separate cleanly. The
figure is reported as `uniqueness` in the results.

With the fixture corrected, the real recipe holds to the limits tested:
offsets to +-250 px and rotations to +-4 degrees, all six checks passing, all
seven screws and all three labels found.

**Choosing a fixture region:** pick something that appears exactly once and has
structure -- a corner, a connector, a printed block. Not one of a row of
identical screws. If the uniqueness figure is small, enlarge the region.

## Responsiveness on large images

A live camera frame or a full-resolution photograph was costing 37 ms to put on
screen, which is about 27 frames a second before any inspection work at all.

The canvas now paints a reduced pixmap sized to what the viewport can actually
show, and scales the item back up so every region coordinate is unchanged.
Zooming in re-renders at higher detail, so close inspection is still sharp.

```
  live frame on screen   36.6 ms  ->  8.7 ms
```

Downscaling uses INTER_LINEAR rather than INTER_AREA: AREA is the better
downscaler and costs 24 ms against 2 ms on a 2576x1944 frame, which is most of
the budget for a whole frame. The reduced pixmap is only a display proxy.

**The full-resolution array is what the tools measure.** Only what is drawn is
reduced, and a test asserts it, because a downscale that reached the
measurement path would quietly cost accuracy everywhere while still looking
right.

## A worked example on a real part

`projects/chassis_example` is a complete recipe built from production photographs
of a network-switch chassis taken in a robot inspection cell -- real lighting,
reflections off brushed metal, labels at whatever angle the camera saw them.
Open the model and press F5.

```
  Network chassis - top face                 PASS   2740 ms
    locate_part             pass    833 ms   fixture on the brass insert
    serial_label_present    pass      6 ms   match 1.00
    serial_text             pass   1833 ms   3 blocks, read at 180 degrees
    caution_label_present   pass     23 ms   match 1.00
    sample_tag_present      pass      7 ms   match 1.00
    panel_screws            pass     42 ms   7 of 7
```

It shows the mix you actually need on a part like this: a fixture, three
presence checks on different labels, text extraction from a selected area, and
a fastener count -- all in one recipe on one image.

Two things worth knowing, both learned building it:

**The fixture is on the brass insert, not a screw.** The panel carries eleven
near-identical fasteners, so aligning on one of those invites the matcher to
pick the wrong one. The brass insert is the only high-contrast unique feature
on this face.

**The screw count is 7, not 11.** A circle detector finds eleven round features
across the panel, but they range from 12 to 26 pixels in radius -- they are
different fasteners. Matching one template finds the seven of that type, which
is the right answer to "are all seven of these present" and is stable across a
wide threshold range. Count each fastener type with its own tool rather than
trying to make one tool count all of them.

## Jobs, sides and trained tools

Structured the way VisionPro users work. **One job** inspects one product. The
product is photographed from several **sides**. Each side has a **fixture** --
the first Locate part tool -- that finds the part in that view, and every other
tool's region lives in the fixture's coordinate space, so it follows the part
when it moves or turns.

The job tree shows Job -> Sides -> Tools. Click a side to open it.

```
python build_chassis_job.py      # builds the seven-side chassis job from its photos
```

The job's verdict is PASS, FAIL -- naming the side and tool that failed -- or
**INCOMPLETE** when a side was not photographed. A chassis inspected on four of
its seven sides has not passed.

### Trained tools: OK and NG images

**Trained surface check** learns what a region should look like from example
images instead of being configured by hand. Select the tool and the Training
panel shows its OK and NG columns. Put a part on the canvas, press **Add current
image as OK** or **as NG**, and press **Train**. Right-click an image to move it
between columns or remove it.

Training images are plain files and the folder is the label:
`sides/<side>/training/<tool>/ok/` and `.../ng/`. Fix a mislabelled image in a
file browser and retrain.

The threshold is not picked by hand -- that is where inspections go wrong. It is
set midway between the worst OK and the best NG, and the gap between them is
reported as the margin. An NG image that scores like a good part is set aside
and named: either it is not really defective, or its defect cannot be seen in
that lighting. Letting it count collapsed the threshold to one pixel above the
best good part, which rejects good parts.

It is a per-pixel statistical model, not a neural network: it trains instantly
on a CPU. With few OK images it augments them with the shift, rotation and
lighting variation the fixture and lamp introduce. More real OK images always
beat augmentation.

Measured on the real chassis, on parts it never saw -- moved up to 150 px,
rotated up to 4 degrees, lamp 15% dimmer to 12% brighter:

```
  12 of 12 correct    worst good part scored 0, weakest scratch 33, threshold 17.5
```

**The shipped NG images are demonstration data**: the real photographs with a
scratch drawn on them, since no defective chassis was photographed. Replace them
with real rejects before trusting any threshold.

### Real-image campaign (bench_real.py, real_checks.py)

`python bench_real.py [job folder]` takes each side's real photograph through
line conditions -- shifted, turned, 25% darker and brighter, lower contrast,
slightly out of focus, sensor noise, JPEG 50 -- which must pass, and plants real
defects -- labels and screws removed, holes covered, scratches, stains -- which
must fail on the right check. `python real_checks.py [side folder] [output]`
does the same for screws, part present / absent and measurement on one side,
reports each check's margin between its worst good score and its best defective
one, and writes result photographs with an index.html (see real_results/).

**Seven chassis photographs: 82 of 84 good conditions pass, 40 of 40 defects are
caught.** Screws: 7 on every good image, and each of the 7 removed in turn is
caught. Presence: serial label margin 0.83, sample tag 0.64, caution label 0.93.
Hole spacing: 116.16 to 116.58 px on every good image; a hole covered or moved
3 px fails. The two false rejects are the serial label's print, ~10 px tall --
at the limit of any OCR; dependable reading wants ~20 px characters.

What the campaign found and fixed in the software: fixtured regions jittered by
up to a pixel (truncated, not rounded) and moved a hole's fit 1.4 px; the
trained surface check worked at 160 px and could not see a 3 px scratch on a
large region; several lines of print merged into one unreadable block when soft;
an overexposed region -- 31-38% clipped at white on a 25%-brighter photograph --
was reported as a surface defect, and is now reported as overexposure. The Top
right view is 7% clipped even as photographed: turn its exposure down.

And in the example recipe: the caution label check used a 1180 x 760 region
running off the image, so its worst good score was 0.73 -- now a tight region on
the warning symbols (worst good 0.947, missing 0.016); the side-panel surface
regions took in foam and the black base -- now drawn on the panel only; the
serial label is read at its known orientation (upside down in the photograph),
which halved the front-top job from 1.76 s to 0.90 s with identical readings.
The serial check still passes on any text: set your real part number as its
expected text. Two test defects were flawed and fixed first: a label "removed"
by inpainting its region left the label's layout behind, and a stain of fixed
colour was the side panel's own grey.

### Undo and redo in QuickBuild

Edit -> Undo / Redo, Ctrl+Z / Ctrl+Y, for the job in front -- the menu says what
it will undo ("Undo move lot_code region", "Undo delete lot_code"). Every change
is recorded at the one place all edits already end, not at each of the dozens
of places that change a recipe, so none can be missed; a drag or a spinbox sweep
is one step. Open editors follow the restored recipe, an editor whose tool no
longer exists closes, and training's tuned threshold can be undone like any
edit. Locked in Operator Mode.

### Running jobs in parallel

Run -> Run Jobs in Parallel overlaps a cycle's jobs on a multi-core PC -- OpenCV
and the OCR and inference engines release Python's lock while they work. Off by
default and saved per PC: it cannot help on one core, and `python
bench_parallel.py <application or .vsp>` measures on the line PC whether it
does there, and checks the results are identical. (This build machine has one
core, so the gain could not be measured here -- only that results match.)

Making it safe found a real race: a cached model is shared by every job using
it, and OpenVINO's single inference request is not safe from two threads, so
two jobs using the same text detector could read each other's results. Each
model's calls are now locked (different models still run side by side), model
creation is locked, and OCR engines that hold one model are serialised; a test
calling one model from eight threads failed one run in three without the lock.

### Every tool through the interface

Each of the 40 tool types is taken through the whole interface by a test, in
both front ends: added from the Toolbox, its editor opened, every setting
changed, run, every record and tab shown, saved and reopened, deleted. Qt does
not fail a test when an exception is raised inside one of its event handlers --
it prints it and carries on -- so the tests capture those too; a fault planted
in a settings handler to prove it is caught was.

What it found: in the classic window, **unticking "Run this tool" deselected
the tool** -- its region was hidden, Qt deselects a hidden item, and the
property panel went blank, so the tool could not be switched back on from the
same place. A disabled tool's region is now drawn faded instead: still
selectable, and you can see where it sits.

### 40 tools: the rest of VisionPro's catalogue

| VisionPro tool | Here | What it does |
|---|---|---|
| CogAffineTransformTool / CopyRegion | Affine transform | the region resampled upright, at a chosen scale, into its own image |
| CogPolarUnwrapTool | Polar unwrap | a ring unrolled into a strip; turns with the part |
| CogIPTwoImage tools | Two-image operation | difference, subtraction, min, max against the reference, lined up first |
| CogFixtureNPointToNPointTool | Fixture from points | the part's pose from found features -- two fiducial holes |
| CogCalibNPointToNPointTool | N-point calibration | pixels to world units, rotation and skew included |
| CogCreateLine / Circle / Segment | Create geometry | line through points, midpoint, perpendicular, parallel, circle through three points |
| CogBeadInspectTool | Bead inspect | a glue bead's width, position and gaps at stations along a path |
| CogDataAnalysisTool | Data analysis | a value's mean, spread, Cp and Cpk over runs; flags drift before a reject |

Fit line / circle also fits an ellipse. Deliberately not here: the C# **Script**
tool (a job file that runs code is a risk on a line PC -- your own code does that
part, through the exported file) and **Image Average** (several frames per trigger).

**Images with their own coordinates.** Affine transform and Polar unwrap make
an image whose coordinates are not the camera's. A tool reading it draws and
edits its region on that image; the engine gives it no fixture (the transform
already followed the part) and a reference transformed the same way; and if the
image was not produced, it is an error -- never a silent fall back to the camera
image, which would inspect the wrong place.

**Features from failed tools are not used.** A Find circle that fails still
publishes its best guess. Fixture from points used such a guess and set a pose
8 px out with a 2.6% scale error; Distance, Intersect and Create geometry read
features the same way. All four now use a feature only if its tool passed.

### What your code gets from a result

```python
r = app.run(image, job="Front top")
r.verdict, r.failed_tools            # 'PASS' / 'FAIL', and which checks failed
r.pose()                             # where the part is: x, y, angle, full transform
r.to_reference(x, y)                 # an image point in the part's own coordinates
r.shapes()                           # what the tools found, as geometry in image coordinates
r.shapes("hole_a")                   #   [{"tool", "kind", "points", "status", "label"}]
r.values("hole_spacing")             # every measurement one tool made
r.to_dict()                          # the whole result as JSON-ready data, no images
r.annotated()                        # the image with the results drawn
```

Region outlines and search guides are left out of `shapes()` unless asked for
(`include_regions=True`, `include_search=True`): they show where a tool looked,
not what it found.

### 32 tools

Four more from VisionPro's catalogue: **Find ellipse** (CogFindEllipseTool --
radial calipers, a robust ellipse fit), **Image sharpness**
(CogImageSharpnessTool -- focus against the reference, so a knocked lens is
named instead of making every other tool fail), **Color segmenter**
(CogColorSegmenterTool -- the share of a region within a colour range; hue wraps
past red), and **Classify** (VisionPro Deep Learning's Green Classify -- an ONNX
classifier with accepted classes and a confidence floor). **Intersect** now also
does line-circle and circle-circle, as VisionPro's intersect tools do.

Still not here, on purpose: tools that change the image's geometry (polar
unwrap, affine transform) -- every later tool's region would be in the wrong
coordinates -- and YOLO, the AI surface check and your own classifiers, which
need models trained on your parts.

### The gallery: how each tool detects

`tool_gallery/index.html` beside this folder: 58 photographs of the application
-- QuickBuild, a Job Editor, the Train and Graphics tabs, and every tool that
runs without a model of your own (37 of 40) detecting on the demo board, a
rejected board, the real chassis and a QR label, plus their diagnostic
records. `python make_gallery.py` rebuilds it; a test checks that every tool in
it works and the rejected board fails for exactly its planted defects.

## Teach once, run every part: the Python job file

**1. Teach each job's images.** Job Editor -> Image Source: an image file, a list
of files, a folder (every image in it; re-scanned, so images a line camera keeps
writing are picked up, continuing after the last one inspected), or a camera.
The choice is saved with the job and used again when the application reopens.

**2. Teach where results go.** File -> Result Output: a folder for annotated
result images, whether to show each result in a window, and whether to keep
only rejected parts.

**3. Export.** File -> Export Job File (Python) writes `<name>.py` and the
`<name>.vsp` package it runs, side by side. Then:

```
python chassis_app.py                     every job on its taught images
python chassis_app.py --no-show           no windows
python chassis_app.py --output D:\results
python chassis_app.py --source "Front top=D:\parts\top"
```

or from other code: `import chassis_app; results = chassis_app.run(show=False,
on_result=my_display)` -- `on_result` receives each result for your own window.
The job file opens with a plain settings section -- each job's images, the
result folder, show, only-failures -- to edit for a particular PC. Relative
paths are taken from the job file's own folder, so a job file and its images
can be copied together. The exit code is 0 only when every part passed.

`chassis_job/` beside this folder is the seven-job chassis, exported this way
with its photographs in `images/`: copied to another folder and run from a
third, all seven jobs pass and the annotated results appear in `chassis_job/results/`.

A job whose image cannot be read is reported as ERROR and the others still
run; the overall verdict is then INCOMPLETE, never PASS.

## The workspace: windows inside the window

Job Editors, Tool Edit Controls, floating displays and results grids now open
as windows inside QuickBuild's workspace, beside the Run View -- Window ->
Cascade / Tile / Close all editors. One **Toolbox** serves the application and
adds to the job in front; a Toolbox inside every Job Editor squeezed the tool
tree to a sliver once editors shared the workspace.

F5, F6 and Ctrl+S belong to QuickBuild and act on the window in front: F5 in a
Job Editor or Tool Edit Control runs that job, from the Run View every job.
Inside one main window an editor's own F5 and QuickBuild's F5 would be the same
window's shortcut -- ambiguous, so Qt would have fired neither.

## Using an application from your own Python code

**File -> Export Package** writes the whole application to one `.vsp` file:
every job, its tools, reference images, trained models, lens calibration, and
the runtime that runs them. Your program needs only that file, OpenCV and NumPy
(plus an OCR engine if the application reads text, and openvino or onnxruntime
if it uses trained models):

```python
import sys, cv2
sys.path.insert(0, "chassis.vsp")     # the package carries the runtime
import vsruntime

app = vsruntime.load("chassis.vsp")
result = app.run(cv2.imread("part.jpg"), job="Front top")

print(result.verdict)                         # PASS / FAIL
print(result.value("hole_spacing", "value_px"))
print(result.failed_tools)
cv2.imshow("result", result.annotated())      # the image with results drawn
cv2.waitKey(0)
```

`app.run_all({"Front top": img1, "Top right": img2})` runs several jobs and
`app.verdict(results)` gives PASS only if every job passed (INCOMPLETE if one
had no image). `result.annotated(labels=False)` draws shapes without text, and
`only_failures=True` just what failed. From the command line:
`python vsruntime.py chassis.vsp part.jpg --job "Front top" --save out.jpg`.
See `example_run_package.py`. **File -> Import Package** opens a `.vsp` back in
QuickBuild as a new application.

`chassis.vsp` alongside this folder is the seven-job chassis application,
exported and verified this way: loaded in a clean Python process with nothing
but the package, the front-top photograph passes with the hole spacing at
116.58 px and all 7 screws -- the same as in the application.

**About .vpp:** Cognex's `.vpp` is proprietary serialized .NET that only
VisionPro reads, so it cannot be produced here, and naming this format `.vpp`
would only mislead -- VisionPro could not open it. `.vsp` is Vision Station's
own, documented format: a zip with `manifest.json` (jobs, tools, what they
need, versions), `app/` and `runtime/`. Packages are unpacked once into
`~/.vision_station/packages/`, named by their content hash; a package made by a
newer version is refused with a clear message, and one containing paths that
would escape that folder is refused outright.

### More VisionPro tools

- **Results analysis** (Logic) -- accept or reject from an expression over
  earlier results: `hole_spacing.value_px > 115 and panel_screws.count == 7`.
  Every tool offers `.score`, `.status` and `.passed` besides its values.
  Expressions are interpreted, never passed to Python's `eval`: only numbers,
  comparisons, arithmetic, and/or/not, abs/min/max/round and tool.value
  references are allowed, so a job file cannot run code on the line PC.
- **Intersect lines** -- the corner where two Find line results meet and the
  angle between them, published as a point Distance can measure from.
- **Fit line / circle** -- through the centres Count by brightness or Count by
  shape found, reporting the worst deviation: are the screws in a row, do the
  holes lie on their bolt circle. Count by shape now reports where each match
  is, in image coordinates.
- **Color match** -- the region's median colour against the reference as CIE
  delta E, unmoved by a glint or a small mark.

## The VisionPro-style application  (run.bat, or python vp_app.py)

The default window is now laid out the way Cognex VisionPro QuickBuild is, in
its three levels. The classic single window is still there: `run_classic.bat`,
or File -> Classic editor.

**QuickBuild** -- the application. The Jobs list (one job per side, with
Accept/Reject and cycle time), a run display for every job, Run Once (F5) and
Run Continuously (F6), accept/reject counts and yield, and the **results
queue**: the last 32 runs, or the last 32 failures (View menu), each with its
image and graphics. Double-click a job, or its run display, to edit it.

**Job Editor** -- one job. The Image Source first, then every tool in run order
with a pass/fail dot, its status and its time; expand a tool to see its
terminals -- which image it reads, which fixture it follows, what it measured.
A Toolbox of every tool by category (double-click to add), and a display of the
selected tool's last run. Run Job Once / Continuously, Image Source (a list of
image files, cycled one per run), Save. Right-click a tool to open, rename,
disable, reorder or delete it; renaming keeps every link to it pointing at it.

**Tool Edit Control** -- one tool, in its own window: Run, **Electric** (re-run
automatically after each change, once per burst of edits), Floating display,
Open / Save the tool's settings, Reset (defaults, keeping the region), Results
grid, Help. The settings tabs on the left; on the right a display with a
**record selector** -- Current.InputImage, LastRun.InputImage, and
LastRun.OutputImage for image-processing tools. Regions are edited on
Current.InputImage and shown with their results on the LastRun records. The
status bar gives Accept/Reject and both the tool's time and the job's.

Every image in all three levels is one display control, **VisionDisplay**:
pointer, pan, zoom in and out, fit, 1:1, pixel grid (8x and above), graphics on
or off, the record selector, and a status bar with the coordinate space, the
pointer position, the zoom level and the exact pixel value.

### The look: Windows 11, light and dark

`theme.py` holds the whole look: Windows 11 Fluent design tokens for a light
and a dark mode, and one stylesheet built from them. It follows Windows'
app-mode setting; **View -> Theme** picks one, kept for next time, and applies
at once. The title bar follows the mode too. Widgets carry no colours of their
own any more; they set an object name or a `tone` property (pass, fail, warn,
idle) and the stylesheet colours them, so the classic window, QuickBuild and
every dialog match.

What changed in the layout, and why:

- **Editors open as tabs**, beside the Run View, not as windows inside the
  workspace. The windows opened larger than the workspace, covered each other,
  cut off their own displays and squashed the Run View to its title bar.
  Window -> Tabbed documents switches back to floating windows; the Run View
  tab cannot be closed.
- **The command bar** puts Run Once first, as the one accent-coloured button,
  and shows Accepted, Rejected, Yield and Last cycle as cards on the right.
  Run Continuously turns red while it runs. On a narrow window the secondary
  commands show icons only, so the numbers stay on screen.
- **The Run View** shows each job as a card: its name, cycle time, failed
  tools and an ACCEPT / REJECT / ERROR badge, with the card's border in the
  verdict's colour. A first start, with no applications, shows a welcome card
  instead of an empty grey area.
- **The Toolbox has a search box.** Type part of a name or description; Enter
  adds the best match to the job in front.
- **Panels dock but do not float**, so one cannot be torn off over the image
  by accident. The x on a panel hides it; View brings it back.
- **Property tabs scroll**, so a tool with many settings fits a laptop screen
  without pushing the editor's status bar out of sight.
- **Help** has Getting started, Keyboard shortcuts and Check installation.

### Bugs found by using it

- **Every chassis part was rejected.** PaddleOCR 3.7 on PaddlePaddle 3.3 raises
  `NotImplementedError (ConvertPirAttribute2RuntimeAttribute)` from its oneDNN
  executor on the first read, after the engine had started without complaint,
  so the fallback to Tesseract never happened and every text tool reported
  ERROR. oneDNN is now off, and the engine reads a test line when it starts:
  one that cannot read is not chosen.
- **Text reading took 8 s per label.** Without oneDNN the server-size text
  detector took 1.2 s per crop. The mobile detector finds the same lines in
  0.17 s; the full-size recogniser still does the reading. The four chassis
  fields read identically, in half the time. The document unwarping models,
  which are for photographed pages, are no longer loaded.
- **Start-up waited on the internet.** PaddleOCR checked it could reach its
  model hosts on every start -- over a minute offline. `vp_app.py` now sets
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`; a missing model still downloads.
- **Operator Mode crashed**, and Undo raised on window switches. On PySide6
  6.11, `QAction.menu()` returns a second Python wrapper for the menu; when it
  is garbage-collected, every action wrapper of that menu is invalidated. The
  menus are now held, and walked directly.
- **Remove job failed on Windows** with "Access is denied": the job's results
  database was still open. It is closed first.
- **Production History kept the previous application's numbers** after opening
  another one. It is closed with the application.
- **Wide-spaced print read as nothing.** Find and read all text split
  `AB123456` into `AB1` and `23456`, both under the minimum length. Pieces of
  one line are joined when the gap is narrower than the print is tall.
- **The allowed-characters setting did nothing with PaddleOCR.** It was passed
  to Tesseract only; it now filters every engine's reading.
- Known: with PaddleOCR the text is read the right way up, but because Paddle
  turns lines itself, the `quarter_turn` a text tool records can be 0 for a
  label that was upside down. Four orientation and confidence tests in
  `tests.py` were written against Tesseract and still fail for this reason.
- The Jobs list cut names to "Network chas..."; the results-queue preview showed
  a 4000-pixel frame at 4%; preflight told people to start `app.py`.

### Terminals and links

Expand a tool in the Job Editor to see its terminals: its inputs, each with
where it comes from, and its outputs, with the values from the last run.
**Drag an output onto an input** to link them. Only legal links are accepted
while you drag -- the types must match (image, fixture, feature, blobs, value)
and the source must run earlier, since a later tool's output does not exist yet
when this one runs. Right-click an input for **Link to** and **Unlink**.

Every tool now has an explicit **Fixture** input: the most recent Locate part
(the default, and what every recipe did before), none (image space), or a
specific Locate tool. **Add Terminals** (right-click a tool) turns one of its
numeric settings into an input driven by an earlier tool's result -- VisionPro's
Add Input. A linked value that is missing is an error, never a silent default.

### Training in the Tool Edit Control

The **Train** tab holds the tool's OK and NG images, shown as thumbnails of the
tool's region. **Train** for a trained surface check, **Tune threshold** for a
presence or locate tool, **Test** for any tool: each thumbnail is framed green
or red and the report counts false rejects and escapes. Unsaved edits are saved
before training, which reads and writes the job on disk. Testing an empty set
says so rather than reporting 0/0 in green, and a set with no NG images is
flagged: nothing in it shows the tool can reject a bad part.

### Image Source

Job Editor -> Image Source: the reference image, image files or a folder cycled
one per run, or a camera. **A source that cannot deliver is an acquisition
error, shown in amber on that job, not an inspection.** It used to fall back to
the reference image -- so a dropped camera would inspect the stored golden image
and pass, the worst failure a vision system can have. Continuous runs keep
retrying while the camera recovers.

### Jobs and Posted Items

File -> New Application makes a multi-job application from a first image, with
a fixture ready. Add Job, and right-click a job to rename or remove it. A
removed job's folder is moved to removed/, not deleted, so a misclick cannot
destroy a trained job. **Configure Posted Items** picks result values to show
as extra columns in the results queue for every run; they are saved with the
application.

### Repetition removed

A scanner for repeated code found 50 blocks copied between files; 22 remain,
all structural boilerplate (import headers, the two lines every tool's run
starts with). Several copies had already drifted into bugs: only one YOLO tool
explained a missing runtime; only one text reader turned light-on-dark print
the right way round, and the other would crash on an empty detection box;
polygon regions followed a different name-tag rule from rectangles; and the
two front ends reported an empty test set differently. Shared code now lives in
uikit.py (UI pieces), terminals.py (links), job.py (training thumbnails and
summaries), and vstools/base.py (NMS, runtime loading). Every tool's settings
and defaults were checked identical before and after, and the geometry
benchmark gives the same numbers.

### Connecting to the line: TCP/IP

Run -> Communication turns on a TCP/IP interface for a PLC or robot controller
-- the role of VisionPro's Communication Explorer, over the one thing every
controller can do: open a socket and exchange text lines.

```
TRIGGER                  inspect every job      TRIGGER Front top    one job
  -> RESULT frt_top PASS 2122 -                 one line per job: verdict, ms, failed tools
  -> DONE PASS 7                                PASS / FAIL / INCOMPLETE, number of jobs
GET frt_top.hole_spacing.value_px   -> VALUE 116.579
STATUS -> READY | BUSY     PING -> PONG     VERSION -> VISIONSTATION 1
```

A trigger that arrives while an inspection is running is answered `BUSY`,
never queued silently -- a controller has to know its trigger was not acted on.
A job whose image cannot be read reports `ERROR` and the run `INCOMPLETE`. With
broadcast on, every run -- from the line, a button or continuous mode -- is sent
to every connected client, for controllers that only listen. The settings are
saved with the application and the server starts whenever it opens; the status
bar shows whether the line is connected. `example_line_client.py` checks the
connection from the line PC before the controller is wired.

The session above is the real chassis: seven jobs triggered over TCP, all
passing, the hole spacing and screw count read back.

Two more fixes found building it:

- **A missing part image inspected the golden image instead.** Choosing image
  files that did not exist silently fell back to the reference image, and the
  line was told PASS -- the same hazard fixed earlier in acquisition, surviving
  in another place. Missing files are now an acquisition error.
- **Relative taught paths meant different things in two places**: the exported
  job file took them relative to its own folder, QuickBuild relative to wherever
  it was started, so an application with relative paths found no images. Both
  now take them relative to the application's folder.

### Production records and Operator Mode

**Every inspection is logged**, per job, to that job's results database
(`results/results.db`), with images of rejected parts archived. The queue on
screen holds only the last 32 runs and is gone when the application closes; the
log keeps everything. Run -> Production History shows yield by job, the checks
rejecting parts most often, for the last hour to everything, and exports CSV.
The log writer has its own thread and never delays an inspection.

**Operator Mode** (Run menu) is for running, not changing: editors close and
cannot be reopened, the Toolbox and every build action are put away, and the
operator's name is recorded with each inspection. Running, the results queue,
history and display controls remain. Set Supervisor Password makes leaving it
require a password, stored only as a salted PBKDF2 hash.

### Two crashes fixed

- **Closing within seconds of opening an application aborted the process.**
  The engine warm-up ran on the inspection thread, and starting the OCR engine
  takes seconds (over a minute when PaddleOCR checks for a network); shutdown
  waited 3 s, and Qt aborts when a running thread is destroyed. The warm-up now
  has its own background thread, a lock stops two threads starting two engines,
  and the inspection thread -- running only inspections, which end -- is waited
  for properly. Closing during warm-up now takes 0.6 s and exits cleanly; a test
  runs it as a real process, since the symptom was the process dying.
- A sub-window that closed itself scheduled its frame to close after Qt had
  already deleted it, raising an error on every such close.

Editors also remember how their panes were divided.

### Seeing where tools search

Tools draw **search guides** -- thin dashed sky-blue, so they never read as
something found: Find Circle its inner and outer search radii, Find Line each
caliper strip, the edge-pair Caliper its measuring direction. They are drawn
when a tool fails too -- that is when you most need to see where it looked; the
first version only drew them on success. The display toolbar toggles them;
exported result images leave them out unless asked (`annotated(search=True)`).

### Graphics tab

Each tool's Graphics tab switches its region outline, results, labels and
search guides on or off -- VisionPro's Graphics tab. It is applied once, in the
engine, so the run view, the editors, the results queue and exported job files
all agree. Switching off a large fixture's outline is how a busy run view stays
readable.

### Coordinate spaces

The display's status bar reads the pointer in a chosen coordinate space: `#`
is image pixels; each Locate tool offers its fixture's space -- the part's own
coordinates, so a point on the part reads the same however the part moved or
turned. Checked on the demo board rotated 3 degrees and shifted: the red LED
reads within 1.5 px of its reference position.

(While writing that test the fixture looked 10 px wrong. It was not: the test
had moved good.png, a different photograph sitting 9 px right and 5 px up of
the reference. The fixture was exact; the assumption was wrong.)

### Icons

An original set of modern line icons (icons.py): drawn as SVG on a 24-unit grid,
crisp at any size and on high-DPI screens. Every toolbar action has one, shown
beside its name; the display toolbars are icon-only with names in tooltips.
Every tool type has its own icon coloured by its family -- locate blue,
geometry indigo, text green, surface amber, trained models pink, colour orange,
logic slate -- shown in the Toolbox and the Job Editor tree, where the pass/fail
dot now has its own Status column. Cognex's icons are their copyright, so none
are copied; the family colours follow the same idea of telling tool kinds apart
at a glance.

### The fixture followed the background, not the part

The features method of Locate part matched features across the **whole
image**, ignoring its region. On a line only the part moves -- the robot,
conveyor and cell stay put -- and their features all vote for "nothing moved".
On the real chassis with a fixture region a third of the frame, every movement
of the part was reported as (0, 0): the fixture passed while every tool after
it inspected the wrong place. The earlier robustness tests missed it because
they moved the whole photograph, background included.

Reference features now come from the fixture region only, and the live image is
searched in that region plus a margin for how far the part may move (35% of the
region by default; set it with search_pad_px). The reference's features are
computed once instead of on every inspection, and the windows are cropped
rather than masked, since a mask still builds the detector's image pyramid over
the whole frame. On the chassis: the small region went from wrong at every
offset to exact, and from 360 to 140 ms; the chassis's own large region from
370 to 310 ms.

### Diagnostic records

The Tool Edit Control's record selector offers each tool's diagnostics as well
as its images, as VisionPro's does -- how you see *why* a tool missed:

- **LastRun.RegionImage** -- every tool: its region exactly as the tool saw it,
  fixtured and deskewed.
- **LastRun.Projection** -- Caliper and Measure length: the brightness profile
  along the tool, its edge strength, every edge found, and each measured pair
  shaded. Edges not in a pair are marked red.
- **LastRun.Histogram** -- Histogram: the grey-level distribution, the band and
  the mean.
- **LastRun.BlobImage** -- Count by brightness: the thresholded mask.

On the demo board a caliper across the LED row reported five pairs where four
LEDs were expected; the projection shows why at a glance -- the dark rims of the
green and orange LEDs form narrow pairs of their own, and the blue LED is a
light bar, not a dark one.

The plots are drawn from the tool's own data, never a second search with
separate settings (the first draft of the Measure length plot did exactly that,
with a different contrast threshold, and could have disagreed with the
measurement). They are produced only when a display asks for them: a production
run pays nothing, and QuickBuild's full seven-job cycle on the chassis went from
3.23 to 3.31 s with them on. Regions are drawn only on full-frame records --
outlines over a plot were meaningless.

### Quality pass: static analysis

Every source file checked with pyflakes: no undefined names anywhere. What it
did find:

- **Polygon regions only scaled from one corner.** The scale was measured
  against the bounding box's right and bottom edges whatever handle was
  grabbed, so the top-left, top-right and bottom-left handles did nothing.
  Surfaced by two unused variables; now tested from all four corners.
- **A method defined twice** in the tool base class, the second silently
  replacing the first. The one in effect was the right one; the shadowed copy
  would have double-rotated polygons had the file ever been reordered. Removed.
- Dead code: an unused colour conversion run on every LED inspection, an unused
  border value, and nine unused imports left over from refactoring.

### PaddleOCR 3

**PaddleOCR never started with the version pip installs today.** The engine
was constructed with PaddleOCR 2's arguments, and PaddleOCR 3 rejects
`show_log` outright, so every read silently fell back to Tesseract -- with or
without network. (Earlier notes blamed the missing network alone; that is only
the second failure, after this one.) Both generations are now supported:
PaddleOCR 3's constructor is tried first, and results are read from either
`predict()` (version 3) or `ocr()` (version 2). Why a better engine did not
start is recorded and shown by the installation check -- `python preflight.py`
-- rather than the fallback happening silently.

This was tested against stand-ins that reproduce each version's interface,
including version 3 rejecting `show_log`. Real PaddleOCR reading was not
verified here: its models download on first use and this build machine has no
network. On a machine with internet the first read will download them.

### How this was designed

From Cognex's documentation of QuickBuild, the Job Editor and the tool edit
controls, and from the open-source projects that imitate VisionPro and
VisionMaster -- VisionEdit (VisionPro-style job flow and tool links),
WPF-VisionMaster / GVision (a separate control library with one ZoomBox display
reused everywhere), PY2CV (PyQt In-Sight imitation) and the Qt node editors.
They agree on one display control built once and reused, and VisionPro itself
is a tree of tools with terminals rather than a node canvas, so that is what
this is. The engine, tools and job files underneath are the same tested code
the classic window uses.

### What building it turned up

**Edits in a live editor never reached the inspection.** Tools copy their
region and settings when they are made, and the engine made them once, when a
job loaded. So a region dragged in a tool editor was inspected at its old
position, and a tool added from the Toolbox never ran, until the job was saved
and reopened. Every run now builds its tools from a snapshot of the recipe as it
stands -- a snapshot so the worker thread never reads data a window is editing.

**Stop let one more inspection through.** A finished continuous run schedules
the next; Stop pressed in that gap did not cancel it. The scheduled run now
checks again when it fires.

**A new run reset the record you had chosen.** Rebuilding the job tree fired
selection changes, which looked like choosing a different tool.

**Viewer:** Fit is now a mode -- it follows window resizes until you zoom by
hand -- instead of a one-off; the pixel readout floors rather than truncating,
so a pointer just outside the image no longer reports pixel 0; and mono images
no longer break the readout.

**The first part was slow**: the text engine takes seconds to start, over a
minute if an OCR package tries to fetch models with no network. It now starts in
the background when the application opens.

Inspection runs on a worker thread, and results carry a run number so a late
result from an older run is never shown over a newer one. Results-queue images
are kept JPEG-encoded: 32 full frames from a 5-megapixel camera would hold half
a gigabyte. Layout is saved per version with named docks and toolbars, and View
-> Reset Layout restores the default.

### The classic window

**Job tree** (top left): the application, its jobs, and inside the open job its
**Image source** followed by every tool in the order they run, each with its
live result. Click a job to open it; click Image source for the camera, a file,
test images or the reference. Every job is listed, so the whole application is
visible at once.

**Toolbox** (below the tree, tabbed with Test images): every tool the
application has, by category. Double-click one to add it to the open job. They
used to be reachable only through a menu, which hides the catalogue.

**Properties** and **Training** on the right, **Results** across the bottom,
drawing tools down the left edge.

### Tool editor windows

Double-click a tool in the job tree -- or select it and press Ctrl+E -- and it
opens in its own window, as VisionPro's tool edit control does: its own image
display, opened framed on the tool's region, showing only that tool and its
results; the tool's Settings, Region and Last result tabs; a table of everything
it measured; and Run job. Each tool has at most one window; asking again brings
it forward. Several can be open, and they can sit on a second monitor.

Nothing is copied between windows. An editor edits the same tool record as the
main window, and its image sends every move and resize to the same handlers the
main canvas uses, undo included -- so a change in either window is one change to
one recipe, and both redraw. Deleting a tool closes its editor; opening another
model closes them all.

### Keyboard shortcuts that did nothing

Ten keys had two owners after the toolbars were reorganised -- F5, F6, Ctrl+S,
Ctrl+O, Ctrl+Y, F7, F8, F9, V and H -- because toolbar buttons re-registered
shortcuts the menus already held. Qt fires neither action of an ambiguous
shortcut, so pressing F5 did not inspect and Ctrl+S did not save. The menus now
own every shortcut, and a test fails if any key ever has two owners again. On
Windows, redo by keyboard had also stopped working: the standard Redo key there
is Ctrl+Y, which a second action also claimed.

### Zoom

Wheel zooms at the pointer; the menu, the keys and the buttons zoom about the
centre of the view -- anchoring those at the pointer moved the image away from
whatever you were looking at, because the pointer was on the menu at the time.
Zoom is bounded: the image can never be smaller than 80 px on screen nor
magnified past 32x. Without limits, sixty scroll clicks took a 640 px image
down to 0.1 px -- it vanished, with nothing to say where it had gone. The
current level shows in the status bar; Ctrl+0 fits, Ctrl+1 is actual size.

### Build mode and Run mode

**Build** (Ctrl+B) is everything for making and tuning a model: the job tree,
properties, training panel, drawing tools. **Run** (Ctrl+R) is the operator's
screen: the image, the verdict, the counters, large Inspect / Start / Reset
buttons, a strip of the last thirty verdicts, and History. Every editing panel is
put away and the regions are locked, so a stray click on the line cannot move a
tool. **Build mode** on the right of the run bar returns to building. VisionPro
keeps these apart the same way: QuickBuild is where a job is made, the
application is where it runs.

### Image processing  (Tools -> Image -> Image processing)

One operator per tool: smooth, median, sharpen, equalize, threshold, erode,
dilate, open, close, edges, invert, channel, gain. Its output is an image any
later tool can read -- choose the processing tool's id under **Reads image** on
that tool's Region tab. Chain two by pointing the second at the first. Every
operator keeps the image the same size with pixels where they were, so the
fixture and every region stay valid; operators that move pixels, like a polar
unwrap, are deliberately left out because they would silently misplace every
region drawn after them.

It earns its place. Find Circle on a hole with salt-and-pepper speckle, over
eight noise patterns:

```
  speckle   read raw: mean / worst     through a median filter
    12%       0.66 / 1.61 px              0.11 / 0.17 px
    20%       1.83 / 3.04 px              0.18 / 0.41 px
```

### Caliper edge pairs  (Tools -> Geometry -> Caliper)

VisionPro's edge-pair caliper: a pair is a bar or a gap bounded by two edges of
opposite polarity. It reports every pair, left to right, with each width and the
pitch between them -- so it counts pins and checks their spacing as well as
measuring them. On six pins at sub-pixel positions beside a lighter step that is
not a pin: all six found, the step ignored, widths within 0.024 px, pitch 50.251
against 50.25 -- and a part with one pin 26.5 px wide against 20.25 +- 1 failed.

### Lens distortion calibration  (Tools -> Calibrate lens)

Photograph a printed checkerboard 8-15 times, tilted and moved so the corners
of the frame are covered, and give the number of inner corners and the square
size. Every frame -- and the reference, which tools compare against -- is then
corrected before any tool sees it. Measured against a synthetic lens of known
distortion:

```
  focal length 819.7 (true 820)   centre within 0.4 px   reprojection 0.04 px
  straight edge near the top of the frame, bow as fitted by Find Line:
      as the lens sees it                        6.04 px
      corrected, boards never near the edge      0.75 px
      corrected, boards reaching the edges       0.07 px   = correcting with the true lens
```

Covering the corners is the difference between those last two lines: distortion
is strongest at the edges, and the calibration can only learn what it sees.

The scale a calibration reports is averaged over boards at several distances.
For millimetres, calibrate the scale with the board lying where the part sits.

### Geometry and analysis tools, modelled on VisionPro

The core VisionPro families were already here -- pattern location and fixture,
blob, caliper, OCR and ID. These fill the gaps: **Find Line**, **Find Circle**,
**Histogram** and **Distance / angle**.

**Find Line / Find Circle** work the way VisionPro's do: many small calipers are
laid across the expected edge, each finds one edge point to sub-pixel precision,
and a line or circle is fitted through them, discarding the worst few points as
outliers (VisionPro's NumToIgnore). Measured against edges of exactly known
position, with blur and noise (`python bench_geometry.py`):

```
  Find Circle   centre 0.08 px mean, 0.14 worst    radius 0.02 px mean
  Find Line     angle 0.02 deg mean, 0.06 worst    position 0.08 px mean
  a chip in the rim:  centre error 0.69 px trusting every caliper,
                      0.11 px discarding the worst three (the default)
```

**Distance / angle** measures between features other tools found, as VisionPro's
analysis tools do: centre to centre, point to line, the angle between two lines,
or a circle's diameter. Choose the features from a dropdown of the geometry tools
that run before it. Results are in millimetres once the scale is calibrated.
Because features are found through the fixture, the measurement does not depend
on where the part sits: two holes 150 px apart measured 150.00 +- 0.04 px moved,
turned 7 degrees, or both.

On the real chassis, the front-top side now measures two fastener holes and the
spacing between them. It held at 116.3-116.6 px (std 0.1 px) across moves of
150 px, 4 degrees of rotation and a lamp 10% brighter or dimmer. The rotated test
images were made by resampling a JPEG, which softens edges, so a real rotated
part is likely somewhat better than that.

**Histogram** is VisionPro's standard presence/absence check: grey-level mean,
spread and the share of pixels in a grey band, judged against limits. Only
pixels inside the region's shape are counted.

Edge tools pick the **colour channel** with most contrast. Greyscale conversion
throws colour contrast away: on the demo board a green LED is 142 grey against a
138 background -- invisible to an edge tool -- while each colour channel shows 60
to 90 levels of contrast. Set the channel by hand if you prefer.

Found while building these: Find Circle measured edge strength per half pixel
while Find Line measured per pixel, so the same contrast setting meant different
things in the two tools; and a project opened by a relative path resolved its
trained tools' model paths twice, so they reported "not trained yet" beside their
own models. Both fixed and tested.

### Every tool has OK and NG images

Select any tool and the panel holds its OK and NG images. What they are for
depends on the tool:

- **Test** (every tool) runs the side's recipe on each image and checks the tool
  passes the OKs and rejects the NGs. Each thumbnail gets a green or red frame.
  An OK that fails is a false reject; an NG that passes is an **escape** -- a bad
  part shipped -- and the report says so in red. The whole recipe runs, not the
  tool alone, because some tools read what earlier ones found and a failed
  fixture really does fail the tool on the line.
- **Tune threshold** (Part present, Locate part) sets the threshold midway
  between the lowest-scoring OK and the highest-scoring NG. On the real chassis
  the hand-picked 0.55 on the serial-label check sat far nearer the
  missing-label parts than the good ones; tuned from examples it is 0.691, the
  largest margin both ways. If no threshold separates the images, it changes
  nothing and says so.
- **Train** (Trained surface check) learns the region from the images.

### Presence checks in plain regions

A presence region drawn wholly inside a plain sticker -- the neat, natural way to
draw it -- used to score 1.000 against anything, including the sticker being
missing. Normalised correlation divides by the template's variance, and a plain
patch has none, so the check passed every part. A region with no texture is now
judged on brightness instead, and the result reports which mode was used.
Found while writing the tuning tests; no existing recipe here was affected,
because their labels carry print.

### Editing regions

**Ellipses are edited by their centre and radii**, the way VisionPro does it:
four round handles sit on the curve itself, at the ends of the two axes, and
dragging one changes that radius while the centre stays exactly where it was. A
cross marks the centre and the readout shows the radius, or both radii. Hold
**Shift** to make it a circle. The previous box-style editing put handles on the
bounding box's empty corners and moved the centre whenever a side was dragged.

**Rectangles** resize from the opposite corner as before; hold **Alt** to resize
about the centre instead, and **Shift** to keep the proportions. Modifiers held
when a drag begins now apply for the whole drag -- they were being dropped part
way through.

The Training panel's thumbnails show the tool's region as the tool sees it --
fixtured and deskewed -- not the whole photograph, so a scratch that would be
invisible in a thumbnail of the full frame is plain to see. The property sheet
now scrolls, which lets the training panel have real space beneath it.

### The fixture bug this uncovered

A region with no rotation of its own, under a rotated fixture, was being cut as
the axis-aligned bounding box of where it landed. Every tool therefore saw the
part still rotated, in an oversized box, whenever the part turned. It stayed
hidden because presence checks and template counting tolerate a few degrees; the
pixel-by-pixel trained tool rejected a good part rotated 3.7 degrees and exposed
it. Regions are now always cut in the part's own coordinate space, which is the
point of a fixture.

## The stack: OpenCV, OpenVINO, Qt - trained on Colab

**OpenCV** does every classical tool: alignment, counting, colour, defects,
measurement, text preprocessing. **Qt** (through PySide6) is the interface.
**OpenVINO** runs the trained models - YOLO, AI surface check, AI text
detection - on the CPU, with onnxruntime as the fallback. Models are trained on
**Colab**, so the line PC needs no GPU at all.

### Training on Colab

Open `train_on_colab.ipynb` in Colab and set the runtime to T4 GPU. It covers:

1. **YOLO** from images labelled in the Label workspace: upload the exported
   dataset zip, train, download `best.onnx`, drop it into
   `projects/<model>/models/`. Augmentation is set for inspection -- no
   mirroring, since a mirrored board is a different part; modest rotation;
   strong brightness variation, because lighting drift is the usual enemy.
2. **The text detector**, retrained for longer on a GPU than the fifteen CPU
   minutes it shipped with.

The notebook was checked here by running its GPU patch against the real
training script end to end: every patch target matches exactly once, and the
patched script trains and exports. It has not been run on Colab itself.

### OpenVINO, and why precision defaults to accuracy

Measured on an Intel Xeon, on the text detector:

```
  onnxruntime                54 ms
  OpenVINO, accuracy (fp32)  50 ms   outputs identical to 1e-5
  OpenVINO, speed            12 ms   4x faster, outputs move by up to 0.08
```

**The whole speedup is reduced precision.** At full precision OpenVINO is barely
faster than onnxruntime. In speed mode, on a real chassis label, one detected
box grew by a pixel -- immaterial for reading text, but exactly the kind of drift
that matters on a measurement, and an inspection system should not change its
verdict because a runtime chose a number format.

So every model tool defaults to **accuracy**, and **speed** is a setting on its
Settings tab. How much speed buys depends entirely on the processor: this Xeon
has reduced-precision hardware, and an older line PC may gain little. Run

```
python bench_inference.py
```

**on the line PC itself.** It times every model under each runtime and precision,
reports how far speed mode moves the outputs, and says whether switching is
worth it on that machine.

## AI text detection

**Tools -> Text and codes -> AI text detection.** A trained network finds every
line of print in the region, then each line is read.

The model is not downloaded or pretrained. `textnet_train.py` builds it from
generated data in about fifteen minutes on a CPU, and the result is a 41 kB
ONNX file that runs in roughly 16 ms:

```
python textnet_train.py --minutes 15      # resumes from any existing checkpoint
```

It is a small encoder-decoder, about 100k parameters, predicting per pixel
whether that pixel belongs to a line of print. Boxes come from connected
components of that map afterwards. Segmentation rather than box regression,
because industrial labels carry text at any orientation.

**Why it exists.** A gradient detector cannot tell a line of print from a
barcode -- both are dense clusters of short vertical edges -- and the
hand-written discriminator measured on a real label had overlapping
distributions for the two classes, so it could not be tuned. A network learns
the boundary, provided barcodes appear in the training data labelled as
not-text. They do: `textnet_data.py` renders labels on brushed metal with
barcodes, screw heads, panel seams, ventilation slots, glare and rotations,
every one of them because something in the real photographs was mistaken for
text.

Measured on a real chassis label, against ground truth:

```
  detector              boxes   text lines found   spurious (barcodes)
  classical, gradient       6           4 of 4              2
  trained network           3           4 of 4              0
```

Both find all the text. Only the network rejects the barcodes.

Orientation is chosen once per region rather than per line. Per-line was tried
and is worse: the upside-down reading of a single line is often longer and no
less confident than the correct one, so nothing local separates them. Summed
across a whole label the difference is unambiguous, and on the real label the
tool selects 180 degrees by itself.

**The limit is still recognition, not detection.** All four fields come through
recognisable -- `WHO2d08L200sysve3` for `NX40-24Q8L2DD-SYS V03` -- which
confirms a label is present and roughly right but will not capture a serial
exactly. Install PaddleOCR for that; the detector feeds it just as well.

## Find and read all text

**Tools -> Text and codes -> Find and read all text.** Draw a loose region over
a label and it locates each line of print inside and reads them separately,
rather than needing one box per field.

It exists because the single-box tools could not do this job. Handed a real
network-switch chassis label carrying four fields, `Read text` returned `':'` -- it was
being asked to read a whole multi-field label as one line.

Detection works on the morphological gradient, not on brightness. Thresholding
by brightness cannot separate printing from its surroundings on a metal
chassis: the dark metal around a label and the dark ink on it come out the
same, and the first attempt returned a single block covering the whole region.
Edge density does separate them.

Blocks are filtered by **length, not confidence**. On that same real label
Tesseract's confidence ran backwards: the barcode rows it misread as "LAWN" and
"VAM" scored 0.38 and 0.36, while the two rows it read substantially correctly
scored 0.14 and 0.00. Filtering on confidence therefore threw away precisely
the results worth keeping.

**Honest limits, measured on a real label:** finding the text works -- all six
blocks located, and the 180-degree orientation detected correctly. Reading it
is only about 80% character-accurate with Tesseract: `NX40-24Q8L2DD-SYS V03`
came back as `NS40-2408L200-SYSV03`. Good enough to confirm a label is present
and roughly right, not good enough to capture a serial number exactly. For that
install PaddleOCR, which is far stronger on small industrial print:

```
pip install paddleocr paddlepaddle
```

Check which engine is actually live with **Help -> Check installation**. That
now asks the text tool rather than guessing from what imports, because
PaddleOCR downloads its models on first use: on a machine with no route to its
model host it imports, looks available, and then quietly falls back to
Tesseract.

## Text at any angle

`Read text` and `Verify text` estimate the baseline angle of the marking,
correct it, then try the four right-angle orientations and keep whichever reads
best. Plain OCR collapses past about five degrees:

```
  rotation    before          after
      5 deg   AB12345X        AB12345X
     12 deg   (nothing)       readable
     25 deg   (nothing)       readable
     45 deg   "A"             readable
     90 deg   (nothing)       readable
    180 deg   XSPCCLEV        readable
```

Deskew fixes the baseline but cannot tell upright text from upside-down, since
both have a horizontal baseline -- hence the quarter turns. Orientation scoring
is additive rather than multiplicative: Tesseract reports zero confidence on a
noisy crop, and a multiplicative score collapses to zero for every orientation,
so length never broke the tie and a garbage upside-down read beat a correct
upright one. Switch either behaviour off in the tool settings.

## Multiple captures per inspection

**File -> Captures** defines the exposures a model needs. Each tool then picks
one on its Region tab under **Reads capture**.

This exists because one frame genuinely cannot do the job on a lit faceplate.
Any exposure long enough to show the panel clips the LEDs to white; any exposure
short enough to keep the LED colours leaves the panel invisible. On a real
photograph the LED band wanted roughly a twentieth of the faceplate's exposure.
Rather than compromise and do both badly, the camera takes both frames in
sequence and each tool reads the one it needs.

A typical two-capture model:

```
  panel   12.0 ms   reads the silkscreen, the lot code, the connectors
  leds     0.9 ms   short enough that the LED cores are not clipped
```

Use F7 with the LED region selected to find that second number.

Three details that matter:

**Existing recipes are untouched.** A model defining no captures behaves exactly
as before -- one frame, called `default`.

**A missing capture falls back rather than fails.** A recipe built on a
two-capture station still runs when you open a single image from disk to try
something out. The run reports which captures were missing, because falling back
silently would be worse: the tool would use the wrong exposure and return a
plausible, wrong answer.

**Frames after an exposure change are discarded.** A camera in free-running
acquisition already has one or two frames in flight taken with the previous
setting. Using one would give a capture the wrong exposure, which shows up later
as an intermittent false reject rather than an error.

## Automatic exposure for colour work  (F7)

Select the region over the LEDs, then **Run -> Find exposure for selected
region**. The app searches for the longest exposure that keeps that region
unclipped and sets it.

Why a search rather than a calculation: once a region clips, the image contains
no information about how far past 255 the true values went. On the real
faceplate the 90th percentile of the LED band was already 255, so any
histogram-based correction says "reduce by 6%" when the truth was closer to a
twentieth. Changing the exposure and looking again is the only honest method.
Bisection converges in about eleven frames, and it targets the *longest* usable
exposure -- an unnecessarily short one throws away signal and makes everything
noisy.

The end-to-end effect, on simulated LEDs of known colour:

```
  over-exposed  (10.000 ms)   26.9% clipped   red green blue amber  ->  all wrong
  after search  ( 0.981 ms)    0.0% clipped   red green blue amber  ->  4 of 4
```

Diagnostics (F8) now also measures clipping **over the selected region**, not
just the whole frame. That distinction matters: the real faceplate was 0.02%
clipped overall and 10.6% clipped across the LED band, so the whole-image figure
said everything was fine.

Automatic exposure needs a Basler camera. Exposure cannot be set through OpenCV
on an ordinary webcam reliably enough to search.

## Blown-out LEDs: what a real photograph exposed

Run against a real switch faceplate, `LED colour` called **all thirteen LEDs
white**, greens and reds included.

The cause was not the swatches. A lit LED photographed at an exposure that shows
anything else at all is clipped at its centre: the core reads about
(253, 249, 255), which is white whatever colour the LED is. Taking the median of
the blob samples that core. The colour only survives in the halo around it.

`LED colour` now prefers unclipped pixels: first inside the blob, then from a
ring around it when the whole blob is clipped, falling back to the plain median
for LEDs that are not clipped at all. Two new settings expose it -- **Treat as
blown out above** and **Halo sample width**.

This does not make a badly exposed photograph good enough. On the real image the
tool moved from "everything is white" to plausible-but-imperfect: reds and ambers
came out right, greens read as yellow, because the halo blends the LED with bloom
from its neighbours. Colour work needs an exposure where the LED cores are not
clipped. Diagnostics (F8) will tell you: if **clipped highlights** is above a
fraction of a percent over the LEDs, shorten the exposure until it is not.

## Colour: stock settings rejected good LEDs

`python bench_defaults.py` builds every tool from `schema.defaults()` exactly as
the *Add tool* menu does, and runs it against a good part. Anything that cannot
pass with stock settings is a tool nobody can use out of the box.

It found one: **colour classification rejected perfectly good LEDs.** The
reference swatches were written by eye and sat 66 to 74 LAB units away from a
true green or blue, against a default acceptance of 28. Nearest-neighbour still
chose the right name, so colours were *identified* correctly and the code looked
fine -- but every distance was enormous and the verdict was a reject.

The first fix was wrong in the opposite direction. Replacing the estimates with
fully saturated primaries gave a perfect dE of 0.0 on pure colours and failed
every realistic LED, because a real LED is never that saturated. The swatches now
sit at representative *observed* appearances.

A second change matters more than the numbers. Lightness is weighted at half of
chroma, because drive current, viewing angle and lens diffusion all move L a long
way while a and b barely shift -- so an LED dimmed by half was reading as a
different colour. Across a spread of realistic LED appearances that takes the
worst correct-answer distance from 28.4 to 20.4 against an acceptance of 28: the
difference between marginal and comfortable. L is not dropped altogether because
white and "off" differ only in lightness.

## OCR: two bugs fixed, accuracy 9x better

`python bench_ocr.py` measures character error rate across eight degradations
that mirror real marking: small text, low contrast, defocus, sensor noise,
glare, rotation and dot-matrix printing.

```
  mean character error rate    53.9%  ->  6.1%
  cycle time (demo recipe)    1297 ms -> 717 ms
```

Three changes, each chosen by measurement rather than intuition:

**Preprocessing order.** Denoise, upscale, close, then threshold. The old
pipeline upscaled and ran CLAHE with no binarisation, which amplified sensor
noise into texture the recogniser read as characters -- on a noisy sample it
returned `: : OB ee` for `AB12345X`, worse than reading nothing. Adaptive
thresholding rather than Otsu, because a global threshold gives up under glare
when one bright patch drags it past the text on the dim side of the part.

**Character whitelist.** New *Allowed characters* setting, defaulting to
uppercase, digits and hyphen. Telling the recogniser that a lot code cannot
contain lowercase or punctuation removes a whole class of confusions, and it is
the cheapest accuracy win available on industrial marking.

**Real confidence.** Tesseract confidence was hardcoded to 0.5, which made
*Minimum confidence* a control that did nothing: set it above 0.5 and every read
failed, at or below and every read passed. That fake number also fed the margin
report in Diagnostics, so a fragile OCR step always looked perfectly healthy.
Confidence now comes from the engine per word.

**Bug: every OCV tool built in the UI failed every part.** The verification
checked `expect_text is not None`, and the property sheet supplies an empty
string, so the tool demanded the part be blank. Any pattern you typed was
ignored and the result was always a reject. Only the bundled demo escaped it,
because its JSON omits the field. Empty now means "not set".

Honest limit: dot-matrix marking still reads at about 26% CER. Joining the dots
with *Join dot-matrix (px)* set to 5 helps a great deal, but tesseract is weak on
non-solid strokes. If your parts are inkjet-coded, install PaddleOCR
(`pip install paddleocr paddlepaddle`), which the tool prefers automatically and
which handles dot-matrix far better.

## Accuracy audit of every measuring tool

After two real bugs turned up in `align` -- code that had been tested and
declared working -- the other tools got the same treatment: synthetic scenes
with exactly known ground truth, measure the error, then mutation-test the tests.

```
python bench_measure.py     measurement and calibration
python bench_tools.py       counting, colour, defect, presence
python bench_align.py       alignment
```

Results, all against exact ground truth:

| tool | measured accuracy |
|---|---|
| measure_length, clean edges | 0.023 px |
| measure_length, subpixel steps | 0.015 px mean -- genuinely subpixel |
| measure_length, blur 3 + noise sigma 8 | 0.324 px worst |
| calibrate_scale into measure_length | 20.000 mm recovered from a 400 px bar |
| count_blob, 1 to 10 features | exact, including noise, small features, low contrast |
| count_template, 2 to 6 features | exact |
| color_classify | full 4x4 confusion matrix correct |
| defect_diff, good parts | no false rejects under noise, +8% gain, +10 offset |
| defect_diff, defects | 6 px dent, thin scratch and missing feature all caught |
| presence | correct both ways |

Nothing was found wrong in these tools. That is a real result, not a formality:
the same method had already found `align` was 147 px out, so a clean sheet here
means something.

Two caveats the audit did surface, neither a defect:

**Colour margins are thin.** A correctly named red came back at deltaE 33.5
against a threshold of 40, and blue at 26.7. The verdicts are right, but they
are one lighting change from being wrong. Recalibrate the swatches in
`vstools/color.py` against the actual lamp on your station.

**Polarity on a single edge pair.** Selecting `dark_to_light` on a single bar
fails, because a bar has only one edge of that polarity. That is correct -- it
refuses rather than inventing a second edge -- and a two-bar scene measures
correctly with either polarity.

## Alignment accuracy  (fixed)

`Locate part` had two real bugs. Every region in a recipe is positioned by it,
so both put every downstream ROI in the wrong place while reporting a healthy
match score -- a confidently wrong fixture, which is the worst failure this app
can produce, because nothing downstream can tell it happened.

1. The rotation matrix was built about the image origin instead of the template
   centre.
2. The rotation sign was inverted. `cv2.getRotationMatrix2D` returns
   `[[cos, sin], [-sin, cos]]`, the transpose of the textbook form, so the
   recovered angle came out negated -- an error of exactly twice the true angle.

Measured against known transforms, worst corner displacement across eight poses:

```
                        before      after
  worst error         147.45 px    0.06 px
  mean cycle time        231 ms     148 ms
```

Faster as well as correct, because the intensity refinement step now converges
immediately from a good starting estimate instead of dragging one across tens of
pixels.

Also added: coarse-to-fine angle search on an image pyramid, subpixel peak
interpolation, and an ECC polish that is accepted only if it measurably improves
the match.

`python bench_align.py` reproduces the table.

## Tests

179 tests, two of them against a crop of a real switch faceplate. Coverage: 83% of the inspection code, 77% of the canvas, 57% of the
application window.

Coverage was measured rather than assumed, and it pointed at the weakest spot:
the YOLO tools sat at 18% because testing them looked like it needed a trained
network. It does not. `make_test_onnx.py` emits a genuine ONNX file in YOLOv8
layout -- fixed detections, class names in metadata -- which loads through real
onnxruntime. That exercises the parts that can plausibly be wrong (metadata
parsing, letterbox un-padding, non-maximum suppression, region offsets, count
logic) without any weights. YOLO coverage went from 18% to 90%.

Three behaviours documented for sessions but never actually tested are now
covered: that a failed `Locate part` stops the rest of the recipe instead of
reporting a page of meaningless failures, that the fixture transform really does
move downstream regions with the part, and that it does so for rotated and
elliptical regions too -- which is a separate code path, and mutation testing
showed it was entirely uncovered.

There is also a test that drives the real window: create a model, add tools, edit
regions through the property-sheet spinboxes, save, reload, run. That path
deserves its own test because the one bug that reached a shipped release lived
exactly there -- correct through hand-written JSON, broken through the property
sheet, and every test at the time used hand-written JSON.


```
pip install pytest
python -m pytest tests.py -v
```

Or double-click `run_tests.bat`. 24 tests, about a second.

These are deliberately not per-feature unit tests -- each feature was checked
when it was written. They target the *interactions*, which is where regressions
actually live: rotation combined with elliptical regions, undo restoring both,
recipes round-tripping every field through save and reload, logging surviving a
rotated recipe.

The suite was itself checked by mutation testing -- deliberately breaking the
code to confirm the tests notice. That found a real hole: the original rotated-
graphics test only used a rectangle, and rectangles take a separate branch. With
the general mapping path sabotaged, the whole suite stayed green. A test for
line, point and polygon graphics now covers it. Five further mutations (undo
coalescing, snapshot depth, ellipse masking, focus normalisation, class
reindexing) were all caught.

Mutation testing earned its keep twice more on alignment. The first set of align
tests passed even with the original origin-rotation bug restored, because the
refinement step was quietly rescuing a broken estimate. Only a test with
refinement disabled exposed it -- and that test then failed against the
supposedly fixed code, which is how the second bug, the inverted sign, was
found. Any matcher that leans on a polish step to hide a broken estimate will
fail the moment that polish does not converge, which on a real part is exactly
when things are already difficult.

Every dialog in the application is now opened by the suite with its modals
auto-dismissed, and the history tables are checked against the database behind
them. Those were the last paths only ever exercised by a hand-written script. A
dialog that raises is invisible to every other kind of test: the traceback goes
to a console nobody is watching and the window simply never appears.

Run these before trusting a change. A green suite is not proof the app is
correct, but a red one is proof something broke.

## Production history and traceability  (F9)

Every inspection is now recorded to a SQLite database at
`projects\<model>\results\results.db`. Until now results vanished when the app
closed, which cannot answer the question every real line eventually asks: what
did we ship on Tuesday, and which check rejected it?

**Runs** -- every inspection with timestamp, verdict, cycle time, which checks
failed, and the archived image.

**What is rejecting parts** -- a Pareto of failing checks, worst first. Note the
percentages can total more than 100%: one bad part commonly trips several checks,
and the column is the share of rejected parts in which each check failed. That is
deliberate. Presenting it as a share of blame would imply a single cause that
usually is not there.

**Yield by hour** -- yield trend, colour-coded. A yield that slides hour by hour
is the classic signature of lighting drift or a lens creeping, not of parts
getting worse.

**Export CSV** for QA, and a purge button for records older than 30 days.

**Image archiving** (Run menu) defaults to failures only. Saving every image
fills a drive in days. The archive is capped at 2000 images, oldest deleted
first -- but only the images are pruned, never the database records, so
traceability survives even after the pictures are gone.

Logging never slows inspection. Results go onto a queue and a background thread
commits them; measured cost on the inspection thread is 0.013 ms per run. If the
disk ever stalls, the queue drops its oldest entries rather than blocking, on the
principle that a late inspection is worse than an incomplete log.

## Diagnostics  (F8)

Three questions that decide whether a recipe survives a real shift.

**Image quality.** Focus, exposure clipping, contrast, dynamic range and motion
blur. Most "it worked yesterday" failures are not the recipe -- they are a lens
that crept, lighting someone changed, or a conveyor running faster than the
exposure allows.

Focus is contrast-normalised on purpose. Raw Laplacian variance, the usual
sharpness measure, collapses when an image is merely dark, which produces
exactly the wrong advice: it sends you to refocus a lens that never moved.
Dividing by intensity variance cancels the contrast term. Verified against a
darkened-but-sharp image, which scores 0.79 against 0.81 for the original and is
correctly reported as a lighting problem, not a focus one.

**Recipe margins.** For every tool with an accept threshold, how far the score
sat from it. A check passing at 0.71 against a 0.70 threshold is not passing, it
is a failure that has not happened yet -- normal part-to-part variation will push
it over, usually on a night shift. Anything inside 10% of its threshold is
flagged FRAGILE.

**Cycle time.** Per-tool timing, worst first. One tool almost always dominates
and it is rarely the expected one. On the demo recipe, OCR is 79% of a 1.0
second cycle -- if that needed to be faster, nothing else is worth touching.

**Run -> Stability check** runs the recipe over every test image and reports
which tool's score wanders most. Point it at images of parts you consider good:
the widest spread is where your first false reject will come from.

## Circular and elliptical regions

Region tab -> Shape -> Ellipse / circle. It uses the same X/Y/W/H box, inscribed,
so resizing and rotating work exactly as before.

Use it for anything round: pads, screw heads, LEDs, connector barrels, lens
apertures. A rectangle around a round feature drags in four corners of whatever
happens to sit next to it, and those corners are a common cause of phantom blobs
and false defect hits.

How it works underneath: the tool is still handed a rectangular patch, because
every tool expects one. Pixels outside the ellipse are replaced with the mean of
the pixels inside. That is deliberate -- blacking the corners out would invent
four hard edges and a large dark area that both thresholding and golden-image
differencing would react to. A neutral fill does not threshold as a blob, and in
differencing the identical fill appears on both sides and cancels.

Verified on a synthetic pad with bright junk in the box corners: a rectangle
region picks up all four pieces of junk, the ellipse picks up none, and the pad
itself is untouched. Works together with rotation.

## Undo and redo

Ctrl+Z and Ctrl+Shift+Z, with Ctrl+Y as an alternative redo. The Edit menu names
the specific action, so it reads "Undo delete 2 regions" rather than just "Undo".

One gesture is one undo step. Dragging a region emits geometry updates
continuously and a held arrow key repeats, so edits of the same kind arriving
close together collapse into a single entry -- otherwise one drag would become
fifty presses of Ctrl+Z and the feature would be useless.

History is per model and resets when you switch models. It covers adding,
deleting, pasting, duplicating, reordering, renaming, moving, resizing, rotating
and settings changes. It does not cover labelling, which saves to disk
immediately, or the test image list.

## Files

```
history.py   undo/redo snapshots and gesture coalescing
diagnostics.py  image quality, recipe margin and cycle time analysis
results.py   SQLite result logging, archiving and traceability queries
tests.py     integration test suite (python -m pytest tests.py)
bench_align.py  alignment accuracy benchmark against known transforms
bench_measure.py  measurement and calibration accuracy audit
bench_tools.py  counting, colour, defect and presence accuracy audit
bench_ocr.py  OCR character error rate across eight degradations
bench_defaults.py  every tool built the way the UI builds it
textnet_data.py  synthetic training data for the text detector
textnet_train.py  trains the detector and exports it to ONNX
models/textnet.onnx  the trained detector, 41 kB
inference.py  OpenVINO / onnxruntime backend shared by every model tool
job.py  jobs, sides and per-tool OK/NG training sets
vstools/geometry.py  Find Line, Find Circle, Histogram, Distance / angle
bench_geometry.py  geometry accuracy against exact sub-pixel ground truth
vstools/imageproc.py  image processing operators whose output later tools read
calibration.py  lens distortion calibration from checkerboard photographs
tool_editor.py  the classic window's separate tool editor
vp_app.py  the VisionPro-style application: QuickBuild, Job Editor, Tool Edit Control
vp_display.py  the one display control every window uses
tests_vp.py  tests for the VisionPro-style application
terminals.py  tool inputs, outputs and the rules for linking them
uikit.py  UI pieces shared by both front ends
comms.py  the TCP/IP line interface
example_line_client.py  check the line connection from the line PC
icons.py  the icon set: actions and every tool type
vstools/plots.py  projection and histogram plots for diagnostic records
vstools/analysis.py  Results analysis, Intersect, Fit line/circle, Color match
vstools/more_tools.py  Find ellipse, Image sharpness, Color segmenter, Classify
vstools/transform.py  Affine transform, Polar unwrap, Two-image operation
vstools/construct_tools.py  Fixture from points, N-point calibration, Create geometry, Bead inspect, Data analysis
make_gallery.py  rebuild the gallery of every tool detecting
bench_parallel.py  measure whether parallel jobs help on this PC
bench_real.py  real-photo campaign: line conditions must pass, planted defects must fail
real_checks.py  screws, presence and measurement on real photos, with result photos
vsruntime.py  run an exported package from your own code, no GUI
package.py  export and import .vsp packages
example_run_package.py  a complete example of using a package
tests_package.py  tests for packages and the runtime
run_classic.bat  the classic single window
build_chassis_job.py  builds the seven-side chassis job from its photographs
bench_inference.py  run on the line PC to choose runtime and precision
train_on_colab.ipynb  trains YOLO and the text detector on a free Colab GPU
exposure.py  clipping measurement and the automatic exposure search
preflight.py  environment check: what is installed and what it costs you
testdata/    a real faceplate crop used as a regression fixture
install.py   the installer (tested); install.bat just launches it
make_test_onnx.py  generates a YOLOv8-shaped ONNX file for testing
fake_pylon.py  simulates four Basler camera families for testing
```

## Not built yet

Polygon and circular regions, segmentation models (boxes only for now), lens
distortion calibration.
