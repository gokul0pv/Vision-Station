"""Run a Vision Station package from your own code.

    python example_run_package.py chassis.vsp part.jpg "Front top"

Needs only the .vsp file, OpenCV and NumPy -- plus an OCR engine if the
application reads text, and openvino or onnxruntime if it uses trained models.
"""
import sys

import cv2

package, image_path = sys.argv[1], sys.argv[2]
job = sys.argv[3] if len(sys.argv) > 3 else None

sys.path.insert(0, package)          # the package carries the runtime
import vsruntime                      # noqa: E402

app = vsruntime.load(package)
print(app)                           # the application and its jobs

result = app.run(cv2.imread(image_path), job=job)
print(result)                        # verdict, time, failed tools

for tool_id, t in result.tools.items():
    print("  %-24s %-5s %s" % (tool_id, t["status"], t.get("message") or ""))

# any measurement by tool and name, e.g. result.value("hole_spacing", "value_px")

annotated = result.annotated()       # labels=False / only_failures=True for less clutter
cv2.imwrite("result.jpg", annotated)
cv2.imshow("Vision Station", cv2.resize(annotated, None, fx=0.5, fy=0.5))
cv2.waitKey(0)
