"""Vision Core package: everything a clean Windows PC needs, in one folder
(and a zip of it). The camera additionally needs Basler pylon installed (its
driver), exactly as Vision Station does.

    python package.py            -> dist/VisionCore/ and dist/VisionCore-<date>.zip

Each program's imports are checked with dumpbin: a DLL it needs that is
neither shipped nor part of Windows stops the build, so a package cannot
leave the line PC with a missing file."""
import glob
import datetime
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(HERE, "build")
DIST = os.path.join(HERE, "dist", "VisionCore")
LOCAL = os.environ["LOCALAPPDATA"]
DUMPBIN = os.path.join(LOCAL, r"msvc-portable\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe")

PROGRAMS = ["vc-studio.exe", "vc-engine.exe", "vc-supervisor.exe", "vc-run.exe"]
DLLS = ["opencv_world4100.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll", "DirectML.dll", "slint_cpp.dll"]
# the Microsoft C++ runtime (redistributable by Microsoft's licence)
RUNTIME = ["msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"]
# provided by Windows itself, or by the pylon driver installation
SYSTEM_OK = {"kernel32.dll", "user32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll", "gdi32.dll",
             "ws2_32.dll", "comdlg32.dll", "imm32.dll", "dwmapi.dll", "uxtheme.dll", "shlwapi.dll", "winmm.dll",
             "bcrypt.dll", "crypt32.dll", "dxgi.dll", "d3d11.d3d12.dll", "d3d12.dll", "opengl32.dll", "setupapi.dll",
             "version.dll", "ntdll.dll", "userenv.dll", "ncrypt.dll", "secur32.dll", "iphlpapi.dll", "propsys.dll",
             "comctl32.dll", "dbghelp.dll", "powrprof.dll", "cfgmgr32.dll", "dwrite.dll", "d2d1.dll", "mf.dll",
             "mfplat.dll", "mfreadwrite.dll", "pylonbase_v12.dll", "pylonutility_v12.dll",
             "gcbase_md_vc142_v3_5_basler_pylon_v1.dll", "genapi_md_vc142_v3_5_basler_pylon_v1.dll",
             "dcomp.dll", "wininet.dll", "msvcrt.dll", "rpcrt4.dll", "psapi.dll", "netapi32.dll", "wtsapi32.dll",
             "dxcore.dll", "uiautomationcore.dll", "combase.dll", "bcryptprimitives.dll", "d3dcompiler_47.dll"}


def imports(path):
    out = subprocess.run([DUMPBIN, "/nologo", "/dependents", path], capture_output=True, text=True).stdout
    return [ln.strip().lower() for ln in out.splitlines()
            if ln.strip().lower().endswith(".dll") and not ln.strip().lower().startswith("dump of file")]


def main():
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    os.makedirs(DIST)
    for f in PROGRAMS + DLLS:
        shutil.copy2(os.path.join(BUILD, f), DIST)
    # numpy's OpenBLAS: fits computed exactly as Vision Station computes them (src/core/linalg.cpp)
    for f in glob.glob(os.path.join(BUILD, "libscipy_openblas64_*.dll")):
        shutil.copy2(f, DIST)
    # the barcode decoders (src/core/codes.cpp): zbar + its iconv, libdmtx
    for f in ("libzbar-64.dll", "libiconv.dll", "libdmtx-64.dll"):
        if os.path.exists(os.path.join(BUILD, f)):
            shutil.copy2(os.path.join(BUILD, f), DIST)
    for f in RUNTIME:
        shutil.copy2(os.path.join(os.environ["SystemRoot"], "System32", f), DIST)
    # the text reader's models
    shutil.copytree(os.path.join(ROOT, "models", "ocr"), os.path.join(DIST, "models", "ocr"),
                    ignore=shutil.ignore_patterns("*.yml"))
    # the trained surface check's deep model network
    bb = os.path.join(ROOT, "models", "backbone")
    if os.path.isdir(bb):
        shutil.copytree(bb, os.path.join(DIST, "models", "backbone"))
    # licences of everything shipped
    lic = os.path.join(DIST, "licenses")
    os.makedirs(lic)
    for src, name in ((os.path.join(LOCAL, "opencv-4.10.0", "LICENSE"), "OpenCV-LICENSE.txt"),
                      (os.path.join(LOCAL, "onnxruntime-dml-1.24.4", "LICENSE"), "ONNXRuntime-LICENSE.txt"),
                      (os.path.join(LOCAL, "onnxruntime-dml-1.24.4", "ThirdPartyNotices.txt"), "ONNXRuntime-ThirdPartyNotices.txt"),
                      (os.path.join(LOCAL, "slint-cpp-1.19.1", "licenses", "LICENSE.md"), "Slint-LICENSE.md"),
                      (os.path.join(HERE, "ui", "fonts", "Inter-LICENSE.txt"), "Inter-LICENSE.txt"),
                      (os.path.join(HERE, "ui", "icons", "LICENSE.txt"), "Lucide-LICENSE.txt"),
                      (os.path.join(LOCAL, "vc-tools", "openblas", "LICENSE.txt"), "OpenBLAS-numpy-LICENSE.txt"),
                      (os.path.join(LOCAL, "vc-tools", "codes", "zbar-LICENSE.txt"), "zbar-LICENSE.txt"),
                      (os.path.join(LOCAL, "vc-tools", "codes", "libdmtx-LICENSE.txt"), "libdmtx-LICENSE.txt"),
                      (os.path.join(HERE, "third_party", "x86-simd-sort", "LICENSE.md"), "x86-simd-sort-LICENSE.md"),
                      (os.path.join(HERE, "third_party", "miniz", "LICENSE"), "miniz-LICENSE.txt")):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(lic, name))
    with open(os.path.join(DIST, "Start Vision Core.bat"), "w") as f:
        f.write('@echo off\r\ncd /d "%~dp0"\r\nstart "" vc-supervisor.exe --studio\r\n')
    with open(os.path.join(DIST, "README.txt"), "w") as f:
        f.write("Vision Core\r\n\r\nStart: double-click 'Start Vision Core.bat' (the supervisor starts the\r\n"
                "engine and the Studio, and restarts the engine if it ever stops).\r\n\r\n"
                "Camera: install Basler pylon (the camera driver) first.\r\n"
                "Line PCs: vc-engine listens on \\\\.\\pipe\\vision-core-engine; the TCP line\r\n"
                "interface is started with the engine command line_start.\r\n")
    # every program's DLLs must be shipped or part of Windows
    shipped = {f.lower() for f in os.listdir(DIST) if f.lower().endswith(".dll")}
    missing = {}
    for f in PROGRAMS + DLLS:
        for d in imports(os.path.join(DIST, f)):
            if d not in shipped and d not in SYSTEM_OK and not d.startswith("api-ms-win-") and not d.startswith("ext-ms-"):
                missing.setdefault(f, []).append(d)
    if missing:
        print("MISSING DLLs:", missing)
        sys.exit(1)
    stamp = datetime.date.today().isoformat()
    zpath = os.path.join(HERE, "dist", "VisionCore-%s.zip" % stamp)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _d, files in os.walk(DIST):
            for n in files:
                p = os.path.join(base, n)
                z.write(p, os.path.relpath(p, os.path.dirname(DIST)))
    size = sum(os.path.getsize(os.path.join(b, n)) for b, _d, fs in os.walk(DIST) for n in fs)
    print("package %s: %.0f MB, zip %.0f MB, all DLL imports satisfied" % (DIST, size / 1e6, os.path.getsize(zpath) / 1e6))


if __name__ == "__main__":
    main()
