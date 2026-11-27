cmake_minimum_required(VERSION 3.24)
project(VisionCore LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# Dependencies (see build.bat): the official OpenCV 4.10.0 Windows build --
# the same library Vision Station's cv2 is.
set(OPENCV_DIR "$ENV{LOCALAPPDATA}/opencv-4.10.0/build" CACHE PATH "OpenCV build folder")
add_library(opencv_world SHARED IMPORTED)
set_target_properties(opencv_world PROPERTIES
    IMPORTED_LOCATION "${OPENCV_DIR}/x64/vc16/bin/opencv_world4100.dll"
    IMPORTED_IMPLIB "${OPENCV_DIR}/x64/vc16/lib/opencv_world4100.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENCV_DIR}/include")

if(MSVC)
    # /EHa: Windows faults (access violation...) reach the check guard as exceptions
    add_compile_options(/EHa /W3 /utf-8 /permissive- /Zc:__cplusplus)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    # keep debug info in release too: crash dumps must be readable
    add_compile_options($<$<CONFIG:Release>:/Zi>)
    add_link_options($<$<CONFIG:Release>:/DEBUG> $<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
endif()

# ONNX Runtime 1.24.4: headers + import library from the NuGet package; at run
# time the very DLLs Vision Station's onnxruntime-directml wheel ships
set(ORT_DIR "$ENV{LOCALAPPDATA}/onnxruntime-dml-1.24.4" CACHE PATH "ONNX Runtime package")
set(ORT_RUNTIME "$ENV{LOCALAPPDATA}/Programs/Python/Python312/Lib/site-packages/onnxruntime/capi" CACHE PATH "onnxruntime.dll + DirectML.dll to ship")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_RUNTIME}/onnxruntime.dll"
    IMPORTED_IMPLIB "${ORT_DIR}/runtimes/win-x64/native/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/build/native/include")

# the engine: every check, the job loader, the run loop
add_library(vc_core STATIC
    src/core/types.cpp
    src/core/npy.cpp
    src/core/linalg.cpp
    src/core/nprandom.cpp
    src/core/npsort.cpp
    src/core/npz.cpp
    src/core/codes.cpp
    src/core/pyre.cpp
    src/core/annotate.cpp
    src/core/advanced.cpp
    third_party/miniz/miniz.c
    src/core/npsort_avx2.cpp
    src/core/npsort_avx512.cpp
    src/core/pyfmt.cpp
    src/core/tool.cpp
    src/core/engine.cpp
    src/core/onnx.cpp
    src/core/paths.cpp
    src/core/results.cpp
    third_party/sqlite/sqlite3.c
    src/tools/align.cpp
    src/tools/count.cpp
    src/tools/color.cpp
    src/tools/image.cpp
    src/tools/geometry.cpp
    src/tools/plots.cpp
    src/tools/analysis.cpp
    src/tools/more_tools.cpp
    src/tools/construct.cpp
    src/tools/models.cpp
    src/tools/learned.cpp
    src/tools/measure.cpp
    src/tools/ocr_engine.cpp
    src/tools/text.cpp
)
target_include_directories(vc_core PUBLIC third_party third_party/sqlite)
set_source_files_properties(third_party/sqlite/sqlite3.c PROPERTIES COMPILE_DEFINITIONS "SQLITE_THREADSAFE=1;SQLITE_DEFAULT_WAL_SYNCHRONOUS=1")
target_link_libraries(vc_core PUBLIC opencv_world onnxruntime)

# numpy's argsort: x86-simd-sort built once per instruction set, picked at run time
set_source_files_properties(src/core/npsort_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/wd4244")
set_source_files_properties(src/core/npsort_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512;/wd4244")

# numpy's own OpenBLAS (loaded at run time by src/core/linalg.cpp) so fits match numpy bit for bit
set(OPENBLAS_DIR "$ENV{LOCALAPPDATA}/vc-tools/openblas" CACHE PATH "libscipy_openblas64_*.dll from numpy.libs")
file(GLOB OPENBLAS_DLLS "${OPENBLAS_DIR}/libscipy_openblas64_*.dll")
# the barcode decoders pyzbar / pylibdmtx ship (loaded at run time by src/core/codes.cpp)
set(CODES_DIR "$ENV{LOCALAPPDATA}/vc-tools/codes" CACHE PATH "libzbar-64.dll, libiconv.dll, libdmtx-64.dll")
file(GLOB CODES_DLLS "${CODES_DIR}/*.dll")

# command-line runner; WHOLEARCHIVE keeps every check's self-registration
add_executable(vc-run src/apps/vc_run.cpp)
target_link_libraries(vc-run PRIVATE vc_core)
if(MSVC)
    target_link_options(vc-run PRIVATE /WHOLEARCHIVE:vc_core)
endif()
add_custom_command(TARGET vc-run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_DIR}/x64/vc16/bin/opencv_world4100.dll"
        "${ORT_RUNTIME}/onnxruntime.dll" "${ORT_RUNTIME}/onnxruntime_providers_shared.dll" "${ORT_RUNTIME}/DirectML.dll"
        ${OPENBLAS_DLLS} ${CODES_DLLS}
        $<TARGET_FILE_DIR:vc-run>)

# ---- engine process, supervisor, Studio
add_library(vc_ipc STATIC src/core/client.cpp src/core/paths.cpp src/core/types.cpp)
target_include_directories(vc_ipc PUBLIC third_party)
target_link_libraries(vc_ipc PUBLIC opencv_world)

# Basler pylon (C++ SDK, installed); its headers link its libraries themselves
set(PYLON_DEV "C:/Program Files/Basler/pylon/Development" CACHE PATH "pylon SDK")
add_library(vc_hw STATIC src/hw/camera.cpp src/hw/light.cpp src/hw/line.cpp)
target_include_directories(vc_hw PUBLIC "${PYLON_DEV}/include")
target_link_directories(vc_hw PUBLIC "${PYLON_DEV}/lib/x64")
target_link_libraries(vc_hw PUBLIC opencv_world ws2_32)
target_include_directories(vc_hw PUBLIC third_party)

add_executable(vc-engine src/apps/vc_engine.cpp)
target_link_libraries(vc-engine PRIVATE vc_core vc_hw)
if(MSVC)
    target_link_options(vc-engine PRIVATE /WHOLEARCHIVE:vc_core)
endif()

add_executable(vc-supervisor src/apps/vc_supervisor.cpp)
target_link_libraries(vc-supervisor PRIVATE vc_ipc)

set(Slint_DIR "$ENV{LOCALAPPDATA}/slint-cpp-1.19.1/lib/cmake/Slint" CACHE PATH "Slint package")
find_package(Slint REQUIRED)
add_executable(vc-studio WIN32 src/apps/vc_studio.cpp)
slint_target_sources(vc-studio ui/studio.slint)
target_link_libraries(vc-studio PRIVATE vc_ipc Slint::Slint ole32 shell32)
if(MSVC)
    target_link_options(vc-studio PRIVATE /ENTRY:mainCRTStartup)
endif()
add_custom_command(TARGET vc-studio POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$ENV{LOCALAPPDATA}/slint-cpp-1.19.1/lib/slint_cpp.dll" $<TARGET_FILE_DIR:vc-studio>)
