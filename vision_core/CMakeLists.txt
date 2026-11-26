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
    src/core/pyfmt.cpp
    src/core/tool.cpp
    src/core/engine.cpp
    src/core/onnx.cpp
    src/core/paths.cpp
    src/tools/align.cpp
    src/tools/count.cpp
    src/tools/color.cpp
    src/tools/measure.cpp
    src/tools/ocr_engine.cpp
    src/tools/text.cpp
)
target_include_directories(vc_core PUBLIC third_party)
target_link_libraries(vc_core PUBLIC opencv_world onnxruntime)

# command-line runner; WHOLEARCHIVE keeps every check's self-registration
add_executable(vc-run src/apps/vc_run.cpp)
target_link_libraries(vc-run PRIVATE vc_core)
if(MSVC)
    target_link_options(vc-run PRIVATE /WHOLEARCHIVE:vc_core)
endif()
add_custom_command(TARGET vc-run POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_DIR}/x64/vc16/bin/opencv_world4100.dll"
        "${ORT_RUNTIME}/onnxruntime.dll" "${ORT_RUNTIME}/onnxruntime_providers_shared.dll" "${ORT_RUNTIME}/DirectML.dll"
        $<TARGET_FILE_DIR:vc-run>)
