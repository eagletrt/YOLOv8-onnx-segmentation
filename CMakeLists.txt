cmake_minimum_required(VERSION 3.5)

project(YOLO_INTERFACE)

# OpenCV
find_package(OpenCV REQUIRED)
# !OpenCV

set(YOLO_INCLUDE_DIR_INTERNAL
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.17.1/include
    ${OpenCV_INCLUDE_DIRS}
)

set(YOLO_INCLUDE_DIR ${YOLO_INCLUDE_DIR_INTERNAL} PARENT_SCOPE)

link_directories(${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.17.1/lib)
include_directories(
   ${YOLO_INCLUDE_DIR_INTERNAL}
)

add_library(${PROJECT_NAME} SHARED
    ${PROJECT_SOURCE_DIR}/src/inference_onnx.cpp
)
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.17.1/lib/libonnxruntime.so")


target_link_libraries(${PROJECT_NAME}
   ${OpenCV_LIBS}
   onnxruntime
)
