cmake_minimum_required(VERSION 3.5)

project(${PROJECT_NAME})

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

include_directories(
   ${YOLO_INCLUDE_DIR_INTERNAL}
)

add_library(${PROJECT_NAME} SHARED
    ${PROJECT_SOURCE_DIR}/src/inference_onnx.cpp
)

target_link_libraries(${PROJECT_NAME}
   ${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.17.1/lib
   ${OpenCV_LIBS}
)
