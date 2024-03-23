cmake_minimum_required(VERSION 3.5)

project(yolo_onnx)

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

#link_directories(${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.17.1/lib)

add_library(${PROJECT_NAME} SHARED
    ${PROJECT_SOURCE_DIR}/src/inference_onnx.cpp
)

target_link_libraries(yolo_onnx
   ${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.17.1/lib
   ${OpenCV_LIBS}
)

#set(PROJECT_SOURCES
#    ${PROJECT_SOURCE_DIR}/main.cpp 
#    ${PROJECT_SOURCE_DIR}/src/inference_onnx.cpp 
#)
#add_executable(yolo_interface ${PROJECT_SOURCES})
#target_link_libraries(yolo_interface
#    ${OpenCV_LIBS}
#    onnxruntime
#)
