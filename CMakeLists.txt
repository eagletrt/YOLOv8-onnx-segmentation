cmake_minimum_required(VERSION 3.5)

project(YOLO_INTERFACE)

# CUDA
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
#find_package(CUDA 11 REQUIRED)

#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA

# OpenCV
find_package(OpenCV REQUIRED)
# !OpenCV

set(YOLO_INCLUDE_DIR_INTERNAL
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
set(YOLO_INCLUDE_DIR ${YOLO_INCLUDE_DIR_INTERNAL} PARENT_SCOPE)

include_directories(
   ${YOLO_INCLUDE_DIR_INTERNAL}
)

add_library(${PROJECT_NAME} SHARED
    ${PROJECT_SOURCE_DIR}/src/inference.cpp
)

target_link_libraries(${PROJECT_NAME}
   ${OpenCV_LIBS}
)

#set(PROJECT_SOURCES
#    main.cpp 
#    inference.cpp 
#)
#add_executable(yolo_interface ${PROJECT_SOURCES})
#target_link_libraries(yolo_interface ${OpenCV_LIBS})
