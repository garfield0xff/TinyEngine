cmake_minimum_required(VERSION 3.10)
project(CodeGenerator)

set(CMAKE_CXX_STANDARD 17)

set(TFLITE_LIB_PATH "${CMAKE_SOURCE_DIR}/tensorflow/bazel-bin/tensorflow/lite")
set(TFLITE_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/tensorflow")

set(FLATBUFFERS_INCLUDE_DIR "/opt/homebrew/include")
set(FLATBUFFERS_LIB_DIR "/opt/homebrew/lib")

include_directories(${TFLITE_INCLUDE_PATH} ${FLATBUFFERS_INCLUDE_DIR})
link_directories(${TFLITE_LIB_PATH} ${FLATBUFFERS_LIB_DIR})

set(SOURCES
    src/codegen.cpp
    src/conv2d.cpp
    src/main.cpp
)

add_executable(CodeGenerator ${SOURCES})

target_link_libraries(CodeGenerator tensorflowlite flatbuffers)

target_include_directories(CodeGenerator PRIVATE ${PROJECT_SOURCE_DIR}/include ${FLATBUFFERS_INCLUDE_DIR})
