cmake_minimum_required(VERSION 3.12)
project(qt_vehicle_pano_stitch)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "Release")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# qt
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

# glm
find_package(glm REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

# opengl
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# cuda
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_NVCC_FLAGS
        ${CMAKE_NVCC_FLAGS};
        -Xcompiler
        -fPIC
        -use_fast_math
        -gencode arch=compute_86,code=sm_86)
include_directories(${CUDA_INCLUDE_DIRS})

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# eigen
include_directories(/usr/local/include/eigen3)

# assimp
find_package(ASSIMP REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})

include_directories(include include/gui include/SV include/Utils include/common)

include_directories(build/gui_autogen/include)

set(THIRD_LIBRARIES
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        ${GLM_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        ${ASSIMP_LIBRARIES})

add_subdirectory(src)
add_subdirectory(include)

add_executable(gui main.cpp)
target_link_libraries(gui PUBLIC ${THIRD_LIBRARIES} surround_system)
target_compile_options(gui PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_NVCC_FLAGS}>)
