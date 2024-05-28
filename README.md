# Qt Vehicle Panoramic Stitching System

## Introduction
Qt-based implementation of the vehicle panoramic stitching system, the system input is 4 fisheye images, including color and depth maps, and camera calibration parameters. The output is a panoramic effect that changes in real time around the vehicle.

## Demonstration
<div align="center"> 
  <img width="80%" src="assets/demo.gif"/>
</div>

## Dependency
To run this project, the following dependencies need to be installed.
- OpenCV - 4.5.5 with cuda
- Qt - 5.14.2
- CUDA - 11.4
- glm - use apt install
- Eigen - 3.4.0
- assimp - use apt install

## Compile and Run
You need to modify the following in CMakeLists.txt:
```shell
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) # cuda library path
set(CMAKE_CUDA_ARCHITECTURES 86) # change 86 to adapt your device

find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_NVCC_FLAGS
        ${CMAKE_NVCC_FLAGS};
        -Xcompiler
        -fPIC
        -use_fast_math
        -gencode arch=compute_86,code=sm_86) # this 86 place change too

include_directories(/usr/local/include/eigen3) # change this path to adapt your eigen library
```
Now you can compile and run this project.
```shell
cd [source_directory]
mkdir build
cd build
cmake ..
make
./gui
```

## Controls
- Keyboard W ===== forward
- Keyboard A ===== left
- Keyboard S ===== backward
- Keyboard D ===== right
- left mouse button ===== rotation
- Double-click on the panoramic module to show the panoramic image in full screen, and double-click again to resume.
