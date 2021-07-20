# Install (LINUX)

## Requirements (important!!)
* CMake >= 3.18: https://cmake.org/download/
* CUDA >= 10.2: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do Post-installation Actions)
* OpenCV >= 2.4: use your preferred package manager (brew, apt), build from source using vcpkg or download from OpenCV official site (on Windows set system variable OpenCV_DIR = C:\opencv\build - where are the include and x64 folders image)
* cuDNN >= 8.0.2 https://developer.nvidia.com/rdp/cudnn-archive (on Linux copy cudnn.h,libcudnn.so... as described here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on Windows copy cudnn.h,cudnn64_7.dll, cudnn64_7.lib as described here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
* GPU with CC >= 3.0: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* GCC >= 6.0

## Compile (using cmake)
```
git clone https://github.com/AlexeyAB/darknet
cd darknet
mkdir build_release
cd build_release
cmake ..
cmake --build . --target install --parallel 8
```

## Build Error
### [issue 1](https://github.com/AlexeyAB/darknet/issues/7486)
```
./src/network_kernels.cu(364): warning: variable "l" was declared but never referenced
./src/network_kernels.cu(694): error: identifier "cudaGraphExec_t" is undefined
./src/network_kernels.cu(697): error: identifier "cudaGraph_t" is undefined
./src/network_kernels.cu(706): error: identifier "cudaStreamCaptureModeGlobal" is undefined
./src/network_kernels.cu(706): error: identifier "cudaStreamBeginCapture" is undefined
./src/network_kernels.cu(714): error: identifier "cudaStreamEndCapture" is undefined
./src/network_kernels.cu(715): error: identifier "cudaGraphInstantiate" is undefined
./src/network_kernels.cu(725): error: identifier "cudaGraphLaunch" is undefined

7 errors detected in the compilation of "/tmp/tmpxft_00003086_00000000-9_network_kernels.compute_70.cpp1.ii".
Makefile:185: recipe for target 'obj/network_kernels.o' failed
make: *** [obj/network_kernels.o] Error 1
```
How to solve? ==> use old version darknet [Link](https://github.com/AlexeyAB/darknet/archive/64efa721ede91cd8ccc18257f98eeba43b73a6af.zip)
