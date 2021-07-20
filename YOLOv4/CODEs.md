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

