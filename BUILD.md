# Build Instructions

## Requirements

- CMake 3.20+
- C++ compiler which supports C++20 (e.g., clang-13, gcc-11, msvc-17)
   - MSVC and Clang (with GNU-style command-line options) are recommended and tested on Windows
- On Linux, `uuid-dev` is required to build the core libraries and the following libraries are required for the GUI module:
   - libopencv-dev
   - libglfw3-dev
   - libxinerama-dev
   - libxcursor-dev
   - libxi-dev
- On macOS with M1, you need to install `embree` since a pre-built binary is not provided by the official embree repo. We recommend using [Homebrew](https://brew.sh/) to install it: `brew install embree`.


### Backend Requirements

- CUDA
    - CUDA 11.2 or higher
    - RTX-compatible graphics cards with appropriate drivers
    - OptiX 7.1 or higher
- DirectX
    - DirectX 12 with ray tracing support
    - RTX-compatible graphics card with appropriate drivers
- ISPC
    - x86-64 CPU with AVX256 or Apple M1 CPU with ARM Neon
    - (Optional) LLVM 12+ with the corresponding targets and features enabled (for JIT executing the IR emitted by ISPC)
- Metal
    - macOS 12 or higher
    - Apple M1 chips are recommended (older GPUs are probably supported but not tested)
- LLVM
    - x86-64 CPU with AVX256 or Apple M1 CPU with ARM Neon
    - LLVM 13+ with the corresponding targets and features enabled
      - CMake seems to have trouble with LLVM 15 on Ubuntu, so we recommend using LLVM 13/14; please install LLVM 14 via `wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 14` and use CMake flag `-D LLVM_ROOT=/usr/lib/llvm-14` to specify the LLVM installation directory if you already have LLVM 15 installed

## Build Commands

```bash
cmake  -S . -B build # optionally with CMake flags behind
cmake --build build
```

## CMake Flags

The ISPC backend is disabled by default. Other backends will automatically be enabled if the corresponding APIs are detected. You can override the default settings by supplying CMake flags manually, in form of `-DFLAG=value` behind the first cmake command.

In case you need to run the ISPC backend, download the [ISPC compiler executable](https://ispc.github.io/downloads.html) of your platform and copy it to `src/backends/ispc/ispc_support/` before compiling.

- `CMAKE_BUILD_TYPE`: Set to Debug/Release to configure build type
- `LUISA_COMPUTE_ENABLE_CUDA`: Enable CUDA backend
- `LUISA_COMPUTE_ENABLE_DX`: Enable DirectX backend
- `LUISA_COMPUTE_ENABLE_ISPC`: Enable ISPC backend
- `LUISA_COMPUTE_ENABLE_METAL`: Enable Metal backend
- `LUISA_COMPUTE_ENABLE_GUI`: Enable GUI display in C++ tests (enabled by default)

Note: On Windows, please remember to replace the backslashes `\\` in the paths with `/` when passing arguments to CMake.

## Usage

1. The renderer's executable file is named `luisa-render-cli`, which locates in build directory
2. Usage can be accessed by running `luisa-render-cli --help`
3. The renderer can be simply run as
    ```bash
    luisa-render-cli -b <backend> <scene-file>
    ```
4. Example scene files are under `data/scenes`
