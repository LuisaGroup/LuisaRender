# LuisaRender

LuisaRender is a high-performance cross-platform Monte-Carlo renderer for stream architectures based
on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute).

LuisaRender is also the *rendering application* described in the **SIGGRAPH Asia 2022** paper
> ***LuisaRender: A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures***.

See also [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) for the underlying *framework* as described in the paper; and please visit the [project page](https://luisa-render.com) for other information about the paper and the project.

## Building

LuisaRender follows the standard CMake build process. Basically these steps:

- Check your hardware and platform. Currently, we support CUDA on Linux and Windows; DirectX on Windows; Metal on macOS; and ISPC and LLVM on all the major platforms. For CUDA and DirectX, an RTX-enabled graphics card, e.g., NVIDIA RTX 20 and 30 series, is required.

- Prepare the environment and dependencies. We recommend using the latest IDEs, Compilers, CMake, CUDA drivers, etc. Since we aggressively use new technologies like C++20 and OptiX 7.1+, you may need to, for example, upgrade your VS to 2019 or 2022, and install CUDA 11.0+. Note that if you would like to enable the CUDA backend, [OptiX](https://developer.nvidia.com/designworks/optix/download) is required. For some tests like the toy path tracer, [OpenCV](opencv.org) is also required.

- Clone the repo with the `--recursive` option:
```bash
git clone --recursive https://github.com/LuisaGroup/LuisaRender.git
```
Since we use Git submodules to manage third-party dependencies, a `--recursive` clone is required. Also, as we are not allowed to provide the OptiX headers in tree, you have to copy them from `<optix-installation>/include` to `src/compute/src/backends/cuda/optix`, so that the latter folder *directly* contains `optix.h`. We applogize for this inconvenience.

- Configure the project using CMake. E.g., for command line, `cd` into the project folder and type `cmake -S . -B <build-folder>`. You might also want to specify your favorite generators and build types using options like `-G Ninja` and `-D CMAKE_BUILD_TYPE=Release`. A typical, full command sequence for this would be like
```bash
cd LuisaRender
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
```

- If the configuration succeeds, you are now able to build the project. Type `cmake --build <build-folder>` in the command line, or push the build button if you generated, e.g., a VS project. (And in case the configuration step unluckily failed :-(, please file an [issue](https://github.com/LuisaGroup/LuisaRender/issues)). 

- After building, you will find the CLI executable at `<build-folder>/bin/luisa-render-cli`.

See also [BUILD.md](BUILD.md) for details on platform requirements, configuration options, and other precautions.

# Usage

Use command line to execute LuisaRender:
```bash
<build-fodler>/bin/luisa-render-cli -b <backend> [-d <device-index>] <scene-file>
```

To print the help message about the command line arguments, simply type
```bash
<build-fodler>/bin/luisa-render-cli -h
```
or
```bash
<build-fodler>/bin/luisa-render-cli --help
```

# Scenes

LuisaRender supports a JSON-based and a custom text-based formats for scene description. Please visit [LuisaRenderScenes](https://github.com/LuisaGroup/LuisaRenderScenes) for demo scenes and their renderings.

We also provide a simple script at `tools/tungsten2luisa.py` to convert Tungsten scenes into LuisaRender's custom scene description language; and a CLI application at `src/apps/export.cpp` (compiled to `<build-folder>/bin/luisa-render-export`) to convert glTF scenes to LuisaRender's JSON-based format. But please note that both tools are not perfect. Manual tweaks is necessary to fix conversion errors and/or align the converted scenes to their original appearances.
