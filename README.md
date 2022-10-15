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

LuisaRender supports a JSON-based and a custom text-based formats for scene description. We maintain the demo scenes in a separate [repo](https://github.com/LuisaGroup/LuisaRenderScenes). We sincerely thank all the authors, [Rendering Resources](https://benedikt-bitterli.me/resources), [Poly Heaven](https://polyhaven.com), and [Blender Demo Files](https://download.blender.org/demo/cycles/lone-monk_cycles_and_exposure-node_demo.blend) for sharing these amazing resources.

We also provide a simple script at `tools/tungsten2luisa.py` to convert Tungsten scenes into LuisaRender's custom scene description language; and a CLI application at `src/apps/export.cpp` (compiled to `<build-folder>/bin/luisa-render-export`) to convert glTF scenes to LuisaRender's JSON-based format. But please note that both tools are not perfect. Manual tweaks is necessary to fix conversion errors and/or align the converted scenes to their original appearances.

## Contemporary Bathroom

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/bathroom.zip)

- Credit: [Mareck](http://www.blendswap.com/users/view/Mareck) (CC0)
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1024x1024
- Samples: 65536
- Tonemapping: ACES (exposure = -0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/bathroom.png" width="80%" alt="Bathroom"/>

## Bedroom

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/bedroom.zip)

- Credit: [SlykDrako](http://www.blendswap.com/user/SlykDrako) (CC0)
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1280x720
- Samples: 65536
- Tonemapping: ACES (exposure = -0.5)

![Bedroom](https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/bedroom.png)

## Camera

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/camera.zip)

- Credit: All resources in the scene are from [Poly Heaven](https://polyhaven.com) (CC0, see the contained `README.txt` for the detail of each resource)
- Resolution: 3840x2160
- Samples: 65536
- Tonemapping: Uncharted2

![Camera](https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/camera.png)

## Kitchen

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/kitchen.zip)

- Credit: [Jay-Artist](http://www.blendswap.com/user/Jay-Artist) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1280x720
- Samples: 65536
- Tonemapping: ACES (exposure = -0.5)

![Kitchen](https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/kitchen.png)

## Spaceship

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/spaceship.zip)

- Credit: [thecali](http://www.blendswap.com/user/thecali) (CC0)
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1920x1080
- Samples: 16384
- Tonemapping: Uncharted2

![Kitchen](https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/spaceship.png)

## Modern Hall

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/staircase2.zip)

- Credit: [NewSee2l035](http://www.blendswap.com/user/NewSee2l035) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1024x1024
- Samples: 65536
- Tonemapping: ACES (exposure = -0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/staircase2.png" width="80%" alt="Staircase2"/>

## The Wooden Staircase

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/staircase.zip)

- Credit: [Wig42](http://www.blendswap.com/users/view/Wig42) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1080x1920
- Samples: 16384
- Tonemapping: Uncharted2 (exposure = 0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/staircase.png" width="60%" alt="Staircase"/>

## Coffee Maker

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/coffee.zip)

- Credit: [cekuhnen](http://www.blendswap.com/user/cekuhnen) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1200x1800
- Samples: 16384
- Tonemapping: Uncharted2 (exposure = 0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/coffee.png" width="70%" alt="Coffee"/>

## Japanese Classroom

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/classroom.zip)

- Credit: [NovaZeeke](http://www.blendswap.com/users/view/NovaZeeke) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1920x1080
- Samples: 16384
- Tonemapping: Uncharted2 (exposure = 0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/classroom.png" alt="Classroom"/>

## The Breakfast Room

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/dining-room.zip)

- Credit: [Wig42](http://www.blendswap.com/users/view/Wig42) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1920x1080
- Samples: 16384
- Tonemapping: Uncharted2 (exposure = 0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/dining-room.png" alt="Dining Room"/>

## The Grey & White Room

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/living-room.zip)

- Credit: [Wig42](http://www.blendswap.com/users/view/Wig42) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1920x1080
- Samples: 16384
- Tonemapping: Uncharted2 (exposure = 0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/living-room.png" alt="Living Room"/>

## The White Room

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/living-room-2.zip)

- Credit: [Jay-Artist](http://www.blendswap.com/user/Jay-Artist) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1280x720
- Samples: 65536
- Tonemapping: ACES (exposure = 0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/living-room-2.png" alt="Living Room 2"/>

## The Modern Living Room

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/living-room-3.zip)

- Credit: [Wig42](http://www.blendswap.com/users/view/Wig42) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1280x720
- Samples: 65536
- Tonemapping: ACES (exposure = 0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/living-room-3.png" alt="Living Room 3"/>

## Glass of Water

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/glass-of-water.zip)

- Credit: [aXel](http://www.blendswap.com/user/aXel) (CC0)
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1920x1080
- Samples: 16384
- Tonemapping: Uncharted2

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/glass-of-water.png" alt="Glass of Water"/>

## Salle de bain

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/bathroom2.zip)

- Credit: [nacimus](http://www.blendswap.com/users/view/nacimus) ([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/))
- Converted from Tungsten version at [Rendering Resources](https://benedikt-bitterli.me/resources)
- Resolution: 1280x720
- Samples: 65536
- Tonemapping: ACES (exposure = -0.5)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/bathroom2.png" alt="Salle de bain"/>


## Lone Monk

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/lone-monk.zip)

- Credit: Carlo Bergonzini / [Monorender](http://www.monorender.com) ([CC-BY](https://creativecommons.org/licenses/by/2.0/))
- Converted from Blender Cycles format at [Blender Demo Files](https://download.blender.org/demo/cycles/lone-monk_cycles_and_exposure-node_demo.blend)
- Resolution: 6000x4000
- Samples: 65536
- Tonemapping: Uncharted2 (exposure = +1)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/lone-monk.jpg" alt="Lone Monk"/>


## Sky Texture Demo

Download: [LuisaRender](https://github.com/LuisaGroup/LuisaRenderScenes/releases/download/scenes/sky-texture-demo.zip)

- Credit: Blender Studio (CC0)
- Converted from Blender Cycles format at [Blender Demo Files](https://download.blender.org/demo/cycles/lone-monk_cycles_and_exposure-node_demo.blend)
- Resolution: 3840x2160
- Samples: 1024
- Tonemapping: Uncharted2 (exposure = +1)

<img src="https://github.com/LuisaGroup/LuisaRenderScenes/raw/main/renders/sky-texture-demo.png" alt="Sky Texture Demo"/>

