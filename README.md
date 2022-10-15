# LuisaRender

LuisaRender is a high-performance cross-platform Monte-Carlo renderer for stream architectures based
on [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute).

LuisaRender is also the *rendering application* described in the **SIGGRAPH Asia 2022** paper
> ***LuisaRender: A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures***.

See also [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) for the underlying *framework* as described in the paper; and please visit the [project page](https://luisa-render.com) for other information about the paper and the project.

# Scenes

LuisaRender supports a JSON-based and a custom text-based formats for scene description. Please visit [LuisaRenderScenes](https://github.com/LuisaGroup/LuisaRenderScenes) for demo scenes and their renderings.

We also provide a simple script at `tools/tungsten2luisa.py` to convert Tungsten scenes into LuisaRender's custom scene description language; and a CLI application at `src/apps/export.cpp` to convert glTF scenes to LuisaRender's JSON-based format. But please note that both tools are not perfect. Manual tweaks is necessary to fix conversion errors and/or align the converted scenes to their original appearances.
