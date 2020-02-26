# LuisaRender
High-performance renderer on GPU.

## Features

### Backends
- [x] Apple Metal
- [ ] Intel Embree
- [ ] NVIDIA OptiX & OptiX-Prime

### Cameras
- [x] Pinhole Cameras
- [x] Thin-Lens Cameras
- [x] Camera Transform, Animation & Motion-Blur
- [ ] Shutter Curves

### Geometry
- [x] Triangle Meshes
- [x] Catmull-Clarke Subdivision (with ASSIMP)
- [x] Instancing
- [x] Static and Dynamic Transforms & Motion-Blur
- [ ] Out-of-Core Ray Tracing

### Illumination
- [x] Point Lights
- [x] Diffuse Area Lights
- [ ] HDRI Environment Maps
- [ ] Procedural Skylights
- [x] Uniform-Distribution Light Selection Strategy
- [ ] Power-Distribution Light Selection Strategy

### Appearance
- [ ] BSDFs
- [ ] Materials
- [ ] Textures & Filters
- [ ] Shading Language Integration
- [ ] Texture Caches & Streaming

### Samplers
- [x] Independent Sampler
- [ ] Halton Sampler

### Filters
- [x] Mitchell-Netravali Filter

### Integrators
- [x] Normal Visualizer
- [ ] Path Tracing
- [ ] SPPM
- [ ] PSSMLT

### Postprocessing
- [ ] Colorspace Management & Tone Mapping
- [ ] Postprocess Effects
- [ ] Denoisers

### Renders
- [x] Single-Shot Rendering
- [ ] Animation Rendering
- [ ] Interactive Rendering
- [ ] Remote Rendering

## Gallery

- Normal Visualizer + Subdivision + Instancing + Dynamic Transforms + Thin-Lens Camera + Motion-Blur (7.0s @ 1024spp)
![](gallery/bunny-motion-blur-normal.png)