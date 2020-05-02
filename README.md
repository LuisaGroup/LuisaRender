# LuisaRender
High-Performance Renderer on GPU.

## Features

### Backends
- [x] Apple Metal
- [ ] Intel Embree
- [ ] NVIDIA OptiX & OptiX-Prime
- [ ] ...

### Cameras
- [x] Pinhole Cameras
- [x] Thin-Lens Cameras
- [ ] Realistic Cameras
- [ ] Fish-Eye Cameras
- [ ] ...
- [x] Camera Transform, Animation & Motion-Blur
- [ ] Shutter Curves

### Geometry
- [x] Triangle Meshes
- [x] Catmull-Clark Subdivision (with ASSIMP)
- [x] Instancing
- [x] Static and Dynamic Transforms & Motion-Blur
- [ ] Curves
- [ ] Out-of-Core Ray Tracing

### Illumination
- [x] Point Lights
- [x] Diffuse Area Lights
- [ ] Realistic Lights
- [ ] HDRI Environment Maps
- [ ] Procedural Skylights
- [ ] ...
- [x] Uniform-Distribution Light Selection Strategy
- [ ] Power-Distribution Light Selection Strategy

### Appearance
- [ ] BSDFs
- [ ] Materials
- [ ] Spectral Rendering
- [ ] Textures & Filters
- [ ] Shading Language Integration
- [ ] Texture Caches & Streaming

### Samplers
- [x] Independent Sampler
- [ ] Halton Sampler
- [ ] ...

### Reconstruction Filters
- [x] Filter Importance Sampling
- [x] Mitchell-Netravali Filter
- [x] Box Filter
- [x] Triangle Filter
- [x] Gaussian Filter
- [x] Lanczos Windowed Sinc Filter 

### Integrators
- [x] Normal Visualizer
- [ ] Path Tracing
- [ ] SPPM
- [ ] PSSMLT
- [ ] ...
- [ ] AOV Support

### Postprocessing
- [ ] Colorspace Management & Tone Mapping
- [ ] Postprocess Effects
- [ ] Denoisers

### Renders
- [x] Scene Description Language
- [x] Single-Shot Rendering
- [ ] Animation Rendering
- [ ] Interactive Rendering
- [ ] Remote Rendering

## Gallery

- Normal Visualizer + Subdivision + Instancing + Dynamic Transforms + Thin-Lens Camera + Motion-Blur (7.0s @ 1024spp)
![](gallery/bunny-motion-blur-normal.png)