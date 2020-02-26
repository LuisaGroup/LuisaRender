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

### Lights
- [x] Point Lights
- [x] Diffuse Area Lights
- [ ] HDRI Environment Maps
- [ ] Procedural Skylights

### Materials
Nothing

### Samplers
- [x] Independent Sampler
- [ ] Halton Sampler

### Integrators
- [x] Normal Visualizer
- [ ] Path Tracing

### Renders
- [x] Single-Shot Rendering
- [ ] Animation Rendering

## Galary

- Normal Visualizer + Subdivision + Instancing + Dynamic Transforms + Thin-Lens Camera + Motion-Blur (7.0s @ 1024spp)
![](gallery/bunny-motion-blur-normal.png)