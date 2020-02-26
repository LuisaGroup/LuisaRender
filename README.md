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

### Animated Subdivided Bunny Instances Rendered with Normal Visualizer using a Thin-Lens Camera
![](gallery/bunny-motion-blur-normal.png)