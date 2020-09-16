# LuisaRender
High-Performance Renderer on GPU.

## Features

### Architecture
| Feature                                                  | Progress  |
|----------------------------------------------------------|-----------|
| Plugin System                                            | Done      |
| Embedded DSL for Runtime Kernel Generation & Compilation | Done      |
| Scene Description Language                               | Done      |
| Wavefront Path Tracing                                   | Planned   |

### Backends
| Feature                    | Progress    |
|----------------------------|-------------|
| Apple Metal                | Done        |
| NVIDIA OptiX & OptiX-Prime | Partial     |
| Intel Embree               |             |
| ...                        |             |

### Cameras
| Feature                                   | Progress    |
|-------------------------------------------|-------------|
| Pinhole Cameras                           | Done        |
| Thin-Lens Cameras                         | Refactoring |
| Realistic Cameras                         |             |
| Fish-Eye Cameras                          |             |
| ...                                       |             |
| Camera Transform, Animation & Motion-Blur | Partial     |
| Shutter Curves                            |             |

### Geometry
| Feature                                     | Progress    |
|---------------------------------------------|-------------|
| Triangle Meshes (Wavefront OBJ Format)      | Partial     |
| Catmull-Clark Subdivision                   |             |
| Instancing                                  | Partial     |
| Static and Dynamic Transforms & Motion-Blur | Partial     |
| Curves                                      |             |
| Out-of-Core Ray Tracing                     |             |

### Illumination
| Feature                                       | Progress    |
|-----------------------------------------------|-------------|
| Point Lights                                  | Refactoring |
| Diffuse Area Lights                           | Refactoring |
| Realistic Lights                              |             |
| HDRI Environment Maps                         |             |
| Procedural Skylights                          |             |
| ...                                           |             |
| Uniform-Distribution Light Selection Strategy | Refactoring |
| Power-Distribution Light Selection Strategy   |             |

### Appearance
| Feature                      | Progress |
|------------------------------|----------|
| BSDFs                        | Planned  |
| Materials                    | Planned  |
| Spectral Rendering           |          |
| Textures & Filters           |          |
| Shading Language Integration |          |
| Texture Caches & Streaming   |          |

### Samplers
| Feature             | Progress    |
|---------------------|-------------|
| Independent Sampler | Done        |
| Halton Sampler      | Planned     |
| ...                 |             |

### Reconstruction Filters
| Feature                      | Progress    |
|------------------------------|-------------|
| Filter Importance Sampling   | Done        |
| Mitchell-Netravali Filter    | Done        |
| Box Filter                   | Done        |
| Triangle Filter              | Done        |
| Gaussian Filter              | Done        |
| Lanczos Windowed Sinc Filter | Done        |

### Integrators
| Feature           | Progress    |
|-------------------|-------------|
| Normal Visualizer | Done        |
| Ambient Occlusion | Done        |
| Path Tracing      | Planned     |
| SPPM              |             |
| PSSMLT            |             |
| ...               |             |
| AOV Support       | Planned     |

### Postprocessing
| Feature                              | Progress |
|--------------------------------------|----------|
| Colorspace Management & Tone Mapping |          |
| Postprocessing Effects               |          |
| Denoising                            |          |

### Renders
| Feature               | Progress    |
|-----------------------|-------------|
| Single-Shot Rendering | Partial     |
| Animation Rendering   |             |
| Interactive Rendering |             |
| Remote Rendering      |             |

## Gallery

- Normal Visualizer + Subdivision + Instancing + Dynamic Transforms + Thin-Lens Camera + Motion-Blur (7.0s @ 1024spp)
![](gallery/bunny-motion-blur-normal.png)
