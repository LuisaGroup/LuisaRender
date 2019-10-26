#import <chrono>
#import <iostream>
#import <opencv2/opencv.hpp>
#import <glm/glm.hpp>
#import <glm/ext.hpp>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import <compatibility.h>
#import <mesh.h>
#import <ray_data.h>
#import <intersection_data.h>
#import <camera_data.h>
#import <frame_data.h>
#import <light_data.h>

#import <core/device.h>
#import <util/resource_manager.h>

int main(int argc [[maybe_unused]], char *argv[]) {
    
    ResourceManager::instance().set_working_directory(std::filesystem::current_path());
    ResourceManager::instance().set_binary_directory(std::filesystem::absolute(argv[0]).parent_path());
    
    auto dev = Device::create("Metal");
    
    auto generate_rays_kernel = dev->create_kernel("pinhole_camera_generate_rays");
    auto sample_lights_kernel = dev->create_kernel("sample_lights");
    auto trace_radiance_kernel = dev->create_kernel("trace_radiance");
    auto reconstruct_kernel = dev->create_kernel("mitchell_natravali_filter");
    auto accumulate_kernel = dev->create_kernel("accumulate");
    
    auto working_dir = std::filesystem::current_path();
    auto binary_dir = std::filesystem::absolute(argv[0]).parent_path();
    
    id<MTLDevice> device = nullptr;
    for (id<MTLDevice> potential_device in MTLCopyAllDevices()) {
        if (!potential_device.isLowPower) {
            device = potential_device;
            break;
        }
    }
    NSLog(@"%@", device);
    
    auto command_queue = [device newCommandQueue];
    [command_queue autorelease];
    
    auto library_path = [[[NSString alloc] initWithCString:(binary_dir / "kernels.metallib").c_str() encoding:NSUTF8StringEncoding] autorelease];
    auto library = [device newLibraryWithFile:library_path error:nullptr];
    [library autorelease];
    
    auto pipeline_desc = [[MTLComputePipelineDescriptor alloc] init];
    [pipeline_desc autorelease];
    pipeline_desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
    
    pipeline_desc.computeFunction = [library newFunctionWithName:@"pinhole_camera_generate_rays"];
    auto generate_ray_pso = [device newComputePipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionNone reflection:nullptr error:nullptr];
    [generate_ray_pso autorelease];
    
    pipeline_desc.computeFunction = [library newFunctionWithName:@"sample_lights"];
    auto sample_lights_pso = [device newComputePipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionNone reflection:nullptr error:nullptr];
    [sample_lights_pso autorelease];
    
    pipeline_desc.computeFunction = [library newFunctionWithName:@"trace_radiance"];
    auto trace_radiance_pso = [device newComputePipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionNone reflection:nullptr error:nullptr];
    [trace_radiance_pso autorelease];
    
    pipeline_desc.computeFunction = [library newFunctionWithName:@"accumulate"];
    auto accumulate_pso = [device newComputePipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionNone reflection:nullptr error:nullptr];
    [accumulate_pso autorelease];
    
    pipeline_desc.computeFunction = [library newFunctionWithName:@"mitchell_natravali_filter"];
    MTLAutoreleasedComputePipelineReflection reflection;
    auto filter_pso = [device newComputePipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionArgumentInfo reflection:&reflection error:nullptr];
    [filter_pso autorelease];
    NSLog(@"%@", reflection);
    
    std::vector<MeshDescriptor> mesh_list;
    auto cube_obj_path = std::filesystem::path{working_dir}.append("data").append("meshes").append("cube").append("cube.obj");
    auto scaling = glm::scale(glm::mat4{1.0f}, glm::vec3{10.1f});
    auto transform_back = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, -10.0f}) * scaling;
    auto transform_top = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 10.0f, 0.0f}) * scaling;
    auto transform_bottom = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, -10.0f, 0.0f}) * scaling;
    auto transform_left = glm::translate(glm::mat4{1.0f}, glm::vec3{-10.0f, 0.0f, 0.0f}) * scaling;
    auto transform_right = glm::translate(glm::mat4{1.0f}, glm::vec3{10.0f, 0.0f, 0.0f}) * scaling;
    mesh_list.emplace_back(MeshDescriptor{cube_obj_path, transform_back, glm::vec3{1.0f}, false});
    mesh_list.emplace_back(MeshDescriptor{cube_obj_path, transform_top, glm::vec3{1.0f}, false});
    mesh_list.emplace_back(MeshDescriptor{cube_obj_path, transform_bottom, glm::vec3{1.0f}, false});
    mesh_list.emplace_back(MeshDescriptor{cube_obj_path, transform_left, glm::vec3{1.0f, 0.0f, 0.0f}, false});
    mesh_list.emplace_back(MeshDescriptor{cube_obj_path, transform_right, glm::vec3{0.0f, 1.0f, 0.0f}, false});
//    auto bunny_obj_path = std::filesystem::path{working_dir}.append("data").append("meshes").append("bunny").append("bunny.obj");
    auto bunny_obj_path = std::filesystem::path{working_dir}.append("data").append("meshes").append("nanosuit").append("nanosuit.obj");
    auto bunny_transform = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, -5.0f, -1.0f}) *
                           glm::rotate(glm::mat4{1.0f}, glm::radians(30.0f), glm::vec3{0.0f, 1.0f, 0.0f}) *
                           glm::scale(glm::mat4{1.0f}, glm::vec3{0.5f});
    mesh_list.emplace_back(MeshDescriptor{bunny_obj_path, bunny_transform, glm::vec3{1.0f}, true});
    auto mesh = Mesh::load(mesh_list);
    
    auto position_buffer = [device newBufferWithBytes:mesh.positions.data() length:mesh.positions.size() * sizeof(Vec3f) options:MTLResourceStorageModeManaged];
    [position_buffer autorelease];
    
    auto normal_buffer = [device newBufferWithBytes:mesh.normals.data() length:mesh.normals.size() * sizeof(Vec3f) options:MTLResourceStorageModeManaged];
    [normal_buffer autorelease];
    
    auto material_id_buffer = [device newBufferWithBytes:mesh.material_ids.data() length:mesh.material_ids.size() * sizeof(uint) options:MTLResourceStorageModeManaged];
    [material_id_buffer autorelease];
    
    auto material_buffer = [device newBufferWithBytes:mesh.materials.data() length:mesh.materials.size() * sizeof(MaterialData) options:MTLResourceStorageModeManaged];
    [material_buffer autorelease];
    
    auto ray_intersector = [[MPSRayIntersector alloc] initWithDevice:device];
    [ray_intersector autorelease];
    ray_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    ray_intersector.intersectionDataType = MPSIntersectionDataTypeDistancePrimitiveIndexCoordinates;
    ray_intersector.rayStride = sizeof(RayData);
    
    auto shadow_ray_intersector = [[MPSRayIntersector alloc] initWithDevice:device];
    [shadow_ray_intersector autorelease];
    shadow_ray_intersector.rayDataType = MPSRayDataTypeOriginMinDistanceDirectionMaxDistance;
    shadow_ray_intersector.intersectionDataType = MPSIntersectionDataTypeDistance;
    shadow_ray_intersector.rayStride = sizeof(ShadowRayData);
    
    auto accelerator = [[MPSTriangleAccelerationStructure alloc] initWithDevice:device];
    [accelerator autorelease];
    accelerator.vertexBuffer = position_buffer;
    accelerator.vertexStride = sizeof(Vec3f);
    accelerator.triangleCount = mesh.material_ids.size();
    [accelerator rebuild];
    
    constexpr auto width = 1024u;
    constexpr auto height = 768u;
    
    constexpr auto ray_count = width * height;
    auto ray_buffer = [device newBufferWithLength:ray_count * sizeof(RayData) options:MTLResourceStorageModePrivate];
    [ray_buffer autorelease];
    auto shadow_ray_buffer = [device newBufferWithLength:ray_count * sizeof(ShadowRayData) options:MTLResourceStorageModePrivate];
    [shadow_ray_buffer autorelease];
    auto its_buffer = [device newBufferWithLength:ray_count * sizeof(IntersectionData) options:MTLResourceStorageModePrivate];
    [its_buffer autorelease];
    auto shadow_its_buffer = [device newBufferWithLength:ray_count * sizeof(ShadowIntersectionData) options:MTLResourceStorageModePrivate];
    [shadow_its_buffer autorelease];
    
    auto texture_desc = [[MTLTextureDescriptor alloc] init];
    [texture_desc autorelease];
    texture_desc.pixelFormat = MTLPixelFormatRGBA32Float;
    texture_desc.textureType = MTLTextureType2D;
    texture_desc.width = width;
    texture_desc.height = height;
    texture_desc.storageMode = MTLStorageModePrivate;
    texture_desc.allowGPUOptimizedContents = true;
    texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    auto filter_texture = [device newTextureWithDescriptor:texture_desc];
    auto result_texture = [device newTextureWithDescriptor:texture_desc];
    [filter_texture autorelease];
    [result_texture autorelease];
    
    CameraData camera_data{};
    camera_data.position = {0.0f, -1.0f, 15.0f};
    camera_data.front = {0.0f, 0.0f, -1.0f};
    camera_data.left = {-1.0f, 0.0f, 0.0f};
    camera_data.up = {0.0f, 1.0f, 0.0f};
    camera_data.near_plane = 0.1f;
    camera_data.fov = glm::radians(35.0f);
    
    FrameData frame{};
    frame.size = {width, height};
    frame.index = 0;
    
    LightData light;
    light.position = {0.0f, 4.0f, 0.0f};
    light.emission = {10.0f, 10.0f, 10.0f};
    auto light_buffer = [device newBufferWithBytes:&light length:sizeof(LightData) options:MTLResourceStorageModeManaged];
    [light_buffer autorelease];
    auto light_count = 1u;
    
    auto threads_per_group = MTLSizeMake(32, 32, 1);
    auto thread_groups = MTLSizeMake((width + threads_per_group.width - 1) / threads_per_group.width, (height + threads_per_group.height - 1) / threads_per_group.height, 1);
    
    constexpr auto spp = 128u;
    
    static auto available_frame_count = 8u;
    static std::mutex mutex;
    static std::condition_variable cond_var;
    static auto count = 0u;
    
    std::cout << "Rendering..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    
    for (auto i = 0u; i < spp; i++) {
        
        // wait until max_frames_in_flight not exceeded
        {
            std::unique_lock lock{mutex};
            cond_var.wait(lock, [] { return available_frame_count != 0; });
        }
        
        auto command_buffer = [command_queue commandBuffer];
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
            {
                std::lock_guard guard{mutex};
                available_frame_count++;
            }
            cond_var.notify_one();
            std::cout << "Progress: " << (++count) << "/" << spp << std::endl;
        }];
        
        frame.index = i;
        
        auto command_encoder = [command_buffer computeCommandEncoder];
        [command_encoder setBuffer:ray_buffer offset:0 atIndex:0];
        [command_encoder setBytes:&camera_data length:sizeof(CameraData) atIndex:1];
        [command_encoder setBytes:&frame length:sizeof(FrameData) atIndex:2];
        [command_encoder setComputePipelineState:generate_ray_pso];
        [command_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:threads_per_group];
        [command_encoder endEncoding];
        
        for (auto bounce = 0u; bounce < 10u; bounce++) {
            
            // intersection
            [ray_intersector encodeIntersectionToCommandBuffer:command_buffer
                                              intersectionType:MPSIntersectionTypeNearest
                                                     rayBuffer:ray_buffer
                                               rayBufferOffset:0
                                            intersectionBuffer:its_buffer
                                      intersectionBufferOffset:0
                                                      rayCount:ray_count
                                         accelerationStructure:accelerator];
            
            // sample lights
            command_encoder = [command_buffer computeCommandEncoder];
            [command_encoder setBuffer:ray_buffer offset:0 atIndex:0];
            [command_encoder setBuffer:its_buffer offset:0 atIndex:1];
            [command_encoder setBuffer:light_buffer offset:0 atIndex:2];
            [command_encoder setBuffer:position_buffer offset:0 atIndex:3];
            [command_encoder setBuffer:shadow_ray_buffer offset:0 atIndex:4];
            [command_encoder setBytes:&light_count length:sizeof(uint) atIndex:5];
            [command_encoder setBytes:&frame length:sizeof(FrameData) atIndex:6];
            [command_encoder setComputePipelineState:sample_lights_pso];
            [command_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:threads_per_group];
            [command_encoder endEncoding];
            
            // intersection
            [shadow_ray_intersector encodeIntersectionToCommandBuffer:command_buffer
                                                     intersectionType:MPSIntersectionTypeAny
                                                            rayBuffer:shadow_ray_buffer
                                                      rayBufferOffset:0
                                                   intersectionBuffer:shadow_its_buffer
                                             intersectionBufferOffset:0
                                                             rayCount:ray_count
                                                accelerationStructure:accelerator];
            
            // trace radiance
            command_encoder = [command_buffer computeCommandEncoder];
            [command_encoder setBuffer:ray_buffer offset:0 atIndex:0];
            [command_encoder setBuffer:shadow_ray_buffer offset:0 atIndex:1];
            [command_encoder setBuffer:its_buffer offset:0 atIndex:2];
            [command_encoder setBuffer:shadow_its_buffer offset:0 atIndex:3];
            [command_encoder setBuffer:position_buffer offset:0 atIndex:4];
            [command_encoder setBuffer:normal_buffer offset:0 atIndex:5];
            [command_encoder setBuffer:material_id_buffer offset:0 atIndex:6];
            [command_encoder setBuffer:material_buffer offset:0 atIndex:7];
            [command_encoder setBytes:&frame length:sizeof(FrameData) atIndex:8];
            [command_encoder setComputePipelineState:trace_radiance_pso];
            [command_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:threads_per_group];
            [command_encoder endEncoding];
        }
        
        // filter
        auto pixel_radius = 1u;
        command_encoder = [command_buffer computeCommandEncoder];
        [command_encoder setBuffer:ray_buffer offset:0 atIndex:0];
        [command_encoder setBytes:&frame length:sizeof(FrameData) atIndex:1];
        [command_encoder setBytes:&pixel_radius length:sizeof(uint) atIndex:2];
        [command_encoder setTexture:filter_texture atIndex:0];
        [command_encoder setComputePipelineState:filter_pso];
        [command_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:threads_per_group];
        [command_encoder endEncoding];
        
        // accumulate
        command_encoder = [command_buffer computeCommandEncoder];
        [command_encoder setBytes:&frame length:sizeof(FrameData) atIndex:0];
        [command_encoder setTexture:filter_texture atIndex:0];
        [command_encoder setTexture:result_texture atIndex:1];
        [command_encoder setComputePipelineState:accumulate_pso];
        [command_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:threads_per_group];
        [command_encoder endEncoding];
        
        [command_buffer commit];
        std::lock_guard guard{mutex};
        available_frame_count--;
    }
    
    auto result_buffer = [device newBufferWithLength:ray_count * sizeof(Vec4f) options:MTLResourceStorageModeManaged];
    [result_buffer autorelease];
    
    auto command_buffer = [command_queue commandBuffer];
    auto blit_encoder = [command_buffer blitCommandEncoder];
    [blit_encoder copyFromTexture:result_texture
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(width, height, 1)
                         toBuffer:result_buffer
                destinationOffset:0
           destinationBytesPerRow:width * sizeof(Vec4f)
         destinationBytesPerImage:width * height * sizeof(Vec4f)
                          options:MTLBlitOptionNone];
    [blit_encoder synchronizeResource:result_buffer];
    [blit_encoder endEncoding];
    
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    auto t1 = std::chrono::steady_clock::now();
    
    using namespace std::chrono_literals;
    std::cout << "Done in " << (t1 - t0) / 1ns * 1e-9f << "s" << std::endl;
    
    cv::Mat image;
    image.create(cv::Size{width, height}, CV_32FC3);
    auto src_data = reinterpret_cast<Vec4f *>(result_buffer.contents);
    auto dest_data = reinterpret_cast<PackedVec3f *>(image.data);
    for (auto row = 0u; row < height; row++) {
        for (auto col = 0u; col < width; col++) {
            auto index = row * width + col;
            auto src = src_data[index];
            dest_data[index] = {src.b, src.g, src.r};
        }
    }
    cv::imwrite("result.exr", image);
//    cv::pow(image, 1.0f / 2.2f, image);
//    cv::imshow("Image", image);
//    cv::waitKey();
    
    return 0;
}