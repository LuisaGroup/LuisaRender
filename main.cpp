#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <compatibility.h>
#include <mesh.h>
#include <ray_data.h>
#include <intersection_data.h>
#include <camera_data.h>
#include <frame_data.h>
#include <light_data.h>
#include <core/device.h>
#include <util/resource_manager.h>

int main(int argc [[maybe_unused]], char *argv[]) {
    
    ResourceManager::instance().set_working_directory(std::filesystem::current_path());
    ResourceManager::instance().set_binary_directory(std::filesystem::absolute(argv[0]).parent_path());
    
    auto device = Device::create("Metal");
    auto generate_rays_kernel = device->create_kernel("pinhole_camera_generate_rays");
    auto sample_light_kernel = device->create_kernel("sample_lights");
    auto trace_radiance_kernel = device->create_kernel("trace_radiance");
    auto sort_rays_kernel = device->create_kernel("sort_rays");
    auto filter_kernel = device->create_kernel("mitchell_natravali_filter");
    auto convert_colorspace_kernel = device->create_kernel("convert_colorspace_rgb");
    
    std::vector<MeshDescriptor> mesh_list;
    auto cube_obj_path = ResourceManager::instance().working_path("data/meshes/cube/cube.obj");
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
    auto bunny_obj_path = ResourceManager::instance().working_path("data/meshes/nanosuit/nanosuit.obj");
    auto bunny_transform = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, -5.0f, -1.0f}) *
                           glm::rotate(glm::mat4{1.0f}, glm::radians(30.0f), glm::vec3{0.0f, 1.0f, 0.0f}) *
                           glm::scale(glm::mat4{1.0f}, glm::vec3{0.5f});
    mesh_list.emplace_back(MeshDescriptor{bunny_obj_path, bunny_transform, glm::vec3{1.0f}, true});
    auto mesh = Mesh::load(mesh_list);
    
    auto position_buffer_size = mesh.positions.size() * sizeof(Vec3f);
    auto position_buffer = device->create_buffer(position_buffer_size, BufferStorageTag::MANAGED);
    position_buffer->upload(mesh.positions.data(), position_buffer_size);
    
    auto normal_buffer_size = mesh.normals.size() * sizeof(Vec3f);
    auto normal_buffer = device->create_buffer(normal_buffer_size, BufferStorageTag::MANAGED);
    normal_buffer->upload(mesh.normals.data(), normal_buffer_size);
    
    auto material_id_buffer_size = mesh.material_ids.size() * sizeof(uint);
    auto material_id_buffer = device->create_buffer(material_id_buffer_size, BufferStorageTag::MANAGED);
    material_id_buffer->upload(mesh.material_ids.data(), material_id_buffer_size);
    
    auto material_buffer_size = mesh.materials.size() * sizeof(MaterialData);
    auto material_buffer = device->create_buffer(material_buffer_size, BufferStorageTag::MANAGED);
    material_buffer->upload(mesh.materials.data(), material_buffer_size);
    
    auto accelerator = device->create_acceleration(*position_buffer, sizeof(Vec3f), mesh.material_ids.size());
    
    constexpr auto width = 1000u;
    constexpr auto height = 800u;
    
    constexpr auto max_ray_count = width * height;
    auto ray_buffer = device->create_buffer(max_ray_count * sizeof(RayData), BufferStorageTag::DEVICE_PRIVATE);
    auto swap_ray_buffer = device->create_buffer(max_ray_count * sizeof(RayData), BufferStorageTag::DEVICE_PRIVATE);
    auto gather_ray_buffer = device->create_buffer(max_ray_count * sizeof(GatherRayData), BufferStorageTag::DEVICE_PRIVATE);
    auto shadow_ray_buffer = device->create_buffer(max_ray_count * sizeof(ShadowRayData), BufferStorageTag::DEVICE_PRIVATE);
    auto its_buffer = device->create_buffer(max_ray_count * sizeof(IntersectionData), BufferStorageTag::DEVICE_PRIVATE);
    auto shadow_its_buffer = device->create_buffer(max_ray_count * sizeof(ShadowIntersectionData), BufferStorageTag::DEVICE_PRIVATE);
    
    auto result_texture = device->create_texture(uint2{width, height}, TextureFormatTag::RGBA32F, TextureAccessTag::READ_WRITE);
    
    CameraData camera_data{};
    camera_data.position = {0.0f, 0.0f, 15.0f};
    camera_data.front = {0.0f, 0.0f, -1.0f};
    camera_data.left = {-1.0f, 0.0f, 0.0f};
    camera_data.up = {0.0f, 1.0f, 0.0f};
    camera_data.near_plane = 0.1f;
    camera_data.fov = glm::radians(42.7f);
    
    FrameData frame{};
    frame.size = {width, height};
    frame.index = 0;
    
    auto light_count = 1u;
    LightData light;
    light.position = {0.0f, 4.0f, 0.0f};
    light.emission = {10.0f, 10.0f, 10.0f};
    auto light_buffer = device->create_buffer(sizeof(LightData), BufferStorageTag::MANAGED);
    light_buffer->upload(&light, sizeof(LightData));
    
    auto threadgroup_size = uint2(16, 16);
    auto threadgroups = uint2((width + threadgroup_size.x - 1) / threadgroup_size.x, (height + threadgroup_size.y - 1) / threadgroup_size.y);
    
    constexpr auto spp = 128u;
    constexpr auto max_depth = 31u;
    
    static auto available_frame_count = 16u;
    static std::mutex mutex;
    static std::condition_variable cond_var;
    static auto count = 0u;
    
    std::cout << "Rendering..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    
    std::vector<uint> initial_ray_counts;
    initial_ray_counts.resize(spp * (max_depth + 1u), 0u);
    for (auto i = 0u; i < spp; i++) {
        initial_ray_counts[i * (max_depth + 1u)] = max_ray_count;
    }
    
    auto ray_count_buffer_size = initial_ray_counts.size() * sizeof(uint);
    auto ray_count_buffer = device->create_buffer(ray_count_buffer_size, BufferStorageTag::MANAGED);
    ray_count_buffer->upload(initial_ray_counts.data(), ray_count_buffer_size);
    
    for (auto i = 0u; i < spp; i++) {
        
        // wait until max_frames_in_flight not exceeded
        {
            std::unique_lock lock{mutex};
            cond_var.wait(lock, [] { return available_frame_count != 0; });
        }
        
        device->launch_async(
            [&](KernelDispatcher &dispatch) {
                
                frame.index = i;
                
                dispatch(*generate_rays_kernel, threadgroups, threadgroup_size, [&](KernelArgumentEncoder &encoder) {
                    encoder["ray_buffer"]->set_buffer(*ray_buffer);
                    encoder["camera_data"]->set_bytes(&camera_data, sizeof(CameraData));
                    encoder["frame_data"]->set_bytes(&frame, sizeof(FrameData));
                });
                
                for (auto bounce = 0u; bounce < max_depth; bounce++) {
                    
                    auto curr_ray_count_offset = (i * (max_depth + 1u) + bounce) * sizeof(uint);
                    auto next_ray_count_offset = curr_ray_count_offset + sizeof(uint);
                    
                    accelerator->trace_nearest(dispatch, *ray_buffer, *its_buffer, *ray_count_buffer, curr_ray_count_offset);
                    
                    dispatch(*sample_light_kernel, threadgroups, threadgroup_size, [&](KernelArgumentEncoder &encoder) {
                        encoder["ray_buffer"]->set_buffer(*ray_buffer);
                        encoder["intersection_buffer"]->set_buffer(*its_buffer);
                        encoder["light_buffer"]->set_buffer(*light_buffer);
                        encoder["p_buffer"]->set_buffer(*position_buffer);
                        encoder["shadow_ray_buffer"]->set_buffer(*shadow_ray_buffer);
                        encoder["light_count"]->set_bytes(&light_count, sizeof(uint));
                        encoder["ray_count"]->set_buffer(*ray_count_buffer, curr_ray_count_offset);
                    });
                    
                    accelerator->trace_any(dispatch, *shadow_ray_buffer, *shadow_its_buffer, *ray_count_buffer, curr_ray_count_offset);
                    
                    dispatch(*trace_radiance_kernel, threadgroups, threadgroup_size, [&](KernelArgumentEncoder &encoder) {
                        encoder["ray_buffer"]->set_buffer(*ray_buffer);
                        encoder["shadow_ray_buffer"]->set_buffer(*shadow_ray_buffer);
                        encoder["its_buffer"]->set_buffer(*its_buffer);
                        encoder["shadow_its_buffer"]->set_buffer(*shadow_its_buffer);
                        encoder["p_buffer"]->set_buffer(*position_buffer);
                        encoder["n_buffer"]->set_buffer(*normal_buffer);
                        encoder["material_id_buffer"]->set_buffer(*material_id_buffer);
                        encoder["material_buffer"]->set_buffer(*material_buffer);
                        encoder["ray_count"]->set_buffer(*ray_count_buffer, curr_ray_count_offset);
                    });
                    
                    dispatch(*sort_rays_kernel, threadgroups, threadgroup_size, [&](KernelArgumentEncoder &encoder) {
                        encoder["ray_buffer"]->set_buffer(*ray_buffer);
                        encoder["ray_count"]->set_buffer(*ray_count_buffer, curr_ray_count_offset);
                        encoder["output_ray_buffer"]->set_buffer(*swap_ray_buffer);
                        encoder["output_ray_count"]->set_buffer(*ray_count_buffer, next_ray_count_offset);
                        encoder["frame_data"]->set_bytes(&frame, sizeof(FrameData));
                        encoder["gather_ray_buffer"]->set_buffer(*gather_ray_buffer);
                    });
                    
                    std::swap(ray_buffer, swap_ray_buffer);
                }
                
                // filter
                auto pixel_radius = 1u;
                dispatch(*filter_kernel, threadgroups, threadgroup_size, [&](KernelArgumentEncoder &encoder) {
                    encoder["ray_buffer"]->set_buffer(*gather_ray_buffer);
                    encoder["frame_data"]->set_bytes(&frame, sizeof(FrameData));
                    encoder["pixel_radius"]->set_bytes(&pixel_radius, sizeof(uint));
                    encoder["result"]->set_texture(*result_texture);
                });
            }, [] {
                {
                    std::lock_guard guard{mutex};
                    available_frame_count++;
                }
                cond_var.notify_one();
                std::cout << "Progress: " << (++count) << "/" << spp << std::endl;
            });
        
        std::lock_guard guard{mutex};
        available_frame_count--;
    }
    
    auto result_buffer = device->create_buffer(max_ray_count * sizeof(Vec4f), BufferStorageTag::MANAGED);
    device->launch([&] (KernelDispatcher &dispatch) {
        dispatch(*convert_colorspace_kernel, threadgroups, threadgroup_size, [&](KernelArgumentEncoder &encoder) {
            encoder["frame_data"]->set_bytes(&frame, sizeof(FrameData));
            encoder["result"]->set_texture(*result_texture);
        });
        result_texture->copy_to_buffer(dispatch, *result_buffer);
        result_buffer->synchronize(dispatch);
        ray_count_buffer->synchronize(dispatch);
    });
    
    auto t1 = std::chrono::steady_clock::now();
    
    auto total_ray_count = 0ul;
    auto final_ray_count_buffer = reinterpret_cast<const uint *>(ray_count_buffer->data());
    for (auto i = 0ul; i < initial_ray_counts.size(); i++) {
        total_ray_count += final_ray_count_buffer[i];
    }
    total_ray_count *= 2;  // two rays per bounce
    
    using namespace std::chrono_literals;
    auto total_time = (t1 - t0) / 1ns * 1e-9f;
    std::cout << "Total time:      " << total_time << "s" << std::endl;
    std::cout << "Total rays:      " << total_ray_count * 1e-6 << "M" << std::endl;
    std::cout << "Rays per second: " << static_cast<double>(total_ray_count) * 1e-6 / total_time << "M" << std::endl;
    
    cv::Mat image;
    image.create(cv::Size{width, height}, CV_32FC3);
    auto src_data = reinterpret_cast<const Vec4f *>(result_buffer->data());
    auto dest_data = reinterpret_cast<PackedVec3f *>(image.data);
    for (auto row = 0u; row < height; row++) {
        for (auto col = 0u; col < width; col++) {
            auto index = row * width + col;
            auto src = src_data[index];
            dest_data[index] = {src.b, src.g, src.r};
        }
    }
    cv::imwrite("result.exr", image);
    
    return 0;
}