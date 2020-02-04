#include <chrono>
#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <mesh.h>
#include <intersection_data.h>
#include <camera_data.h>
#include <frame_data.h>
#include <light_data.h>
#include <core/ray.h>
#include <core/device.h>
#include <util/resource_manager.h>

#include <core/data_types.h>
#include <core/mathematics.h>
#include <core/camera.h>
#include <core/parser.h>

#include <cameras/thin_lens_camera.h>

#include <glm/glm.hpp>

using namespace luisa;

int main(int argc, char *argv[]) {
    
    std::unique_ptr<ParameterSet> p;
    p->parse_child<Camera>("");
    
    luisa::float3 v;
    v = luisa::math::cos(v);
    
    ResourceManager::instance().set_working_directory(std::filesystem::current_path());
    ResourceManager::instance().set_binary_directory(std::filesystem::absolute(argv[0]).parent_path());
    
    auto device = Device::create("Metal");
    auto generate_rays_kernel = device->create_kernel("pinhole_camera_generate_rays");
    auto sample_light_kernel = device->create_kernel("sample_lights");
    auto update_ray_state_kernel = device->create_kernel("update_ray_states");
    auto trace_radiance_kernel = device->create_kernel("trace_radiance");
    auto sort_rays_kernel = device->create_kernel("sort_rays");
    auto film_clear_kernel = device->create_kernel("rgb_film_clear");
    auto film_gather_rays_kernel = device->create_kernel("rgb_film_gather_rays");
    auto film_convert_colorspace_kernel = device->create_kernel("rgb_film_convert_colorspace");
    
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
    auto bunny_transform = glm::translate(glm::mat4{1.0f}, glm::vec3{1.0f, -5.0f, -1.0f}) *
                           glm::rotate(glm::mat4{1.0f}, glm::radians(-30.0f), glm::vec3{0.0f, 1.0f, 0.0f}) *
                           glm::scale(glm::mat4{1.0f}, glm::vec3{0.25f});
    mesh_list.emplace_back(MeshDescriptor{bunny_obj_path, bunny_transform, glm::vec3{1.0f}, true});
    
    auto house_obj_path = "data/meshes/cow/cow.obj";
    auto cow_transform = math::translation(-1.5f, -5.0f, 0.0f) *
                         math::rotation(make_float3(0.0f, 1.0f, 0.0f), math::radians(40.0f)) *
                         math::scaling(25.0f);
    mesh_list.emplace_back(MeshDescriptor{house_obj_path, cow_transform, glm::vec3{0.5f, 0.4f, 0.3f}, false});
    
    auto sphere_obj_path = ResourceManager::instance().working_path("data/meshes/sphere/sphere.obj");
    auto sphere_transform = glm::translate(glm::mat4{1.0f}, glm::vec3{3.0f, -4.0f, 1.0f}) *
                            glm::rotate(glm::mat4{1.0f}, glm::radians(0.0f), glm::vec3{0.0f, 1.0f, 0.0f}) *
                            glm::scale(glm::mat4{1.0f}, glm::vec3{1.0f});
    mesh_list.emplace_back(MeshDescriptor{sphere_obj_path, sphere_transform, glm::vec3{1.0f}, true});
    
    auto mesh = Mesh::load(mesh_list);
    
    auto position_buffer_size = mesh.positions.size() * sizeof(float3);
    auto position_buffer = device->allocate_buffer(position_buffer_size, BufferStorage::MANAGED);
    position_buffer->upload(mesh.positions.data(), position_buffer_size);
    
    auto normal_buffer_size = mesh.normals.size() * sizeof(float3);
    auto normal_buffer = device->allocate_buffer(normal_buffer_size, BufferStorage::MANAGED);
    normal_buffer->upload(mesh.normals.data(), normal_buffer_size);
    
    auto material_id_buffer_size = mesh.material_ids.size() * sizeof(uint);
    auto material_id_buffer = device->allocate_buffer(material_id_buffer_size, BufferStorage::MANAGED);
    material_id_buffer->upload(mesh.material_ids.data(), material_id_buffer_size);
    
    auto material_buffer_size = mesh.materials.size() * sizeof(MaterialData);
    auto material_buffer = device->allocate_buffer(material_buffer_size, BufferStorage::MANAGED);
    material_buffer->upload(mesh.materials.data(), material_buffer_size);
    
    auto accelerator = device->create_acceleration(*position_buffer, sizeof(float3), mesh.material_ids.size());
    
    constexpr auto width = 1280u;
    constexpr auto height = 720u;
    
    constexpr auto max_ray_count = width * height;
    auto ray_index_buffer = device->allocate_buffer(max_ray_count * sizeof(uint), BufferStorage::DEVICE_PRIVATE);
    auto output_ray_index_buffer = device->allocate_buffer(max_ray_count * sizeof(uint), BufferStorage::DEVICE_PRIVATE);
    auto ray_buffer = device->allocate_buffer(max_ray_count * sizeof(Ray), BufferStorage::DEVICE_PRIVATE);
    auto ray_throughput_buffer = device->allocate_buffer(max_ray_count * sizeof(float3), BufferStorage::DEVICE_PRIVATE);
    auto ray_seed_buffer = device->allocate_buffer(max_ray_count * sizeof(uint), BufferStorage::DEVICE_PRIVATE);
    auto ray_radiance_buffer = device->allocate_buffer(max_ray_count * sizeof(float3), BufferStorage::DEVICE_PRIVATE);
    auto ray_depth_buffer = device->allocate_buffer(max_ray_count * sizeof(uint), BufferStorage::DEVICE_PRIVATE);
    auto ray_pixel_buffer = device->allocate_buffer(max_ray_count * sizeof(float2), BufferStorage::DEVICE_PRIVATE);
    auto ray_pdf_buffer = device->allocate_buffer(max_ray_count * sizeof(float), BufferStorage::DEVICE_PRIVATE);
    auto light_sample_buffer = device->allocate_buffer(max_ray_count * sizeof(LightSample), BufferStorage::DEVICE_PRIVATE);
    auto shadow_ray_buffer = device->allocate_buffer(max_ray_count * sizeof(Ray), BufferStorage::DEVICE_PRIVATE);
    auto its_buffer = device->allocate_buffer(max_ray_count * sizeof(IntersectionData), BufferStorage::DEVICE_PRIVATE);
    auto shadow_its_buffer = device->allocate_buffer(max_ray_count * sizeof(ShadowIntersectionData), BufferStorage::DEVICE_PRIVATE);
    auto accum_buffer = device->allocate_buffer(max_ray_count * sizeof(glm::ivec4), BufferStorage::DEVICE_PRIVATE);
    auto result_buffer = device->allocate_buffer(max_ray_count * sizeof(packed_float3), BufferStorage::MANAGED);
    
    CameraData camera_data{};
    camera_data.position = {0.8f, -2.5f, 15.0f};
    camera_data.front = {0.0f, 0.0f, -1.0f};
    camera_data.left = {-1.0f, 0.0f, 0.0f};
    camera_data.up = {0.0f, 1.0f, 0.0f};
    camera_data.near_plane = 0.1f;
    camera_data.fov = glm::radians(23.7f);
    camera_data.focal_distance = 16.0f;
    camera_data.aperture = 0.035f;
    
    FrameData frame_data{};
    frame_data.size = {width, height};
    frame_data.index = 0;
    
    auto light_count = 1u;
    LightData light{};
    light.position = {0.0f, 4.0f, 0.0f};
    light.emission = {10.0f, 10.0f, 10.0f};
    auto light_buffer = device->allocate_buffer(sizeof(LightData), BufferStorage::MANAGED);
    light_buffer->upload(&light, sizeof(LightData));
    
    auto threadgroup_size = uint2(16, 16);
    auto threadgroups = uint2((width + threadgroup_size.x - 1) / threadgroup_size.x, (height + threadgroup_size.y - 1) / threadgroup_size.y);
    auto threadgroup_size_1D = 256u;
    auto threadgroups_1D = (width * height + threadgroup_size_1D - 1) / threadgroup_size_1D;
    
    constexpr auto spp = 128u;
    constexpr auto max_depth = 11u;
    
    static auto available_frame_count = 4u;
    static std::mutex mutex;
    static std::condition_variable cond_var;
    static auto count = 0u;
    
    std::cout << "Rendering..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    
    std::vector<uint> initial_ray_counts(spp * (max_depth + 1u), 0u);
    auto ray_count_buffer_size = initial_ray_counts.size() * sizeof(uint);
    auto ray_count_buffer = device->allocate_buffer(ray_count_buffer_size, BufferStorage::MANAGED);
    ray_count_buffer->upload(initial_ray_counts.data(), ray_count_buffer_size);
    
    device->launch_async([&](KernelDispatcher &dispatch) {
        dispatch(*film_clear_kernel, threadgroups_1D, threadgroup_size_1D, [&](KernelArgumentEncoder &encode) {
            encode("accum_buffer", *accum_buffer);
            encode("ray_count", max_ray_count);
        });
    });
    
    std::mt19937 random{std::random_device{}()};
    std::uniform_real_distribution<float> distribution;
    
    for (auto i = 0u; i < spp; i++) {
        
        // wait until max_frames_in_flight not exceeded
        {
            std::unique_lock lock{mutex};
            cond_var.wait(lock, [] { return available_frame_count != 0; });
        }
        
        device->launch_async([&](KernelDispatcher &dispatch) {
            
            frame_data.index = i;
            
            auto frame_camera_data = camera_data;
            frame_camera_data.position.z += -2.0f + 4.0f * (distribution(random) + i) / spp;
            
            dispatch(*generate_rays_kernel, threadgroups, threadgroup_size, [&](KernelArgumentEncoder &encode) {
                encode("ray_index_buffer", *ray_index_buffer);
                encode("ray_buffer", *ray_buffer);
                encode("ray_count", ray_count_buffer->view<uint>(i * (max_depth + 1u)));
                encode("ray_throughput_buffer", *ray_throughput_buffer);
                encode("ray_seed_buffer", *ray_seed_buffer);
                encode("ray_radiance_buffer", *ray_radiance_buffer);
                encode("ray_depth_buffer", *ray_depth_buffer);
                encode("ray_pixel_buffer", *ray_pixel_buffer);
                encode("ray_pdf_buffer", *ray_pdf_buffer);
                encode("camera_data", frame_camera_data);
                encode("frame_data", frame_data);
            });
            
            for (auto bounce = 0u; bounce < max_depth; bounce++) {
                
                auto curr_ray_count_offset = (i * (max_depth + 1u) + bounce);
                auto next_ray_count_offset = curr_ray_count_offset + 1u;
                
                // intersection
                accelerator->trace_nearest(dispatch, *ray_buffer, *ray_index_buffer, *its_buffer, *ray_count_buffer, curr_ray_count_offset * sizeof(uint));
                
                // sample lights
                dispatch(*sample_light_kernel, threadgroups_1D, threadgroup_size_1D, [&](KernelArgumentEncoder &encode) {
                    encode("ray_index_buffer", *ray_index_buffer);
                    encode("ray_buffer", *ray_buffer);
                    encode("ray_seed_buffer", *ray_seed_buffer);
                    encode("intersection_buffer", *its_buffer);
                    encode("light_buffer", *light_buffer);
                    encode("light_sample_buffer", *light_sample_buffer);
                    encode("p_buffer", *position_buffer);
                    encode("shadow_ray_buffer", *shadow_ray_buffer);
                    encode("light_count", light_count);
                    encode("ray_count", ray_count_buffer->view<uint>(curr_ray_count_offset));
                });
                
                // shadow
                accelerator->trace_any(dispatch, *shadow_ray_buffer, *shadow_its_buffer, *ray_count_buffer, curr_ray_count_offset * sizeof(uint));
                
                // trace radiance
                dispatch(*trace_radiance_kernel, threadgroups_1D, threadgroup_size_1D, [&](KernelArgumentEncoder &encode) {
                    encode("ray_index_buffer", *ray_index_buffer);
                    encode("ray_buffer", *ray_buffer);
                    encode("ray_radiance_buffer", *ray_radiance_buffer);
                    encode("ray_throughput_buffer", *ray_throughput_buffer);
                    encode("ray_seed_buffer", *ray_seed_buffer);
                    encode("ray_depth_buffer", *ray_depth_buffer);
                    encode("light_sample_buffer", *light_sample_buffer);
                    encode("its_buffer", *its_buffer);
                    encode("shadow_its_buffer", *shadow_its_buffer);
                    encode("p_buffer", *position_buffer);
                    encode("n_buffer", *normal_buffer);
                    encode("material_id_buffer", *material_id_buffer);
                    encode("material_buffer", *material_buffer);
                    encode("ray_count", ray_count_buffer->view<uint>(curr_ray_count_offset));
                });
                
                // sort rays
                dispatch(*sort_rays_kernel, threadgroups_1D, threadgroup_size_1D, [&](KernelArgumentEncoder &encode) {
                    encode("ray_index_buffer", *ray_index_buffer);
                    encode("ray_buffer", *ray_buffer);
                    encode("output_ray_index_buffer", *output_ray_index_buffer);
                    encode("ray_count", ray_count_buffer->view<uint>(curr_ray_count_offset));
                    encode("output_ray_count", ray_count_buffer->view<uint>(next_ray_count_offset));
                });
                std::swap(ray_index_buffer, output_ray_index_buffer);
            }
            
            dispatch(*film_gather_rays_kernel, threadgroups_1D, threadgroup_size_1D, [&](KernelArgumentEncoder &encode) {
                encode("ray_radiance_buffer", *ray_radiance_buffer);
                encode("ray_pixel_buffer", *ray_pixel_buffer);
                encode("filter_radius", 1.5f);
                encode("frame_data", frame_data);
                encode("accum_buffer", *accum_buffer);
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
    
    device->launch([&](KernelDispatcher &dispatch) {
        dispatch(*film_convert_colorspace_kernel, threadgroups, threadgroup_size, [&](KernelArgumentEncoder &encoder) {
            encoder("frame_data", frame_data);
            encoder("accum_buffer", *accum_buffer);
            encoder("result_buffer", *result_buffer);
        });
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
    
    cv::Mat image(cv::Size{width, height}, CV_32FC3, result_buffer->data());
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite("result.exr", image);
    
    return 0;
}