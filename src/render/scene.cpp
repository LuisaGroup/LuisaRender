//
// Created by Mike Smith on 2020/9/4.
//

#include <compute/dsl.h>
#include "scene.h"

namespace luisa::render {

using namespace compute;
using namespace compute::dsl;

Scene::Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, std::shared_ptr<Background> background, float initial_time)
    : _device{device},
      _background{std::move(background)} {
    
    _process_geometry(shapes, initial_time);
}

void Scene::_encode_geometry_buffers(const std::vector<std::shared_ptr<Shape>> &shapes,
                                     float3 *positions,
                                     float3 *normals,
                                     float2 *uvs,
                                     TriangleHandle *triangles,
                                     EntityHandle *entities,
                                     std::vector<EntityRange> &entity_ranges,
                                     std::vector<Material *> &instance_materials,
                                     uint *instances) {
    
    size_t vertex_count = 0u;
    size_t triangle_count = 0u;
    size_t instance_count = 0u;
    
    std::queue<std::tuple<Shape *, TransformTree *, Material *>> queue;
    for (auto &&shape: shapes) { queue.emplace(shape.get(), &_transform_tree, nullptr); }
    
    std::unordered_map<Shape *, uint> entity_to_id;
    while (!queue.empty()) {
        
        auto[shape, transform_tree, material] = queue.front();
        queue.pop();
        
        if (material == nullptr) { material = shape->material(); }
        
        if (shape->is_entity()) {  // leaf node, containing one instance of entity
            
            auto instance_id = instance_count++;
            transform_tree->add_leaf(shape->transform(), instance_id);
            
            auto iter = entity_to_id.find(shape);
            if (iter == entity_to_id.end()) {  // unvisited entity
                
                auto vertex_offset = vertex_count;
                auto triangle_offset = triangle_count;
                
                // copy vertices
                auto &&vertices = shape->vertices();
                for (auto i = 0u; i < vertices.size(); i++) {
                    positions[vertex_offset + i] = make_float3(vertices[i].position);
                    normals[vertex_offset + i] = make_float3(vertices[i].normal);
                    uvs[vertex_offset + i] = vertices[i].uv;
                }
                vertex_count += vertices.size();
                
                // copy indices
                auto indices = shape->triangles();
                std::copy(indices.cbegin(), indices.cend(), triangles + triangle_offset);
                triangle_count += shape->triangles().size();
                
                shape->clear();
                
                auto entity_id = static_cast<uint>(entity_ranges.size());
                entity_ranges.emplace_back(EntityRange{static_cast<uint>(vertex_offset), static_cast<uint>(triangle_offset), static_cast<uint>(indices.size())});
                entities[entity_id] = {static_cast<uint>(vertex_offset), static_cast<uint>(triangle_offset)};
                
                iter = entity_to_id.emplace(shape, entity_id).first;
            }
            auto entity_id = iter->second;
            instances[instance_id] = entity_id;
            entities[instance_id] = {entity_ranges[entity_id].vertex_offset, entity_ranges[entity_id].triangle_offset};
            instance_materials.emplace_back(material);
        } else {  // inner node, visit children
            for (auto &&child : shape->children()) {
                queue.emplace(child.get(), transform_tree->add_inner_node(child->transform()), material);
            }
        }
    }
}

void Scene::_update_geometry(Pipeline &pipeline, float time) {
    if (!_is_static) {  // add update stage only if the scene is dynamic and time changed
        pipeline << [this, time](Dispatcher &dispatch) {
            dispatch(_instance_transforms.modify([this, time](float4x4 *matrices) { _transform_tree.update(matrices, time); }));
            dispatch(_acceleration->refit());
        };
    }
}

void Scene::_intersect_any(Pipeline &pipeline, const BufferView<Ray> &rays) {
    auto ray_count = rays.size();
    if (_any_hit_buffer.empty()) {
        _any_hit_buffer = _device->allocate_buffer<AnyInteraction>(ray_count);
    }
    pipeline << _acceleration->intersect_any(rays, _any_hit_buffer);
}

void Scene::_intersect_closest(Pipeline &pipeline, const BufferView<Ray> &ray_buffer) {
    
    auto ray_count = static_cast<uint>(ray_buffer.size());
    constexpr auto threadgroup_size = 256u;
    
    if (_closest_hit_buffer.empty()) {
        
        _closest_hit_buffer = _device->allocate_buffer<ClosestHit>(ray_count);
        _interaction_buffers.create(_device, ray_count);
        
        _evaluate_interactions_kernel = _device->compile_kernel("scene_evaluate_interactions", [&] {
            auto tid = thread_id();
            If (ray_count % threadgroup_size == 0u || tid < ray_count) {
                Var<ClosestHit> hit = _closest_hit_buffer[tid];
                If (hit.distance() <= 0.0f) {
                    _interaction_buffers.valid[tid] = false;
                } Else {
                    
                    _interaction_buffers.valid[tid] = true;
                    
                    Var instance_id = hit.instance_id();
                    _interaction_buffers.material[tid] = _instance_materials[instance_id];
                    
                    Var entity = _instance_entities[instance_id];
                    Var triangle_id = entity.triangle_offset() + hit.triangle_id();
                    Var i = _triangles[triangle_id].i() + entity.vertex_offset();
                    Var j = _triangles[triangle_id].j() + entity.vertex_offset();
                    Var k = _triangles[triangle_id].k() + entity.vertex_offset();
                    
                    Var bary_u = hit.bary_u();
                    Var bary_v = hit.bary_v();
                    Var bary_w = 1.0f - (bary_u + bary_v);
                    
                    Var p0 = _positions[i];
                    Var p1 = _positions[j];
                    Var p2 = _positions[k];
                    
                    Var m = _instance_transforms[instance_id];
                    Var nm = transpose(inverse(make_float3x3(m)));
                    
                    Var p = make_float3(m * make_float4(bary_u * p0 + bary_v * p1 + bary_w * p2, 1.0f));
                    _interaction_buffers.pi[tid] = p;
                    
                    // NOTE: DO NOT NORMALIZE!
                    _interaction_buffers.ray_origin_to_hit[tid] = p - make_float3(ray_buffer[tid].origin_x(), ray_buffer[tid].origin_y(), ray_buffer[tid].origin_z());
                    
                    Var ng = normalize(nm * cross(p1 - p0, p2 - p0));
                    Var ns = normalize(bary_u * _normals[i] + bary_u * _normals[j] + bary_w * _normals[k]);
                    _interaction_buffers.ns[tid] = ns;
                    _interaction_buffers.ng[tid] = select(dot(ns, ng) < 0.0f, -ng, ng);
                    _interaction_buffers.uv[tid] = bary_u * _tex_coords[i] + bary_v * _tex_coords[j] + bary_w * _tex_coords[k];
                };
            };
        });
    }
    
    pipeline << _acceleration->intersect_closest(ray_buffer, _closest_hit_buffer)
             << _evaluate_interactions_kernel.parallelize(ray_count, threadgroup_size);
}

void Scene::_process_geometry(const std::vector<std::shared_ptr<Shape>> &shapes, float initial_time) {
    
    // calculate memory usage...
    size_t vertex_count = 0u;
    size_t triangle_count = 0u;
    size_t entity_count = 0u;
    size_t instance_count = 0u;
    
    std::queue<Shape *> queue;
    for (auto &&shape: shapes) { queue.emplace(shape.get()); }
    
    std::unordered_set<Shape *> visited_entities;
    while (!queue.empty()) {
        auto shape = queue.front();
        queue.pop();
        if (shape->is_entity()) {  // leaf node, containing one instance of entity
            instance_count++;
            if (visited_entities.count(shape) == 0u) {  // unvisited entity
                entity_count++;
                vertex_count += shape->vertices().size();
                triangle_count += shape->triangles().size();
                visited_entities.emplace(shape);
            }
        } else {  // inner node, visit children
            for (auto &&child : shape->children()) {
                queue.emplace(child.get());
            }
        }
    }
    LUISA_ERROR_IF_NOT(entity_count == visited_entities.size(), "Something went wrong...");
    
    LUISA_INFO("Creating geometry with ",
               instance_count, " instances, ",
               entity_count, " entities, ",
               triangle_count, " unique triangles and ",
               vertex_count, " unique vertices.");
    
    // allocate buffers
    _positions = _device->allocate_buffer<float3>(vertex_count);
    _normals = _device->allocate_buffer<float3>(vertex_count);
    _tex_coords = _device->allocate_buffer<float2>(vertex_count);
    _triangles = _device->allocate_buffer<TriangleHandle>(triangle_count);
    _instance_entities = _device->allocate_buffer<EntityHandle>(instance_count);
    _instances = _device->allocate_buffer<uint>(instance_count);
    _instance_transforms = _device->allocate_buffer<float4x4>(instance_count);
    
    // encode shapes
    std::vector<EntityRange> meshes;
    std::vector<Material *> materials;
    meshes.reserve(entity_count);
    materials.reserve(entity_count);
    _device->launch([&](Dispatcher &dispatch) {
        dispatch(_positions.modify([&](float3 *positions) {
            dispatch(_normals.modify([&](float3 *normals) {
                dispatch(_tex_coords.modify([&](float2 *uvs) {
                    dispatch(_triangles.modify([&](TriangleHandle *indices) {
                        dispatch(_instance_entities.modify([&](EntityHandle *entities) {
                            dispatch(_instances.modify([&](uint *instances) {
                                _encode_geometry_buffers(shapes, positions, normals, uvs, indices, entities, meshes, materials, instances);
                            }));
                        }));
                    }));
                }));
            }));
        }));
    }, [&] {
        _positions.clear_cache();
        _normals.clear_cache();
        _tex_coords.clear_cache();
        _triangles.clear_cache();
        _instance_entities.clear_cache();
        _instances.clear_cache();
    });
    
    // apply initial transforms and build acceleration structure
    _is_static = _transform_tree.is_static();
    _device->launch(_instance_transforms.modify([&](float4x4 *matrices) {
        _transform_tree.update(matrices, initial_time);
    }), [&] { if (_is_static) { _instance_transforms.clear_cache(); }});
    _acceleration = _device->build_acceleration(_positions, _triangles, meshes, _instances, _instance_transforms, _is_static);
    
    // now it's time to process materials
    // not now...
    _instance_materials = _device->allocate_buffer<MaterialHandle>(entity_count);
}

}
