//
// Created by Mike Smith on 2020/9/4.
//

#include <compute/dsl_syntax.h>
#include "scene.h"

namespace luisa::render {

using namespace compute;
using namespace compute::dsl;

Scene::Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, std::shared_ptr<Background> background, float initial_time)
    : _device{device},
      _background{std::move(background)} {
    
    std::vector<Material *> instance_materials;
    _process_geometry(shapes, initial_time, instance_materials);
    _process_materials(instance_materials);
}

void Scene::_encode_geometry_buffers(const std::vector<std::shared_ptr<Shape>> &shapes,
                                     float3 *positions,
                                     float3 *normals,
                                     float2 *uvs,
                                     TriangleHandle *triangles,
                                     float *triangle_cdf_tables,
                                     EntityHandle *entities,
                                     std::vector<MeshHandle> &meshes,
                                     std::vector<Material *> &instance_materials,
                                     uint *instances) {
    
    size_t vertex_count = 0u;
    size_t triangle_count = 0u;
    size_t instance_count = 0u;
    
    std::queue<std::tuple<Shape *, TransformTree *, Material *>> queue;
    for (auto &&shape: shapes) { queue.emplace(shape.get(), _transform_tree.add_inner_node(shape->transform()), nullptr); }
    
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
                
                // compute cdf table
                auto sum_area = 0.0f;
                for (auto i = 0u; i < shape->triangles().size(); i++) {
                    auto triangle = shape->triangles()[i];
                    auto p0 = shape->vertices()[triangle.i].position;
                    auto p1 = shape->vertices()[triangle.j].position;
                    auto p2 = shape->vertices()[triangle.k].position;
                    auto area = 0.5f * length(cross(p1 - p0, p2 - p0));
                    triangle_cdf_tables[triangle_offset + i] = (sum_area += area);
                }
                auto inv_sum_area = 1.0f / sum_area;
                for (auto i = 0u; i < shape->triangles().size(); i++) {
                    triangle_cdf_tables[triangle_offset + i] *= inv_sum_area;
                }
                triangle_count += shape->triangles().size();
                
                auto entity_id = static_cast<uint>(meshes.size());
                meshes.emplace_back(MeshHandle{static_cast<uint>(vertex_offset), static_cast<uint>(triangle_offset),
                                               static_cast<uint>(shape->vertices().size()), static_cast<uint>(indices.size())});
                entities[entity_id] = {static_cast<uint>(vertex_offset), static_cast<uint>(triangle_offset)};
                
                shape->clear();
                
                iter = entity_to_id.emplace(shape, entity_id).first;
            }
            auto entity_id = iter->second;
            instances[instance_id] = entity_id;
            entities[instance_id] = {meshes[entity_id].vertex_offset, meshes[entity_id].triangle_offset};
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
            dispatch(_instance_transforms.modify([this, time](float4x4 *matrices) {
                _transform_tree.update(matrices, time);
            }));
            dispatch(_acceleration->refit());
        };
    }
}

void Scene::_intersect_any(Pipeline &pipeline, const BufferView<Ray> &rays, BufferView<AnyHit> &hits) {
    pipeline << _acceleration->intersect_any(rays, hits);
}

void Scene::_intersect_closest(Pipeline &pipeline, const BufferView<Ray> &ray_buffer, InteractionBuffers &buffers) {
    
    auto ray_count = static_cast<uint>(ray_buffer.size());
    constexpr auto threadgroup_size = 1024u;
    
    if (_closest_hit_buffer.size() < ray_buffer.size()) {
        _closest_hit_buffer = _device->allocate_buffer<ClosestHit>(ray_count);
    }
    
    pipeline << _acceleration->intersect_closest(ray_buffer, _closest_hit_buffer)
             << _device->compile_kernel("scene_evaluate_interactions", [&] {
                 auto tid = thread_id();
                 If (ray_count % threadgroup_size == 0u || tid < ray_count) {
                     Var<ClosestHit> hit = _closest_hit_buffer[tid];
                     If (hit.distance <= 0.0f) {
                         if (buffers.has_miss()) { buffers.miss()[tid] = true; }
                     } Else {
                
                         if (buffers.has_miss()) { buffers.miss()[tid] = false; }
                
                         Var instance_id = hit.instance_id;
                         if (buffers.has_material()) { buffers.material()[tid] = _instance_materials[instance_id]; }
                
                         Var entity = _entities[_instance_to_entity_id[instance_id]];
                         Var triangle_id = entity.triangle_offset + hit.triangle_id;
                         Var i = _triangles[triangle_id].i + entity.vertex_offset;
                         Var j = _triangles[triangle_id].j + entity.vertex_offset;
                         Var k = _triangles[triangle_id].k + entity.vertex_offset;
                
                         Var bary_u = hit.bary.x;
                         Var bary_v = hit.bary.y;
                         Var bary_w = 1.0f - (bary_u + bary_v);
                
                         Var p0 = _positions[i];
                         Var p1 = _positions[j];
                         Var p2 = _positions[k];
                
                         Var m = _instance_transforms[instance_id];
                         Var nm = transpose(inverse(make_float3x3(m)));
                         
                         if (buffers.has_pi()) { buffers.pi()[tid] = make_float3(m * make_float4(bary_u * p0 + bary_v * p1 + bary_w * p2, 1.0f)); }
                         if (buffers.has_distance()) { buffers.distance()[tid] = hit.distance; }
                         
                         Var wo = make_float3(-ray_buffer[tid].direction_x, -ray_buffer[tid].direction_y, -ray_buffer[tid].direction_z);
                         if (buffers.has_wo()) { buffers.wo()[tid] = wo; }
                
                         Var c = cross(p1 - p0, p2 - p0);
                         Var ng = normalize(c);
                         if (buffers.has_ns()) { buffers.ns()[tid] = normalize(nm * (bary_u * _normals[i] + bary_v * _normals[j] + bary_w * _normals[k])); }
                         if (buffers.has_ng()) { buffers.ng()[tid] = ng; }
                         if (buffers.has_uv()) { buffers.uv()[tid] = bary_u * _tex_coords[i] + bary_v * _tex_coords[j] + bary_w * _tex_coords[k]; }
                         if (buffers.has_pdf()) {
                             Var area = 0.5f * length(c);
                             Var cdf_low = select(hit.triangle_id == 0u, 0.0f, _triangle_cdf_tables[triangle_id - 1u]);
                             Var cdf_high = _triangle_cdf_tables[triangle_id];
                             Var pdf = (cdf_high - cdf_low) * hit.distance * hit.distance / (area * abs(dot(wo, ng)));
                             buffers.pdf()[tid] = pdf;
                         }
                     };
                 };
             }).parallelize(ray_count, threadgroup_size);
}

void Scene::_process_geometry(const std::vector<std::shared_ptr<Shape>> &shapes, float initial_time, std::vector<Material *> &instance_materials) {
    
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
    _triangle_cdf_tables = _device->allocate_buffer<float>(triangle_count);
    _entities = _device->allocate_buffer<EntityHandle>(entity_count);
    _instance_to_entity_id = _device->allocate_buffer<uint>(instance_count);
    _instance_transforms = _device->allocate_buffer<float4x4>(instance_count);
    
    // encode shapes
    std::vector<MeshHandle> meshes;
    meshes.reserve(entity_count);
    instance_materials.reserve(instance_count);
    _device->launch([&](Dispatcher &dispatch) {
        // clang-format off
        dispatch(_positions.modify([&](float3 *positions) {
        dispatch(_normals.modify([&](float3 *normals) {
        dispatch(_tex_coords.modify([&](float2 *uvs) {
        dispatch(_triangles.modify([&](TriangleHandle *indices) {
        dispatch(_triangle_cdf_tables.modify([&](float *cdf_tables) {
        dispatch(_entities.modify([&](EntityHandle *entities) {
        dispatch(_instance_to_entity_id.modify([&](uint *instance_to_entity_id) {
            _encode_geometry_buffers(
                shapes, positions, normals, uvs, indices, cdf_tables,
                entities, meshes, instance_materials, instance_to_entity_id);
        })); })); })); })); })); })); }));
        // clang-format on
    });
    
    // apply initial transforms and build acceleration structure
    _is_static = _transform_tree.is_static();
    _device->launch(_instance_transforms.modify([&](float4x4 *matrices) {
        _transform_tree.update(matrices, initial_time);
    }));
    
    _device->synchronize();
    LUISA_INFO("Done encoding geometry buffers.");
    
    _positions.clear_cache();
    _normals.clear_cache();
    _tex_coords.clear_cache();
    _triangles.clear_cache();
    _triangle_cdf_tables.clear_cache();
    _entities.clear_cache();
    _instance_to_entity_id.clear_cache();
    if (_is_static) { _instance_transforms.clear_cache(); }
    
    LUISA_INFO("Creating acceleration structure.");
    _acceleration = _device->build_acceleration(_positions, _triangles, meshes, _instance_to_entity_id, _instance_transforms, _is_static);
}

void Scene::_process_materials(const std::vector<Material *> &instance_materials) {
    
    // now it's time to process materials
    // not now...
    _instance_materials = _device->allocate_buffer<MaterialHandle>(instance_materials.size());
    
}

}
