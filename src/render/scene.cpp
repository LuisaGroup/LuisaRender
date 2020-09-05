//
// Created by Mike Smith on 2020/9/4.
//

#include "scene.h"

namespace luisa::render {

Scene::Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, float initial_time) {
    
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
                triangle_count += shape->indices().size();
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
    _positions = device->allocate_buffer<float3>(vertex_count);
    _normals = device->allocate_buffer<float3>(vertex_count);
    _tex_coords = device->allocate_buffer<float2>(vertex_count);
    _triangles = device->allocate_buffer<packed_uint3>(triangle_count);
    _entities = device->allocate_buffer<Entity>(entity_count);
    _instances = device->allocate_buffer<uint>(instance_count);
    _instance_transforms = device->allocate_buffer<float4x4>(instance_count);
    
    // encode shapes
    std::vector<packed_uint3> meshes;
    std::vector<Material *> materials;
    meshes.reserve(entity_count);
    materials.reserve(entity_count);
    device->launch(_positions.modify([&](float3 *positions) {
        _normals.modify([&](float3 *normals) {
            _tex_coords.modify([&](float2 *uvs) {
                _triangles.modify([&](packed_uint3 *indices) {
                    _entities.modify([&](Entity *entities) {
                        _instances.modify([&](uint *instances) {
                            _encode(shapes, positions, normals, uvs, indices, entities, meshes, materials, instances);
                        });
                    });
                });
            });
        });
    }), [&] {
        _positions.clear_cache();
        _normals.clear_cache();
        _tex_coords.clear_cache();
        _triangles.clear_cache();
        _entities.clear_cache();
        _instances.clear_cache();
    });
    
    // apply initial transforms and build acceleration structure
    _is_static = _transform_tree.is_static();
    device->launch(_instance_transforms.modify([&](float4x4 *matrices) { _transform_tree.update(matrices, initial_time); }));
    _acceleration = device->build_acceleration(_positions, _triangles, meshes, _instances, _instance_transforms, _is_static);
    
    // now it's time to process materials
    
}

void Scene::_encode(const std::vector<std::shared_ptr<Shape>> &shapes,
                    float3 *positions,
                    float3 *normals,
                    float2 *uvs,
                    packed_uint3 *triangles,
                    Entity *entities,
                    std::vector<packed_uint3> &entity_ranges,
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
                auto indices = shape->indices();
                std::copy(indices.cbegin(), indices.cend(), triangles + triangle_offset);
                triangle_count += shape->indices().size();
                
                shape->clear();
                
                auto entity_id = static_cast<uint>(entity_ranges.size());
                entity_ranges.emplace_back(vertex_offset, triangle_offset, indices.size());
                entities[entity_id] = {static_cast<uint>(vertex_offset), static_cast<uint>(triangle_offset)};
                
                iter = entity_to_id.emplace(shape, entity_id).first;
            }
            auto entity_id = iter->second;
            instances[instance_id] = entity_id;
            instance_materials.emplace_back(material);
        } else {  // inner node, visit children
            for (auto &&child : shape->children()) {
                queue.emplace(child.get(), transform_tree->add_inner_node(child->transform()), material);
            }
        }
    }
    
}

}
