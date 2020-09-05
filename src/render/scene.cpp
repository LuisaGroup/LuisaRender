//
// Created by Mike Smith on 2020/9/4.
//

#include <compute/dsl.h>
#include "scene.h"

namespace luisa::render {

using namespace compute;
using namespace compute::dsl;

Scene::Scene(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes, std::shared_ptr<Background> background, float initial_time, size_t max_ray_count)
    : _device{device}, _background{std::move(background)}, _time{initial_time}, _closest_hit_buffer{device->allocate_buffer<ClosestHit>(max_ray_count)} {
    
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
    _instance_entities = device->allocate_buffer<Entity>(instance_count);
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
                    _instance_entities.modify([&](Entity *entities) {
                        _instances.modify([&](uint *instances) {
                            _process(shapes, positions, normals, uvs, indices, entities, meshes, materials, instances);
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
        _instance_entities.clear_cache();
        _instances.clear_cache();
    });
    
    // apply initial transforms and build acceleration structure
    _is_static = _transform_tree.is_static();
    device->launch(_instance_transforms.modify([&](float4x4 *matrices) {
        _transform_tree.update(matrices, initial_time);
    }), [&] { if (_is_static) { _instance_transforms.clear_cache(); }});
    _acceleration = device->build_acceleration(_positions, _triangles, meshes, _instances, _instance_transforms, _is_static);
    
    // now it's time to process materials
    
}

void Scene::_process(const std::vector<std::shared_ptr<Shape>> &shapes,
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
            entities[instance_id] = {entity_ranges[entity_id].x, entity_ranges[entity_id].y};
            instance_materials.emplace_back(material);
        } else {  // inner node, visit children
            for (auto &&child : shape->children()) {
                queue.emplace(child.get(), transform_tree->add_inner_node(child->transform()), material);
            }
        }
    }
}

void Scene::update_geometry(Pipeline &pipeline, const float &time) {
    if (!_is_static) {
        // add update stage if the scene is not static
        pipeline << [&time, this](Dispatcher &dispatch) {
            if (_time != time) {
                dispatch(_instance_transforms.modify([this](float4x4 *matrices) { _transform_tree.update(matrices, _time); }));
                dispatch(_acceleration->refit());
            }
        };
    }
}

void Scene::intersect_any(Pipeline &pipeline, const BufferView<Ray> &rays, const BufferView<uint> &ray_count, const BufferView<AnyInteraction> &its) {
    pipeline << _acceleration->intersect_any(rays, its, ray_count);
}

void Scene::intersect_closest(Pipeline &pipeline, const BufferView<Ray> &ray_buffer, const BufferView<uint> &ray_count_buffer, const InteractionBuffers &its_buffers) {
    
    auto kernel = _device->compile_kernel("retrieve_interactions", [&] {
        
        Arg<const ClosestHit *> hits{_closest_hit_buffer};
        Arg<const Ray *> rays{ray_buffer};
        Arg<const uint *> p_ray_count{ray_count_buffer};
        
        Arg<bool *> its_valid{its_buffers.valid};
        Arg<float3 *> its_pi{its_buffers.pi};
        Arg<float3 *> its_wo{its_buffers.ray_origin_to_hit};
        Arg<float3 *> its_ng{its_buffers.ng};
        Arg<float3 *> its_ns{its_buffers.ns};
        Arg<float2 *> its_uv{its_buffers.uv};
        Arg<MaterialHandle *> its_material{its_buffers.material};
        
        Arg<const float3 *> positions{_positions};
        Arg<const float3 *> normals{_normals};
        Arg<const float2 *> tex_coords{_tex_coords};
        Arg<const packed_uint3 *> triangles{_triangles};
        Arg<const uint *> instances{_instances};
        Arg<const float4x4 *> transforms{_instance_transforms};
        Arg<const Entity *> instance_entities{_instance_entities};
        Arg<const MaterialHandle *> materials{_instance_materials};
        
        auto tid = thread_id();
        Auto ray_count = *p_ray_count;
        If (tid < ray_count) {
            
            Auto hit = hits[tid];
            If (hit.$(distance) <= 0.0f) {
                its_valid[tid] = false;
                Return;
            };
            
            its_valid[tid] = true;
            
            Auto instance_id = hit.$(instance_id);
            its_material[tid] = materials[instance_id];
            
            Auto entity = instance_entities[instance_id];
            Auto triangle_id = entity.$(triangle_offset) + hit.$(triangle_id);
            Auto indices = triangles[triangle_id] + entity.$(vertex_offset);
            
            Auto bary_u = hit.$(bary_u);
            Auto bary_v = hit.$(bary_v);
            Auto bary_w = literal(1.0f) - (bary_u + bary_v);
            
            Auto p0 = positions[indices.x()];
            Auto p1 = positions[indices.y()];
            Auto p2 = positions[indices.z()];
            
            Auto m = transforms[instance_id];
            Auto nm = transpose(inverse(make_float3x3(m)));
    
            Auto p = m * (bary_u * p0 + bary_v * p1 + bary_w * p2);
            its_pi[tid] = p;
            its_wo[tid] = p - rays[tid].$(origin);
            
            Auto ng = normalize(nm * cross(p1 - p0, p2 - p0));
            Auto ns = normalize(bary_u * normals[indices.x()] + bary_u * normals[indices.y()] + bary_w * normals[indices.z()]);
            its_ns[tid] = ns;
            its_ng[tid] = select(dot(ns, ng) < 0.0f, -ng, ng);
            its_uv[tid] = bary_u * tex_coords[indices.x()] + bary_v * tex_coords[indices.y()] + bary_w * tex_coords[indices.z()];
        };
    });
    
    pipeline << _acceleration->intersect_closest(ray_buffer, _closest_hit_buffer, ray_count_buffer)
             << [kernel = std::move(kernel), size = ray_buffer.size()](Dispatcher &dispatch) { dispatch(kernel->parallelize(size)); };
}

}
