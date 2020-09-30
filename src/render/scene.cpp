//
// Created by Mike Smith on 2020/9/4.
//

#include <compute/dsl_syntax.h>
#include <render/sampling.h>
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
    _entity_triangle_counts = _device->allocate_buffer<uint>(entity_count);
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
    
    _device->launch(_entity_triangle_counts.modify([&](uint *counts) {
        for (auto i = 0u; i < meshes.size(); i++) {
            counts[i] = meshes[i].triangle_count;
        }
    }));
    
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
    
    LUISA_EXCEPTION_IF(std::any_of(instance_materials.cbegin(), instance_materials.cend(), [](auto material) noexcept {
        return material == nullptr;
    }), "Found instance assigned with null material.");
    
    auto shader_count = 0u;
    auto data_block_count = 0u;
    std::vector<Material *> materials;
    std::unordered_map<Material *, MaterialHandle> material_to_handle;
    for (auto material : instance_materials) {
        if (material_to_handle.find(material) == material_to_handle.cend()) {
            auto lobe_count = material->lobe_count();
            materials.emplace_back(material);
            material_to_handle.emplace(material, MaterialHandle{shader_count, lobe_count});
            shader_count += lobe_count;
            data_block_count += material->required_data_block_count();
        }
    }
    
    auto emitter_count = 0u;
    std::vector<Material *> emissive_materials;
    std::unordered_map<Material *, MaterialHandle> emissive_material_to_handle;
    for (auto material : instance_materials) {
        if (material->is_emissive()) {
            emitter_count++;
            if (emissive_material_to_handle.find(material) == emissive_material_to_handle.cend()) {
                auto lobe_count = material->emissive_lobe_count();
                emissive_materials.emplace_back(material);
                emissive_material_to_handle.emplace(material, MaterialHandle{shader_count, lobe_count});
                shader_count += lobe_count;
                data_block_count += material->required_emission_data_block_count();
            }
        }
    }
    
    // encode materials...
    _instance_materials = _device->allocate_buffer<MaterialHandle>(instance_materials.size());
    _device->launch(_instance_materials.modify([&](MaterialHandle *handles) {
        for (auto i = 0u; i < instance_materials.size(); i++) {
            auto material = instance_materials[i];
            handles[i] = material_to_handle.at(material);
        }
    }));
    
    if (emitter_count == 0u) {
        LUISA_WARNING("No emitter found in scene.");
    } else {
        _emitter_materials = _device->allocate_buffer<MaterialHandle>(emitter_count);
        _emitter_to_instance_id = _device->allocate_buffer<uint>(emitter_count);
        _device->launch([&](Dispatcher &dispatch) {
            // clang-format off
            dispatch(_emitter_materials.modify([&](MaterialHandle *handles) {
            dispatch(_emitter_to_instance_id.modify([&](uint *emitter_to_instance) {
                auto offset = 0u;
                for (auto i = 0u; i < instance_materials.size(); i++) {
                    auto material = instance_materials[i];
                    if (material->is_emissive()) {
                        emitter_to_instance[offset] = i;
                        handles[offset] = emissive_material_to_handle.at(material);
                    }
                }
            })); }));
            // clang-format on
        });
    }
    
    _shader_weights = _device->allocate_buffer<float>(shader_count);
    _shader_cdf_tables = _device->allocate_buffer<float>(shader_count);
    _shader_types = _device->allocate_buffer<uint>(shader_count);
    _shader_block_offsets = _device->allocate_buffer<uint>(shader_count);
    _shader_blocks = _device->allocate_buffer<DataBlock>(data_block_count);
    _device->launch([&](Dispatcher &dispatch) {
        // clang-format off
        dispatch(_shader_types.modify([&](uint *shader_types) {
        dispatch(_shader_weights.modify([&](float *shader_weights) {
        dispatch(_shader_cdf_tables.modify([&](float *shader_cdf) {
        dispatch(_shader_block_offsets.modify([&](uint *shader_block_offsets) {
        dispatch(_shader_blocks.modify([&](DataBlock *blocks) {
            auto shader_offset = 0u;
            auto data_block_offset = 0u;
            for (auto material : materials) {
                auto sum = 0.0f;
                auto sum_weight = material->sum_weight();
                for (auto &&lobe : material->lobes()) {
                    shader_types[shader_offset] = lobe.shader->type_uid();
                    shader_weights[shader_offset] = lobe.weight;
                    shader_cdf[shader_offset] = (sum += lobe.weight) / sum_weight;
                    shader_block_offsets[shader_offset] = data_block_offset;
                    lobe.shader->encode_data(blocks + data_block_offset);
                    shader_offset++;
                    data_block_offset += lobe.shader->required_data_block_count();
                    _surface_evaluate_functions.try_emplace(lobe.shader->type_uid(), lobe.shader.get());
                }
            }
            for (auto material : emissive_materials) {
                auto sum = 0.0f;
                auto sum_weight = material->sum_emission_weight();
                for (auto &&lobe : material->lobes()) {
                    if (lobe.shader->is_emissive()) {
                        shader_types[shader_offset] = lobe.shader->type_uid();
                        shader_weights[shader_offset] = lobe.weight;
                        shader_cdf[shader_offset] = (sum += lobe.weight) / sum_weight;
                        shader_block_offsets[shader_offset] = data_block_offset;
                        lobe.shader->encode_data(blocks + data_block_offset);
                        shader_offset++;
                        data_block_offset += lobe.shader->required_data_block_count();
                        _surface_emission_functions.try_emplace(lobe.shader->type_uid(), lobe.shader.get());
                    }
                }
            }
        })); })); })); })); }));
        // clang-format on
    });
}

Scene::LightSample Scene::uniform_sample_light(const LightSelection &selection, Expr<float3> p, Expr<float2> u_shape) const {
    
    using namespace luisa::compute;
    using namespace luisa::compute::dsl;
    
    Var light_index = selection.index;
    Var light_instance_id = _emitter_to_instance_id[light_index];
    Var light_entity_id = _instance_to_entity_id[light_instance_id];
    Var light_entity = _entities[light_entity_id];
    Var light_triangle_count = _entity_triangle_counts[light_entity_id];
    Var triangle_index = sample_discrete(_triangle_cdf_tables, light_entity.triangle_offset, light_entity.triangle_offset + light_triangle_count, u_shape.x);
    u_shape.x = u_shape.x * light_triangle_count - triangle_index;
    Var bary = uniform_sample_triangle(u_shape);
    Var m = _instance_transforms[light_instance_id];
    Var triangle = _triangles[triangle_index];
    Expr uv0 = _tex_coords[triangle.i + light_entity.vertex_offset];
    Expr uv1 = _tex_coords[triangle.j + light_entity.vertex_offset];
    Expr uv2 = _tex_coords[triangle.k + light_entity.vertex_offset];
    Expr uv = bary.x * uv0 + bary.y * uv1 + (1.0f - bary.x - bary.y) * uv2;
    Var p0 = make_float3(m * make_float4(_positions[triangle.i + light_entity.vertex_offset], 1.0f));
    Var p1 = make_float3(m * make_float4(_positions[triangle.j + light_entity.vertex_offset], 1.0f));
    Var p2 = make_float3(m * make_float4(_positions[triangle.k + light_entity.vertex_offset], 1.0f));
    Var p_light = bary.x * p0 + bary.y * p1 + (1.0f - bary.x - bary.y) * p2;
    Var c = cross(p1 - p0, p2 - p0);
    Var area = 0.5f * length(c);
    Var ng = normalize(c);
    Var pdf_area = (_triangle_cdf_tables[triangle_index] - select(triangle_index == light_entity.triangle_offset, 0.0f, _triangle_cdf_tables[triangle_index - 1u])) / area;
    Var d = length(p_light - p);
    Var wi = normalize(p_light - p);
    Var cos_theta = abs(dot(wi, ng));
    Var pdf = d * d * pdf_area / cos_theta;
    Var L = make_float3(0.0f);
    Switch (selection.shader.type) {
        for (auto f : _surface_emission_functions) {
            Case (f.first) {
                auto[eval_L, eval_pdf] = f.second->emission(uv, ng, -wi, _shader_blocks[_shader_block_offsets[selection.shader.index]]);
                pdf *= eval_pdf;
                L = eval_L * selection.shader.weight / selection.shader.prob;
            };
        }
    };
    return LightSample{wi, L, pdf};
}

Scene::LightSelection Scene::uniform_select_light(Expr<float> u_light, Expr<float> u_shader) const {
    
    using namespace luisa::compute;
    using namespace luisa::compute::dsl;
    
    LUISA_ERROR_IF(light_count() == 0u, "Cannot sample lights in a scene without lights.");
    
    Var light_index = dsl::clamp(cast<uint>(u_light * light_count()), 0u, light_count() - 1u);
    Var light_material = _emitter_materials[light_index];
    Var shader_index = sample_discrete(_shader_cdf_tables, light_material.shader_offset, light_material.shader_offset + light_material.shader_count, u_shader);
    Var shader_pdf = _shader_cdf_tables[shader_index] - select(shader_index == light_material.shader_offset, 0.0f, _shader_cdf_tables[shader_index - 1u]);
    Var shader_weight = _shader_weights[shader_index];
    Var shader_type = _shader_types[shader_index];
    return LightSelection{light_index, 1.0f / light_count(), shader_type, shader_index, shader_pdf, shader_weight};
}

Interaction Scene::evaluate_interaction(Expr<Ray> ray, Expr<ClosestHit> hit, Expr<float> u_shader, uint flags) const {
    
    Interaction intr;
    
    If (hit.distance <= 0.0f) {
        if (flags & Interaction::COMPONENT_MISS) { intr.miss = true; }
    } Else {
        
        if (flags & Interaction::COMPONENT_MISS) { intr.miss = false; }
        
        Var instance_id = hit.instance_id;
        Var entity = _entities[_instance_to_entity_id[instance_id]];
        Var triangle_id = entity.triangle_offset + hit.triangle_id;
        Var i = _triangles[triangle_id].i + entity.vertex_offset;
        Var j = _triangles[triangle_id].j + entity.vertex_offset;
        Var k = _triangles[triangle_id].k + entity.vertex_offset;
        
        Var bary_u = hit.bary.x;
        Var bary_v = hit.bary.y;
        Var bary_w = 1.0f - (bary_u + bary_v);
        
        Var m = _instance_transforms[instance_id];
        Var nm = transpose(inverse(make_float3x3(m)));
        
        Var p0 = make_float3(m * make_float4(_positions[i], 1.0f));
        Var p1 = make_float3(m * make_float4(_positions[j], 1.0f));
        Var p2 = make_float3(m * make_float4(_positions[k], 1.0f));
        
        if (flags & Interaction::COMPONENT_PI) { intr.pi = bary_u * p0 + bary_v * p1 + bary_w * p2; }
        if (flags & Interaction::COMPONENT_DISTANCE) { intr.distance = hit.distance; }
        
        Var wo = make_float3(-ray.direction_x, -ray.direction_y, -ray.direction_z);
        if (flags & Interaction::COMPONENT_WO) { intr.wo = wo; }
        
        Var c = cross(p1 - p0, p2 - p0);
        Var ng = normalize(c);
        // if (flags & Interaction::COMPONENT_NS) { interaction.ns = normalize(nm * (bary_u * _normals[i] + bary_v * _normals[j] + bary_w * _normals[k])); }
        // FIXME: Error in Ns, temporally using Ng instead...
        if (flags & Interaction::COMPONENT_NS) { intr.ns = ng; }
        if (flags & Interaction::COMPONENT_NG) { intr.ng = ng; }
        if (flags & Interaction::COMPONENT_UV) { intr.uv = bary_u * _tex_coords[i] + bary_v * _tex_coords[j] + bary_w * _tex_coords[k]; }
        if (flags & Interaction::COMPONENT_PDF) {
            Var area = 0.5f * length(c);
            Var cdf_low = select(hit.triangle_id == 0u, 0.0f, _triangle_cdf_tables[triangle_id - 1u]);
            Var cdf_high = _triangle_cdf_tables[triangle_id];
            intr.pdf = (cdf_high - cdf_low) * hit.distance * hit.distance / (area * abs(dot(wo, ng)));
        }
        
        if (flags & Interaction::COMPONENT_SHADER) {
            Var material = _instance_materials[hit.instance_id];
            intr.shader.index = sample_discrete(_shader_cdf_tables, material.shader_offset, material.shader_offset + material.shader_count, u_shader);
            intr.shader.type = _shader_types[intr.shader.index];
            intr.shader.prob = _shader_cdf_tables[intr.shader.index] - select(intr.shader.index == material.shader_offset, 0.0f, _shader_cdf_tables[intr.shader.index - 1u]);
            intr.shader.weight = _shader_weights[intr.shader.index];
        }
    };
    
    return intr;
}

Scene::Scattering Scene::evaluate_scattering(const Interaction &intr, Expr<float3> wi, Expr<float2> u, uint flags) {
    
    Scattering scattering;
    
    If (!intr.miss) {
        Switch (intr.shader.type) {
            for (auto f : _surface_evaluate_functions) {
                Case (f.first) {
                    scattering = f.second->evaluate(intr.uv, intr.ns, intr.wo, wi, u, _shader_blocks[_shader_block_offsets[intr.shader.index]], flags);
                };
            }
        };
    };
    return scattering;
}

}
