//
// Created by Mike Smith on 2020/9/11.
//

#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <render/shape.h>

namespace luisa::render::shape {

class WavefrontObj : public Shape {

private:
    void _load(const std::filesystem::path &path) {
        
        LUISA_INFO("Loading Wavefront OBJ file: ", path);
        
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        
        std::string warn;
        std::string err;
        
        auto input_file = path.string();
        bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, input_file.c_str(), nullptr, true);
        
        LUISA_WARNING_IF_NOT(warn.empty(), warn);
        LUISA_EXCEPTION_IF_NOT(err.empty(), err);
        LUISA_EXCEPTION_IF_NOT(success, "Error occurred while loading Wavefront OBJ file: ", path);
        
        _vertices.resize(attrib.vertices.size() / 3u);
        for (auto i = 0u; i < _vertices.size(); i++) {
            _vertices[i].position = make_float3(attrib.vertices[i * 3u], attrib.vertices[i * 3u + 1u], attrib.vertices[i * 3u + 2u]);
        }
        
        for (auto &&shape : shapes) {
            for (auto i = 0u; i < shape.mesh.indices.size(); i += 3u) {
                uint triangle[3];
                auto p0 = _vertices[shape.mesh.indices[i].vertex_index].position;
                auto p1 = _vertices[shape.mesh.indices[i + 1u].vertex_index].position;
                auto p2 = _vertices[shape.mesh.indices[i + 2u].vertex_index].position;
                auto ng = normalize(cross(p1 - p0, p2 - p0));
                for (auto j = 0u; j < 3u; j++) {
                    auto idx = shape.mesh.indices[i + j];
                    triangle[j] = idx.vertex_index;
                    if (idx.normal_index < 0) {
                        _vertices[idx.vertex_index].normal = ng;
                    } else {
                        _vertices[idx.vertex_index].normal = make_float3(
                            attrib.normals[3u * idx.normal_index],
                            attrib.normals[3u * idx.normal_index + 1u],
                            attrib.normals[3u * idx.normal_index + 2u]);
                    }
                    if (idx.texcoord_index >= 0) {
                        _vertices[idx.vertex_index].uv = make_float2(
                            attrib.texcoords[2u * idx.texcoord_index],
                            attrib.texcoords[2u * idx.texcoord_index + 1u]);
                    }
                }
                _triangles.emplace_back(TriangleHandle{triangle[0], triangle[1], triangle[2]});
            }
        }
    }

public:
    WavefrontObj(Device *d, const ParameterSet &params)
        : Shape{d, params} {
        
        auto path = std::filesystem::canonical(device()->context().input_path(params["path"].parse_string()));
        _load(path);
    }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::shape::WavefrontObj)
