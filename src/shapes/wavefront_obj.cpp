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
        bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, input_file.c_str());
        
        LUISA_WARNING_IF_NOT(warn.empty(), warn);
        LUISA_EXCEPTION_IF_NOT(err.empty(), err);
        LUISA_EXCEPTION_IF_NOT(success, "Error occurred while loading Wavefront OBJ file: ", path);
        
        _vertices.resize(attrib.vertices.size() / 3u);
        for (auto s = 0u; s < shapes.size(); s++) {
            // Loop over faces(polygon)
            auto index_offset = 0u;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                
                auto fv = shapes[s].mesh.num_face_vertices[f];
                LUISA_EXCEPTION_IF_NOT(fv == 3, "Only triangle primitives are supported.");
                
                // Loop over vertices in the face.
                for (size_t v = 0; v < fv; v++) {
                    // access to vertex
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                    tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                    tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                    tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                    tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
                    tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                }
                index_offset += fv;
                
                // per-face material
                shapes[s].mesh.material_ids[f];
            }
        }
    }

public:
    WavefrontObj(Device *d, const ParameterSet &params)
        : Shape{d, params} {
        
        auto path = std::filesystem::canonical(device()->context().working_path(params["path"].parse_string()));
        _load(path);
    }
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::shape::WavefrontObj)
