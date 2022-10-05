//
// Created by Mike Smith on 2022/9/27.
//

#include <iostream>
#include <filesystem>
#include <fstream>

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <core/stl.h>
#include <core/logging.h>
#include <core/json.h>
#include <core/mathematics.h>

using luisa::uint;

int main(int argc, char *argv[]) {

    // TODO: Parse command line arguments.

    using namespace std::string_view_literals;
    if (argc < 2 || argv[1] == "-h"sv || argv[1] == "--help"sv) {
        std::cout << "Scene exporter for LuisaRender\n"
                  << "Usage: " << argv[0] << " <file>"
                  << std::endl;
        return 0;
    }

    // load
    auto path = std::filesystem::canonical(argv[1]);
    auto folder = path.parent_path();
    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    auto scene = importer.ReadFile(path.string(),
                                   aiProcess_FindInstances | aiProcess_SortByPType |
                                       aiProcess_RemoveComponent | aiProcess_Debone |
                                       aiProcess_GenBoundingBoxes | aiProcess_TransformUVCoords |
                                       aiProcess_RemoveRedundantMaterials | aiProcess_FixInfacingNormals |
                                       aiProcess_FindInvalidData | aiProcess_GenUVCoords);
    LUISA_ASSERT(scene != nullptr, "Failed to load scene: {}.", importer.GetErrorString());
    LUISA_INFO("Loaded scene '{}' with {} camera(s), {} mesh(es), and {} material(s).",
               path.filename().string(), scene->mNumCameras,
               scene->mNumMeshes, scene->mNumMaterials);

    for (auto i = 0u; i < scene->mNumLights; i++) {
        auto light = scene->mLights[i];
        switch (light->mType) {
            case aiLightSource_DIRECTIONAL: LUISA_WARNING("Ignoring punctual light #{}: DIRECTIONAL('{}')", i, light->mName.C_Str()); break;
            case aiLightSource_POINT: LUISA_WARNING("Ignoring punctual light #{}: POINT('{}')", i, light->mName.C_Str()); break;
            case aiLightSource_SPOT: LUISA_WARNING("Ignoring punctual light #{}: SPOT('{}')", i, light->mName.C_Str()); break;
            case aiLightSource_AMBIENT: LUISA_WARNING("Ignoring punctual light #{}: AMBIENT('{}')", i, light->mName.C_Str()); break;
            case aiLightSource_AREA: LUISA_WARNING("Ignoring punctual light #{}: AREA('{}')", i, light->mName.C_Str()); break;
            default: LUISA_WARNING("Ignoring punctual light #{}: Undefined('{}')", i, light->mName.C_Str()); break;
        }
    };

    // check flattened
    //    LUISA_ASSERT(scene->mRootNode->mTransformation.IsIdentity(),
    //                 "Scene must be flattened.");
    //    for (auto i = 0u; i < scene->mRootNode->mNumChildren; i++) {
    //        auto node = scene->mRootNode->mChildren[i];
    //        LUISA_ASSERT(node->mChildren == nullptr &&
    //                         node->mTransformation.IsIdentity(),
    //                     "Scene must be flattened.");
    //    }

    // convert
    luisa::json scene_materials;
    luisa::json scene_geometry;

    // textures
    luisa::unordered_map<uint, luisa::string> embedded_textures;
    std::filesystem::create_directories(folder / "lr_exported_textures");
    for (auto i = 0u; i < scene->mNumTextures; i++) {
        if (auto texture = scene->mTextures[i]; texture->mHeight == 0) {
            auto width = 0;
            auto height = 0;
            auto channels = 0;
            auto image_data = stbi_load_from_memory(
                reinterpret_cast<uint8_t *>(texture->pcData),
                static_cast<int>(texture->mWidth), &width, &height, &channels, 0);
            auto texture_name = texture->mFilename.length > 0u ?
                                    luisa::format("texture_{:05}_{}.png", i,
                                                  std::filesystem::path{texture->mFilename.C_Str()}.stem().string()) :
                                    luisa::format("texture_{:05}.png", i);
            LUISA_ASSERT(image_data != nullptr && width > 0 && height > 0 && channels > 0,
                         "Failed to load embedded texture '{}'.", texture_name);
            auto texture_path = folder / "lr_exported_textures" / texture_name;
            stbi_write_png(texture_path.string().c_str(),
                           width, height, channels, image_data, 0);
            stbi_image_free(image_data);
            embedded_textures.emplace(i, relative(texture_path, folder).string());
        } else {
            LUISA_WARNING_WITH_LOCATION(
                "Unsupported texture format for '{}'.",
                texture->mFilename.C_Str());
        }
    }

    // materials
    luisa::unordered_map<uint64_t, luisa::string> loaded_textures;
    luisa::unordered_map<uint, luisa::string> material_names;
    luisa::unordered_map<uint, luisa::string> light_names;
    for (auto i = 0u; i < scene->mNumMaterials; i++) {
        auto m = scene->mMaterials[i];
        auto mat_name = luisa::format("Surface:{:05}:{}", i, m->GetName().C_Str());
        LUISA_INFO("Converting material '{}'...", mat_name);
        auto parse_texture = [&](auto key, auto type, auto index,
                                 luisa::string_view semantic) noexcept
            -> luisa::optional<luisa::string> {
            if (aiString s; m->Get(key, type, index, s) == AI_SUCCESS) {
                luisa::string tex{s.C_Str(), s.length};
                for (auto &c : tex) {
                    if (c == '\\') { c = '/'; }
                }
                // embedded texture
                if (tex.starts_with('*')) {
                    auto end = tex.data() + tex.length() - 1u;
                    auto id = std::strtoul(tex.data() + 1u, &end, 10);
                    tex = embedded_textures.at(id);
                }
                // external texture
                auto name = luisa::format("Texture:{:05}:{}", loaded_textures.size(), tex);
                try {
                    auto rel_path = std::filesystem::relative(
                        std::filesystem::canonical(folder / tex), folder);
                    auto hash = luisa::hash64(rel_path.string(), luisa::hash64(semantic, luisa::hash64("__external__")));
                    if (auto it = loaded_textures.find(hash); it != loaded_textures.end()) {
                        return luisa::format("@{}", it->second);
                    }
                    scene_materials[name] = {
                        {"type", "Texture"},
                        {"impl", "Image"},
                        {"prop", {{"file", rel_path.string()}, {"semantic", semantic}}}};
                    loaded_textures[hash] = name;
                    return luisa::format("@{}", name);
                } catch (const std::exception &e) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Failed to find texture '{}' for material '{}': {}.",
                        tex, name, e.what());
                }
            }
            return luisa::nullopt;
        };
        auto parse_constant = [&](auto key, auto type, auto index,
                                  luisa::string_view semantic,
                                  bool force_value = false) noexcept
            -> luisa::optional<luisa::string> {
            if (aiColor3D c; (m->Get(key, type, index, c) == AI_SUCCESS &&
                              c != aiColor3D{0.f}) ||
                             force_value) {
                auto hash = luisa::hash64(luisa::make_float4(c.r, c.g, c.b, 1.f),
                                          luisa::hash64(semantic, luisa::hash64("__constant__")));
                if (auto it = loaded_textures.find(hash); it != loaded_textures.end()) {
                    return luisa::format("@{}", it->second);
                }
                auto name = luisa::format("Texture:{:05}", loaded_textures.size());
                scene_materials[name] = {
                    {"type", "Texture"},
                    {"impl", "Constant"},
                    {"prop", {{"v", {c.r, c.g, c.b}}, {"semantic", semantic}}}};
                loaded_textures[hash] = name;
                return luisa::format("@{}", name);
            }
            return luisa::nullopt;
        };
        auto color_map = parse_texture(AI_MATKEY_TEXTURE_DIFFUSE(0), "albedo")
                             .value_or(parse_constant(AI_MATKEY_COLOR_DIFFUSE, "albedo", true).value());
        auto specular_map = parse_texture(AI_MATKEY_TEXTURE_SPECULAR(0), "albedo");
        auto shininess_map = parse_texture(AI_MATKEY_TEXTURE_SHININESS(0), "generic");
        if (specular_map) { LUISA_INFO("Specular: {}", *specular_map); }
        if (shininess_map) { LUISA_INFO("Shininess: {}", *shininess_map); }

        auto rough = 1.f;
        m->Get(AI_MATKEY_ROUGHNESS_FACTOR, rough);
        scene_materials[mat_name] = {
            {"type", "Surface"},
            {"impl", "Substrate"},
            {"prop",
             {{"Kd", color_map},
              {"Ks",
               {{"impl", "Constant"},
                {"prop",
                 {{"v", {0.04f, 0.04f, 0.04f}},
                  {"semantic", "albedo"}}}}},
              {"roughness",
               {{"impl", "Constant"},
                {"prop", {{"v", {rough}}}}}}}}};
        material_names[i] = mat_name;
        if (auto normal_map = parse_texture(AI_MATKEY_TEXTURE_NORMALS(0), "generic")) {
            scene_materials[normal_map->substr(1)]["prop"]["encoding"] = "linear";
            scene_materials[mat_name]["prop"]["normal_map"] = luisa::json{*normal_map};
        }
        // light
        if (auto emission = parse_texture(AI_MATKEY_TEXTURE_EMISSIVE(0), "illuminant")
                                .value_or(parse_constant(AI_MATKEY_COLOR_EMISSIVE, "illuminant").value_or(""));
            !emission.empty()) {
            auto intensity = 1.f;
            m->Get(AI_MATKEY_COLOR_EMISSIVE, intensity);
            auto light_name = luisa::format("Light:{:05}:{}", i, m->GetName().C_Str());
            scene_materials[light_name] = {
                {"type", "Light"},
                {"impl", "Diffuse"},
                {"prop", {{"emission", emission}, {"scale", intensity}}}};
            LUISA_INFO("Found light '{}'.", light_name);
            light_names[i] = light_name;
        }
    }

    // meshes
    std::vector<luisa::string> meshes;
    std::filesystem::create_directories(folder / "lr_exported_meshes");
    for (auto i = 0u; i < scene->mNumMeshes; i++) {
        auto m = scene->mMeshes[i];
        auto file_name = luisa::format("mesh_{:05}.obj", i);
        LUISA_INFO("Converting mesh '{}'...", file_name);
        auto file_path = folder / "lr_exported_meshes" / file_name;
        std::ofstream file{file_path};
        for (auto iv = 0u; iv < m->mNumVertices; iv++) {
            auto v = m->mVertices[iv];
            file << "v " << v.x << ' ' << v.y << ' ' << v.z << '\n';
        }
        if (m->HasNormals()) {
            for (auto iv = 0u; iv < m->mNumVertices; iv++) {
                auto v = m->mNormals[iv];
                file << "vn " << v.x << ' ' << v.y << ' ' << v.z << '\n';
            }
        }
        if (m->HasTextureCoords(0)) {
            for (auto iv = 0u; iv < m->mNumVertices; iv++) {
                auto v = m->mTextureCoords[0][iv];
                file << "vt " << v.x << ' ' << v.y << '\n';
            }
        }
        for (auto f = 0u; f < m->mNumFaces; f++) {
            file << "f";
            for (auto j = 0u; j < 3u; j++) {
                auto idx = m->mFaces[f].mIndices[j] + 1u;
                file << " " << idx;
                if (m->HasTextureCoords(0u) || m->HasNormals()) {
                    if (m->HasTextureCoords(0)) {
                        file << '/' << idx;
                    } else {
                        file << "/";
                    }
                    if (m->HasNormals()) {
                        file << '/' << idx;
                    }
                }
            }
            file << '\n';
        }
        auto mat_id = m->mMaterialIndex;
        auto mat_name = luisa::format("@{}", material_names.at(mat_id));
        auto mesh_name = luisa::format("Mesh:{:05}:{}", i, m->mName.C_Str());
        scene_geometry[mesh_name] = {
            {"type", "Shape"},
            {"impl", "Mesh"},
            {"prop",
             {{"file", relative(file_path, folder).string()},
              {"flip_uv", true},
              {"surface", mat_name}}}};
        if (auto iter = light_names.find(mat_id); iter != light_names.end()) {
            scene_geometry[mesh_name]["prop"]["light"] = luisa::format("@{}", iter->second);
        }
        meshes.emplace_back(std::move(mesh_name));
    }

    // process scene graph
    aiAABB aabb{aiVector3D{1e30f}, aiVector3D{-1e30f}};
    luisa::queue<const aiNode *> node_queue;
    node_queue.emplace(scene->mRootNode);
    std::vector<luisa::string> groups;
    while (!node_queue.empty()) {
        auto node = node_queue.front();
        node_queue.pop();
        for (auto i = 0u; i < node->mNumChildren; i++) {
            node_queue.emplace(node->mChildren[i]);
        }
        if (node->mNumMeshes != 0u) {
            LUISA_INFO("Processing node '{}'...", node->mName.C_Str());
            std::vector<luisa::string> children;
            auto transform = node->mTransformation;
            for (auto n = node->mParent; n != nullptr; n = n->mParent) {
                transform = n->mTransformation * transform;
            }
            for (auto i = 0u; i < node->mNumMeshes; i++) {
                auto mesh_id = node->mMeshes[i];
                auto mesh_name = meshes[mesh_id];
                children.emplace_back(luisa::format("@{}", mesh_name));
                auto mesh_aabb = scene->mMeshes[mesh_id]->mAABB;
                mesh_aabb.mMin = transform * mesh_aabb.mMin;
                mesh_aabb.mMax = transform * mesh_aabb.mMax;
                aabb.mMin.x = std::min(aabb.mMin.x, mesh_aabb.mMin.x);
                aabb.mMin.y = std::min(aabb.mMin.y, mesh_aabb.mMin.y);
                aabb.mMin.z = std::min(aabb.mMin.z, mesh_aabb.mMin.z);
                aabb.mMax.x = std::max(aabb.mMax.x, mesh_aabb.mMax.x);
                aabb.mMax.y = std::max(aabb.mMax.y, mesh_aabb.mMax.y);
                aabb.mMax.z = std::max(aabb.mMax.z, mesh_aabb.mMax.z);
            }
            if (children.size() == 1u && transform.IsIdentity()) {
                groups.emplace_back(children[0]);
            } else {
                auto group_name = luisa::format("Group:{:05}:{}",
                                                groups.size(), node->mName.C_Str());
                scene_geometry[group_name] = {{"type", "Shape"},
                                              {"impl", "Group"},
                                              {"prop", {{"shapes", children}}}};
                if (!transform.IsIdentity()) {
                    scene_geometry[group_name]["prop"]["transform"] = {
                        {"impl", "Matrix"},
                        {"prop",
                         {{"m",
                           {transform[0][0], transform[0][1], transform[0][2], transform[0][3],
                            transform[1][0], transform[1][1], transform[1][2], transform[1][3],
                            transform[2][0], transform[2][1], transform[2][2], transform[2][3],
                            transform[3][0], transform[3][1], transform[3][2], transform[3][3]}}}}};
                }
                groups.emplace_back(luisa::format("@{}", group_name));
            }
        }
    }
    scene_geometry["lr_exported_geometry"] = {
        {"type", "Shape"},
        {"impl", "Group"},
        {"prop", {{"shapes", std::move(groups)}}}};

    LUISA_INFO("Scene AABB: ({}, {}, {}) -> ({}, {}, {}).",
               aabb.mMin.x, aabb.mMin.y, aabb.mMin.z,
               aabb.mMax.x, aabb.mMax.y, aabb.mMax.z);

    // camera
    auto scene_configs = luisa::json::object();
    std::vector<luisa::string> cameras;
    for (auto i = 0u; i < scene->mNumCameras; i++) {
        auto camera = scene->mCameras[i];
        auto name = luisa::format("Camera:{}:{}", i, camera->mName.C_Str());
        LUISA_INFO("Processing camera '{}'...", name);
        auto front = (camera->mLookAt - camera->mPosition).NormalizeSafe();
        scene_configs[name] = {
            {"type", "Camera"},
            {"impl", "Pinhole"},
            {"prop",
             {{"position", {camera->mPosition.x, camera->mPosition.y, camera->mPosition.z}},
              {"front", {front.x, front.y, front.z}},
              {"up", {camera->mUp.x, camera->mUp.y, camera->mUp.z}},
              {"fov", luisa::degrees(camera->mHorizontalFOV)},
              {"spp", 128u},
              {"file", "render.exr"},
              {"film",
               {{"impl", "Color"},
                {"prop",
                 {{"resolution", {1920, 1080}},
                  {"filter", {{"impl", "Gaussian"}}}}}}}}}};
        cameras.emplace_back(luisa::format("@{}", name));
    }
    // create default camera if non-existent
    if (cameras.empty()) {
        auto name = "Camera:0:Default";
        LUISA_INFO("Creating default camera '{}'...", name);
        auto center = (aabb.mMin + aabb.mMax) * .5f;
        auto size = (aabb.mMax - aabb.mMin) * .5f;
        auto position = center + aiVector3D{0.f, 0.f, size.z};
        scene_configs[name] = {
            {"type", "Camera"},
            {"impl", "Pinhole"},
            {"prop",
             {{"position", {position.x, position.y, position.z}},
              {"front", {0, 0, -1}},
              {"up", {0, 1, 0}},
              {"fov", 50},
              {"spp", 128u},
              {"file", "render.exr"},
              {"film",
               {{"impl", "Color"},
                {"prop",
                 {{"resolution", {1920, 1080}},
                  {"filter", {{"impl", "Gaussian"}}}}}}}}}};
        cameras.emplace_back(luisa::format("@{}", name));
    }
    scene_configs["import"] = {"lr_exported_materials.json", "lr_exported_geometry.json"};
    scene_configs["render"] = {{"cameras", std::move(cameras)},
                               {"shapes", {"@lr_exported_geometry"}},
                               {"integrator", {{"impl", "WavePath"}, {"prop", {{"sampler", {{"impl", "PMJ02BN"}}}}}}}};

    // save
    auto save = [&folder](auto &&file_name, auto &&json) noexcept {
        auto export_scene_path = folder / file_name;
        LUISA_INFO("Saving scene to '{}'...", export_scene_path.string());
        std::ofstream file{export_scene_path};
        file << json.dump(4);
    };
    save("lr_exported_materials.json", scene_materials);
    save("lr_exported_geometry.json", scene_geometry);
    save("lr_exported_scene.json", scene_configs);
}
