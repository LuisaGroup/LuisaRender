//
// Created by Mike Smith on 2022/11/8.
//

#include <util/loop_subdiv.h>
#include <core/logging.h>

namespace luisa::render {

// The following code is from PBRT-v4.
// License: Apache 2.0
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.

struct SDFace;
struct SDVertex;

// LoopSubdiv Macros

[[nodiscard]] constexpr auto loop_subdiv_next(uint e) noexcept { return (e + 1u) % 3u; }
[[nodiscard]] constexpr auto loop_subdiv_prev(uint e) noexcept { return (e + 2u) % 3u; }

// LoopSubdiv Local Structures
struct SDVertex {

    float3 p;
    SDFace *startFace{};
    SDVertex *child{};
    bool regular{};
    bool boundary{};

    // SDVertex Constructor
    explicit SDVertex(float3 p = make_float3()) noexcept : p{p} {}

    // SDVertex Methods
    [[nodiscard]] uint valence() noexcept;
    void oneRing(float3 *p) noexcept;
};

struct SDFace {

    SDVertex *v[3]{};
    SDFace *f[3]{};
    SDFace *children[4]{};

    // SDFace Constructor
    SDFace() noexcept = default;

    // SDFace Methods
    [[nodiscard]] auto vnum(SDVertex *vert) const noexcept {
        for (auto i = 0u; i < 3u; i++) {
            if (v[i] == vert) { return i; }
        }
        LUISA_ERROR_WITH_LOCATION("Basic logic error in SDFace::vnum()");
    }
    [[nodiscard]] auto nextFace(SDVertex *vert) noexcept { return f[vnum(vert)]; }
    [[nodiscard]] auto prevFace(SDVertex *vert) noexcept { return f[loop_subdiv_prev(vnum(vert))]; }
    [[nodiscard]] auto nextVert(SDVertex *vert) noexcept { return v[loop_subdiv_next(vnum(vert))]; }
    [[nodiscard]] auto prevVert(SDVertex *vert) noexcept { return v[loop_subdiv_prev(vnum(vert))]; }
    [[nodiscard]] auto otherVert(SDVertex *v0, SDVertex *v1) noexcept {
        for (auto i : v) {
            if (i != v0 && i != v1) { return i; }
        }
        LUISA_ERROR_WITH_LOCATION("Basic logic error in SDFace::otherVert()");
    }
};

struct SDEdge {

    SDVertex *v[2];
    SDFace *f[2];
    uint f0edgeNum;

    // SDEdge Constructor
    explicit SDEdge(SDVertex *v0 = nullptr, SDVertex *v1 = nullptr) noexcept
        : v{std::min(v0, v1), std::max(v0, v1)}, f{}, f0edgeNum{~0u} {}

    // SDEdge Comparison Operators
    [[nodiscard]] auto operator==(const SDEdge &e) const noexcept {
        return v[0] == e.v[0] && v[1] == e.v[1];
    }
};

struct SDEdgeHash {
    [[nodiscard]] auto operator()(SDEdge e) const noexcept {
        return luisa::detail::murmur2_hash64(e.v, sizeof(e.v), 0x19980810u);
    }
};

// LoopSubdiv Local Declarations
[[nodiscard]] static float3 weightOneRing(SDVertex *vert, float beta) noexcept;
[[nodiscard]] static float3 weightBoundary(SDVertex *vert, float beta) noexcept;

// LoopSubdiv Inline Functions
[[nodiscard]] inline uint SDVertex::valence() noexcept {
    auto f = startFace;
    if (!boundary) {
        // Compute valence of interior vertex
        auto nf = 1u;
        while ((f = f->nextFace(this)) != startFace) { ++nf; }
        return nf;
    }
    // Compute valence of boundary vertex
    auto nf = 1u;
    while ((f = f->nextFace(this)) != nullptr) { ++nf; }
    f = startFace;
    while ((f = f->prevFace(this)) != nullptr) { ++nf; }
    return nf + 1u;
}

[[nodiscard]] inline auto beta(uint valence) noexcept {
    return 3.f / (valence == 3u ? 16.f : 8.f * static_cast<float>(valence));
}

[[nodiscard]] inline auto loopGamma(uint valence) noexcept {
    return 1.f / (static_cast<float>(valence) + 3.f / (8.f * beta(valence)));
}

// LoopSubdiv Function Definitions
std::pair<luisa::vector<float3>, luisa::vector<Triangle>>
loop_subdivide(luisa::span<const float3> positions,
               luisa::span<const Triangle> triangles,
               uint level) noexcept {

    if (level == 0u) {
        return std::make_pair(luisa::vector<float3>{positions.begin(), positions.end()},
                              luisa::vector<Triangle>{triangles.begin(), triangles.end()});
    }

    luisa::vector<SDVertex *> vertices;
    luisa::vector<SDFace *> faces;
    // Allocate _LoopSubdiv_ vertices and faces
    auto verts = luisa::make_unique<SDVertex[]>(positions.size());
    for (auto i = 0u; i < positions.size(); ++i) {
        verts[i] = SDVertex(positions[i]);
        vertices.emplace_back(&verts[i]);
    }
    auto nFaces = triangles.size();
    auto fs = luisa::make_unique<SDFace[]>(nFaces);
    for (int i = 0; i < nFaces; ++i) { faces.emplace_back(&fs[i]); }

    // Set face to vertex pointers
    for (auto i = 0u; i < nFaces; i++) {
        auto f = faces[i];
        auto t = make_uint3(triangles[i].i0, triangles[i].i1, triangles[i].i2);
        for (auto j = 0u; j < 3u; j++) {
            auto v = vertices[t[j]];
            f->v[j] = v;
            v->startFace = f;
        }
    }

    // Set neighbor pointers in _faces_
    luisa::unordered_set<SDEdge, SDEdgeHash> edges;
    for (auto i = 0u; i < nFaces; i++) {
        auto f = faces[i];
        for (auto edgeNum = 0u; edgeNum < 3u; edgeNum++) {
            // Update neighbor pointer for _edgeNum_
            auto v0 = edgeNum;
            auto v1 = loop_subdiv_next(edgeNum);
            SDEdge e(f->v[v0], f->v[v1]);
            if (edges.find(e) == edges.end()) {
                // Handle new edge
                e.f[0] = f;
                e.f0edgeNum = edgeNum;
                edges.insert(e);
            } else {
                // Handle previously seen edge
                e = *edges.find(e);
                e.f[0]->f[e.f0edgeNum] = f;
                f->f[edgeNum] = e.f[0];
                edges.erase(e);
            }
        }
    }

    // Finish vertex initialization
    for (auto i = 0u; i < positions.size(); i++) {
        auto v = vertices[i];
        auto f = v->startFace;
        do {
            f = f->nextFace(v);
        } while ((f != nullptr) && f != v->startFace);
        v->boundary = (f == nullptr);
        v->regular = (!v->boundary && v->valence() == 6u) ||
                     (v->boundary && v->valence() == 4u);
    }

    // Refine _LoopSubdiv_ into triangles
    auto f = faces;
    auto v = vertices;
    Pool<SDVertex, false, false> vertexAllocator;
    Pool<SDFace, false, false> faceAllocator;

    for (auto i = 0u; i < level; i++) {
        // Update _f_ and _v_ for next level of subdivision
        luisa::vector<SDFace *> newFaces;
        luisa::vector<SDVertex *> newVertices;

        // Allocate next level of children in mesh tree
        for (auto vertex : v) {
            vertex->child = vertexAllocator.create();
            vertex->child->regular = vertex->regular;
            vertex->child->boundary = vertex->boundary;
            newVertices.push_back(vertex->child);
        }
        for (auto face : f) {
            for (auto &k : face->children) {
                k = faceAllocator.create();
                newFaces.push_back(k);
            }
        }

        // Update vertex positions and create new edge vertices

        // Update vertex positions for even vertices
        for (auto vertex : v) {
            if (!vertex->boundary) {
                // Apply one-ring rule for even vertex
                vertex->child->p = weightOneRing(
                    vertex, vertex->regular ? 1.f / 16.f : beta(vertex->valence()));
            } else {
                // Apply boundary rule for even vertex
                vertex->child->p = weightBoundary(vertex, 1.f / 8.f);
            }
        }

        // Compute new odd edge vertices
        luisa::unordered_map<SDEdge, SDVertex *, SDEdgeHash> edgeVerts;
        for (auto face : f) {
            for (auto k = 0u; k < 3u; k++) {
                // Compute odd vertex on _k_th edge
                SDEdge edge{face->v[k], face->v[loop_subdiv_next(k)]};
                auto vert = edgeVerts[edge];
                if (vert == nullptr) {
                    // Create and initialize new odd vertex
                    vert = vertexAllocator.create();
                    newVertices.push_back(vert);
                    vert->regular = true;
                    vert->boundary = (face->f[k] == nullptr);
                    vert->startFace = face->children[3u];
                    // Apply edge rules to compute new vertex position
                    if (vert->boundary) {
                        vert->p = 0.5f * edge.v[0u]->p;
                        vert->p += 0.5f * edge.v[1u]->p;
                    } else {
                        vert->p = 3.f / 8.f * edge.v[0u]->p;
                        vert->p += 3.f / 8.f * edge.v[1u]->p;
                        vert->p += 1.f / 8.f * face->otherVert(edge.v[0u], edge.v[1u])->p;
                        vert->p +=
                            1.f / 8.f * face->f[k]->otherVert(edge.v[0u], edge.v[1u])->p;
                    }
                    edgeVerts[edge] = vert;
                }
            }
        }

        // Update new mesh topology

        // Update even vertex face pointers
        for (auto vertex : v) {
            auto vertNum = vertex->startFace->vnum(vertex);
            vertex->child->startFace = vertex->startFace->children[vertNum];
        }

        // Update face neighbor pointers
        for (auto face : f) {
            for (auto j = 0u; j < 3u; j++) {
                // Update children _f_ pointers for siblings
                face->children[3]->f[j] = face->children[loop_subdiv_next(j)];
                face->children[j]->f[loop_subdiv_next(j)] = face->children[3];
                // Update children _f_ pointers for neighbor children
                auto f2 = face->f[j];
                face->children[j]->f[j] =
                    f2 != nullptr ? f2->children[f2->vnum(face->v[j])] : nullptr;
                f2 = face->f[loop_subdiv_prev(j)];
                face->children[j]->f[loop_subdiv_prev(j)] =
                    f2 != nullptr ? f2->children[f2->vnum(face->v[j])] : nullptr;
            }
        }

        // Update face vertex pointers
        for (auto face : f) {
            for (int j = 0u; j < 3u; ++j) {
                // Update child vertex pointer to new even vertex
                face->children[j]->v[j] = face->v[j]->child;
                // Update child vertex pointer to new odd vertex
                auto vert = edgeVerts[SDEdge(face->v[j], face->v[loop_subdiv_next(j)])];
                face->children[j]->v[loop_subdiv_next(j)] = vert;
                face->children[loop_subdiv_next(j)]->v[j] = vert;
                face->children[3u]->v[j] = vert;
            }
        }

        // Prepare for next level of subdivision
        f = newFaces;
        v = newVertices;
    }

    // Push vertices to limit surface
    luisa::vector<float3> pLimit(v.size());
    for (auto i = 0u; i < v.size(); i++) {
        pLimit[i] = v[i]->boundary ?
                        weightBoundary(v[i], 1.f / 5.f) :
                        weightOneRing(v[i], loopGamma(v[i]->valence()));
    }
    for (auto i = 0u; i < v.size(); i++) { v[i]->p = pLimit[i]; }

    // Create triangle mesh from subdivision mesh
    luisa::vector<Triangle> tris(f.size());
    luisa::unordered_map<SDVertex *, uint> usedVerts;
    for (auto i = 0u; i < v.size(); i++) { usedVerts[v[i]] = i; }
    for (auto i = 0u; i < f.size(); ++i) {
        tris[i] = {usedVerts[f[i]->v[0u]],
                   usedVerts[f[i]->v[1u]],
                   usedVerts[f[i]->v[2u]]};
    }
    return std::make_pair(std::move(pLimit), std::move(tris));
}

static float3 weightOneRing(SDVertex *vert, float beta) noexcept {
    // Put _vert_ one-ring in _pRing_
    auto valence = vert->valence();
    luisa::fixed_vector<float3, 16u> pRing(valence);

    vert->oneRing(pRing.data());
    float3 p = (1.f - static_cast<float>(valence) * beta) * vert->p;
    for (auto i = 0u; i < valence; i++) { p += beta * pRing[i]; }
    return p;
}

void SDVertex::oneRing(float3 *p) noexcept {
    if (!boundary) {
        // Get one-ring vertices for interior vertex
        SDFace *face = startFace;
        do {
            *p++ = face->nextVert(this)->p;
            face = face->nextFace(this);
        } while (face != startFace);
    } else {
        // Get one-ring vertices for boundary vertex
        SDFace *face = startFace, *f2;
        while ((f2 = face->nextFace(this)) != nullptr) {
            face = f2;
        }
        *p++ = face->nextVert(this)->p;
        do {
            *p++ = face->prevVert(this)->p;
            face = face->prevFace(this);
        } while (face != nullptr);
    }
}

static float3 weightBoundary(SDVertex *vert, float beta) noexcept {
    // Put _vert_ one-ring in _pRing_
    auto valence = vert->valence();
    luisa::fixed_vector<float3, 16u> pRing(valence);
    vert->oneRing(pRing.data());
    float3 p = (1 - 2 * beta) * vert->p;
    p += beta * pRing[0];
    p += beta * pRing[valence - 1];
    return p;
}

}// namespace luisa::render
