//
// Created by Mike Smith on 2019/10/21.
//

#pragma once

#include "compatibility.h"

struct Onb {
    explicit Onb(Vec3f normal) : m_normal{normal} {
        
        using namespace metal;
        
        if (abs(m_normal.x) > abs(m_normal.z)) {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        } else {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }
        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }
    
    [[nodiscard]] Vec3f inverse_transform(Vec3f p) const {
        return p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }
    
    [[nodiscard]] Vec3f transform(Vec3f p) const {
        using namespace metal;
        return {dot(p, m_tangent), dot(p, m_binormal), dot(p, m_normal)};
    }
    
    Vec3f m_tangent{};
    Vec3f m_binormal{};
    Vec3f m_normal{};
};
