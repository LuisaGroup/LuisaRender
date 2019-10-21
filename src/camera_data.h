//
// Created by Mike Smith on 2019/10/20.
//

#pragma once

#include "compatibility.h"

struct CameraData {
    Vec3f position;
    Vec3f front;
    Vec3f left;
    Vec3f up;
    float near_plane;
    float fov;
};
