//
// Created by Mike Smith on 2020/2/1.
//

#pragma once

#include <core/data_types.h>

struct ThinLensCameraParameters {

};

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/camera.h>

namespace luisa {

class ThinLensCamera : public Camera {

private:
    float _fov;
    Device *_device;

public:

};

}

#endif
