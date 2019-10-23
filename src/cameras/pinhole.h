//
// Created by Mike Smith on 2019/10/23.
//

#pragma once

#include <type_traits>
#include <core/camera.h>

class PinholeCamera : public Camera {

public:
    struct Backend {
        virtual void prepare(PinholeCamera &self) = 0;
        virtual void execute(PinholeCamera &self) = 0;
    };

public:
    
};


