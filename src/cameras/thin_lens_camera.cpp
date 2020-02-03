//
// Created by Mike Smith on 2020/2/1.
//

#include "thin_lens_camera.h"

namespace luisa {

void ThinLensCamera::generate_rays(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue) {

    dispatch(*_generate_rays_kernel, ray_queue.capacity(), [&](KernelArgumentEncoder &encode) {
    
    });
}

}
