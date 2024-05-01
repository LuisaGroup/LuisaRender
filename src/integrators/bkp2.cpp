$for(id, path_size-1){
            auto normal_cur_0 = v0->normal();
            auto normal_cur_1 = v1->normal();
            auto normal_cur_2 = v2->normal();
            instance = pipeline().geometry()->instance(inst_ids[id + 1]);
            triangle = pipeline().geometry()->triangle(instance, triangle_ids[id + 1]);
            v_buffer = instance.vertex_buffer_id();
            v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
            v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
            v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
            auto point_nxt_0 = v0->position();
            auto point_nxt_1 = v1->position();
            auto point_nxt_2 = v2->position();
            
            Float3 bary_pre;
            

            auto instance = pipeline().geometry()->instance(inst_ids[id]);
            auto triangle = pipeline().geometry()->triangle(instance, triangle_ids[id]);
            auto v_buffer = instance.vertex_buffer_id();

            auto v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
            auto v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
            auto v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);

            mat_bary[locate(id*2+0,id*2+0)] = dot(v0-v2, grad_p_cur[0]);
            mat_bary[locate(id*2+0,id*2+1)] = dot(v1-v2, grad_p_cur[0]);
            mat_bary[locate(id*2+1,id*2+0)] = dot(v0-v2, grad_p_cur[1]);
            mat_bary[locate(id*2+1,id*2+1)] = dot(v1-v2, grad_p_cur[1]);
            mat_param[id*2+0,1] = grad_p_cur[0];
            mat_param[id*2+1,1] = grad_p_cur[1];
            mat_param[id*2+0,3] = grad_n_cur[0];
            mat_param[id*2+1,3] = grad_n_cur[1];
            mat_param[id*2+0,4] = grad_eta[0];
            mat_param[id*2+1,4] = grad_eta[1];

            ArrayFloat3<2> grad_uv_pre, grad_uv_cur, grad_uv_nxt;

            $autodiff {
                Float3 bary_cur = bary_coords[id];  
                Float3 bary_nxt = bary_coords[id+1];  
                requires_grad(bary_pre, bary_cur, bary_nxt);
                Float3 point_pre = point_pre_0*bary_pre[0]+point_pre_1*bary_pre[1]+point_pre_2*(1-bary_pre[0]-bary_pre[1]);
                Float3 point_cur = point_cur_0*bary_cur[0]+point_cur_1*bary_cur[1]+point_cur_2*(1-bary_cur[0]-bary_cur[1]);
                Float3 point_nxt = point_nxt_0*bary_nxt[0]+point_nxt_1*bary_nxt[1]+point_nxt_2*(1-bary_nxt[0]-bary_nxt[1]);
                Float3 normal_cur = normal_cur_0*bary_cur[0]+normal_cur_1*bary_cur[1]+normal_cur_2*(1-bary_cur[0]-bary_cur[1]);
                requires_grad(point_pre, point_cur, point_nxt, normal_cur);
                auto wi = normalize(point_pre-point_cur);
                auto wo = normalize(point_nxt-point_cur);
                auto f = Frame::make(normal_cur);
                auto wi_local = f.world_to_local(wi);
                auto wo_local = f.world_to_local(wo);
                auto res = normalize(wi_local+wo_local*etas[id]);   
                device_log("res is {}",res);
                backward(res[0]);
                grad_uv_pre[0] = grad(bary_pre);
                grad_uv_cur[0] = grad(bary_cur);
                grad_uv_nxt[0] = grad(bary_nxt);
                mat_param[((((id+1)<<1)|0)<<2)] = grad(point_pre);
                mat_param[((((id+1)<<1)|0)<<2)|1] = grad(point_cur);
                mat_param[((((id+1)<<1)|0)<<2)|2] = grad(point_nxt);
                mat_param[((((id+1)<<1)|0)<<2)|3] = grad(normal_cur);
            };
            auto cust_normalize = [](Float3 x){
                return x/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
            };
            $autodiff {
                Float3 bary_cur = bary_coords[id];  
                Float3 bary_nxt = bary_coords[id+1];  
                requires_grad(bary_pre, bary_cur, bary_nxt);
                Float3 point_pre = point_pre_0*bary_pre[0]+point_pre_1*bary_pre[1]+point_pre_2*(1-bary_pre[0]-bary_pre[1]);
                Float3 point_cur = point_cur_0*bary_cur[0]+point_cur_1*bary_cur[1]+point_cur_2*(1-bary_cur[0]-bary_cur[1]);
                Float3 point_nxt = point_nxt_0*bary_nxt[0]+point_nxt_1*bary_nxt[1]+point_nxt_2*(1-bary_nxt[0]-bary_nxt[1]);
                Float3 normal_cur = normal_cur_0*bary_cur[0]+normal_cur_1*bary_cur[1]+normal_cur_2*(1-bary_cur[0]-bary_cur[1]);
                device_log("point_pre {} point_cur {} point_nxt {} normal_cur {}",point_pre, point_cur, point_nxt, normal_cur);
                requires_grad(point_pre, point_cur, point_nxt, normal_cur);
                auto wi = normalize(point_pre-point_cur);
                auto wo = normalize(point_nxt-point_cur);
                auto s = normalize(make_float3(0.0f, -normal_cur[2], normal_cur[1]));
                requires_grad(s);
                auto ss = normalize(s - normal_cur * dot(normal_cur, s));
                requires_grad(ss);
                auto tt = normalize(cross(normal_cur, ss));
                requires_grad(tt);
                Frame f{ss,tt,normal_cur};
                auto wi_local = f.world_to_local(wi);
                auto wo_local = f.world_to_local(wo);
                backward(wi_local);
                auto res = normalize(wi_local+wo_local*etas[id]);   
                auto grad_s = grad(s);
                auto grad_ss = grad(ss);
                auto grad_tt = grad(tt);
                device_log("grad(s) is {}",grad_s);
                device_log("grad(ss) is {}",grad_ss);
                device_log("grad(tt) is {}",grad_tt);
                device_log("res is {} wi {} wo {} eta {} normal {}",res, wi_local, wo_local, etas[id], normal_cur);
                device_log("grad(normal_cur) is {}",grad(normal_cur));
                device_log("grad(point_pre) is {}",grad(point_pre));
                device_log("grad(point_cur) is {}",grad(point_cur));
                device_log("grad(point_nxt) is {}",grad(point_nxt));
                grad_uv_pre[1] = grad(bary_pre);
                grad_uv_cur[1] = grad(bary_cur);
                grad_uv_nxt[1] = grad(bary_nxt);
                mat_param[((((id+1)<<1)|1)<<2)] = grad(point_pre);
                mat_param[((((id+1)<<1)|1)<<2)|1] = grad(point_cur);
                mat_param[((((id+1)<<1)|1)<<2)|2] = grad(point_nxt);
                mat_param[((((id+1)<<1)|1)<<2)|3] = grad(normal_cur);
            };

            for(uint j=0;j<2;j++)
            {
                $if(id>0)
                {
                    mat_bary[locate(id*2+2+j,id*2-2)] = grad_uv_pre[j][0];
                    mat_bary[locate(id*2+2+j,id*2-1)] = grad_uv_pre[j][1];
                };
                mat_bary[locate(id*2+2+j,id*2+0)] = grad_uv_cur[j][0];
                mat_bary[locate(id*2+2+j,id*2+1)] = grad_uv_cur[j][1];
                mat_bary[locate(id*2+2+j,id*2+2)] = grad_uv_nxt[j][0];
                mat_bary[locate(id*2+2+j,id*2+3)] = grad_uv_nxt[j][1];
            }
            point_pre_0 = point_cur_0;
            point_pre_1 = point_cur_1;
            point_pre_2 = point_cur_2;
            point_cur_0 = point_nxt_0;
            point_cur_1 = point_nxt_1;
            point_cur_2 = point_nxt_2;
        };
        inverse_matrix();
        compute_and_scatter_grad();

        // class PhotonMappingLogger{
    // public:
    //     //Buffer<uint> photon_inst_ids, photon_triangle_ids, photon_scatter_events, photon_sizes;
    //     Buffer<uint> path_inst_ids, path_triangle_ids, path_scatter_events, path_end_type, path_sizes, path_light_insts, path_light_tris;
    //     Buffer<float> path_etas, indirect_betas, path_betas, path_camera_weights, path_light_colors;
    //     Buffer<float3> path_camera_starts, path_light_ends;

    //     Buffer<float3> path_barys, path_light_barys;



    //     // Buffer<float3> photon_light_starts, photon_colors;
    //     // Buffer<float> photon_etas;
    //     //Buffer<UInt> path_photon_connections;
    //     //unique_ptr<Matrix> photon_matrix, photon_matrix_param, path_matrix, path_matrix_param;
    //     //Pipeline _pipeline;
    //     uint path_per_iter, max_path_depth;
    //     //uint photon_per_iter, max_photon_depth;
    //     const Spectrum::Instance *_spectrum;
    //     const uint _dimension;
    //     PhotonMappingLogger(uint path_per_iter, uint max_path_depth, const Spectrum::Instance *spectrum) :
    //         path_per_iter(path_per_iter), max_path_depth(max_path_depth),
    //         _spectrum(spectrum),_dimension(spectrum->node()->dimension()){
    //         auto &&device = spectrum->pipeline().device();
    //         //photon_inst_ids = device.create_buffer<uint>(photon_per_iter * max_photon_depth);
    //         //photon_triangle_ids = device.create_buffer<uint>(photon_per_iter * max_photon_depth);
    //         //photon_etas = device.create_buffer<float>(photon_per_iter * max_photon_depth);
    //         //photon_light_starts = device.create_buffer<float3>(photon_per_iter);
    //         //photon_colors = device.create_buffer<float3>(photon_per_iter * max_photon_depth);
    //         //photon_sizes = device.create_buffer<uint>(photon_per_iter);
    //         //path_photon_connections = device.create_buffer<UInt>>(path_per_iter * max_photon_per_path);
    //         path_inst_ids = device.create_buffer<uint>(path_per_iter * max_path_depth);
    //         path_triangle_ids = device.create_buffer<uint>(path_per_iter * max_path_depth);
    //         path_scatter_events = device.create_buffer<uint>(path_per_iter * max_path_depth);
    //         path_etas = device.create_buffer<float>(path_per_iter * max_path_depth);
    //         path_barys = device.create_buffer<float3>(path_per_iter * max_path_depth);

    //         path_light_ends = device.create_buffer<float3>(path_per_iter * max_path_depth);
    //         path_light_insts = device.create_buffer<uint>(path_per_iter * max_path_depth);
    //         path_light_tris = device.create_buffer<uint>(path_per_iter * max_path_depth);
    //         path_light_barys = device.create_buffer<float3>(path_per_iter * max_path_depth);
    //         path_end_type = device.create_buffer<uint>(path_per_iter * max_path_depth);

    //         path_camera_weights = device.create_buffer<float>(path_per_iter);
    //         path_camera_starts = device.create_buffer<float3>(path_per_iter);

    //         path_sizes = device.create_buffer<uint>(path_per_iter);
    //         indirect_betas = device.create_buffer<float>(path_per_iter * _dimension);
    //         path_betas = device.create_buffer<float>(path_per_iter * max_path_depth * _dimension);
    //         path_light_colors = device.create_buffer<float>(path_per_iter * max_path_depth * _dimension);
    //     }

    //     void add_start_camera(Expr<uint> path_id, Var<Ray> &ray, Float beta) {
    //         path_camera_starts->write(path_id, ray->origin());
    //         path_camera_weights->write(path_id, beta);
    //     }

    //     void add_path_vertex(Expr<uint> path_id, Expr<uint> path_size, shared_ptr<Interaction> it) {
    //         auto index = path_id * max_path_depth + path_size;
    //         path_inst_ids->write(index, it->instance_id());
    //         path_triangle_ids->write(index, it->triangle_id());
    //         path_barys->write(index, it->bary_coord());
    //     }

    //     void set_path_vertex_attrib(Expr<uint> path_id, Expr<uint> path_size, Surface::Sample &s, Float eta) {
    //         auto index = path_id * max_path_depth + path_size;
    //         path_etas->write(index, eta);
    //         path_scatter_events->write(index, s.event);
    //         for (auto i = 0u; i < _dimension; ++i)
    //             path_betas->write(index * _dimension + i, s.eval.f[i]);
    //     }

    //     void add_light_end(Expr<uint> path_id, Expr<uint> path_size, shared_ptr<Interaction> it, SampledSpectrum L) {
    //         //return;
    //         auto index = path_id * max_path_depth + path_size;
    //         path_light_ends->write(index, it->p());
    //         path_light_insts->write(index, it->instance_id());
    //         path_light_tris->write(index, it->triangle_id());
    //         path_light_barys->write(index, it->bary_coord());
    //         for (auto i = 0u; i < _dimension; ++i)
    //             path_light_colors->write(index * _dimension + i, L[i]);
    //         path_end_type->write(index, 1u|path_end_type->read(index));
    //     }

    //     void add_envlight_end(Expr<uint> path_id, Expr<uint> path_size, Float3 end, SampledSpectrum L) {
    //         //return;
    //         auto index = path_id * max_path_depth + path_size;
    //         path_light_ends->write(index, end);
    //         for (auto i = 0u; i < _dimension; ++i)
    //             path_light_colors->write(index * _dimension + i, L[i]);
    //         path_end_type->write(index, 2u|path_end_type->read(index));
    //     }

    //     /*
    //     void add_start_light(Expr<uint> photon_id, LightSampler::Sample &light_sample) {
    //         auto index = photon_id;
    //         auto beta = light_sample.eval.L / light_sample.eval.pdf;
    //         auto start = light_sample.shadow_ray->origin();
    //         photon_light_starts->write(index, start);
    //     }
    //     void add_photon_vertex(Expr<uint> photon_id, Expr<uint> path_size, shared_ptr<Interaction> it, Surface::Sample &s, Float eta) {
    //         auto index = photon_id * max_photon_depth + path_size;
    //         photon_inst_ids->write(index, it->instance_id());
    //         photon_triangle_ids->write(index, it->triangle_id());
    //         photon_etas->write(index, eta);
    //         photon_colors->write(index, s.eval.f);
    //         //photon_scatter_events->write(index, s.eval.events);
    //     }
    //     void connect_path_photon(UInt path_id, UInt photon_id, UInt path_photon_size, Float3 dis, Float3 Phi){
    //         auto index = path_id * max_photon_per_path + path_photon_size;
    //         path_photon_connections->write(index, photon_id);
    //     }

    //     void add_path_sizes(UInt path_id, UInt path_size){
    //         path_sizes->write(path_id, path_size);
    //     }

    //     void add_indirect_end(Expr<uint> path_id, Expr<uint> path_size, SampledSpectrum beta) {
    //         for (auto i = 0u; i < _dimension; ++i)
    //             indirect_betas->write(path_id * _dimension + i, beta[i]);
    //         //path_end_type->write(index, 4u|path_end_type->read(index));
    //     }

    //     auto indirect_beta(Expr<uint> path_id) {
    //         SampledSpectrum s{_dimension};
    //         for (auto i = 0u; i < _dimension; ++i)
    //             s[i] = indirect_betas->read(path_id * _dimension + i);
    //         return s;
    //     }
    //     */

    // };

    [[nodiscard]] Float3 emit_viewpoint_bp(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time, Expr<float> shutter_weight) noexcept {
        
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), shutter_weight * camera_weight};
        SampledSpectrum Li{swl.dimension()};
        SampledSpectrum testbeta{swl.dimension()};
        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);

        auto resolution = camera->film()->node()->resolution();
        auto pixel_id_1d = pixel_id.y*resolution.x+pixel_id.x;

        Float3 start_point = camera_ray->origin();
        Float3 end_point;
        Float3 radiance;
        ArrayUInt<4> triangle_ids, inst_ids;
        ArrayFloat3<4> bary_coords, points, normals;
        ArrayFloat<4> etas;
        ArrayFloat2<4> grad_barys;
        ArrayFloat3<4> grad_betas;
     
        logger->add_start_camera(pixel_id_1d, ray, camera_weight);
        auto path_size = 0u;
        $for(depth, node<MegakernelPhotonMappingDiff>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    logger->add_envlight_end(pixel_id_1d, path_size, ray->direction(), eval.L);
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    logger->add_light_end(pixel_id_1d, path_size, it, eval.L);
                };
            }
            
            $if(!it->shape().has_surface()) { $break; };

            // generate uniform samples
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<MegakernelPhotonMappingDiff>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // sample one light
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            Bool stop_direct = false;
            auto rr_threshold = node<MegakernelPhotonMappingDiff>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { stop_direct = true; };
            };

            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });

            call.execute([&](auto closure) noexcept {
                // if (auto dispersive = closure->is_dispersive()) {
                //     $if(*dispersive) { swl.terminate_secondary(); };
                // }
                // logger->add_path_vertex(pixel_id_1d, path_size, it);
                // direct lighting

                $if(light_sample.eval.pdf > 0.0f & !occluded) {
                    auto light_pos = light_sample.shadow_ray->origin();
                    auto eval = closure->evaluate(wo, wi);
                    auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                light_sample.eval.pdf;
                    Li += w * beta * eval.f * light_sample.eval.L;
                    logger->add_light_end(pixel_id_1d, path_size, it, light_sample.eval.L);
                };

                auto roughness = closure->roughness();
                Bool stop_check = (roughness.x * roughness.y > 0.16f) | stop_direct;
                $if(stop_check) {
                    stop_direct = true;
                    viewpoints->push(it->p(), swl, beta, wo, pixel_id_1d, surface_tag);
                };

                // sample material
                auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                ray = it->spawn_ray(surface_sample.wi);
                pdf_bsdf = surface_sample.eval.pdf;
                auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                beta *= w * surface_sample.eval.f;
                
                // apply eta scale
                auto eta = closure->eta().value_or(1.f);
                

                logger->set_path_vertex_attrib(pixel_id_1d, path_size, surface_sample, eta);
                path_size += 1;
                
                $switch(surface_sample.event) {
                    $case(Surface::event_enter) { eta_scale = sqr(eta); };
                    $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                };

            });

            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };

            $if(stop_direct) {
                auto it_next = pipeline().geometry()->intersect(ray);
                // logger->add_indirect_end(pixel_id_1d, path_size, beta);
                $if(!it_next->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        logger->add_envlight_end(pixel_id_1d, path_size, ray->direction(), eval.L);
                    }
                };
                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it_next->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it_next, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        logger->add_light_end(pixel_id_1d, path_size, it_next, eval.L);
                    };
                }
                $break;
            };
            $if(depth + 1u >= rr_depth) {
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        ArrayUInt<4> triangle_ids, inst_ids;
        ArrayFloat3<4> bary_coords, points, normals;
        Float3 light_pos, camera_pos, radiance;
        Float3 point_pre = camera_pos;
        Float3 grad_rgb;
        Float2 grad_xy;
        ArrayFloat3<4> grad_barys;
        $for(i,path_size)
        {
            Float3 point_cur = points[i], point_nxt;
            Float3 normal_cur = normals[i];
            $if(i<path_size-1){
                
                point_nxt = points[i];
            } $else{
                point_nxt = light_pos;
            }
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            Float3 grad_b_cur, grad_b_nxt, grad_b_pre;
            call.execute([&](auto closure) noexcept {
                Float wi = point_nxt-point_cur;
                Float wo = point_pre-point_cur;
                $autodiff{
                    requires_grad(bary_cur, bary_pre, bary_nxt);
                    auto eval = closure->evaluate(wo, wi);
                    auto devalf = L/detach(eval.f)*grad_rgb;
                    backward(devalf*eval.f);
                    grad_b_cur = grad(point_cur);
                    grad_b_nxt = grad(point_nxt);
                    grad_b_pre = grad(point_pre);
                };
                closure->backward(wo, wi, grad_rgb * L / eval.f);
                $if(i>0){
                    grad_barys[i-1]+=grad_b_pre;
                };
                grad_barys[i]+=grad_b_cur;
                $if(i<path_size-1){
                    grad_barys[i+1]+=grad_b_nxt;
                };
            });
        };
        EPSM_path(grad_barys, camera_pos, light_pos, triangle_ids, inst_ids, bary_coords, etas);
        return spectrum->srgb(swl, Li);
    }
    
    [[nodiscard]] void photon_tracing_bp(const Camera::Instance *camera, Expr<uint> frame_index,
                        Expr<uint2> sampler_id, Expr<float> time, Expr<uint> photon_id_1d, BufferFloat &grad_in) {

        sampler()->start(sampler_id, frame_index);
        // generate uniform samples
        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        auto u_direction = sampler()->generate_2d();
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto light_sample = light_sampler()->sample_le(
            u_light_selection, u_light_surface, u_direction, swl, time);
        //cos term canceled out in pdf
        SampledSpectrum beta = light_sample.eval.L / light_sample.eval.pdf;
        
        auto ray = light_sample.shadow_ray;
        auto pdf_bsdf = def(1e16f);

        //logger->add_start_light(photon_id_1d, light_sample);
        UInt path_size = 0u; 

        ArrayUInt<4> triangle_ids, inst_ids;
        ArrayFloat3<4> bary_coords, points, normals;
        ArrayFloat<4> etas;
        ArrayFloat2<4> grad_barys;
        ArrayFloat3<4> grad_betas;
    

        auto resolution = camera->film()->node()->resolution();
        auto max_depth = min(node<MegakernelPhotonMappingDiff>()->max_depth(), 4u);
        auto tot_neighbors = 0u;

        $for(depth, max_depth) {
            // trace
            //path_size = depth;
            auto wi = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);
            // miss
            grad_barys[path_size] = make_float2(0.f);
            grad_betas[path_size] = make_float3(0.f);
            $if(!it->valid()) {
                $if(node<MegakernelPhotonMappingDiff>()->debug_photon()%2==1)
                {
                    $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
                    {
                        device_log("break it unvalid at size {} id {}", path_size, photon_id_1d);
                        device_log("ray {} {}", ray->origin(), ray->direction());
                    };
                };
                
                $break;
            };

            $if(!it->shape().has_surface()) { 
                $if(node<MegakernelPhotonMappingDiff>()->debug_photon()%2==1)
                {
                    $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
                    {
                        device_log("break it non surface {} ", photon_id_1d);
                    };
                };
                $break; 
            };
            // generate uniform samples
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<MegakernelPhotonMappingDiff>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            $if(depth > 0) {// add diffuse constraint?
                auto grid = viewpoints->point_to_grid(it->p());
                Float3 grad_beta = make_float3(0.f);
                Float2 grad_bary = make_float2(0.f);
                auto count_neighbors = 0u;
                $for(x, grid.x - 1, grid.x + 2) {
                    $for(y, grid.y - 1, grid.y + 2) {
                        $for(z, grid.z - 1, grid.z + 2) {
                            Int3 check_grid{x, y, z};
                            auto viewpoint_index = viewpoints->grid_head(viewpoints->grid_to_index(check_grid));
                            $while(viewpoint_index != ~0u) {
                                auto position = viewpoints->position(viewpoint_index);
                                auto pixel_id = viewpoints->pixel_id(viewpoint_index);
                                
                                auto dis = distance(position, it->p());
                                auto rad = indirect->radius(pixel_id);
                                $if(dis <= rad) {
                                    auto viewpoint_beta = viewpoints->beta(viewpoint_index);
                                    auto viewpoint_wo = viewpoints->wo(viewpoint_index);
                                    auto bary = it->bary_coord();
                                    auto instance = pipeline().geometry()->instance(it->instance_id());
                                    auto triangle = pipeline().geometry()->triangle(instance, it->triangle_id());
                                    auto v_buffer = instance.vertex_buffer_id();
                                    auto v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
                                    auto v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
                                    auto v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
                                    auto point_0 = v0->position();
                                    auto point_1 = v1->position();
                                    auto point_2 = v2->position();
                                    auto surface_tag = viewpoints->surface_tag(viewpoint_index);
                                    SampledSpectrum eval_viewpoint(3u);
                                    PolymorphicCall<Surface::Closure> call;
                                    pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                                        surface->closure(call, *it, swl, viewpoint_wo, 1.f, time);
                                    });
                                    call.execute([&](const Surface::Closure *closure) noexcept {
                                        eval_viewpoint = closure->evaluate(viewpoint_wo, wi).f; 
                                    });
                                    Float2 grad_b{0.0f, 0.0f};
                                    Float grad_pixel_0, grad_pixel_1, grad_pixel_2;
                                    Float rel_dis_diff, grad_dis;
                                    Float weight;
                                    Float3 Phi, Phi_real;
                                    $autodiff {
                                        Float3 beta_diff = make_float3(beta[0u], beta[1u], beta[2u]);
                                        requires_grad(bary, beta_diff);
                                        Float3 photon_pos = point_0 * bary[0] + point_1 * bary[1] + point_2 * (1 - bary[0] - bary[1]);
                                        rel_dis_diff = distance(position, photon_pos) / rad;
                                        requires_grad(rel_dis_diff);
                                        auto rel3 = rel_dis_diff*rel_dis_diff*rel_dis_diff;
                                        weight = 3.5f*(1- 6*rel3*rel_dis_diff*rel_dis_diff + 15*rel3*rel_dis_diff - 10*rel3);
                                        auto wi_local = it->shading().world_to_local(wi);
                                        Phi = spectrum->srgb(swl, viewpoint_beta * eval_viewpoint / abs_cos_theta(wi_local));
                                        Phi_real = spectrum->srgb(swl, beta * viewpoint_beta * eval_viewpoint / abs_cos_theta(wi_local));
                                        auto Phi_beta = Phi * beta_diff * weight / 200000.0f;
                                        auto _grad_dimension = 3u;
                                        grad_pixel_0 = grad_in->read(pixel_id * _grad_dimension + 0);
                                        grad_pixel_1 = grad_in->read(pixel_id * _grad_dimension + 1);
                                        grad_pixel_2 = grad_in->read(pixel_id * _grad_dimension + 2);
                                        auto dldPhi = (Phi_beta[0]*grad_pixel_0 + Phi_beta[1]*grad_pixel_1 + Phi_beta[2]*grad_pixel_2);
                                        // auto dist = photon_pos[0]*photon_pos[0]+photon_pos[1]*photon_pos[1]+photon_pos[2]*photon_pos[2];
                                        // backward(dist);
                                        backward(dldPhi);
                                        grad_b = grad(bary).xy();
                                        grad_bary += grad_b;
                                        grad_beta += grad(beta_diff);
                                        grad_dis = grad(rel_dis_diff);
                                    };
                                    count_neighbors+=1;

                                    $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 2) % 2 == 1) {
                                        $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
                                        {
                                            $if(etas[0]==1.8f)
                                            {
                                                UInt2 pixel_id_2d = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
                                                camera->film()->accumulate(pixel_id_2d, make_float3(grad_dis,0.0f,0.0f));
                                            };
                                        };
                                    };

                                    $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 16) % 2 == 1) {
                                        $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
                                        {
                                            indirect->add_phi(pixel_id, Phi_real*weight);
                                            indirect->add_cur_n(pixel_id, 1u);
                                        };
                                    };
                                };
                                viewpoint_index = viewpoints->nxt(viewpoint_index);
                            };
                        };
                    };
                };
                $if(count_neighbors>0){
                    tot_neighbors+=count_neighbors;
                    grad_betas[path_size] = make_float3(grad_beta[0]/count_neighbors, grad_beta[1]/count_neighbors, grad_beta[2]/count_neighbors);
                    grad_barys[path_size] = make_float2(grad_bary[0]/count_neighbors, grad_bary[1]/count_neighbors);
                };
            };
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wi, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                // apply opacity map
                if (auto dispersive = closure->is_dispersive()) {
                    $if(*dispersive) { swl.terminate_secondary(); };
                }
                // sample material
                auto surface_sample = closure->sample(wi, u_lobe, u_bsdf, TransportMode::IMPORTANCE);
                ray = it->spawn_ray(surface_sample.wi);
                pdf_bsdf = surface_sample.eval.pdf;
                auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                auto bnew = beta * w * surface_sample.eval.f;
                // apply eta scale
                auto eta = closure->eta().value_or(1.f);
                auto roughness = closure->roughness();
                $switch(surface_sample.event) {
                    $case(Surface::event_enter) { eta_scale = sqr(eta);  etas[path_size] = eta;};
                    $case(Surface::event_exit) { eta_scale = sqr(1.f / eta);  etas[path_size] = 1.f / eta;};
                };
                eta_scale *= ite(beta.max() < bnew.max(), 1.f, bnew.max() / beta.max());
                beta = bnew;
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { 
                // $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
                // {
                //     device_log("break beta negative {}",photon_id_1d);
                // };
                $break; 
            };
            auto rr_threshold = node<MegakernelPhotonMappingDiff>()->rr_threshold();
            auto q = max(eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { 
                    $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
                    {
                        device_log("break rr {}",photon_id_1d);
                    };
                    $break; 
                };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
            triangle_ids[path_size] = it->triangle_id();
            inst_ids[path_size] = it->instance_id();
            bary_coords[path_size] = it->bary_coord();
            points[path_size] = it->p();
            normals[path_size] = it->ng();
            path_size = path_size+1u;
        };
        //@Todo: log surface event
        $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
        {
            $if(path_size==2) {
                $if(etas[0]==1.8f)
                {
                    $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 1) % 2 == 1) {
                        device_log("path size is {}",path_size);
                        device_log("photon_id is {}",photon_id_1d);
                        device_log("light sample is {}",light_sample.shadow_ray->origin());
                        $for(i,path_size)
                        {    
                            device_log("{} point {} normal {} eta {} inst {}, bary {} grad_bary {}",i, points[i], normals[i], etas[i], inst_ids[i], bary_coords[i], grad_barys[i]);
                        };
                    };
                    EPSM_photon(path_size, points, normals, inst_ids, triangle_ids, bary_coords, etas, light_sample.shadow_ray->origin(), grad_barys, photon_id_1d);
                };
            };
        };
    }