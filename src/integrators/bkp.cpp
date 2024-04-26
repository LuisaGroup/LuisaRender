//     // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id - 1, grad_uv_cur[0]);
                //     // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 0, grad_uv_cur[1]);
                //     // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 1, grad_uv_nxt[0]);
                //     // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 2, grad_uv_nxt[1]);
                //     // $if(id>0){
                //     // };
                //     // auto point_pre_0_grad = grad(point_pre_0);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 0, point_pre_0_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 1, point_pre_0_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 2, point_pre_0_grad[2]);
                //     // auto point_pre_1_grad = grad(point_pre_1);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 3, point_pre_1_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 4, point_pre_1_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 5, point_pre_1_grad[2]);
                //     // auto point_pre_2_grad = grad(point_pre_2);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 6, point_pre_2_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 7, point_pre_2_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 8, point_pre_2_grad[2]);

                //     // auto point_nxt_0_grad = grad(point_nxt_0);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 0, point_nxt_0_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 1, point_nxt_0_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 2, point_nxt_0_grad[2]);
                //     // auto point_nxt_1_grad = grad(point_nxt_1);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 3, point_nxt_1_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 4, point_nxt_1_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 5, point_nxt_1_grad[2]);
                //     // auto point_nxt_2_grad = grad(point_nxt_2);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 6, point_nxt_2_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 7, point_nxt_2_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 8, point_nxt_2_grad[2]);

                //     // auto point_cur_0_grad = grad(point_cur_0);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 0, point_cur_0_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 1, point_cur_0_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 2, point_cur_0_grad[2]);
                //     // auto point_cur_1_grad = grad(point_cur_1);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 3, point_cur_1_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 4, point_cur_1_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 5, point_cur_1_grad[2]);
                //     // auto point_cur_2_grad = grad(point_cur_2);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 6, point_cur_2_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 7, point_cur_2_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 8, point_cur_2_grad[2]);
                    
                //     // auto normal_cur_0_grad = grad(point_cur_0);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 9, normal_cur_0_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 10, normal_cur_0_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 11, normal_cur_0_grad[2]);
                //     // auto normal_cur_1_grad = grad(point_cur_1);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 12, normal_cur_1_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 13, normal_cur_1_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 14, normal_cur_1_grad[2]);
                //     // auto normal_cur_2_grad = grad(point_cur_2);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 15, normal_cur_2_grad[0]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 16, normal_cur_2_grad[1]);
                //     // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 17, normal_cur_2_grad[2]);
                // }