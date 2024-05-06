
#import luisarender

import torch
import cupy
import numpy as np
import luisarender
import matplotlib.pyplot as plt
import cv2
import imageio
luisarender.init()
def cu_device_ptr_to_torch_tensor(ptr, shape, dtype=cupy.float32):
    """
    Convert a CUdeviceptr to a PyTorch tensor.

    Args:
        ptr (ctypes.c_uint64): CUdeviceptr pointing to the GPU memory.
        shape (tuple): Shape of the tensor.
        dtype (cupy.dtype): Data type of the tensor. Default is cupy.float32.

    Returns:
        torch.Tensor: PyTorch tensor.
    """

    size_bytes = cupy.dtype(dtype).itemsize * np.prod(shape)

    # Create an UnownedMemory view of the CUdeviceptr
    umem = cupy.cuda.memory.UnownedMemory(int(ptr), size_bytes, owner=None)
    memptr = cupy.cuda.memory.MemoryPointer(umem, 0)

    # Convert the MemoryPointer to a CuPy ndarray
    array = cupy.ndarray(shape, dtype=dtype, memptr=memptr)

    # Convert the CuPy ndarray to a DLPack tensor and then to a PyTorch tensor
    return torch.utils.dlpack.from_dlpack(array.toDlpack())
    
def compute_vertex_normals(vertex_pos, face_ids):
    v0 = vertex_pos[face_ids[:, 0]]
    v1 = vertex_pos[face_ids[:, 1]]
    v2 = vertex_pos[face_ids[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    vertex_normals = torch.zeros_like(vertex_pos)
    vertex_normals[face_ids[:, 0]] += face_normals
    vertex_normals[face_ids[:, 1]] += face_normals
    vertex_normals[face_ids[:, 2]] += face_normals
    vertex_normals_norm = vertex_normals/torch.norm(vertex_normals, dim=-1, keepdim=True)
    return vertex_normals_norm


gt_args = ["C:/Users/jiankai/anaconda3/Lib/site-packages/luisarender/dylibs","-b","cuda", "D:/Code/LuisaRender2/data/scenes/cbox_caustic.luisa"]
init_args = ["C:/Users/jiankai/anaconda3/Lib/site-packages/luisarender/dylibs","-b","cuda", "D:/Code/LuisaRender2/data/scenes/cbox_caustic.luisa"]

luisarender.load_scene(gt_args)
x = luisarender.ParamStruct()
x.type = 'geom'
x.id = 0
[geom_ptr,geom_size,face_ids,face_sizes] = luisarender.get_scene_param([x])
geom_ptr_torch = cu_device_ptr_to_torch_tensor(geom_ptr[0], (geom_size[0]//8,8), dtype=cupy.float32)
face_ids_torch= cu_device_ptr_to_torch_tensor(face_ids[0], (face_sizes[0]//3,3), dtype=cupy.int32)
vertex_pos = geom_ptr_torch.clone()[...,:3]
vertex = geom_ptr_torch.clone()

# print(face_ids_torch,vertex)
# exit()

vertex_height = torch.ones((vertex_pos.shape[0]),dtype=torch.float32).cuda()
optimizer = torch.optim.Adam([vertex_height], lr=0.001)
vertex_height.requires_grad_()


#vertex_pos[...,1] = ((vertex_pos[...,1]-1)*0.05)+1
gt_param = vertex_pos[...,1].clone()
vertex_normal = compute_vertex_normals(vertex_pos, face_ids_torch)
vertex[...,0:3] = vertex_pos
vertex[...,3:6] = vertex_normal

pos_ptr = vertex.contiguous().data_ptr()
pos_size = np.prod(vertex.shape)
pos_dtype=float

x.size = pos_size
x.buffer_ptr = pos_ptr

torch.cuda.synchronize()
luisarender.update_scene([x])
target_img = cu_device_ptr_to_torch_tensor(luisarender.render()[0], (512, 512, 4)).clone()
imageio.imwrite("gt_0.05.exr",target_img.detach().cpu().numpy()[...,:3])
imageio.imwrite("gt.png",target_img.detach().cpu().numpy()[...,:3])
exit()
#print(torch.max(target_img), torch.min(target_img), torch.sum(target_img))




torch.cuda.synchronize()
vertex_pos[...,1] = vertex_height
vertex_normal = compute_vertex_normals(vertex_pos, face_ids_torch)
vertex[...,0:3] = vertex_pos
vertex[...,3:6] = vertex_normal
luisarender.update_scene([x])

render_img = cu_device_ptr_to_torch_tensor(luisarender.render()[0], (512, 512,4)).clone()
imageio.imwrite("init.exr",render_img.detach().cpu().numpy()[...,:3])
imageio.imwrite("init.png",render_img.detach().cpu().numpy()[...,:3])

#exit()
cm = plt.get_cmap('viridis')
# Apply the colormap like a function to any array:

for i in range(500):

    vertex_pos = geom_ptr_torch.clone()[...,:3]
    vertex_pos[...,1] = vertex_height
    vertex_normal = compute_vertex_normals(vertex_pos, face_ids_torch)
    vertex[...,0:3] = vertex_pos
    vertex[...,3:6] = vertex_normal
    torch.cuda.synchronize()
    luisarender.update_scene([x])

    render_img = cu_device_ptr_to_torch_tensor(luisarender.render()[0], (512, 512, 4)).clone()
    imageio.imwrite(f"outputs/render{i}.exr",render_img.detach().cpu().numpy()[...,:3])
    render_img.requires_grad_()
    #loss = loss_func(render_img,target_img)
    loss = torch.sum((render_img[400:,...]-target_img[400:,...])**2)
    loss.backward()
    grad = torch.cat([render_img.grad[...,:3],torch.zeros_like(render_img.grad[...,:2])],dim=-1)
    print(grad.shape)
    aux_buffer = luisarender.render_backward([grad.contiguous().data_ptr()],[np.prod(grad.shape)])
    aux_buffer_torch = cu_device_ptr_to_torch_tensor(aux_buffer[0],  (512, 512, 4), dtype=cupy.float32)
    aux_buffer_numpy = aux_buffer_torch.cpu().numpy()

    imageio.imwrite(f"outputs/backwarde_render_{i}.exr",aux_buffer_numpy[...,:3])

    graddis_avg = aux_buffer_numpy[...,0]
    mx = np.max(abs(graddis_avg))
    print(mx, np.max(graddis_avg), np.min(graddis_avg))
    grad_reldis_vis = cm(graddis_avg/mx)
    imageio.imwrite(f"outputs/grad_vis_{i}.png",grad_reldis_vis)
    exit()

    tex_grad, geom_grad = luisarender.get_gradients()
    geom_grad_torch = cu_device_ptr_to_torch_tensor(geom_grad[0], vertex.shape, dtype=cupy.float32)
    #print(loss, geom_grad_torch)

    vertex_height.grad = geom_grad_torch[...,1]
    #vertex_normal.grad = geom_grad_torch[...,3:6]
    vertex_normal.backward(geom_grad_torch[...,3:6])
    optimizer.step()
    print(loss, torch.sum((vertex_height-gt_param)**2))
    #print(vertex_height)


# def torch_to_luisa_scene(args):
#     return tuple(torch_to_lc_buffer(a) if is_torch_tensor(a) else a for a in args)    

# class RenderWithLuisa(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, *args):
#         ctx.args = args
#         ctx.scene_luisa = torch_to_luisa_scene(args)
#         #luisa.enable_grad(ctx.args_luisa)
#         res = luisarender.render(*ctx.scene_luisa)
#         ctx.res_luisa = (res,) if not isinstance(res, tuple) else res
#         return lc_buffer_to_torch(res)

#     @staticmethod
#     @torch.autograd.function.once_differentiable
#     def backward(ctx, *grad_output):
#         luisarender.set_grad(ctx.res_luisa, grad_output)
#         luisarender.render_backward()
#         args_grad = luisarender.get_grad(ctx.scene_luisa)
#         del ctx.scene_luisa, ctx.res_luisa
#         return args_grad



# string param_type;
#     uint param_id;
#     uint param_size;
#     uint64_t param_buffer_ptr;
#     float4 param_value;
    #exit()
    #optimizer.zero_grad()
    #tex.grad = tex_grad_torch
    #optimizer.step()    
    #cv2.imshow("texture", cv2.cvtColor(tex.detach().cpu().numpy()[...,:3], cv2.COLOR_BGR2RGB))
    # print(grad)
    # grad_np = grad[...,1].detach().cpu().numpy()  # Convert the tensor to numpy for visualization
    # plt.imshow(grad_np, cmap='viridis')  # Use the 'viridis' color map
    # plt.colorbar()
    # plt.show()
    # exit()
    #visualize grad with a color map
    # imgplot = plt.imshow(np.hstack([target_img.detach().cpu().numpy()[...,:3],render_img.detach().cpu().numpy()[...,:3],grad.detach().cpu().numpy()[...,:3]]))
    # plt.show()
    # exit()
    # img = luisarender.render() 
    # torch_tensor = cu_device_ptr_to_torch_tensor(img[0], (1024*1024,4))
    # img = torch_tensor.cpu().numpy().reshape((1024, 1024,4))
    # imgplot = plt.imshow(img[...,:3])
    # plt.show()
    #print(grad,torch.nonzero(torch.isnan(grad.view(-1))))
# img = tex_grad_torch.cpu().numpy().reshape(tex.shape)
# imgplot = plt.imshow(img[...,:3])
# print(tex_grad_torch)
# plt.show()
#gt_img = lc_buffer_to_torch(luisarender.render_scene())
#
#luisarender.load_scene(init_args)
#init_img = lc_buffer_to_torch(luisarender.render_scene())
# luisarender.regist_differentiable(differentiable_params_list)

# optimizer = torch.optim.Adam(scene_torch, lr=0.01)

# for i in range(1000):
#     optimizer.zero_grad()
#     image = RenderWithLuisa.apply(scene_torch)
#     loss = (gt_img-image)**2
#     loss.backward()
#     optimizer.step()


# class ToTorch(luisa.CustomOp):
#     def eval(self, *args):
#         self.args = args
#         self.argstorch = drjit_totorch(args, enable_grad=True)
#         self.restorch = func(*self.argstorch)
#         return torch_toluisajit(self.restorch)

#     def forward(self):
#         raise TypeError("warp_ad(): forward-mode AD is not supported!")

#     def backward(self):
#         grad_outtorch = drjit_totorch(self.grad_out())
#         grad_outtorch = torch_ensure_grad_shape(grad_outtorch, self.restorch)
#         def flatten(values):
#             """Flatten structure in a consistent arbitrary order"""
#             result = []
#             def traverse(values):
#                 if isinstance(values, _Sequence):
#                     for v in values:
#                         traverse(v)
#                 elif isinstance(values, _Mapping):
#                     for _, v in sorted(values.items(), key=lambda item: item[0]):
#                         traverse(v)
#                 else:
#                     result.append(values)
#             traverse(values)

#             # Single item should not be wrapped into a list
#             if not isinstance(values, _Sequence) and not isinstance(values, _Mapping):
#                 result = result[0]

#             return result

#         torch.autograd.backward(flatten(self.restorch), flatten(grad_outtorch))

#         def get_grads(args):
#             if isinstance(args, _Sequence) and not isinstance(args, str):
#                 return tuple(get_grads(b) for b in args)
#             elif isinstance(args, _Mapping):
#                 return {k: get_grads(v) for k, v in args.items()}
#             elif istorch_tensor(args):
#                 return getattr(args, 'grad', None)
#             else:
#                 return None

#         args_gradtorch = get_grads(self.argstorch)
#         args_grad = torch_toluisajit(args_gradtorch)
#         self.set_grad_in('args', args_grad)

