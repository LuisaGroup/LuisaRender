
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

# def torch_to_lc_buffer(tensor):
#     assert tensor.dtype is torch.float32  # TODO
#     size = np.prod(tensor.shape)
#     buf = luisa.Buffer.import_external_memory(
#         tensor.contiguous().data_ptr(),
#         size, dtype=float)
#     return buf

# def lc_buffer_to_torch(buf):
#     assert buf.dtype is float  # TODO
#     shape = (buf.size,)
#     return cu_device_ptr_to_torch_tensor(buf.native_handle, shape)

def is_torch_tensor(a):
    return getattr(a, '__module__', None) == 'torch' \
            and type(a).__name__ == 'Tensor'

def torch_ensure_grad_shape(a, b):
    if is_torch_tensor(a) and a.dtype in [torch.float, torch.float32, torch.float64]:
        return a.reshape(b.shape)
    else:
        return a

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


gt_args = ["C:/Users/jiankai/anaconda3/Lib/site-packages/luisarender/dylibs","-b","cuda", "D:/Code/LuisaRender2/data/scenes/cbox_caustic.luisa"]
init_args = ["C:/Users/jiankai/anaconda3/Lib/site-packages/luisarender/dylibs","-b","cuda", "D:/Code/LuisaRender2/data/scenes/cbox_caustic.luisa"]

differentiable_params_list = [
    {"type":"mesh","idx":0,"param":"vertex_position"},
    #{"type":"texture","idx":0,"param":"base_color"}
]

luisarender.load_scene(gt_args)
target_img = cu_device_ptr_to_torch_tensor(luisarender.render()[0], (512, 512, 4)).clone()
imageio.imwrite("gt.exr",target_img.detach().cpu().numpy()[...,:3])

#print(torch.max(target_img), torch.min(target_img), torch.sum(target_img))

x = luisarender.ParamStruct()
x.type = 'geom'
x.id = 0
[geom_ptr,geom_size] = luisarender.get_scene_param([x])
geom_ptr_torch = cu_device_ptr_to_torch_tensor(geom_ptr[0], (geom_size[0]//8,8), dtype=cupy.float32)
vertex_pos = geom_ptr_torch.clone()
vertex_pos[...,1]=1.0
vertex_pos[...,3]=0.0
vertex_pos[...,4]=1.0
vertex_pos[...,5]=0.0
pos_ptr = vertex_pos.contiguous().data_ptr()
pos_size = np.prod(vertex_pos.shape)
pos_dtype=float

optimizer = torch.optim.Adam([vertex_pos], lr=0.001)
x.size = pos_size
x.buffer_ptr = pos_ptr
luisarender.update_scene([x])

render_img = cu_device_ptr_to_torch_tensor(luisarender.render()[0], (512, 512,4)).clone()
imageio.imwrite("init.exr",render_img.detach().cpu().numpy()[...,:3])

loss_func = torch.nn.MSELoss()

for i in range(500):
    render_img = cu_device_ptr_to_torch_tensor(luisarender.render()[0], (512, 512, 4)).clone()
    imageio.imwrite(f"render{i}.exr",render_img.detach().cpu().numpy()[...,:3])
    render_img.requires_grad_()
    #loss = loss_func(render_img,target_img)
    loss = torch.sum((render_img-target_img)**2)
    loss.backward()
    grad = render_img.grad[...,:3]
    luisarender.render_backward([grad.contiguous().data_ptr()],[np.prod(grad.shape)])
    tex_grad, geom_grad = luisarender.get_gradients()
    geom_grad_torch = cu_device_ptr_to_torch_tensor(geom_grad[0], vertex_pos.shape, dtype=cupy.float32)
    print(loss, torch.max(geom_grad_torch), torch.min(geom_grad_torch), geom_grad_torch.shape)
    exit()
    luisarender.update_scene([x])

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

