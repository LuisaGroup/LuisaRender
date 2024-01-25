
#import luisarender

import torch
import cupy
import numpy as np
import luisarender
import matplotlib.pyplot as plt
import cv2
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

class RenderWithLuisa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        ctx.args = args
        ctx.scene_luisa = torch_to_luisa_scene(args)
        #luisa.enable_grad(ctx.args_luisa)
        res = luisarender.render(*ctx.scene_luisa)
        ctx.res_luisa = (res,) if not isinstance(res, tuple) else res
        return lc_buffer_to_torch(res)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_output):
        luisarender.set_grad(ctx.res_luisa, grad_output)
        luisarender.render_backward()
        args_grad = luisarender.get_grad(ctx.scene_luisa)
        del ctx.scene_luisa, ctx.res_luisa
        return args_grad



# string param_type;
#     uint param_id;
#     uint param_size;
#     uint64_t param_buffer_ptr;
#     float4 param_value;

gt_args = ["C:/Users/jiankai/anaconda3/Lib/site-packages/luisarender/dylibs","-b","cuda", "D:/Code/LuisaRender2/cbox-diff/scenes/cbox-diff/cbox-diff-matte.luisa"]
init_args = ["C:/Users/jiankai/anaconda3/Lib/site-packages/luisarender/dylibs","-b","cuda", "C:/Users/jiankai/Downloads/bathroom/scene.luisa"]
differentiable_params_list = [
    #{"type":"mesh","idx":0,"param":"vertex_position"},
    {"type":"texture","idx":0,"param":"base_color"}
]
scene_torch = [#torch.tensor([[-1.01, 0.00,  0.99]]),
               torch.tensor([[0.9, 0.9, 0.9]])
               ]

luisarender.load_scene(gt_args)
target_img = cu_device_ptr_to_torch_tensor(luisarender.render()[0], (1024, 1024,4)).clone()

tex = torch.zeros((853,656,4),device='cuda',requires_grad=True)
tex_ptr = tex.contiguous().data_ptr()
tex_size = np.prod(tex.shape)
tex_dtype=float
optimizer = torch.optim.Adam([tex], lr=0.05)
for i in range(500):
    x = luisarender.ParamStruct()
    x.type = 'texture'
    x.id = 0
    x.size = tex_size
    x.buffer_ptr = tex_ptr
    luisarender.update_scene([x])
    render_img = cu_device_ptr_to_torch_tensor(luisarender.render()[0], (1024, 1024,4))
    render_img.requires_grad_()
    loss = torch.nn.MSELoss()(render_img,target_img)
    loss.backward()
    grad = render_img.grad
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
    luisarender.render_backward([grad.contiguous().data_ptr()],[np.prod(grad.shape)])
    tex_grad, geom_grad = luisarender.get_gradients()
    tex_grad_torch = cu_device_ptr_to_torch_tensor(tex_grad[0], tex.shape, dtype=cupy.float32)
    optimizer.zero_grad()
    tex.grad = tex_grad_torch
    optimizer.step()    
    cv2.imshow("texture", cv2.cvtColor(tex.detach().cpu().numpy()[...,:3], cv2.COLOR_BGR2RGB))
    cv2.imshow("render", cv2.cvtColor(render_img.detach().cpu().numpy()[...,:3], cv2.COLOR_BGR2RGB))
    cv2.waitKey(100)
cv2.waitKey(0)
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

