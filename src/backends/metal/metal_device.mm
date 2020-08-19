//
// Created by Mike Smith on 2019/10/24.
//

#import <cstdlib>
#import <string>
#import <fstream>
#import <queue>
#import <cstring>
#import <map>

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import <compute/device.h>

#import "metal_buffer.h"
#import "metal_texture.h"
#import "metal_kernel.h"
#import "metal_codegen.h"
#import "metal_dispatcher.h"

namespace luisa::metal {

using compute::Device;

class MetalDevice : public Device {

public:
    struct MTLFunctionWrapper {
        id<MTLComputePipelineState> pipeline_state;
        id<MTLArgumentEncoder> argument_encoder;
        std::map<std::string, uint32_t> argument_to_id;
    };
    
    static constexpr auto max_command_queue_size = 8u;

private:
    id<MTLDevice> _handle;
    id<MTLCommandQueue> _command_queue;
    std::map<SHA1::Digest, MTLFunctionWrapper> _kernel_cache;
    std::vector<std::unique_ptr<MetalDispatcher>> _dispatchers;
    uint32_t _next_dispatcher{0u};
    
    [[nodiscard]] MetalDispatcher &next_dispatcher() noexcept {
        auto id = _next_dispatcher;
        _next_dispatcher = (_next_dispatcher + 1u) % max_command_queue_size;
        auto &&dispatcher = *_dispatchers[id];
        dispatcher._synchronize();
        return dispatcher;
    }

protected:
    std::unique_ptr<Kernel> _compile_kernel(const compute::dsl::Function &f) override;
    std::unique_ptr<Buffer> _allocate_buffer(size_t size, size_t max_host_caches) override;
    std::unique_ptr<Texture> _allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format, size_t max_caches) override;
    void _launch(const std::function<void(Dispatcher &)> &dispatch) override;

public:
    explicit MetalDevice(Context *context);
    ~MetalDevice() noexcept override = default;
    void synchronize() override;
};

MetalDevice::MetalDevice(Context *context) : Device{context} {
    _handle = MTLCreateSystemDefaultDevice();
    _command_queue = [_handle newCommandQueue];
    _dispatchers.reserve(max_command_queue_size);
    for (auto i = 0u; i < max_command_queue_size; i++) {
        _dispatchers.emplace_back(std::make_unique<MetalDispatcher>());
    }
}

std::unique_ptr<Kernel> MetalDevice::_compile_kernel(const compute::dsl::Function &f) {
    
    LUISA_INFO("Generating source for kernel \"", f.name(), "\"...");
    
    std::ostringstream os;
    MetalCodegen codegen{os};
    codegen.emit(f);
    auto s = os.str();
    
    auto digest = sha1_digest(s);
    auto iter = _kernel_cache.find(digest);
    if (iter == _kernel_cache.cend()) {
        
        LUISA_INFO("No compilation cache found for kernel \"", f.name(), "\", compiling from source...");
        NSError *error = nullptr;
        auto library = [_handle newLibraryWithSource:@(s.c_str()) options:nullptr error:&error];
        if (error != nullptr) {
            LUISA_WARNING("Compilation output:");
            NSLog(@"%@", error);
        }
        LUISA_INFO("Compilation for kernel \"", f.name(), "\" succeeded.");
        
        // Create PSO
        auto function = [library newFunctionWithName:@(f.name().c_str())];
        auto desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = function;
        desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
        desc.label = @(f.name().c_str());
        
        error = nullptr;
        auto pso = [_handle newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nullptr error:&error];
        if (error != nullptr) {
            LUISA_WARNING("Error occurred while creating pipeline state object, reason:");
            NSLog(@"%@", error);
            LUISA_EXCEPTION("Failed to create pipeline state object for kernel \"", f.name(), "\".");
        }
        
        MTLAutoreleasedArgument reflection;
        auto argument_encoder = [function newArgumentEncoderWithBufferIndex:0 reflection:&reflection];
        MTLFunctionWrapper wrapper{.pipeline_state = pso, .argument_encoder = argument_encoder};
        
        auto arg_struct = reflection.bufferStructType;
        for (auto &&arg : f.arguments()) {
            auto arg_name = serialize("v", arg.uid());
            auto arg_reflection = [arg_struct memberByName:@(arg_name.c_str())];
            if (compute::dsl::is_ptr_or_ref(arg.type())) {
                LUISA_EXCEPTION_IF([arg_reflection pointerType] == nullptr, "Argument \"", arg_name, "\" bound to a buffer is not declared as a pointer or reference.");
            } else if (arg.type()->type == compute::dsl::TypeCatalog::TEXTURE) {
                LUISA_EXCEPTION_IF([arg_reflection textureReferenceType] == nullptr, "Argument \"", arg_name, "\" bound to a texture is not declared as a texture.");
            }
            wrapper.argument_to_id.emplace(arg_name, arg_reflection.argumentIndex);
        }
        iter = _kernel_cache.emplace(digest, wrapper).first;
    } else {
        LUISA_INFO("Cache hit for kernel \"", f.name(), "\", compilation skipped.");
    }
    
    auto pso = iter->second.pipeline_state;
    auto arg_enc = iter->second.argument_encoder;
    auto &&arg_ids = iter->second.argument_to_id;
    
    std::vector<MetalKernel::Uniform> uniforms;
    std::vector<MetalKernel::Argument> arguments;
    for (auto &&arg : f.arguments()) {
        auto arg_name = serialize("v", arg.uid());
        auto arg_index = arg_ids.at(arg_name);
        if (arg.is_buffer_argument()) {
            arguments.emplace_back(MetalKernel::BufferArgument{
                .handle = dynamic_cast<MetalBuffer *>(arg.buffer().buffer())->handle(),
                .usage = compute::dsl::is_const_ptr_or_ref(arg.type()) ? MTLResourceUsageRead : (MTLResourceUsageRead | MTLResourceUsageWrite),
                .offset = static_cast<uint32_t>(arg.buffer().byte_offset()),
                .index = arg_index});
        } else if (arg.is_texture_argument()) {
            MTLResourceUsage usage;
            switch (arg.type()->access) {
                case compute::TextureAccess::READ:
                    usage = MTLResourceUsageRead;
                    break;
                case compute::TextureAccess::WRITE:
                    usage = MTLResourceUsageWrite;
                    break;
                case compute::TextureAccess::READ_WRITE:
                    usage = MTLResourceUsageRead | MTLResourceUsageWrite;
                    break;
                case compute::TextureAccess::SAMPLE:
                    usage = MTLResourceUsageSample;
                    break;
            }
            arguments.emplace_back(MetalKernel::TextureArgument{
                .handle = dynamic_cast<MetalTexture *>(arg.texture())->handle(),
                .usage = usage,
                .index = arg_index});
        } else if (arg.is_immutable_argument()) {
            arguments.emplace_back(MetalKernel::ImmutableArgument{
                .data = arg.immutable_data(),
                .index = arg_index});
        } else if (arg.is_uniform_argument()) {
            uniforms.emplace_back(MetalKernel::Uniform{arg_index, arg.type()->size, arg.uniform_data()});
        }
    }
    
    return std::make_unique<MetalKernel>(pso, std::move(uniforms), std::move(arguments), arg_enc);
}

void MetalDevice::_launch(const std::function<void(Dispatcher &)> &dispatch) {
    auto &&dispatcher = next_dispatcher();
    auto command_buffer = [_command_queue commandBuffer];
    dispatcher.reset(command_buffer);
    dispatch(dispatcher);
    dispatcher._commit();
}

void MetalDevice::synchronize() {
    for (auto i = 0u; i < max_command_queue_size; i++) { next_dispatcher().reset(); }
}

std::unique_ptr<Buffer> MetalDevice::_allocate_buffer(size_t size, size_t max_host_caches) {
    auto buffer = [_handle newBufferWithLength:size options:MTLResourceStorageModePrivate];
    return std::make_unique<MetalBuffer>(buffer, size, max_host_caches);
}

std::unique_ptr<Texture> MetalDevice::_allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format, size_t max_caches) {
    
    using namespace compute;
    
    auto desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType2D;
    desc.width = width;
    desc.height = height;
    desc.allowGPUOptimizedContents = true;
    desc.storageMode = MTLStorageModePrivate;
    switch (format) {
        case PixelFormat::R8U:
            desc.pixelFormat = MTLPixelFormatA8Unorm;
            break;
        case PixelFormat::RG8U:
            desc.pixelFormat = MTLPixelFormatRG8Unorm;
            break;
        case PixelFormat::RGBA8U:
            desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case PixelFormat::R32F:
            desc.pixelFormat = MTLPixelFormatR32Float;
            break;
        case PixelFormat::RG32F:
            desc.pixelFormat = MTLPixelFormatRG32Float;
            break;
        case PixelFormat::RGBA32F:
            desc.pixelFormat = MTLPixelFormatRGBA32Float;
            break;
    }
    auto texture = [_handle newTextureWithDescriptor:desc];
    return std::make_unique<MetalTexture>(texture, width, height, format, max_caches);
}

}

LUISA_EXPORT_DEVICE_CREATOR(luisa::metal::MetalDevice)
