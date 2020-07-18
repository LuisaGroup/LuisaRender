#include <iostream>
#include <string_view>
#include <vector>
#include <functional>
#include <sstream>
#include <memory>

#include <compute/type_desc.h>
#include <compute/scalar.h>

namespace luisa {

enum struct MemorySpace {
    DEVICE,
    MANAGED,
    CONSTANT
};

template<typename T, MemorySpace mspace>
struct Buffer {

};

struct BufferDesc : public TypeDesc {
    
    std::unique_ptr<TypeDesc> element_desc;
    MemorySpace memory_space;
    
    BufferDesc(std::unique_ptr<TypeDesc> element_desc, MemorySpace memory_space) noexcept
        : element_desc{std::move(element_desc)}, memory_space{memory_space} {}
    
    void accept(const TypeDescVisitor &compiler) const override { compiler._visit(*this); }
};

namespace detail {

template<typename T, MemorySpace mspace>
struct TypeDescCreator<Buffer<T, mspace>> {
    static std::unique_ptr<TypeDesc> create() noexcept {
        return std::make_unique<BufferDesc>(TypeDescCreator<T>::create(), mspace);
    }
};

}

template<typename T>
struct Argument {

};

class Kernel {

private:

public:

};

class PipelineState {

};

namespace detail {

template<typename F>
constexpr auto to_std_function(F &&f) noexcept {
    return std::function{f};
}

template<typename F>
struct FunctionArgumentsImpl {};

template<typename R, typename ...Args>
struct FunctionArgumentsImpl<std::function<R(Args...)>> {
    using Type = std::tuple<Args...>;
};

}

template<typename F>
using FunctionArguments = typename detail::FunctionArgumentsImpl<decltype(detail::to_std_function(std::declval<F>()))>::Type;

class TypeDescPrinter : public TypeDescVisitor {

private:
    std::ostream &_os;

public:
    explicit TypeDescPrinter(std::ostream &os) : _os{os} {}
    
    void _visit(const ScalarDesc &desc) const override {
        switch (desc.type) {
            case ScalarType::BYTE:
                _os << "int8_t";
                break;
            case ScalarType::UBYTE:
                _os << "uint8_t";
                break;
            case ScalarType::SHORT:
                _os << "int16_t";
                break;
            case ScalarType::USHORT:
                _os << "uint16_t";
                break;
            case ScalarType::INT:
                _os << "int32_t";
                break;
            case ScalarType::UINT:
                _os << "uint32_t";
                break;
            case ScalarType::LONG:
                _os << "int64_t";
                break;
            case ScalarType::ULONG:
                _os << "uint64_t";
                break;
            case ScalarType::BOOL:
                _os << "bool";
                break;
            case ScalarType::FLOAT:
                _os << "float";
                break;
        }
    }
    
    void _visit(const BufferDesc &desc) const override {
        switch (desc.memory_space) {
            case MemorySpace::DEVICE:
            case MemorySpace::MANAGED:
                _os << "device ";
                break;
            case MemorySpace::CONSTANT:
                _os << "constant ";
                break;
        }
        TypeDescVisitor::visit(*desc.element_desc);
        _os << " *";
    }
};

namespace detail {

template<typename ...Args>
struct ArgumentPrinterImpl {};

template<>
struct ArgumentPrinterImpl<std::tuple<>> {
    static void print(int, const TypeDescPrinter &) noexcept {}
};

template<typename Current, typename ...Others>
struct ArgumentPrinterImpl<std::tuple<Argument<Current>, Others...>> {
    static void print(int counter, const TypeDescPrinter &printer) noexcept {
        auto desc = create_type_desc<Current>();
        std::cout << "arg #" << counter << ": ";
        printer.visit(*desc);
        std::cout << "\n";
        ArgumentPrinterImpl<std::tuple<Others...>>::print(counter + 1, printer);
    }
};

template<typename Arguments>
inline void print_args() noexcept {
    TypeDescPrinter printer{std::cout};
    ArgumentPrinterImpl<Arguments>::print(0, printer);
}

}

class Device {

public:
    template<typename K>
    [[nodiscard]] const PipelineState *compile_kernel(std::string_view name, K &&kernel) const {
        using Arguments = FunctionArguments<K>;
        std::cout << std::tuple_size_v<Arguments> << std::endl;
        detail::print_args<Arguments>();
        return nullptr;
    }
};

}

int main() {
    
    luisa::Device device;
    auto kernel = device.compile_kernel("film::clear", [&](
        luisa::Argument<luisa::Buffer<float, luisa::MemorySpace::DEVICE>> framebuffer,
        luisa::Argument<uint32_t> size) {
        
        
        
    });
    
}
