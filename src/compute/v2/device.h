//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <compute/v2/buffer.h>
#include <compute/v2/pipeline.h>
#include <compute/function.h>

namespace luisa::compute {

class Device : Noncopyable {

protected:
    std::vector<std::unique_ptr<Buffer>> _buffers;
    
    [[nodiscard]] virtual std::unique_ptr<Buffer> _allocate_buffer(size_t size) = 0;
    [[nodiscard]] virtual std::unique_ptr<PipelineStage> _compile_kernel(const dsl::Function &function) = 0;
    
private:
    template<typename Def, std::enable_if_t<std::is_invocable_v<Def, dsl::Function &>, int> = 0>
    [[nodiscard]] std::unique_ptr<PipelineStage> compile_kernel(std::string name, Def &&def) {
        dsl::Function function{std::move(name)};
        def(function);
        return _compile_kernel(function);
    }
    
    template<typename T>
    [[nodiscard]] BufferView<T> create_buffer(size_t size) {
        return _buffers.emplace_back(_allocate_buffer(size * sizeof(T)))->view<T>();
    }
    
    virtual void launch(const std::function<void(Dispatcher &)> &dispatch) = 0;
    virtual void synchronize() = 0;
};

}
