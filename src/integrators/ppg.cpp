//
// Created by Hercier on 2024/3/1.
//

#include <util/sampling.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {
using namespace compute;
struct Sample {
    float4 pos;
    float2 dir;
    float val;
    float weight;
};
struct DTreeNode{
    uint4 children;
    float4 sum;
};
struct STreeNode{
    uint2 children;//if is_leaf--> children, else dtree id
    uint data;// &3 == 0,1,2  axis, 3 is_leaf, >>2== weight
    float weight;
};
}
LUISA_STRUCT(luisa::render::Sample, pos, dir, val,weight){};
LUISA_STRUCT(luisa::render::STreeNode, children, data, weight){};
LUISA_STRUCT(luisa::render::DTreeNode, children, sum){};
namespace luisa::render{
class PPGPathTracing final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    PPGPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PPGPathTracingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;
protected:
    class Dtree{
        Buffer<DTreeNode> dnode;
        static constexpr uint MAX_DEPTH=20u;
        static constexpr float SPLIT_THRESHOLD=0.01f;
    public:
        Buffer<uint> counter;
        Dtree(){}
        Dtree(Device &device, uint max_size=10000000){
            dnode=device.create_buffer<DTreeNode>(max_size);
            counter=device.create_buffer<uint>(1);
        }
        auto child(Expr<float2> pos) noexcept{
            return ite(pos.x<0.5f,def(0u),def(2u))+ite(pos.y<0.5f,def(0u),def(1u));
        }
        Float2 subspace(Expr<float2> pos) noexcept{
            return make_float2(ite(pos.x<0.5f,pos.x,pos.x-.5f),ite(pos.y<0.5f,pos.x,pos.x-.5f))*2.f;
        }

        auto sph2vec(Expr<float2> pos) {
            auto theta=pos.x*2.f-1.f;
            auto phi=pos.y*2.f*pi;
            return make_float3(cos(phi)*sqrt(1.f-theta*theta),theta,sin(phi)*sqrt(1.f-theta*theta));
        }

        void insert(Expr<uint> id, Expr<Sample> sample) {
            auto pos=def(sample.dir);
            auto cur=def(id);
            $while(id!=-1){
                auto node=dnode->read(cur);
                auto child_id=child(pos);
                node.sum[child_id]+=sample.val;
                dnode->write(cur,node);
                cur=node.children[child_id];
                pos=subspace(pos);
            };
        }

        auto get_child(Expr<float4> sum, Expr<float> u) noexcept{
            auto total=sum.x+sum.y+sum.z+sum.w;
            auto half=sum.x+sum.y;
            return ite(u<half,ite(u<sum.x,def(0u),def(1u)),ite(u<half+sum.z,def(2u),def(3u)));
        }
        auto sample(Expr<uint> id, Sampler::Instance &sampler) noexcept {
            auto cur=def(id);
            auto pos=def<float2>(.0f,.0f);
            auto size=def(1.f);
            auto pdf=def(1.f/(4.f*pi));
            $while(id!=-1){
                auto node=dnode->read(cur);
                auto child_id=get_child(node.sum,sampler.generate_1d());
                pdf=pdf*4.f*node.sum[child_id]/(node.sum[0]+node.sum[1]+node.sum[2]+node.sum[3]);
                size*=0.5f;
                pos=pos+make_float2(ite((child_id&2u)==2u,size,def(0.f)),ite((child_id&1u)==1u,size,def(0.f)));
                cur=node.children[child_id];
            };
            pos+=sampler.generate_2d()*size;
            return make_float4(sph2vec(pos),pdf);
        }
        auto pdf(Expr<uint> id, Expr<float2> dir) noexcept{
            auto pos=def(dir);
            auto cur=def(id);
            auto pdf=def(1.f/(4.f*pi));
            $while(id!=-1){
                auto node=dnode->read(cur);
                auto child_id=child(pos);
                cur=node.children[child_id];
                pdf=pdf*4.f*node.sum[child_id]/(node.sum[0]+node.sum[1]+node.sum[2]+node.sum[3]);
                pos=subspace(pos);
            };
            return pdf;
        }
        auto copy_node(Expr<uint> id) noexcept{
            auto node=dnode->read(id);
            auto bit=make_uint4(ite(node.children.x!=-1,1u,0u),ite(node.children.y!=-1,1u,0u),
                                    ite(node.children.z!=-1,1u,0u),ite(node.children.w!=-1,1u,0u));
            auto prev=bit.x+bit.y;
            auto bit_sum=make_uint4(0u,bit.x,prev,prev+bit.z);
            auto new_id=counter->atomic(0u).fetch_add(bit_sum.w+bit.w);
            auto ans=new_id+bit_sum;
            node.children=make_uint4(ite(bit.x==1u,ans.x,def(-1u)),ite(bit.y==1u,ans.y,def(-1u)),
                                    ite(bit.z==1u,ans.z,def(-1u)),ite(bit.w==1u,ans.w,def(-1u)));

            return node;
        }
        auto update_node(Expr<uint> dst,Expr<uint> src) noexcept{
            auto node=dnode->read(src);
            auto res=dnode->read(dst);
            auto bit=make_uint4(ite((node.children.x!=-1)&(res.children.x==-1),1u,0u),
                                ite((node.children.y!=-1)&(res.children.y==-1),1u,0u),
                                ite((node.children.z!=-1)&(res.children.z==-1),1u,0u),
                                ite((node.children.w!=-1)&(res.children.w==-1),1u,0u));
            auto prev=bit.x+bit.y;
            auto bit_sum=make_uint4(0u,bit.x,prev,prev+bit.z);
            auto new_id=counter->atomic(0u).fetch_add(bit_sum.w+bit.w);
            auto ans=new_id+bit_sum;
            node.children=make_uint4(ite(bit.x==1u,ans.x,node.children.x),ite(bit.y==1u,ans.y,node.children.y),
                                    ite(bit.z==1u,ans.z,node.children.z),ite(bit.w==1u,ans.w,node.children.w));
            return node;
        }
        auto empty_node() noexcept{
            auto id=counter->atomic(0u).fetch_add(1u);
            auto node=def<DTreeNode>();
            node.sum=make_float4(0.1f);
            node.children=make_uint4(-1u);
            dnode->write(id,node);
            return id;
        }
        void refine(Expr<uint> id) noexcept{
            ArrayUInt3<MAX_DEPTH> stack;//(src_id,dst_id,child)
            uint top=1;
            stack[0]=make_uint3(id,id,0u);
            auto sum=dnode->read(id).sum;
            auto tot=sum.x+sum.y+sum.z+sum.w;
            $while(top!=0){
                auto now=stack[top-1];
                auto prev_src=dnode->read(now.x);
                auto son=prev_src.children[now.z];
                stack[top-1].z+=1;
                $if(son==-1){//consider split
                    $if(prev_src.sum[now.z] > tot * SPLIT_THRESHOLD) {
                        auto new_id = counter->atomic(0u).fetch_add(1u);
                        auto new_node = def<DTreeNode>();
                        new_node.sum = make_float4(prev_src.sum[now.z]/4);
                        new_node.children= make_uint4(-1u);
                        prev_src.children[now.z] = new_id;
                        dnode->write(now.x,prev_src);
                        dnode->write(new_id, new_node);
                        son = new_id;
                    };
                };
                $if(son!=-1) {// not leaf
                    stack[top] = make_uint3(son, son, 0u);
                    top += 1;
                };
                $while((top!=0)){
                    $if(stack[top-1].z>3u){
                        top -= 1;
                    }
                    $else{
                        $break;
                    };
                };
            };
        }
        void update(Expr<uint> dst, Expr<uint> src) noexcept{
            ArrayUInt3<MAX_DEPTH> stack;//(src_id,dst_id,child)
            uint top=1;
            stack[0]=make_uint3(src,dst,0u);
            auto node=update_node(dst,src);
            dnode->write(dst,node);
            $while(top!=0){
                auto now=stack[top-1];
                auto prev_src=dnode->read(now.x);
                auto prev_dst=dnode->read(now.y);
                auto src_son=prev_src.children[now.z];
                auto dst_son=prev_dst.children[now.z];
                stack[top-1].z+=1;
                $if(src_son!=-1) {
                    auto new_node=def<DTreeNode>();
                    $if(dst_son!=-1){
                        new_node = update_node(dst_son, src_son);
                    }$else {
                        new_node = copy_node(src_son);
                    };
                    dnode->write(dst_son, new_node);
                    stack[top] = make_uint3(src_son, dst_son, 0u);
                    top += 1;
                };
                $while((top!=0)){
                    $if(stack[top-1].z>3u){
                        top -= 1;
                    }
                    $else{
                        $break;
                    };
                };
            };
        }
        auto copy(Expr<uint> id) noexcept{//copy tree from copy
            auto new_id=counter->atomic(0u).fetch_add(1u);
            auto cur=def(id);
            auto new_cur=def(new_id);
            ArrayUInt3<MAX_DEPTH> stack;//(src_id,dst_id,child)
            uint top=1;
            stack[0]=make_uint3(id,new_id,0u);
            auto node=copy_node(id);
            dnode->write(new_id,node);
            $while(top!=0){
                auto now=stack[top-1];
                auto prev_src=dnode->read(now.x);
                auto prev_dst=dnode->read(now.y);
                auto src_son=prev_src.children[now.z];
                auto dst_son=prev_dst.children[now.z];
                stack[top-1].z+=1;
                $if(dst_son!=-1) {
                    auto new_node = copy_node(src_son);
                    dnode->write(dst_son, new_node);
                    stack[top] = make_uint3(src_son, dst_son, 0u);
                    top += 1;
                };
                $while((top!=0)){
                    $if(stack[top-1].z>3u){
                        top -= 1;
                    }
                    $else{
                        $break;
                    };
                };
            };
            return new_id;
        }
    };
    class SDTree{
        Buffer<STreeNode> snode;

        static constexpr uint MAX_DEPTH=20u;
    public:
        Buffer<float3> _aabb;
        Buffer<uint> counter;
        Dtree dtree;
        SDTree():
            dtree(){}
        SDTree(Device &device, uint max_size=10000000):
            dtree(device,max_size){
            _aabb=device.create_buffer<float3>(2);
            snode=device.create_buffer<STreeNode>(max_size);
            counter=device.create_buffer<uint>(1);
        }
        auto sample_id(Expr<STreeNode> node){
            return node.children.x;
        }
        auto train_id(Expr<STreeNode> node){
            return node.children.y;
        }
        auto dtrees(Expr<STreeNode> node){
            return node.children;
        }
        auto is_leaf(Expr<STreeNode> node){
            return (node.data&4u)==4u;
        }
        auto weight(Expr<STreeNode> node){
            return node.weight;
        }
        auto axis(Expr<STreeNode> node){
            return node.data&3u;
        }
        auto build_data(Expr<uint> axis,Expr<uint> is_leaf){
            return axis|(is_leaf<<2);
        }
        auto create_node(Expr<uint> data){
            auto node=def<STreeNode>(make_uint2(-1u,-1u),data,0.0f);
            $if(data!=3u){
                auto id=counter->atomic(0u).fetch_add(2u);
                node.children=make_uint2(id,id+1);
            }$else{
                node.children=make_uint2(dtree.empty_node(),dtree.empty_node());

            };
            return node;
        }
        void build(Expr<uint> depth){//build an empty Stree with depth
            counter->atomic(0u).fetch_add(1u);
            auto new_id=0u;
            ArrayUInt2<MAX_DEPTH> stack;//(src_id,dst_id,child)
            uint top=1;
            stack[0]=make_uint2(new_id,0u);
            auto node=create_node(0u);
            snode->write(new_id,node);
            $while(top!=0){
                auto now=stack[top-1];
                auto prev=snode->read(now.x);
                auto son=prev.children[now.y];
                stack[top-1].y+=1;
                device_log("traverse depth:{}, now id:{}, son {} id:{}",depth,now.x,now.y,son);
                $if(!is_leaf(prev)) {
                    auto new_node = create_node(ite(top<=depth,ite(prev.data==2u,0u,prev.data+1u),3u));
                    snode->write(son, new_node);
                    stack[top] = make_uint2(son, 0u);
                    top += 1;
                };
                $while((top!=0)){
                    $if(stack[top-1].y>1u){
                        top -= 1;
                    }
                    $else{
                        $break;
                    };
                };
            };
        }
        void insert(Expr<Sample> sample) noexcept{
            auto id=def(0u);
            auto node=snode->read(id);
            auto pos=(sample.pos.xyz()-_aabb->read(0u))/(_aabb->read(1u)-_aabb->read(0u));
            $while(!is_leaf(node)){
                snode->write(id,node);
                auto ax=axis(node);
                id=ite(pos[ax]<0.5f,node.children.x,node.children.y);
                pos=ite(pos[ax]<0.5f,pos*2.f,(pos-0.5f)*2.f);
                node=snode->read(id);
            };
            node.weight+=1.f;
            snode->write(id,node);
            auto d_id=train_id(node);
            dtree.insert(d_id,sample);
        }
        void refine(Expr<uint> id,Expr<float> threshold){
            auto node=snode->read(id);
            $if((is_leaf(node))){
                dtree.update(node.children.x,node.children.y);//update sample with train
                $if(weight(node)>threshold){
                    auto new_id = counter->atomic(0u).fetch_add(2u);
                    auto new_dtree = dtree.copy(sample_id(node));
                    node.children = make_uint2(new_dtree, id);
                    auto new_node = def<STreeNode>();
                    //new middle
                    new_node.children = make_uint2(new_id, new_id + 1);
                    new_node.data = build_data(axis(node),1u);
                    snode->write(id, new_node);
                    //new_left
                    new_node.children = dtrees(node);
                    new_node.data = ite(node.data == 2u, 0u, node.data + 1u);
                    new_node.weight = weight(node) / 2;
                    snode->write(new_id, new_node);
                    //new_right
                    new_node.children = dtrees(node);
                    new_node.data = ite(node.data == 0u, 2u, node.data + 1u);
                    snode->write(new_id + 1, new_node);
                }$else{
                    snode->write(id,node);
                };
            };
        }
        void dtree_refine(Expr<uint> id){
            dtree.refine(id);
        }
        auto sample(Expr<float3> p,Sampler::Instance &sampler){
            auto id=def(0u);
            auto node=snode->read(id);
            auto pos=(p-_aabb->read(0u))/(_aabb->read(1u)-_aabb->read(0u));
            $while(!is_leaf(node)){
                snode->write(id,node);
                auto ax=axis(node);
                id=ite(pos[ax]<0.5f,node.children.x,node.children.y);
                pos=ite(pos[ax]<0.5f,pos*2.f,(pos-0.5f)*2.f);
                node=snode->read(id);
            };
            auto d_id=sample_id(node);
            return dtree.sample(d_id,sampler);
        }
    }_stree;
    Buffer<Sample> _samples;
    Buffer<float> _dbeta;
    Buffer<float> _bsdf;
    Buffer<float3> _sec_mom;
    Buffer<float3> _var;
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();
        auto pixel_count = resolution.x * resolution.y;
        sampler()->reset(command_buffer, resolution, pixel_count, spp);
        command_buffer << pipeline().printer().reset();
        command_buffer << compute::synchronize();

        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);

        using namespace luisa::compute;
        _stree=SDTree(pipeline().device(),10000000);
        _sec_mom=pipeline().device().create_buffer<float3>(pixel_count);
        _samples=pipeline().device().create_buffer<Sample>(pixel_count*node<PPGPathTracing>()->max_depth());
        _dbeta=pipeline().device().create_buffer<float>(pixel_count*node<PPGPathTracing>()->max_depth()*pipeline().spectrum()->node()->dimension());
        _bsdf=pipeline().device().create_buffer<float>(pixel_count*node<PPGPathTracing>()->max_depth()*pipeline().spectrum()->node()->dimension());
        _var=pipeline().device().create_buffer<float3>(1u);
        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = Li(camera, frame_index, pixel_id, time);
            camera->film()->accumulate(pixel_id, shutter_weight * L);
        };
        LUISA_INFO("kernel build finish!");
        Kernel1D clear_float3_kernel = [&](Var<Buffer<float3>> buffer) noexcept {
            buffer.write(dispatch_x(),make_float3(0.0f));
        };
        Kernel2D variance_kernel = [&]() noexcept{
            auto pixel=dispatch_id().xy();
            auto pid=dispatch_id().x*dispatch_size().y+dispatch_id().y;
            auto smom=_sec_mom->read(pid);
            auto accum=camera->film()->read(pixel);
            auto var=(smom-accum.sample_count*accum.average*accum.average)/(accum.sample_count-1);
            _var->atomic(0u).x.fetch_add(var.x);
            _var->atomic(0u).y.fetch_add(var.y);
            _var->atomic(0u).z.fetch_add(var.z);
            _sec_mom->write(pid,make_float3(0.f));
        };
        Kernel1D initial_kernel = [&](UInt depth) noexcept {
            _stree.build(depth);
        };
        Kernel1D dtree_refine_kernel = [&]() noexcept {
            _stree.dtree_refine(dispatch_x());
        };
        Kernel1D stree_refine_kernel = [&](Float threshold) noexcept {
            _stree.refine(dispatch_x(),threshold);
        };
        Kernel1D clear_uint_kernel = [&](Var<Buffer<uint>> buffer) noexcept {
            buffer.write(dispatch_x(), 0u);
        };
        Clock clock_compile;
        auto render = pipeline().device().compile(render_kernel);
        auto build_initial = pipeline().device().compile(initial_kernel);
        auto dtree_refine = pipeline().device().compile(dtree_refine_kernel);
        auto stree_refine = pipeline().device().compile(stree_refine_kernel);
        auto calc_variance= pipeline().device().compile(variance_kernel);
        auto clear_float3 = pipeline().device().compile(clear_float3_kernel);
        auto clear_uint = pipeline().device().compile(clear_uint_kernel);
        auto integrator_shader_compilation_time = clock_compile.toc();
        LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
        auto shutter_samples = camera->node()->shutter_samples();
        command_buffer << synchronize();

        LUISA_INFO("Rendering started.");
        Clock clock;
        ProgressBar progress;
        progress.update(0.);
        auto dispatch_count = 0u;
        auto sample_id = 0u;
        LUISA_INFO("initialize sd tree...");
        vector<float3> aabb(2);
        aabb[0]=pipeline().geometry()->world_min();
        aabb[1]=pipeline().geometry()->world_max();
        command_buffer<<_stree._aabb.copy_from(aabb.data());
        command_buffer<<clear_uint(_stree.counter).dispatch(1u);
        command_buffer<<clear_uint(_stree.dtree.counter).dispatch(1u);
        command_buffer<<synchronize();
        LUISA_INFO("finish build tree");
        command_buffer<<build_initial(5u).dispatch(1u)<<synchronize();
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            auto rem_spp=s.spp;
            auto prev_var=1e20f;
            for (auto k=1;k<= s.spp;k<<=1){
                LUISA_INFO("running {} samples for training...");
                //switch radiance cache
                rem_spp-=k;
                for(int j=0;j<k;j++){
                    command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                          .dispatch(resolution);
                }
                LUISA_INFO("render finish, start training...");
                //build tree
                uint dtree_counter, stree_counter;
                command_buffer<<_stree.dtree.counter.copy_to(&dtree_counter)
                              <<_stree.counter.copy_to(&stree_counter)
                              <<synchronize();
                LUISA_INFO("counter read back finish! start refining...");
                command_buffer<<dtree_refine().dispatch(dtree_counter)
                              <<stree_refine(200*std::sqrt(k)).dispatch(stree_counter);
                LUISA_INFO("refine finish!");
                //get variance
                float3 spec_var=float3(0.f);
                auto img_var=1.0f;
                LUISA_INFO("read back variance...");
                command_buffer<<clear_float3(_var).dispatch(1u);
                command_buffer<<calc_variance().dispatch(resolution);
                command_buffer<<_var.copy_to(&spec_var)<<synchronize();
                img_var=spec_var.x+spec_var.y+spec_var.z;
                if(isnan(img_var)){
                    img_var=1e20f;
                }
                //multiply contribution with varaince
                auto target_var=img_var*k/rem_spp;
                LUISA_INFO("variance: {} expected final variance: {}, previous variance: {}",img_var,target_var, prev_var);
                if(target_var>prev_var){
                    break;
                }
                prev_var=target_var;
            }
            for(int j=0;j<rem_spp;j++){
                command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                      .dispatch(resolution);
            }
            //            for (auto i = 0u; i < s.spp; i++) {
            //                command_buffer << render(sample_id++, s.point.time, s.point.weight)
            //                                      .dispatch(resolution);
            //                if (auto &&p = pipeline().printer(); !p.empty()) {
            //                    command_buffer << p.retrieve();
            //                }
            //                dispatch_count++;
            //                if (camera->film()->show(command_buffer)) { dispatch_count = 0u; }
            //                auto dispatches_per_commit = 4u;
            //                if (dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
            //                    dispatch_count = 0u;
            //                    auto p = sample_id / static_cast<double>(spp);
            //                    command_buffer << [&progress, p] { progress.update(p); };
            //                }
            //            }
        }
        command_buffer << synchronize();
        progress.done();

        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }
    auto read_spec(Expr<uint> id,Expr<BufferView<float>> _spec){
        auto dim=pipeline().spectrum()->node()->dimension();
        SampledSpectrum ans{dim};
        for(auto i=0u;i<dim;i++){
            ans[i]=_spec->read(id*ans.dimension()+i);
        }
        return ans;
    }
    auto write_spec(Expr<uint> id,SampledSpectrum spec,Expr<BufferView<float>> _spec){
        auto dim=pipeline().spectrum()->node()->dimension();
        SampledSpectrum ans{dim};
        for(auto i=0u;i<dim;i++){
            _spec->write(id*ans.dimension()+i,spec[i]);
        }
        return ans;
    }
    void update_sample(Expr<uint> id, Expr<uint> depth, SampledWavelengths swl, SampledSpectrum Li){
        auto i=def<int>(depth-1);
        auto val=Li;
        $while(i>=0){
            auto buffer_id=id*node<PPGPathTracing>()->max_depth()+i;
            auto dbeta=read_spec(buffer_id,_dbeta);
            auto bsdf=read_spec(buffer_id,_bsdf);
            val*=dbeta;
            auto v=pipeline().spectrum()->cie_y(swl,val*bsdf);
            _samples->atomic(buffer_id).val.fetch_add(v);
            i-=1;
        };
    }
    auto vec2sph(Expr<float3> vec) {
        auto dir=normalize(vec);
        auto theta=(dir.y+1.f)/2.f;
        auto phi=atan2(dir.z,dir.x)/(2*pi);
        phi=ite(phi<0.f,phi+1.f,phi);
        return make_float2(theta,phi);
    }
    void add_sample(Expr<uint> id, Expr<uint> depth, Var<Ray> ray, Expr<float> time,SampledSpectrum dbeta, SampledSpectrum bsdf){
        auto buffer_id=id*node<PPGPathTracing>()->max_depth()+depth;
        write_spec(buffer_id,dbeta,_dbeta);
        write_spec(buffer_id,bsdf,_bsdf);
        auto sample=def<Sample>(make_float4(ray->origin(),time),vec2sph(ray->direction()),0.f,1.f);
        _samples->write(buffer_id,sample);
    }
    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time)noexcept{

        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto pid=dispatch_id().x*dispatch_size().y+dispatch_id().y;
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};

        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        auto path_length= def(0u);
        $for(depth, node<PPGPathTracing>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    update_sample(pid,depth,swl, eval.L * balance_heuristic(pdf_bsdf, eval.pdf));
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $outline {
                    $if(it->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        update_sample(pid,depth,swl,eval.L * balance_heuristic(pdf_bsdf, eval.pdf));
                    };
                };
            }

            $if(!it->shape().has_surface()) { $break; };

            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();

            auto u_rr = def(0.f);
            auto rr_depth = node<PPGPathTracing>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // generate uniform samples
            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            $outline {
                // sample one light
                light_sample = light_sampler()->sample(
                    *it, u_light_selection, u_light_surface, swl, time);
            };

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            SampledSpectrum bsdf{swl.dimension()};
            $outline {
                PolymorphicCall<Surface::Closure> call;
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                    surface->closure(call, *it, swl, wo, 1.f, time);
                });
                call.execute([&](const Surface::Closure *closure) noexcept {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    // direct lighting
                    $if(light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                        update_sample(pid,depth+1,swl, w * eval.f * light_sample.eval.L);
                    };
                    // sample material
                    auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    bsdf=surface_sample.eval.f*w;
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                });
            };

            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            auto rr_threshold = node<PPGPathTracing>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
            add_sample(pid,depth,ray,time,ite(q < rr_threshold, 1.0f / q, 1.f)*bsdf,bsdf);
            //auto sample=def<Sample>(ray->origin(),ray->direction(),Li,1.0f);
            //_stree.insert(sample);
        };
        auto res=spectrum->srgb(swl,Li);
        _sec_mom->atomic(pid).x.fetch_add(res.x*res.x);
        _sec_mom->atomic(pid).y.fetch_add(res.y*res.y);
        _sec_mom->atomic(pid).z.fetch_add(res.z*res.z);
        return res;
    }
};

luisa::unique_ptr<Integrator::Instance> PPGPathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<PPGPathTracingInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PPGPathTracing)
