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
    uint data;// &3 == 0,1,2  axis, 4 is_leaf
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
    bool _visualize;
public:
    PPGPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 3u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _visualize{desc->property_bool_or_default("visualize", false)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto visualize() const noexcept { return _visualize; }
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
        bool clear_on_update=true;
    public:
        Buffer<uint> counter;
        Dtree(){}
        Dtree(Device &device, uint max_size=10000000){
            dnode=device.create_buffer<DTreeNode>(max_size);
            counter=device.create_buffer<uint>(1);
        }
        auto child(Expr<float2> pos) noexcept{//get the child index pos belong
            return ite(pos.x<0.5f,def(0u),def(2u))+ite(pos.y<0.5f,def(0u),def(1u));
        }
        Float2 subspace(Expr<float2> pos) noexcept{//rescale pos to [0,1] in son
            return make_float2(ite(pos.x<0.5f,pos.x,pos.x-.5f),ite(pos.y<0.5f,pos.y,pos.y-.5f))*2.f;
        }

        auto sph2vec(Expr<float2> pos) {
            auto theta=pos.x*2.f-1.f;
            auto phi=pos.y*2.f*pi;
            return make_float3(cos(phi)*sqrt(1.f-theta*theta),theta,sin(phi)*sqrt(1.f-theta*theta));
        }

        void insert(Expr<uint> id, Expr<Sample> sample) {
            auto pos=def(sample.dir);
            auto cur=def(id);
            $if(!(isnan(sample.val)|isinf(sample.val))){
                $while(cur != -1u) {
                    auto node = dnode->read(cur);
                    auto child_id = child(pos);
                    //node.sum[child_id]+=sample.val;
                    //
                    //device_log("id {}, node {}, son {}, v {}",cur,node,child_id,sample.val);
                    dnode->atomic(cur).sum[child_id].fetch_add(sample.val);
                    cur = node.children[child_id];
                    pos = subspace(pos);
                };
            };
        }

        auto get_child(Expr<float4> sum, Expr<float> u) noexcept{
            auto total=sum.x+sum.y+sum.z+sum.w;
            auto half=sum.x+sum.y;
            auto v=total*u;
            return ite(v<half,ite(v<sum.x,def(0u),def(1u)),ite(v<half+sum.z,def(2u),def(3u)));
        }
        auto sample(Expr<uint> id, Sampler::Instance &sampler) noexcept {
            auto cur=def(id);
            auto pos=def<float2>(.0f,.0f);
            auto size=def(1.f);
            auto pdf=def(1.f/(4.f*pi));
            auto node=dnode->read(cur);
            $if(node.sum[0]+node.sum[1]+node.sum[2]+node.sum[3]!=0.f) {
                $while(cur != -1) {
                    node = dnode->read(cur);
                    auto u = sampler.generate_1d();
                    auto child_id = get_child(node.sum, u);
                    //device_log("pdf{},sum{},sampler.generate_1d(){},child{}",pdf,node.sum,u,child_id);
                    pdf = pdf * 4.f * node.sum[child_id] / (node.sum[0] + node.sum[1] + node.sum[2] + node.sum[3]);
                    size *= 0.5f;
                    pos = pos + make_float2(ite((child_id & 2u) == 2u, size, def(0.f)), ite((child_id & 1u) == 1u, size, def(0.f)));
                    cur = node.children[child_id];
                };
            };
            pos+=sampler.generate_2d()*size;
            return make_float4(sph2vec(pos),pdf);
        }
        auto pdf(Expr<uint> id, Expr<float2> dir) noexcept{
            auto pos=def(dir);
            auto cur=def(id);
            auto pdf=def(1.f/(4.f*pi));
            auto node=dnode->read(cur);
            $if(node.sum[0]+node.sum[1]+node.sum[2]+node.sum[3]!=0.f) {
                $while(cur != -1) {
                    node = dnode->read(cur);
                    auto child_id = child(pos);
                    cur = node.children[child_id];
                    pdf = ite(node.sum[child_id]==0.f,0.f,pdf * 4.f * node.sum[child_id] / (node.sum[0] + node.sum[1] + node.sum[2] + node.sum[3]));
                    pos = subspace(pos);
                };
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
        auto update_node(Expr<uint> dst,Expr<uint> src,Expr<uint> last_counter) noexcept{
            auto node=dnode->read(src);
            if(clear_on_update){
                auto new_node=node;
                new_node.sum=make_float4(0.0000f);
                dnode->write(src,new_node);
            }
            auto res=dnode->read(dst);
            res.children=ite(dst<last_counter,res.children,make_uint4(-1u));
            auto bit=make_uint4(ite((node.children.x!=-1)&(res.children.x==-1),1u,0u),
                                ite((node.children.y!=-1)&(res.children.y==-1),1u,0u),
                                ite((node.children.z!=-1)&(res.children.z==-1),1u,0u),
                                ite((node.children.w!=-1)&(res.children.w==-1),1u,0u));
            auto prev=bit.x+bit.y;
            auto bit_sum=make_uint4(0u,bit.x,prev,prev+bit.z);
            auto new_id=counter->atomic(0u).fetch_add(bit_sum.w+bit.w);
            auto ans=new_id+bit_sum;
            node.children=make_uint4(ite(bit.x==1u,ans.x,res.children.x),ite(bit.y==1u,ans.y,res.children.y),
                                    ite(bit.z==1u,ans.z,res.children.z),ite(bit.w==1u,ans.w,res.children.w));
            return node;
        }
        auto empty_node() noexcept{
            auto id=counter->atomic(0u).fetch_add(1u);
            auto node=def<DTreeNode>();
            node.sum=make_float4(0.0001f);
            node.children=make_uint4(-1u);
            dnode->write(id,node);
            return id;
        }
        void refine(Expr<uint> id) noexcept{
            ArrayUInt3<MAX_DEPTH> stack;//(src_id,dst_id,child)
            auto top=def(1u);
            stack[0]=make_uint3(id,id,0u);
            auto sum=dnode->read(id).sum;
            auto tot=sum.x+sum.y+sum.z+sum.w;
            $if(tot>0.0f) {
                $while(top != 0) {
                    auto now = stack[top - 1];
                    auto prev_src = dnode->read(now.x);
                    auto son = prev_src.children[now.z];
                    stack[top - 1].z += 1;
                    $if(son == -1) {//consider split
                        $if((prev_src.sum[now.z] > tot * SPLIT_THRESHOLD) & (top < MAX_DEPTH)) {
                            auto new_id = counter->atomic(0u).fetch_add(1u);
                            auto new_node = def<DTreeNode>();
                            new_node.sum = make_float4(prev_src.sum[now.z] / 4);
                            new_node.children = make_uint4(-1u);
                            prev_src.children[now.z] = new_id;
                            dnode->write(now.x, prev_src);
                            dnode->write(new_id, new_node);
                            son = new_id;
                        };
                    };
                    $if(son != -1) {// not leaf
                        stack[top] = make_uint3(son, son, 0u);
                        top += 1;
                    };
                    $while((top != 0)) {
                        $if(stack[top - 1].z > 3u) {
                            top -= 1;
                        }
                        $else {
                            $break;
                        };
                    };
                };
            };
        }
        void update(Expr<uint> dst, Expr<uint> src,Expr<uint> last_counter) noexcept{
            ArrayUInt3<MAX_DEPTH> stack;//(src_id,dst_id,child)
            auto top=def(1u);
            stack[0]=make_uint3(src,dst,0u);
            auto node=update_node(dst,src,last_counter);
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
                    new_node = update_node(dst_son, src_son,last_counter);
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
            auto top=def(1u);
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
                $if(src_son!=-1) {
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
            return (node.data>>2)!=0u;
        }
        auto weight(Expr<STreeNode> node){
            return node.weight;
        }
        auto axis(Expr<STreeNode> node){
            return node.data&3u;
        }
        auto next_axis(Expr<uint> axis){
            return ite(axis+1u>2u,0u,axis+1u);
        }
        auto build_data(Expr<uint> axis,Expr<uint> is_leaf){
            return axis|(is_leaf<<2);
        }
        auto create_node(Expr<uint> data){
            auto node=def<STreeNode>(make_uint2(-1u,-1u),data,0.0f);
            $if(!is_leaf(node)){
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
            auto top=def(1u);
            stack[0]=make_uint2(new_id,0u);
            auto node=create_node(0u);
            snode->write(new_id,node);
            $while(top!=0){
                auto now=stack[top-1];
                auto prev=snode->read(now.x);
                auto son=prev.children[now.y];
                stack[top-1].y+=1;
                $if(!is_leaf(prev)) {
                    auto new_node = create_node(build_data(next_axis(axis(prev)),ite(top<depth,0u,1u)));
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
                auto ax=axis(node);
                id=ite(pos[ax]<0.5f,node.children.x,node.children.y);
                pos[ax]=ite(pos[ax]<0.5f,pos[ax]*2.f,(pos[ax]-0.5f)*2.f);
                node=snode->read(id);
            };
            snode->atomic(id).weight.fetch_add(1.f);
            auto d_id=train_id(node);
            dtree.insert(d_id,sample);
        }
        auto pdf(Expr<float3> p,Expr<float2> dir) noexcept{
            auto id=def(0u);
            auto node=snode->read(id);
            auto pos=(p-_aabb->read(0u))/(_aabb->read(1u)-_aabb->read(0u));
            $while(!is_leaf(node)){
                auto ax=axis(node);
                id=ite(pos[ax]<0.5f,node.children.x,node.children.y);
                pos[ax]=ite(pos[ax]<0.5f,pos[ax]*2.f,(pos[ax]-0.5f)*2.f);
                node=snode->read(id);
            };
            auto d_id=sample_id(node);
            return dtree.pdf(d_id,dir);
        }
        void refine(Expr<uint> id,Expr<float> threshold,Expr<uint> last_counter){
            auto node=snode->read(id);
            $if((is_leaf(node))){
                dtree.update(sample_id(node),train_id(node),last_counter);//update sample with train
                //device_log("id {}, node {}, weight {}",id,node,weight(node));
                $if(weight(node)>threshold){
                    auto new_id = counter->atomic(0u).fetch_add(2u);
                    auto new_node = def<STreeNode>();
                    //new middle
                    new_node.children = make_uint2(new_id, new_id + 1);
                    new_node.data = build_data(axis(node),0u);
                    snode->write(id, new_node);
                    //new_left
                    new_node.children = dtrees(node);
                    new_node.data = build_data(next_axis(axis(node)),1u);
                    new_node.weight = 0.f;
                    snode->write(new_id, new_node);
                    //new_right
                    new_node.children = make_uint2(dtree.copy(sample_id(node)),dtree.copy(train_id(node)));
                    new_node.data = build_data(next_axis(axis(node)),1u);
                    snode->write(new_id + 1, new_node);
                }$else{
                    node.weight=0.f;
                    snode->write(id,node);
                };
            };
        }
        void dtree_refine(Expr<uint> id){
            auto node=snode->read(id);
            $if((is_leaf(node))){
                //device_log("start_refine {}",node);
                dtree.refine(train_id(node));
            };
        }
        auto sample(Expr<float3> p,Sampler::Instance &sampler){
            auto id=def(0u);
            auto node=snode->read(id);
            auto pos=(p-_aabb->read(0u))/(_aabb->read(1u)-_aabb->read(0u));
            $while(!is_leaf(node)){
                snode->write(id,node);
                auto ax=axis(node);
                id=ite(pos[ax]<0.5f,node.children.x,node.children.y);
                pos[ax]=ite(pos[ax]<0.5f,pos[ax]*2.f,(pos[ax]-0.5f)*2.f);
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
    Buffer<float4> _prev_film;
    vector<float> _prev_var;
    uint2 debug_pixel=uint2(778u,990u);
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
        auto prev_count=4u;
        _stree=SDTree(pipeline().device(),20000000);
        _sec_mom=pipeline().device().create_buffer<float3>(pixel_count);
        _samples=pipeline().device().create_buffer<Sample>(pixel_count*node<PPGPathTracing>()->max_depth());
        _dbeta=pipeline().device().create_buffer<float>(pixel_count*node<PPGPathTracing>()->max_depth()*pipeline().spectrum()->node()->dimension());
        _bsdf=pipeline().device().create_buffer<float>(pixel_count*node<PPGPathTracing>()->max_depth()*pipeline().spectrum()->node()->dimension());
        _var=pipeline().device().create_buffer<float3>(1u);
        _prev_film=pipeline().device().create_buffer<float4>(pixel_count*prev_count);
        _prev_var.resize(prev_count);
        for(auto i=0u;i<prev_count;i++){
            _prev_var[i]=-1;
        }
        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight,BufferFloat4 film) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto pid=dispatch_id().x*dispatch_size().y+dispatch_id().y;
            auto L = Li(camera, frame_index, pixel_id, time);
            L*=shutter_weight;
            L=min(L,camera->film()->node()->clamp());
            //camera->film()->accumulate(pixel_id, shutter_weight * L);
            $if(!any(isnan(L) || isinf(L))) {
                film.atomic(pid).x.fetch_add(L.x);
                film.atomic(pid).y.fetch_add(L.y);
                film.atomic(pid).z.fetch_add(L.z);
                film.atomic(pid).w.fetch_add(1.f);
                _sec_mom->atomic(pid).x.fetch_add(L.x * L.x);
                _sec_mom->atomic(pid).y.fetch_add(L.y * L.y);
                _sec_mom->atomic(pid).z.fetch_add(L.z * L.z);
            };
        };
        Kernel1D clear_float3_kernel = [&](BufferFloat3 buffer) noexcept {
            $if(dispatch_x()==0u){
                buffer.write(dispatch_x(),make_float3(0.0f));
            };
            //buffer.write(dispatch_x(),make_float3(0.0f));
        };
        Kernel2D variance_kernel = [&](BufferFloat4 film) noexcept{
            auto pixel=dispatch_id().xy();
            auto pid=dispatch_id().x*dispatch_size().y+dispatch_id().y;
            auto smom=_sec_mom->read(pid);
            auto accum=film->read(pid);
            auto var=(smom-(accum.xyz()*accum.xyz())/accum.w)/(accum.w-1);
            $if(!any(isnan(var) || isinf(var))){
                _var->atomic(0u).x.fetch_add(var.x);
                _var->atomic(0u).y.fetch_add(var.y);
                _var->atomic(0u).z.fetch_add(var.z);
            };
            _sec_mom->write(pid, make_float3(0.f));
        };
        Kernel2D final_accum_kernel = [&](BufferFloat4 film,Float weight) noexcept{
            auto pixel=dispatch_id().xy();
            auto pid=dispatch_id().x*dispatch_size().y+dispatch_id().y;
            auto accum=film->read(pid);
            camera->film()->accumulate(pixel, accum.xyz()/accum.w*weight,weight);
        };
        Kernel1D initial_kernel = [&](UInt depth) noexcept {
            _stree.build(depth);
        };
        Kernel1D dtree_refine_kernel = [&]() noexcept {
            _stree.dtree_refine(dispatch_x());
        };
        Kernel1D stree_refine_kernel = [&](Float threshold,UInt last_counter) noexcept {
            _stree.refine(dispatch_x(),threshold,last_counter);
        };
        Kernel1D clear_uint_kernel = [&](BufferUInt buffer) noexcept {
            buffer.write(dispatch_x(), 0u);
        };
        Kernel1D clear_float4_kernel = [&](BufferFloat4 buffer) noexcept {
            buffer.write(dispatch_x(), make_float4(0.f));
        };
        Kernel2D visualize_kernel = [&](UInt2 pos) noexcept {
            auto u_filter = make_float2(.5f);
            auto u_lens = make_float2(.5f);
            auto [camera_ray, _, camera_weight] = camera->generate_ray(pos, 0.0f, u_filter, u_lens);
            auto ray = camera_ray;
            auto it = pipeline().geometry()->intersect(ray);
            $if(dispatch_x()+dispatch_y()==0){
                device_log("evaluating pos:{}",it->p());
            };
            auto dir=def<float2>(dispatch_id().xy())/def<float2>(dispatch_size().xy());
            auto pdf=_stree.pdf(it->p(),dir);
            sampler()->start(dispatch_id().xy(), 233u);
            auto sample=_stree.sample(it->p(),*sampler());
            //device_log("sample:{}",sample);
            camera->film()->accumulate(dispatch_id().xy(),make_float3(pdf));
        };
        Clock clock_compile;
        auto render = pipeline().device().compile(render_kernel);
        auto build_initial = pipeline().device().compile(initial_kernel);
        auto dtree_refine = pipeline().device().compile(dtree_refine_kernel);
        auto stree_refine = pipeline().device().compile(stree_refine_kernel);
        auto calc_variance= pipeline().device().compile(variance_kernel);
        auto clear_float3 = pipeline().device().compile(clear_float3_kernel);
        auto clear_uint = pipeline().device().compile(clear_uint_kernel);
        auto final_accum = pipeline().device().compile(final_accum_kernel);
        auto clear_float4= pipeline().device().compile(clear_float4_kernel);
        auto visualize = pipeline().device().compile(visualize_kernel);
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
        command_buffer<<build_initial(12u).dispatch(1u)<<synchronize();
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            auto rem_spp=s.spp;
            auto prev_var=1e20f;
            auto now_film=0;
            for (auto k=1;rem_spp!=0;k<<=1){
                now_film=(now_film+1)%prev_count;
                //camera->film()->clear(command_buffer);
                command_buffer<<clear_float4(_prev_film.view(now_film*pixel_count,pixel_count)).dispatch(pixel_count);
                //switch radiance cache
                if(rem_spp<k){
                    k=rem_spp;
                }
                rem_spp-=k;
                LUISA_INFO("running {} samples for training...", k);
                for(int j=0;j<k;j++){
                    command_buffer << render(sample_id++, s.point.time, s.point.weight,_prev_film.view(now_film*pixel_count,pixel_count))
                                          .dispatch(resolution);
                }
                command_buffer<<synchronize();
                LUISA_INFO("render finish, start training...");
                //build tree
                uint dtree_counter, stree_counter;
                command_buffer<<_stree.dtree.counter.copy_to(&dtree_counter)
                              <<_stree.counter.copy_to(&stree_counter)
                              <<synchronize();
                LUISA_INFO("counter read back finish! dtree node: {}, stree node: {}, start refining...",dtree_counter,stree_counter);
                if(rem_spp!=0) {
                    command_buffer << dtree_refine().dispatch(stree_counter) << synchronize();
                    command_buffer << stree_refine(5000 * std::sqrt(k), dtree_counter).dispatch(stree_counter) << synchronize();
                }
                LUISA_INFO("refine finish!");
                //get variance
                float3 spec_var=float3(0.f);
                auto img_var=1.0f;
                LUISA_INFO("read back variance...");
                command_buffer<<clear_float3(_var).dispatch(1u);
                command_buffer<<calc_variance(_prev_film.view(now_film*pixel_count,pixel_count)).dispatch(resolution);
                command_buffer<<_var.copy_to(&spec_var)<<synchronize();
                LUISA_INFO("spec_var {}",spec_var);
                img_var=(spec_var.x+spec_var.y+spec_var.z)/(3*pixel_count);
                _prev_var[now_film]=img_var/k;
                if(isnan(img_var)){
                    img_var=1e20f;
                }
                //multiply contribution with variance
                auto target_var=img_var/rem_spp;
                LUISA_INFO("variance: {} expected final variance: {}, previous variance: {}",img_var,target_var, prev_var);
                prev_var=target_var;
            }
            auto tot_weight=0.0f;
            for(int i=0;i<prev_count;++i){
                LUISA_INFO("var {}",_prev_var[i]);
                if(_prev_var[i]>1e-6f){
                    tot_weight+=1.0f/_prev_var[i];
                }
            }
            for(int i=0;i<prev_count;++i){
                if(_prev_var[i]>1e-6f){
                    auto weight=1.0f/_prev_var[i]/tot_weight;
                    LUISA_INFO("final accumulate film {}, variance:{}, weight {}",i,_prev_var[i],weight);
                    command_buffer<<final_accum(_prev_film.view(i*pixel_count,pixel_count),weight).dispatch(resolution);
                }
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
        if(node<PPGPathTracing>()->visualize()) {
            camera->film()->clear(command_buffer);
            command_buffer << visualize(debug_pixel).dispatch(resolution);
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
    void update_sample(Expr<uint> id, Expr<uint> depth, SampledWavelengths swl, SampledSpectrum Li,Expr<bool> first){
        bool direct=false;
        $if(first&direct) {
            auto first_id = id * node<PPGPathTracing>()->max_depth() + depth;
            auto v=pipeline().spectrum()->cie_y(swl,Li);
            _samples->atomic(first_id).val.fetch_add(v);
        };
        auto i=def<int>(depth-1);
        auto val=Li;
        Bool flag=first|direct;
        $while(i>=0){
            auto buffer_id=id*node<PPGPathTracing>()->max_depth()+i;
            auto dbeta=read_spec(buffer_id,_dbeta);
            auto bsdf=read_spec(buffer_id,_bsdf);
//            $if(all(dispatch_id().xy()==debug_pixel)){
//                device_log("depth:{},i:{},val:{} {} {},sample:{},bsdf:{} {} {},dbeta:{} {} {}",depth,i,val[0u],val[1u],val[2u],
//                           _samples->read(buffer_id),bsdf[0u],bsdf[1u],bsdf[2u],dbeta[0u],dbeta[1u],dbeta[2u]);
//            };
            $if(flag) {
                auto v = pipeline().spectrum()->cie_y(swl, val * bsdf);
                _samples->atomic(buffer_id).val.fetch_add(v);
            };
            flag=true;
            val*=dbeta;
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
        auto alpha=0.80f;
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
        auto count=def(0u);
        $for(depth, node<PPGPathTracing>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    update_sample(pid,depth,swl, eval.L * balance_heuristic(pdf_bsdf, eval.pdf),false);
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $outline {
                    $if(it->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        update_sample(pid,depth,swl,eval.L * balance_heuristic(pdf_bsdf, eval.pdf),false);
                    };
                };
            }
//            $if(all(dispatch_id().xy()==debug_pixel)) {
//                device_log("fid {}, depth:{}. hit point:{}, direction:{}, Li {} {} {}, beta {} {} {}",frame_index, depth,it->p(),-wo, Li[0u],Li[1u],Li[2u],beta[0u],beta[1u],beta[2u]);
//            };
            $if(!it->shape().has_surface()) { $break; };

            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();

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

                    auto roughness=closure->roughness();
                    auto is_glossy=(roughness.x * roughness.y < 0.16f);
                    $if(light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        $if(!is_glossy) {
                            auto pg_pdf = _stree.pdf(light_sample.shadow_ray->origin(), vec2sph(wi));
                            eval.pdf = pg_pdf * alpha + eval.pdf * (1.f - alpha);
                        };
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
//                        $if(all(dispatch_id().xy()==make_uint2(902,970))) {
//                            device_log("depth:{}. sample beta:{} {} {}, eval.f{} {} {}, light_sample.eval.L{} {} {}, w{},lspdf{},eval.pdf {}",
//                                       depth,beta[0u],beta[1u],beta[2u],eval.f[0u],eval.f[1u],eval.f[2u],light_sample.eval.L[0u],light_sample.eval.L[1u],light_sample.eval.L[2u],w,light_sample.eval.pdf,eval.pdf);
//                        };
                        update_sample(pid,depth,swl, w * eval.f*light_sample.eval.L,true);
                    };
                    // sample material
                    auto u_lobe = sampler()->generate_1d();
                    auto surface_sample=Surface::Sample::zero(swl.dimension());
                    $if(!is_glossy) {
                        $if((u_lobe < alpha)) {
                            u_lobe /= alpha;
                            auto wi_sample = _stree.sample(it->p(), *sampler());
                            surface_sample.wi = normalize(wi_sample.xyz());
                            surface_sample.eval = closure->evaluate(wo, surface_sample.wi);
                            surface_sample.event = -1;

//                            $if(all(dispatch_id().xy()==debug_pixel)){
//                                device_log("pg sample! fid:{} depth:{}. sample bsdfpdf {}, pgpdf {}, wi {} {} {}",frame_index, depth,surface_sample.eval.pdf,wi_sample.w,
//                                surface_sample.wi,vec2sph(surface_sample.wi),_stree.dtree.sph2vec(vec2sph(surface_sample.wi)));
//                            };
                            surface_sample.eval.pdf = wi_sample.w * alpha + surface_sample.eval.pdf * (1.f - alpha);
                        }
                        $else {
                            auto u_bsdf = sampler()->generate_2d();
                            u_lobe = (u_lobe - alpha) / (1.f - alpha);
                            surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                            auto pg_pdf = _stree.pdf(it->p(), vec2sph(surface_sample.wi));
                            surface_sample.eval.pdf = pg_pdf * alpha + surface_sample.eval.pdf * (1.f - alpha);
//                            $if(all(dispatch_id().xy()==debug_pixel)){
//                                device_log("bsdf sample! fid:{} depth:{}. sample bsdfpdf {}, pgpdf {}, wi {} {} {}",frame_index, depth,surface_sample.eval.pdf,pg_pdf,
//                                           surface_sample.wi,vec2sph(surface_sample.wi),_stree.dtree.sph2vec(vec2sph(surface_sample.wi)));
//                            };
                        };
                    }
                    $else{
                        auto u_bsdf = sampler()->generate_2d();
                        surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    };
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
            auto rr_fact=def(1.f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                rr_fact=ite(q < rr_threshold, 1.0f / q, 1.f);
                beta *= rr_fact;
            };
            $if(depth!=node<PPGPathTracing>()->max_depth()-1) {
                add_sample(pid, depth, ray, time, rr_fact * bsdf, bsdf);
                count+=1;
            };
            //auto sample=def<Sample>(ray->origin(),ray->direction(),Li,1.0f);
            //_stree.insert(sample);
        };
        $for(i,count){
            auto sample=_samples->read(pid*node<PPGPathTracing>()->max_depth()+i);
            //device_log("res:{},tot:{}, depth:{}, sample {}", dispatch_id().xy(),count,i,sample);
            _stree.insert(sample);
        };
        auto res=spectrum->srgb(swl,Li);
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
