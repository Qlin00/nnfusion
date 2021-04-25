// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "block_quantize_pass.hpp"
#include <map>
#include <queue>
#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

DEFINE_string(fblockquantize_cfg, "", "Block Sparse Quantize cfg for the target model");
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::element;
using namespace nnfusion;
using namespace std;
// using namespace std;

class BlockQuantizeKernelOptimizer
{
public:
    BlockQuantizeKernelOptimizer(std::shared_ptr<Graph> g, const std::string& cfg_path)
    {
        this->m_graph = g;
        this->cfg_path = cfg_path;
        load_quantize_cfg();
        cache_manager = std::make_shared<nnfusion::cache::KernelCacheManager>();
    }
    void load_quantize_cfg()
    {
        // Load the quantize config
        // pytorch name, onnx name, quantize bit
        ifstream cfg_file(cfg_path.c_str());
        assert(cfg_file.good());
        string line;
        while (std::getline(cfg_file, line))
        {
            std::istringstream iss(line);
            string name, row_file, col_file, value_file;
            int n_bit;
            int need_converter;
            iss >> name >> n_bit >> need_converter >>row_file >> col_file >> value_file;
            std::cout<<"Layer:"<<name<<" bit:"<<n_bit<< " row_file:"<<row_file<<" col_file:"<<col_file<<std::endl;
            quantize_cfg[name] = n_bit;
            csr_row[name] = row_file;
            csr_col[name] = col_file;
            csr_values[name] = value_file;
            fneed_converter[name] = need_converter;

        }
    }

    vector<std::shared_ptr<GNode>> find_successors(std::shared_ptr<GNode> gnode)
    {
        vector<std::shared_ptr<GNode>> successors;
        const std::set<std::shared_ptr<nnfusion::graph::Edge>>& out_edges = gnode->get_out_edges();
        for (auto edge : out_edges)
        {
            successors.push_back(edge->get_dst());
        }
        return successors;
    }
    vector<std::shared_ptr<GNode>> find_predecessors(std::shared_ptr<GNode> gnode)
    {
        vector<std::shared_ptr<GNode>> predecessors;
        const std::set<std::shared_ptr<nnfusion::graph::Edge>>& in_edges = gnode->get_in_edges();
        for (auto edge : in_edges)
        {
            predecessors.push_back(edge->get_src());
        }
        return predecessors;
    }
    vector<std::shared_ptr<GNode>> find_all_predecessors(std::shared_ptr<GNode> gnode)
    {
        vector<std::shared_ptr<GNode>> result;
        auto predecessors = find_predecessors(gnode);
        result.insert(result.end(), predecessors.begin(), predecessors.end());
        for (auto father : predecessors)
        {
            auto grandfathers = find_all_predecessors(father);
            result.insert(result.end(), grandfathers.begin(), grandfathers.end());
        }
        return result;
    }

    nnfusion::cache::KernelEntry_p
        fetch_quantize_kernel(shared_ptr<cache::KernelCacheManager> cache_manager,
                              shared_ptr<GNode> node,
                              NNFusion_DeviceType devtype,
                              vector<std::shared_ptr<GNode>> fused_op)
    {
        std::cout << node->get_unique_name() << " " << node->get_name() << std::endl;

        int quantize_bit = quantize_cfg[node->get_name()];
        std::shared_ptr<KernelContext> ctx(new KernelContext(node));
        std::string identifier = ctx->generate_identifier();
        std::cout << "Identifier: " << identifier << std::endl;
        // last node of the fused kernel
        std::shared_ptr<GNode> last_node = node;
        if (fused_op.size())
            last_node = fused_op[fused_op.size() - 1];
        std::vector<std::shared_ptr<GNode>> successors = find_successors(last_node);
        int succ_bit = 32;
        int succ_quantized = 0;
        for (auto succ_node : successors)
        {
            std::string succ_name = succ_node->get_name();
            std::cout << "Following nodes: " << succ_name << std::endl;
            if (quantize_cfg.count(succ_name))
            {
                if (succ_quantized)
                    // following nodes should have the same quantize config
                    NNFUSION_CHECK(succ_bit == quantize_cfg[succ_name]);
                succ_quantized += 1;
                succ_bit = quantize_cfg[succ_name];
            }
        }
        NNFUSION_CHECK(succ_quantized == 0 || succ_quantized == successors.size());
        // Generate the new identifier: ori_identifier + ${in_quantize}bit+${out_quatize}bit

        const std::vector<std::string> SUPPORT_PLATFORM = {"CUDA_GPU"};
        if (identifier != "" &&
            find(SUPPORT_PLATFORM.begin(), SUPPORT_PLATFORM.end(), get_device_str(devtype)) !=
                SUPPORT_PLATFORM.end())
        {
            identifier = "BlockQuantize" + identifier + "quantize" + to_string(quantize_bit) + "bit_" +
                         to_string(succ_bit) + "bit";

            for (auto tmp_node : fused_op)
            {
                identifier += tmp_node->get_op_type();
            }
            std::cout << "New Identifier:" << identifier << std::endl;
            std::cout << "Device String: " << get_device_str(devtype) << std::endl;
            auto fetched = cache_manager->fetch_all(identifier, get_device_str(devtype));
            nnfusion::cache::KernelEntry_p kernel_entry = nullptr;
            double kernel_time = 1000000000;
            std::cout << "Fetch" << fetched.size() << " Kernels from Kernel Cache!!!!!"
                      << std::endl;
            for (auto fetch_entry : fetched)
            {
                std::cout << "Find Matched quantize kernel" << std::endl;
                if (kernel_entry == nullptr)
                //fetch_entry->miscs["time"] < kernel_time)
                {
                    kernel_entry = fetch_entry;
                    break;
                    // kernel_time = fetch_entry->miscs["time"];
                }
            }
            std::cout << "Debug point 1" << std::endl;
            if (kernel_entry)
                NNFUSION_CHECK(kernel_entry->tags.find("CudaEmitter") != kernel_entry->tags.end());
            return kernel_entry;
            // if (kernel_entry != nullptr)
            // {
            //     NNFUSION_CHECK(kernel_entry->tags.find("CudaEmitter") != kernel_entry->tags.end());
            //     auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
            //     if (kernel->get_or_emit_source())
            //     {
            //         return std::make_pair(devtype, kernel);
            //     }
            // }
        }
        return nullptr;
    }

    vector<std::shared_ptr<GNode>> get_dot_fuse_option(std::shared_ptr<GNode> dot_node)
    {
        vector<std::shared_ptr<GNode>> fused_op;
        auto succs = find_successors(dot_node);
        if (succs.size() == 0)
            // return
            return vector<std::shared_ptr<GNode>>();
        auto son_node = succs[0];
        if (son_node->get_op_type() == "Add")
        {
            fused_op.push_back(son_node);
            auto grandsons = find_successors(son_node);
            if (grandsons.size() > 0)
            {
                if (grandsons[0]->get_op_type() == "Relu")
                {
                    fused_op.push_back(grandsons[0]);
                }
            }
        }
        else if (son_node->get_op_type() == "Relu")
        {
            fused_op.push_back(son_node);
        }
        return fused_op;
    }
    bool optimize()
    {
        if (!cache_manager->is_valid())
        {
            NNFUSION_LOG(INFO) << "No valid kernel cache, cannot find quantized kernel";
            return true;
        }
        auto gnodes = m_graph->get_ordered_ops();
        for (auto node : gnodes)
        {
            std::cout << "###### Node name:" << node->get_name() << " Unique name"
                      << node->get_unique_name() << "  Type:" << node->get_op_type() << std::endl;
        }
        for (auto node : gnodes)
        {
            if ((*node)["Kernel_Selection_Result"].is_valid())
                // already have a matched kernel
                continue;
            if (!(*node)["DeviceType"].is_valid())
            {
                NNFUSION_CHECK_FAIL()
                    << "GNode DeviceType should be assigned before this pass：" << node->get_name();
            }
            auto n_device_type = (*node)["DeviceType"].as<NNFusion_DeviceType>();
            NNFUSION_CHECK(n_device_type != UNKNOWN);
            std::cout << "Quantize Node name: " << node->get_name()
                      << "  Type: " << node->get_op_type();
            if (node->get_op_type()=="Dot"){
                bool has_bias=false;
                for(auto in_edge: node->get_in_edges()){
                    auto src_node = in_edge->get_src();
                    if(src_node->is_constant()){
                        has_bias=true;
                        std::cout<<" IsLinear:"<<has_bias<<" Shape:"<<src_node->get_output_shape(0);
                        std::cout<<std::endl;
                    }
                }
            }
            std::cout<<std::endl;
            if (quantize_cfg.count(node->get_name()) == 0)
                continue;
            int quantize_bit = quantize_cfg[node->get_name()];
            vector<std::shared_ptr<GNode>> fused_op;
            if (node->get_op_type() == "Dot")
            {
                fused_op = get_dot_fuse_option(node);
            }
            else if (node->get_op_type() == "Conv2d")
            {
            }
            else
            {
                throw "Not supported OP";
            }

            auto kernel_entry = fetch_quantize_kernel(cache_manager, node, n_device_type, fused_op);
            if (kernel_entry == nullptr)
                // No matched kernel found in the kernel cache
                continue;
            // Modify the model graph here according to the quantization config
            if (node->get_op_type() == "Dot")
            {
                // update the model graph
                if (quantize_bit == 8)
                {
                    DotQuantizeOptimize8bit(node, n_device_type, kernel_entry, fused_op);
                }
                else
                {
                }
            }
            else if (node->get_op_type() == "Conv2d")
            {
            }
            // Get the corresponding kernel from the Kernel Cache
        }
        return true;
    }
    void load_from_file(char* ptr, size_t buff_size, string filepath)
    {
        std::ifstream fin(filepath, ios::in|ios::binary);
        fin.read(ptr, buff_size);
    }
            nnfusion::cache::KernelEntry_p
        fetch_kernel(shared_ptr<cache::KernelCacheManager> cache_manager,
                              shared_ptr<GNode> node,
                              NNFusion_DeviceType devtype)
    {
        std::cout << "Fetching kernel for"<<node->get_unique_name() << " " << node->get_name() << std::endl;
        std::shared_ptr<KernelContext> ctx(new KernelContext(node));
        std::string identifier = ctx->generate_identifier();
        std::cout << "Identifier: " << identifier << std::endl;
        
        const std::vector<std::string> SUPPORT_PLATFORM = {"CUDA_GPU"};
        if (identifier != "" &&
            find(SUPPORT_PLATFORM.begin(), SUPPORT_PLATFORM.end(), get_device_str(devtype)) !=
                SUPPORT_PLATFORM.end())
        {
            auto fetched = cache_manager->fetch_all(identifier, get_device_str(devtype));
            nnfusion::cache::KernelEntry_p kernel_entry = nullptr;
            double kernel_time = 1000000000;
            std::cout << "Fetch" << fetched.size() << " Kernels from Kernel Cache!!!!!"
                      << std::endl;
            for (auto fetch_entry : fetched)
            {
                std::cout << "Find Matched quantize kernel" << std::endl;
                if (kernel_entry == nullptr)
                //fetch_entry->miscs["time"] < kernel_time)
                {
                    kernel_entry = fetch_entry;
                    break;
                    // kernel_time = fetch_entry->miscs["time"];
                }
            }
            std::cout << "Debug point 1" << std::endl;
            if (kernel_entry)
                NNFUSION_CHECK(kernel_entry->tags.find("CudaEmitter") != kernel_entry->tags.end());
            return kernel_entry;
            // if (kernel_entry != nullptr)
            // {
            //     NNFUSION_CHECK(kernel_entry->tags.find("CudaEmitter") != kernel_entry->tags.end());
            //     auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
            //     if (kernel->get_or_emit_source())
            //     {
            //         return std::make_pair(devtype, kernel);
            //     }
            // }
        }
        return nullptr;
    }

    void DotQuantizeOptimize8bit(std::shared_ptr<GNode> cur_node,
                                 NNFusion_DeviceType dt,
                                 nnfusion::cache::KernelEntry_p kernel_entry,
                                 vector<std::shared_ptr<GNode>> fused_ops)
    {
        std::cout << "In DotQuantizeOptimize 8bit" << std::endl;
        bool has_constant = false;
        bool has_bias = false;
        bool has_relu = false;
        vector<std::shared_ptr<GNode>> need_remove;
        std::shared_ptr<GNode> add_node = nullptr;
        std::shared_ptr<GNode> bias_broadcast = nullptr;
        std::shared_ptr<GNode> relu_node = nullptr;
        for (auto node : fused_ops)
        {
            if (node->get_op_type() == "Add")
            {
                add_node = node;
                for (auto in_edge : add_node->get_in_edges())
                {
                    auto src_node = in_edge->get_src();
                    if (src_node->is_constant())
                    {
                        has_bias = true;
                        auto ori_bias_weight = src_node;
                        auto bias_related = find_all_predecessors(src_node);
                        need_remove.push_back(add_node);
                        need_remove.push_back(ori_bias_weight);
                        need_remove.insert(
                            need_remove.end(), bias_related.begin(), bias_related.end());
                    }
                    else if (src_node->get_op_type() == "Broadcast")
                    {
                        has_bias = true;
                        bias_broadcast = src_node;
                        auto bias_related = find_all_predecessors(bias_broadcast);
                        //ori_bias_weight = bias_broadcast->get_in_edge(0)->get_src();
                        need_remove.push_back(add_node);
                        need_remove.push_back(bias_broadcast);
                        need_remove.insert(
                            need_remove.end(), bias_related.begin(), bias_related.end());
                    }
                }
            }
            if (node->get_op_type() == "Relu")
            {
                has_relu = true;
                assert(has_bias == true);
                relu_node = node;
                need_remove.push_back(relu_node);
            }
        }

        int need_converter = fneed_converter[cur_node->get_name()];

        if (need_converter){
            std::shared_ptr<GNode> activation_node;
            std::shared_ptr<Edge> activation_edge;
            for(auto in_edge : cur_node->get_in_edges())
            {
                auto src_node = in_edge->get_src();
                if(!src_node->is_constant()){
                    // input activation
                    activation_node = src_node;
                    activation_edge = in_edge;
                }
            }
            int ori_device_id = (*activation_node)["DeviceID"];

            float * convert_scale_integer_data = (float*)malloc(sizeof(float));
            float * convert_scale_shift_data = (float*)malloc(sizeof(float));
            auto convert_scale_integer =
                std::make_shared<op::Constant>(from<float>(),
                                                nnfusion::Shape(vector<size_t>({1})),
                                                static_cast<void*>(convert_scale_integer_data));
            auto convert_scale_integer_node = std::make_shared<GNode>(convert_scale_integer, GNodeVector({}));
            convert_scale_integer->revalidate_and_infer_types(convert_scale_integer_node->shared_from_this());
            convert_scale_integer_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
            convert_scale_integer_node->Set<int>("DeviceID", move(ori_device_id));

            auto convert_scale_shift =
                std::make_shared<op::Constant>(from<float>(),
                                                nnfusion::Shape(vector<size_t>({1})),
                                                static_cast<void*>(convert_scale_shift_data));
            auto convert_scale_shift_node = std::make_shared<GNode>(convert_scale_shift, GNodeVector({}));
            convert_scale_shift->revalidate_and_infer_types(convert_scale_shift_node->shared_from_this());
            convert_scale_shift_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
            convert_scale_shift_node->Set<int>("DeviceID", move(ori_device_id));

            auto converter = std::make_shared<nnfusion::op::BitConverter>(32, 8);
            int src_out = activation_edge->get_src_output();
            int dst_input = activation_edge->get_dst_input();
            m_graph->remove_edge(activation_edge);
            auto convert_input = GNodeVector({activation_node, convert_scale_integer_node, convert_scale_shift_node});
            auto converter_node = std::make_shared<GNode>(converter, convert_input);
            converter_node->set_output_size(1);
            auto shape = activation_node->get_output_shape(src_out);
            converter_node->set_output_type_and_shape(0, from<float>(), shape);
            converter_node->get_op_ptr()->revalidate_and_infer_types(converter_node->shared_from_this());
            converter_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
            converter_node->Set<int>("DeviceID", move(ori_device_id));
            m_graph->add_node(converter_node);
            m_graph->add_node(convert_scale_integer_node);
            m_graph->add_node(convert_scale_shift_node);
            m_graph->add_edge(activation_node, src_out, converter_node, 0);
            m_graph->add_edge(convert_scale_integer_node, 0, converter_node, 1);
            m_graph->add_edge(convert_scale_shift_node, 0, converter_node, 2);
            m_graph->add_edge(converter_node, 0, cur_node, dst_input);
            auto convert_kernel = fetch_kernel(cache_manager, converter_node, dt);
            assert (convert_kernel!=nullptr);
            std::shared_ptr<KernelContext> ctx(new KernelContext(converter_node));
            auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, convert_kernel);
            KernelEmitter::Pointer pkernel = kernel;

            // need to emit the source before bind the kernel
            kernel->get_or_emit_source();
            (*converter_node)["Kernel_Selection_Result"] = std::make_pair(dt, pkernel);
            std::cout << "###############################" << std::endl;
            std::cout << kernel->get_or_emit_source()->body_unit->get_code() << std::endl;
            std::cout << kernel->get_or_emit_source()->signature_unit->get_code() << std::endl;
        }


        // NNFusion_DeviceType dt = nnfusion::get_device_type("CUDA_GPU");
        for (auto in_edge : cur_node->get_in_edges())
        {
            auto src_node = in_edge->get_src();
            if (src_node->is_constant())
            {
                int ori_device_id = (*src_node)["DeviceID"];
                int constant_pos = in_edge->get_dst_input();
                if (constant_pos != 1)
                {
                    NNFUSION_LOG(NNFUSION_WARNING)
                        << "The constant input is the first input of Dot, skip this node";
                    assert(constant_pos == 1);
                    continue;
                }
                auto weight_constant =
                    std::dynamic_pointer_cast<nnfusion::op::Constant>(src_node->get_op_ptr());
                auto w_shape = weight_constant->get_shape();
                size_t weight_count = 1, out_count = 1;
                for (int i : w_shape)
                    weight_count *= i;
                auto out_shape = cur_node->get_output_shape(0);
                for (int i : out_shape)
                    out_count *= i;
                // TODO unique_name vs name
                int quantize_bit = quantize_cfg[cur_node->get_name()];
                // we filled the ramdom data temporarily
                // float* quan_weight_data = (float*)malloc(sizeof(float) * weight_count);
                float* block_weight_rows = (float*)malloc(sizeof(float) * weight_count);
                float* block_weight_cols = (float*)malloc(sizeof(float) * weight_count);
                float* block_weight_values = (float*) malloc(sizeof(float)*weight_count);
                load_from_file((char*)block_weight_rows, weight_count, csr_row[cur_node->get_name()]);
                load_from_file((char*)block_weight_cols, weight_count, csr_col[cur_node->get_name()]);
                load_from_file((char*)block_weight_values, weight_count, csr_values[cur_node->get_name()]);

                float* scale_integer_data = (float*)malloc(sizeof(float));
                float* scale_shift_data = (float*)malloc(sizeof(float));
                float* bias_data =
                    (float*)malloc(sizeof(float) * weight_count); // TODO use the correct size here
                float* output_zp_data = (float*)malloc(sizeof(float));
                auto dense_op = std::dynamic_pointer_cast<op::Dot>(cur_node->get_op_ptr());
                // quantized weight of the weight
                // The values is not right and will be filled after nnfusion.
                auto weight_values = std::make_shared<op::Constant>(
                    from<float>(), nnfusion::Shape(w_shape), static_cast<void*>(block_weight_values));
                auto weight_values_node = std::make_shared<GNode>(weight_values, GNodeVector({}));
                //update the output
                weight_values_node->get_op_ptr()->revalidate_and_infer_types(
                    weight_values_node->shared_from_this());
                weight_values_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
                weight_values_node->Set<int>("DeviceID", move(ori_device_id));
              
                auto weight_row = std::make_shared<op::Constant>(
                    from<float>(), nnfusion::Shape(w_shape), static_cast<void*>(block_weight_rows));
                auto weight_row_node = std::make_shared<GNode>(weight_row, GNodeVector({}));
                //update the output
                weight_row_node->get_op_ptr()->revalidate_and_infer_types(
                    weight_row_node->shared_from_this());
                weight_row_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
                weight_row_node->Set<int>("DeviceID", move(ori_device_id));

                auto weight_col = std::make_shared<op::Constant>(
                    from<float>(), nnfusion::Shape(w_shape), static_cast<void*>(block_weight_cols));
                auto weight_col_node = std::make_shared<GNode>(weight_col, GNodeVector({}));
                //update the output
                weight_col_node->get_op_ptr()->revalidate_and_infer_types(
                    weight_col_node->shared_from_this());
                weight_col_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
                weight_col_node->Set<int>("DeviceID", move(ori_device_id));


                auto scale_integer =
                    std::make_shared<op::Constant>(from<float>(),
                                                   nnfusion::Shape(vector<size_t>({1})),
                                                   static_cast<void*>(scale_integer_data));
                auto scale_integer_node = std::make_shared<GNode>(scale_integer, GNodeVector({}));
                scale_integer->revalidate_and_infer_types(scale_integer_node->shared_from_this());
                scale_integer_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
                scale_integer_node->Set<int>("DeviceID", move(ori_device_id));

                auto scale_shift =
                    std::make_shared<op::Constant>(from<float>(),
                                                   nnfusion::Shape(vector<size_t>({1})),
                                                   static_cast<void*>(scale_shift_data));
                auto scale_shift_node = std::make_shared<GNode>(scale_shift, GNodeVector({}));
                scale_shift->revalidate_and_infer_types(scale_shift_node->shared_from_this());
                scale_shift_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
                scale_shift_node->Set<int>("DeviceID", move(ori_device_id));

                auto activate_node = cur_node->get_in_edge(1 - constant_pos)->get_src();
                GNodeVector input_gv({activate_node,
                                      weight_values_node,
                                      weight_row_node,
                                      weight_col_node,
                                      scale_integer_node,
                                      scale_shift_node});

                m_graph->add_node(weight_values_node);
                m_graph->add_node(weight_row_node);
                m_graph->add_node(weight_col_node);

                m_graph->add_node(scale_integer_node);
                m_graph->add_node(scale_shift_node);
                // Handle the fuse option here
                if (has_bias)
                {
                    auto bias_shape = nnfusion::Shape(vector<size_t>(
                        {weight_count})); // TODO currently the memory space for bias is wasted
                    auto bias = std::make_shared<op::Constant>(
                        from<float>(), bias_shape, static_cast<void*>(bias_data));
                    auto bias_node = std::make_shared<GNode>(bias, GNodeVector({}));
                    bias->revalidate_and_infer_types(bias_node->shared_from_this());
                    bias_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
                    bias_node->Set<int>("DeviceID", move(ori_device_id));
                    input_gv.push_back(bias_node);
                    m_graph->add_node(bias_node);
                }

                if (has_relu)
                {
                    auto output_zp =
                        std::make_shared<op::Constant>(from<float>(),
                                                       nnfusion::Shape(vector<size_t>({1})),
                                                       static_cast<void*>(output_zp_data));
                    auto output_zp_node = std::make_shared<GNode>(output_zp, GNodeVector({}));
                    output_zp->revalidate_and_infer_types(output_zp_node->shared_from_this());
                    output_zp_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
                    output_zp_node->Set<int>("DeviceID", move(ori_device_id));
                    input_gv.push_back(output_zp_node);
                    m_graph->add_node(output_zp_node);
                }

                auto quan_dot = std::make_shared<op::QuantizeDot>(dense_op, quantize_bit);

                auto quan_dot_node = std::make_shared<GNode>(quan_dot, input_gv);
                quan_dot_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
                quan_dot_node->Set<int>("DeviceID", move(ori_device_id));
                /// Remember after set the input node vector, we still need to set the edge manually!
                for (int i = 0; i < input_gv.size(); i++)
                {
                    m_graph->add_edge(input_gv.at(i), 0, quan_dot_node, i);
                }

                // replace node will revalidate and infer the output tensor
                auto last_node = cur_node;
                if (fused_ops.size())
                    last_node = fused_ops[fused_ops.size() - 1];

                auto ori_outputs = last_node->get_outputs();
                //???
                for (int i = 0; i < ori_outputs.size(); i++)
                {
                    quan_dot_node->set_output(i, ori_outputs[i]);
                }

                m_graph->replace_node(last_node, quan_dot_node, false);
                m_graph->remove_node(src_node);
                need_remove.push_back(cur_node);
                for (auto tmp_node : need_remove)
                {
                    std::cout << " Removing " << tmp_node->get_name() << " "
                              << tmp_node->get_op_type() << std::endl;
                }
                for (auto tmp_node : need_remove)
                {
                    if (tmp_node != last_node)
                    {
                        m_graph->remove_node(tmp_node);
                    }
                }
                // if(cur_node!=last_node) m_graph->remove_node(cur_node);
                // if(has_bias){
                //     m_graph->remove_node(ori_bias_weight);
                //     if(bias_broadcast)
                // 	    m_graph->remove_node(bias_broadcast);
                //     if(last_node!=add_node) m_graph->remove_node(add_node);
                // }
                // if(has_relu){
                //     if(last_node!=relu_node) m_graph->remove_node(relu_node);
                // }

                // Bind the fetched kernel here with the new kernel context
                std::shared_ptr<KernelContext> ctx(new KernelContext(quan_dot_node));
                auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
                KernelEmitter::Pointer pkernel = kernel;

                // need to emit the source before bind the kernel
                kernel->get_or_emit_source();
                (*quan_dot_node)["Kernel_Selection_Result"] = std::make_pair(dt, pkernel);
                std::cout << "###############################" << std::endl;
                std::cout << kernel->get_or_emit_source()->body_unit->get_code() << std::endl;
                std::cout << kernel->get_or_emit_source()->signature_unit->get_code() << std::endl;
                //exit(-1);
                std::cout << "Bind the Quantized kernel!" << std::endl;
                has_constant = true;
                break;
            }
        }
    }

private:
    std::shared_ptr<Graph> m_graph;
    std::string cfg_path;
    std::map<string, int> quantize_cfg;
    std::map<string, string> csr_row;
    std::map<string, string> csr_col;
    std::map<string, string> csr_values;
    std::map<string, int> fneed_converter;
    std::shared_ptr<nnfusion::cache::KernelCacheManager> cache_manager;
};

bool BlockQuantizeKernelPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool enable_quantize_kernel = FLAGS_fblockquantize_cfg.size() > 0;
    if (!enable_quantize_kernel)
        return true;
    NNFUSION_LOG(INFO) << "Enable the BlockQuantized kernels";
    BlockQuantizeKernelOptimizer optimizer(graph, FLAGS_fblockquantize_cfg);
    return optimizer.optimize();
}
