// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "spargen_pass.hpp"
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

DEFINE_string(fspargen_cfg, "", "Configuration to enable the SparGen optimization");
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::element;
using namespace nnfusion;
using namespace std;

class SparGenOptimizer
{
public:
    SparGenOptimizer(std::shared_ptr<Graph> g, const std::string& cfg_path)
    {
        this->m_graph = g;
        this->cfg_path = cfg_path;
        parse_cfg();
        this->cache_manager = std::make_shared<nnfusion::cache::KernelCacheManager>();
    }
    void parse_cfg()
    {
        // each line configures a kernel in the format of
        // "TesaID SparsityType Kernel_ID Parameters_for_corresponding sparsity pattern"
        ifstream cfg_file(this->cfg_path.c_str());
        assert(cfg_file.good());
        std::string line;
        while (std::getline(cfg_file, line))
        {
            std::istringstream iss(line);
            int tesa_id;
            string sparse_type, kernel_id;
            iss >> tesa_id >> sparse_type >> kernel_id;
            this->sparse_type[tesa_id] = sparse_type;
            this->kernel_id[tesa_id] = kernel_id;

            if (sparse_type == "BlockSparse")
            {
                string row_f, col_f, value_f, bias_f;
                iss >> row_f >> col_f >> value_f >> bias_f;
                this->csr_rows[tesa_id] = row_f;
                this->csr_cols[tesa_id] = col_f;
                this->csr_values[tesa_id] = value_f;
                this->bias_data_path[tesa_id] = bias_f;
            }
            else if (sparse_type == "BlockQuantize")
            {
                int in_bit, out_bit;
                iss >> in_bit >> out_bit;
                this->in_quan_bit[tesa_id] = in_bit;
                this->out_quan_bit[tesa_id] = out_bit;
                string row_f, col_f, value_f, scale_integer_f, scale_shift_f, bias_f;
                iss >> row_f >> col_f >> value_f >> scale_integer_f >> scale_shift_f >> bias_f;
                this->csr_rows[tesa_id] = row_f;
                this->csr_cols[tesa_id] = col_f;
                this->csr_values[tesa_id] = value_f;
                this->scale_integer[tesa_id] = scale_integer_f;
                this->scale_shift[tesa_id] = scale_shift_f;
                this->bias_data_path[tesa_id] = bias_f;
            }
            else if (sparse_type == "Quantize")
            {
                int in_bit, out_bit;
                iss >> in_bit >> out_bit;
                this->in_quan_bit[tesa_id] = in_bit;
                this->out_quan_bit[tesa_id] = out_bit;
                string q_weight, scale_integer_f, scale_shift_f, bias_f;
                iss >> q_weight >> scale_integer_f >> scale_shift_f >> bias_f;
                // quantized weight
                this->weight_data_path[tesa_id] = q_weight;
                this->scale_integer[tesa_id] = scale_integer_f;
                this->scale_shift[tesa_id] = scale_shift_f;
                this->bias_data_path[tesa_id] = bias_f;
            }
            else
            {
                throw std::invalid_argument("Not supported Sparse Type");
            }
        }
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
            if ((*node)["Kernel_Selection_Result"].is_valid())
                continue;
            if (!(*node)["DeviceType"].is_valid())
            {
                NNFUSION_CHECK_FAIL()
                    << "GNode DeviceType should be assigned before this pass：" << node->get_name();
            }
            auto n_device_type = (*node)["DeviceType"].as<NNFusion_DeviceType>();
            NNFUSION_CHECK(n_device_type != UNKNOWN);
            if ( (*node)["TESAID"].is_valid() && (*node)["TESAID"].as<int>()>0 )
            {
                std::cout << "SparGen!!! " << node->get_name() << " " << (*node)["TESAID"].as<int>()
                          << std::endl;
                // build the map
                int tesaid = (*node)["TESAID"].as<int>();
                this->name2tesaid[node->get_name()] = tesaid;
                this->tesaid2name[tesaid] = node->get_name();
                optimize_kernel(node);
            }
        }
        // std::cout<<"Exit the SparGen optimize flow"<<std::endl;
        // exit(-1);
        return true;
    }

    void optimize_kernel(std::shared_ptr<GNode> target_node)
    {
        NNFUSION_LOG(INFO) << "Optimize the Node " << target_node->get_name() << " by SparGen  Op_type:" << target_node->get_op_type();
        if (target_node->get_op_type() == "Dot")
        {
            DotOptimize(target_node);
        }
        else if (target_node->get_op_type() == "Convolution")
        {
            ConvOptimize(target_node);
        }
        else if (target_node->get_op_type() == "DepthwiseConv2dNative")
        {
            DepthConvOptimize(target_node);
        }
    }
    void ConvOptimize(std::shared_ptr<GNode> conv_node)
    {
        std::cout << "In ConvOptimize" << std::endl;
        assert(conv_node->get_op_type() == "Convolution");
        int tesa_id = (*conv_node)["TESAID"].as<int>();
        std::string sparse_type = this->sparse_type[tesa_id];
        vector<std::shared_ptr<GNode>> fusible_nodes = get_conv_fusible_nodes(conv_node);
        std::string identifier = this->kernel_id[tesa_id];
        auto n_device_type = (*conv_node)["DeviceType"].as<NNFusion_DeviceType>();
        auto kernel_entry = fetch_kernel(this->cache_manager, identifier, n_device_type);
        if (kernel_entry == nullptr)
            return;
        if (sparse_type == "Quantize")
        {
            ConvQuantizeOptimize(conv_node, kernel_entry, fusible_nodes, n_device_type);
        }
        else
        {
            throw std::invalid_argument("Not supported Sparse Type");
        }
        
    }
    void DepthConvOptimize(std::shared_ptr<GNode> conv_node)
    {
        std::cout << "In DepthWise ConvOptimize" << std::endl;
        assert(conv_node->get_op_type() == "DepthwiseConv2dNative");
        int tesa_id = (*conv_node)["TESAID"].as<int>();
        std::string sparse_type = this->sparse_type[tesa_id];
        vector<std::shared_ptr<GNode>> fusible_nodes = get_depth_conv_fusible_nodes(conv_node);
        std::string identifier = this->kernel_id[tesa_id];
        auto n_device_type = (*conv_node)["DeviceType"].as<NNFusion_DeviceType>();
        auto kernel_entry = fetch_kernel(this->cache_manager, identifier, n_device_type);
        if (kernel_entry == nullptr)
            return;
        if (sparse_type == "Quantize")
        {
            DepthConvQuantizeOptimize(conv_node, kernel_entry, fusible_nodes, n_device_type);
        }
        else
        {
            throw std::invalid_argument("Not supported Sparse Type");
        }
    }

    void DotOptimize(std::shared_ptr<GNode> dot_node)
    {
        assert(dot_node->get_op_type() == "Dot");
        int tesa_id = (*dot_node)["TESAID"].as<int>();
        std::string sparse_type = this->sparse_type[tesa_id];
        vector<std::shared_ptr<GNode>> fusible_nodes = get_dot_fusible_nodes(dot_node);
        std::string identifier = this->kernel_id[tesa_id];
        auto n_device_type = (*dot_node)["DeviceType"].as<NNFusion_DeviceType>();
        if (sparse_type == "BlockSparse")
        {
            auto kernel_entry = fetch_kernel(this->cache_manager, identifier, n_device_type);
            if (kernel_entry == nullptr)
                return;
            BlockDotOptimize(dot_node, kernel_entry, fusible_nodes, n_device_type);
        }
        else if (sparse_type == "BlockQuantize")
        {
            auto kernel_entry = fetch_kernel(this->cache_manager, identifier, n_device_type);
            if (kernel_entry == nullptr)
                return;
            BlockQuantizeDotOptimize(dot_node, kernel_entry, fusible_nodes, n_device_type);
        }
        else
        {
            std::cout << "Skip this Dot node:" << tesa_id << std::endl;
            // throw std::invalid_argument("Not supported Sparse Type");
        }
    }

private:
    void insert_converter(std::shared_ptr<GNode> node, int in_bit, int out_bit)
    {
        // TODO complete this function
        return;
    }
    void ConvQuantizeOptimize(std::shared_ptr<GNode> cur_node,
                              nnfusion::cache::KernelEntry_p kernel_entry,
                              vector<std::shared_ptr<GNode>> fused_ops,
                              NNFusion_DeviceType dt)
    {
        int kernel_h = cur_node->get_input_shape(1)[2];
        int kernel_w = cur_node->get_input_shape(1)[3];
        if (kernel_h == 1 && kernel_w == 1)
        {
            // Optimize for Conv1x1
            Conv1x1QuantizeOptimize(cur_node, kernel_entry, fused_ops, dt);
        }
    }

    void Conv1x1QuantizeOptimize(std::shared_ptr<GNode> cur_node,
                                 nnfusion::cache::KernelEntry_p kernel_entry,
                                 vector<std::shared_ptr<GNode>> fused_ops,
                                 NNFusion_DeviceType dt)
    {
        std::cout << "In Conv1x1 Quantize Optimization" << std::endl;
        int ori_device_id = (*cur_node)["DeviceID"];
        int tesaid = (*cur_node)["TESAID"].as<int>();
        int quan_bit = this->out_quan_bit[tesaid];
        vector<std::shared_ptr<GNode>> need_remove;
        vector<std::shared_ptr<GNode>> input_gv;
        auto activation_node = cur_node->get_in_edge(0)->get_src();
        auto weight_node = cur_node->get_in_edge(1)->get_src();
        auto weight_shape = cur_node->get_input_shape(1);
        auto output_shape = cur_node->get_output_shape(0);

        size_t weight_count = 1, output_count=1;
        for (auto i:weight_shape)
            weight_count *= i;
        for (auto i:output_shape)
            output_count *= i;
        auto weight_constant = std::dynamic_pointer_cast<nnfusion::op::Constant>(weight_node->get_op_ptr());
        char * weight_data_ptr = (char*)weight_constant->get_data_ptr();
        load_from_file(weight_data_ptr, sizeof(float)*weight_count, this->weight_data_path[tesaid]);

        input_gv.push_back(activation_node);
        input_gv.push_back(weight_node);
        int tmpvalue;
        // These three parameters are abandoned in the current version of kernel
        auto w_mul_zp_node = create_constant_node(dt, ori_device_id, tmpvalue);
        auto w_zp_node = create_constant_node(dt, ori_device_id, tmpvalue);
        auto zp_acc_node = create_constant_node(dt, ori_device_id, tmpvalue);
        float * scale_integer_data, *scale_shift_data, *bias_data;
        scale_integer_data = (float*) malloc(sizeof(float));
        scale_shift_data = (float*) malloc(sizeof(float));
        bias_data = (float*)malloc(sizeof(float)*output_count);
        memset(bias_data, 0, sizeof(float)*output_count);
    
        auto scale_integer_node = create_constant_node(dt, ori_device_id, *((int*)scale_integer_data));
        auto scale_shift_node = create_constant_node(dt, ori_device_id, *((int*)scale_shift_data));
        auto bias_node = create_constant_node(dt, ori_device_id, output_shape, bias_data);
    
        input_gv.push_back(w_mul_zp_node);
        input_gv.push_back(w_zp_node);
        input_gv.push_back(zp_acc_node);
        input_gv.push_back(scale_integer_node);
        input_gv.push_back(scale_shift_node);
        input_gv.push_back(bias_node);
        for (int i=2;i<input_gv.size();i++)
            m_graph->add_node(input_gv[i]);
        auto conv1x1 = std::make_shared<op::QuantizeConv1x1>(quan_bit);
        auto conv1x1_node = std::make_shared<GNode>(conv1x1, input_gv);
        conv1x1_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
        conv1x1_node->Set<int>("DeviceID", move(ori_device_id));
        for(int i=0;i<input_gv.size();i++){
            m_graph->add_edge(input_gv.at(i), 0, conv1x1_node, i);
        }
        auto last_node = cur_node;
        if(fused_ops.size())
            last_node = fused_ops[fused_ops.size()-1];
        
        auto ori_outputs = last_node->get_outputs();
        //???
        for (int i = 0; i < ori_outputs.size(); i++)
        {
            conv1x1_node->set_output(i, ori_outputs[i]);
        }
        fused_ops.push_back(cur_node);
        m_graph->replace_node(last_node, conv1x1_node, false);
        for(auto tmp_node:fused_ops){
            if(tmp_node!=last_node){
                m_graph->remove_node(tmp_node);
            }
        }
        std::shared_ptr<KernelContext> ctx(new KernelContext(conv1x1_node));
        auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
        KernelEmitter::Pointer pkernel = kernel;

        // need to emit the source before bind the kernel
        kernel->get_or_emit_source();
        (*conv1x1_node)["Kernel_Selection_Result"] = std::make_pair(dt, pkernel);
        std::cout << "###############################" << std::endl;
        // std::cout << kernel->get_or_emit_source()->body_unit->get_code() << std::endl;
        // std::cout << kernel->get_or_emit_source()->signature_unit->get_code() << std::endl;
        //exit(-1);
        std::cout << "Bind the Quantized kernel!" << std::endl;


    }

    void DepthConvQuantizeOptimize(std::shared_ptr<GNode> cur_node,
                                   nnfusion::cache::KernelEntry_p kernel_entry,
                                   vector<std::shared_ptr<GNode>> fused_ops,
                                   NNFusion_DeviceType dt)
    {
        std::cout << "In DepthConvQuantizeOptimize" << std::endl;
        vector<std::shared_ptr<GNode>> need_remove;
        int ori_device_id = (*cur_node)["DeviceID"];
        bool has_bias = false;
        bool has_relu = false;
        int tesaid = (*cur_node)["TESAID"].as<int>();
        int quan_bit = this->out_quan_bit[tesaid];
        int need_converter = this->need_converter[tesaid];
        if (need_converter)
        {
            // TODO get the input and output bit number for the converter.
            int in_bit = 32, out_bit = 8;
            insert_converter(cur_node, in_bit, out_bit);
        }
        vector<std::shared_ptr<GNode>> input_gv;
        auto activation_node = cur_node->get_in_edge(0)->get_src();

        auto cur_op = cur_node->get_op_ptr();
        auto _op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(cur_op);
        const auto& dilation_h = int64_t(_op->localOpConfig.getRoot()["dilations"][0]);
        const auto& dilation_w = int64_t(_op->localOpConfig.getRoot()["dilations"][1]);
        const auto& stride_h = int64_t(_op->localOpConfig.getRoot()["strides"][0]);
        const auto& stride_w = int64_t(_op->localOpConfig.getRoot()["strides"][1]);
        const auto& padding_h = int64_t(_op->localOpConfig.getRoot()["padding_before"][0]);
        const auto& padding_w = int64_t(_op->localOpConfig.getRoot()["padding_before"][1]);
        const auto& kernel_size_h = cur_node->get_input_shape(1)[2];
        const auto& kernel_size_w = cur_node->get_input_shape(1)[3];
        const auto& in_shape = cur_node->get_input_shape(0);
        const auto& out_shape = cur_node->get_output_shape(0);
        const auto& channels = out_shape[1]; //NCHW
        // activation node
        input_gv.push_back(cur_node->get_in_edge(0)->get_src());
        // delete the original weight node
        // need_remove.push_back(cur_node->get_in_edge(1)->get_src());
        m_graph->remove_node(cur_node->get_in_edge(1)->get_src());
        float *q_weight, *q_bias, *scale_integer_data, *scale_shift_data;
        auto w_shape = cur_node->get_input_shape(1);
        size_t weight_count = 1;
        for (int i : w_shape)
            weight_count *= i;
        q_weight = (float*)malloc(sizeof(float) * weight_count);
        memset(q_weight, 0, sizeof(float) * weight_count);
        load_from_file(
            (char*)q_weight, sizeof(float) * weight_count, this->weight_data_path[tesaid]);
        auto weight_node =
            create_constant_node(dt, ori_device_id, w_shape, q_weight);

        q_bias = (float*)malloc(sizeof(float) * channels);
        memset(q_bias, 0, sizeof(float) * channels);
        load_from_file((char*)q_bias, sizeof(float) * channels, this->bias_data_path[tesaid]);
        vector<size_t> bias_shape({1, channels});
        auto bias_node = create_constant_node(dt, ori_device_id, bias_shape, q_bias);

        scale_integer_data = (float*)malloc(sizeof(float));
        memset(scale_integer_data, 0, sizeof(float));
        auto integer_node = create_constant_node(dt, ori_device_id, *((int*)scale_integer_data));

        scale_shift_data = (float*)malloc(sizeof(float));
        memset(scale_shift_data, 0, sizeof(float));
        auto shift_node = create_constant_node(dt, ori_device_id, *((int*)scale_shift_data));
        input_gv.push_back(weight_node);
        input_gv.push_back(bias_node);
        input_gv.push_back(integer_node);
        input_gv.push_back(shift_node);

        for (int i = 1; i < input_gv.size(); i++)
            m_graph->add_node(input_gv[i]);

        auto quan_depth_conv = std::make_shared<op::QuantizeDepthwiseConv2dNative>(quan_bit);
        auto quan_conv_node = std::make_shared<GNode>(quan_depth_conv, input_gv);
        quan_conv_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
        quan_conv_node->Set<int>("DeviceID", move(ori_device_id));

        for (int i = 0; i < input_gv.size(); i++)
        {
            m_graph->add_edge(input_gv.at(i), 0, quan_conv_node, i);
        }
        auto last_node = cur_node;
        if (fused_ops.size())
            last_node = fused_ops[fused_ops.size() - 1];
        auto ori_outputs = last_node->get_outputs();
        for (int i = 0; i < ori_outputs.size(); i++)
        {
            quan_conv_node->set_output(i, ori_outputs[i]);
        }
        fused_ops.push_back(cur_node);
        m_graph->replace_node(last_node, quan_conv_node, false);
        for (auto tmp_node : fused_ops)
        {
            if (tmp_node != last_node)
            {
                m_graph->remove_node(tmp_node);
            }
        }
        std::shared_ptr<KernelContext> ctx(new KernelContext(quan_conv_node));
        auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
        KernelEmitter::Pointer pkernel = kernel;

        // need to emit the source before bind the kernel
        kernel->get_or_emit_source();
        (*quan_conv_node)["Kernel_Selection_Result"] = std::make_pair(dt, pkernel);
        std::cout << "###############################" << std::endl;
        std::cout << "Bind the Quantized Depthwise kernel!" << std::endl;
    }
    void BlockQuantizeDotOptimize(std::shared_ptr<GNode> dot_node,
                                  nnfusion::cache::KernelEntry_p kernel_entry,
                                  vector<std::shared_ptr<GNode>> fusible_nodes,
                                  NNFusion_DeviceType n_device_type)
    {
        std::cout << "In SparGen BlockDotOptimize" << std::endl;
        assert(kernel_entry != nullptr);
        assert(dot_node != nullptr);

        bool has_constant = false;
        bool has_bias = false;
        bool has_relu = false;
        vector<shared_ptr<GNode>> need_remove;
        shared_ptr<GNode> add_node = nullptr;
        shared_ptr<GNode> bias_broadcast = nullptr;
        shared_ptr<GNode> relu_node = nullptr;
        int tesaid = (*dot_node)["TESAID"].as<int>();
        for (auto node : fusible_nodes)
        {
            if (node->get_op_type() == "Add")
            {
                add_node = node;
                has_bias = true;
                for (auto in_edge : add_node->get_in_edges())
                {
                    auto src_node = in_edge->get_src();
                    if (src_node->is_constant())
                    {
                        auto ori_bias_weight = src_node;
                        auto bias_related = find_all_predecessors(src_node);
                        need_remove.push_back(add_node);
                        need_remove.push_back(ori_bias_weight);
                        need_remove.insert(
                            need_remove.end(), bias_related.begin(), bias_related.end());
                    }
                    else if (src_node->get_op_type() == "Broadcast")
                    {
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
            else if (node->get_op_type() == "Relu")
            {
                has_relu = true;
                assert(has_bias = true);
                relu_node = node;
                need_remove.push_back(relu_node);
            }
        }
        // TODO: complete the pass to get the flag of whethers need a converter.

        int need_converter = this->need_converter[this->name2tesaid[dot_node->get_name()]];
        if (need_converter)
        {
            std::shared_ptr<GNode> activation_node;
            std::shared_ptr<Edge> activation_edge;
            for (auto in_edge : dot_node->get_in_edges())
            {
                auto src_node = in_edge->get_src();
                if (!src_node->is_constant())
                {
                    // input activation
                    activation_node = src_node;
                    activation_edge = in_edge;
                }
            }
            int ori_device_id = (*activation_node)["DeviceID"];
            // TODO: load the specific value according to the config
            float* convert_scale_integer_data = (float*)malloc(sizeof(float));
            float* convert_scale_shift_data = (float*)malloc(sizeof(float));
            int tmp_value; // currently use a random value, value doesn't affect the speed
            auto convert_scale_integer_node =
                create_constant_node(n_device_type, ori_device_id, tmp_value);
            auto convert_scale_shift_node =
                create_constant_node(n_device_type, ori_device_id, tmp_value);
            // TODO: load the specific convert bit configuration accordingly
            auto converter = std::make_shared<nnfusion::op::BitConverter>(32, 8);
            int src_out = activation_edge->get_src_output();
            int dst_input = activation_edge->get_dst_input();
            m_graph->remove_edge(activation_edge);
            auto convert_input = GNodeVector(
                {activation_node, convert_scale_integer_node, convert_scale_shift_node});
            auto converter_node = std::make_shared<GNode>(converter, convert_input);
            converter_node->set_output_size(1);
            auto shape = activation_node->get_output_shape(src_out);
            converter_node->set_output_type_and_shape(0, from<float>(), shape);
            converter_node->get_op_ptr()->revalidate_and_infer_types(
                converter_node->shared_from_this());
            converter_node->Set<NNFusion_DeviceType>("DeviceType", move(n_device_type));
            converter_node->Set<int>("DeviceID", move(ori_device_id));
            m_graph->add_node(converter_node);
            m_graph->add_node(convert_scale_integer_node);
            m_graph->add_node(convert_scale_shift_node);
            m_graph->add_edge(activation_node, src_out, converter_node, 0);
            m_graph->add_edge(convert_scale_integer_node, 0, converter_node, 1);
            m_graph->add_edge(convert_scale_shift_node, 0, converter_node, 2);
            m_graph->add_edge(converter_node, 0, dot_node, dst_input);

            /////////////////////////////////////////////////////
            std::string convert_identifier =
                this->convert_identifier[this->name2tesaid[dot_node->get_name()]];
            auto convert_kernel = fetch_kernel(cache_manager, convert_identifier, n_device_type);
            assert(convert_kernel != nullptr);
            std::shared_ptr<KernelContext> ctx(new KernelContext(converter_node));
            auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, convert_kernel);
            KernelEmitter::Pointer pkernel = kernel;

            // need to emit the source before bind the kernel
            kernel->get_or_emit_source();
            (*converter_node)["Kernel_Selection_Result"] = std::make_pair(n_device_type, pkernel);
            // std::cout << "###############################" << std::endl;
            // std::cout << kernel->get_or_emit_source()->body_unit->get_code() << std::endl;
            // std::cout << kernel->get_or_emit_source()->signature_unit->get_code() << std::endl;
        }
        // Start to replace the block sparse kernel
        auto src_node = dot_node->get_in_edge(1)->get_src();
        if (!src_node->is_constant())
            return;

        int ori_device_id = (*src_node)["DeviceID"];

        auto weight_constant =
            std::dynamic_pointer_cast<nnfusion::op::Constant>(src_node->get_op_ptr());
        auto w_shape = weight_constant->get_shape();
        size_t weight_count = 1, out_count = 1;
        for (int i : w_shape)
            weight_count *= i;
        auto out_shape = dot_node->get_output_shape(0);
        for (int i : out_shape)
            out_count *= i;

        // we filled the ramdom data temporarily
        // float* quan_weight_data = (float*)malloc(sizeof(float) * weight_count);
        float* block_weight_rows = (float*)malloc(sizeof(float) * (w_shape[0] + 1));
        memset(block_weight_rows, 0, sizeof(float) * (w_shape[0] + 1));
        float* block_weight_cols = (float*)malloc(sizeof(float) * weight_count);
        memset(block_weight_cols, 0, sizeof(float) * weight_count);
        float* block_weight_values = (float*)malloc(sizeof(float) * weight_count);
        memset(block_weight_values, 0, sizeof(float) * weight_count);
        float* scale_integer_data = (float*)malloc(sizeof(float));
        memset(scale_integer_data, 0, sizeof(float));
        float* scale_shift_data = (float*)malloc(sizeof(float));
        memset(scale_shift_data, 0, sizeof(float));

        load_from_file(
            (char*)block_weight_rows, sizeof(float) * (w_shape[0] + 1), this->csr_rows[tesaid]);
        load_from_file(
            (char*)block_weight_cols, sizeof(float) * weight_count, this->csr_cols[tesaid]);
        load_from_file(
            (char*)block_weight_values, sizeof(float) * weight_count, this->csr_values[tesaid]);
        load_from_file((char*)scale_integer_data, sizeof(float), this->scale_integer[tesaid]);
        load_from_file((char*)scale_shift_data, sizeof(float), this->scale_shift[tesaid]);
        // TODO load the right value according to the config

        float* bias_data =
            (float*)malloc(sizeof(float) * weight_count); // TODO use the correct size here
        memset(bias_data, 0, sizeof(float) * weight_count);
        auto dense_op = std::dynamic_pointer_cast<op::Dot>(dot_node->get_op_ptr());
        auto weight_values_node =
            create_constant_node(n_device_type, ori_device_id, w_shape, block_weight_values);
        auto weight_row_node = create_constant_node(
            n_device_type, ori_device_id, vector<size_t>({w_shape[0] + 1}), block_weight_rows);
        auto weight_col_node =
            create_constant_node(n_device_type, ori_device_id, w_shape, block_weight_cols);
        auto scale_integer_node =
            create_constant_node(n_device_type, ori_device_id, *((int*)scale_integer_data));
        auto scale_shift_node =
            create_constant_node(n_device_type, ori_device_id, *((int*)scale_shift_data));

        auto activate_node = dot_node->get_in_edge(0)->get_src();
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
            // TODO also load the correct bias weights
            auto bias = std::make_shared<op::Constant>(
                from<float>(), bias_shape, static_cast<void*>(bias_data));
            if (this->bias_data_path[tesaid].size() > 0)
            {
                load_from_file(
                    (char*)bias_data, sizeof(float) * weight_count, this->bias_data_path[tesaid]);
            }
            auto bias_node = std::make_shared<GNode>(bias, GNodeVector({}));
            bias->revalidate_and_infer_types(bias_node->shared_from_this());
            bias_node->Set<NNFusion_DeviceType>("DeviceType", move(n_device_type));
            bias_node->Set<int>("DeviceID", move(ori_device_id));
            input_gv.push_back(bias_node);
            m_graph->add_node(bias_node);
        }
        auto quan_dot = std::make_shared<op::QuantizeDot>(dense_op, this->out_quan_bit[tesaid]);
        // auto sparse_dot = std::make_shared<op::SparseDot>(dense_op);
        // auto quan_dot = std::make_shared<op::QuantizeDot>(dense_op, quantize_bit);

        auto sparse_dot_node = std::make_shared<GNode>(quan_dot, input_gv);
        sparse_dot_node->Set<NNFusion_DeviceType>("DeviceType", move(n_device_type));
        sparse_dot_node->Set<int>("DeviceID", move(ori_device_id));
        /// Remember after set the input node vector, we still need to set the edge manually!
        for (int i = 0; i < input_gv.size(); i++)
        {
            m_graph->add_edge(input_gv.at(i), 0, sparse_dot_node, i);
        }

        // replace node will revalidate and infer the output tensor
        auto last_node = dot_node;
        if (fusible_nodes.size())
            last_node = fusible_nodes[fusible_nodes.size() - 1];

        auto ori_outputs = last_node->get_outputs();
        //???
        for (int i = 0; i < ori_outputs.size(); i++)
        {
            sparse_dot_node->set_output(i, ori_outputs[i]);
        }

        m_graph->replace_node(last_node, sparse_dot_node, false);
        m_graph->remove_node(src_node);
        need_remove.push_back(dot_node);
        for (auto tmp_node : need_remove)
        {
            std::cout << " Removing " << tmp_node->get_name() << " " << tmp_node->get_op_type()
                      << std::endl;
        }
        for (auto tmp_node : need_remove)
        {
            if (tmp_node != last_node)
            {
                m_graph->remove_node(tmp_node);
            }
        }

        // Bind the fetched kernel here with the new kernel context
        std::shared_ptr<KernelContext> ctx(new KernelContext(sparse_dot_node));
        auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
        KernelEmitter::Pointer pkernel = kernel;

        // need to emit the source before bind the kernel
        kernel->get_or_emit_source();
        (*sparse_dot_node)["Kernel_Selection_Result"] = std::make_pair(n_device_type, pkernel);
        std::cout << "###############################" << std::endl;
        std::cout << kernel->get_or_emit_source()->body_unit->get_code() << std::endl;
        std::cout << kernel->get_or_emit_source()->signature_unit->get_code() << std::endl;
        //exit(-1);
        std::cout << "Bind the Quantized kernel!" << std::endl;
        has_constant = true;
    }
    void BlockDotOptimize(std::shared_ptr<GNode> dot_node,
                          nnfusion::cache::KernelEntry_p kernel_entry,
                          vector<std::shared_ptr<GNode>> fusible_nodes,
                          NNFusion_DeviceType n_device_type)
    {
        std::cout << "In SparGen BlockDotOptimize" << std::endl;
        assert(kernel_entry != nullptr);
        assert(dot_node != nullptr);

        bool has_constant = false;
        bool has_bias = false;
        bool has_relu = false;
        vector<shared_ptr<GNode>> need_remove;
        shared_ptr<GNode> add_node = nullptr;
        shared_ptr<GNode> bias_broadcast = nullptr;
        shared_ptr<GNode> relu_node = nullptr;
        int tesaid = (*dot_node)["TESAID"].as<int>();
        for (auto node : fusible_nodes)
        {
            if (node->get_op_type() == "Add")
            {
                add_node = node;
                has_bias = true;
                for (auto in_edge : add_node->get_in_edges())
                {
                    auto src_node = in_edge->get_src();
                    if (src_node->is_constant())
                    {
                        auto ori_bias_weight = src_node;
                        auto bias_related = find_all_predecessors(src_node);
                        need_remove.push_back(add_node);
                        need_remove.push_back(ori_bias_weight);
                        need_remove.insert(
                            need_remove.end(), bias_related.begin(), bias_related.end());
                    }
                    else if (src_node->get_op_type() == "Broadcast")
                    {
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
            else if (node->get_op_type() == "Relu")
            {
                has_relu = true;
                assert(has_bias = true);
                relu_node = node;
                need_remove.push_back(relu_node);
            }
        }
        // Start to replace the block sparse kernel
        auto src_node = dot_node->get_in_edge(1)->get_src();
        if (!src_node->is_constant())
            return;

        int ori_device_id = (*src_node)["DeviceID"];

        auto weight_constant =
            std::dynamic_pointer_cast<nnfusion::op::Constant>(src_node->get_op_ptr());
        auto w_shape = weight_constant->get_shape();
        size_t weight_count = 1, out_count = 1;
        for (int i : w_shape)
            weight_count *= i;
        auto out_shape = dot_node->get_output_shape(0);
        for (int i : out_shape)
            out_count *= i;

        // we filled the ramdom data temporarily
        // float* quan_weight_data = (float*)malloc(sizeof(float) * weight_count);
        float* block_weight_rows = (float*)malloc(sizeof(float) * (w_shape[0] + 1));
        memset(block_weight_rows, 0, sizeof(float) * (w_shape[0] + 1));
        float* block_weight_cols = (float*)malloc(sizeof(float) * weight_count);
        memset(block_weight_cols, 0, sizeof(float) * weight_count);
        float* block_weight_values = (float*)malloc(sizeof(float) * weight_count);
        memset(block_weight_values, 0, sizeof(float) * weight_count);
        load_from_file(
            (char*)block_weight_rows, sizeof(float) * (w_shape[0] + 1), this->csr_rows[tesaid]);
        load_from_file(
            (char*)block_weight_cols, sizeof(float) * weight_count, this->csr_cols[tesaid]);
        load_from_file(
            (char*)block_weight_values, sizeof(float) * weight_count, this->csr_values[tesaid]);
        // TODO load the right value according to the config

        float* bias_data =
            (float*)malloc(sizeof(float) * weight_count); // TODO use the correct size here
        memset(bias_data, 0, sizeof(float) * weight_count);
        auto dense_op = std::dynamic_pointer_cast<op::Dot>(dot_node->get_op_ptr());
        auto weight_values_node =
            create_constant_node(n_device_type, ori_device_id, w_shape, block_weight_values);
        auto weight_row_node = create_constant_node(
            n_device_type, ori_device_id, vector<size_t>({w_shape[0] + 1}), block_weight_rows);
        auto weight_col_node =
            create_constant_node(n_device_type, ori_device_id, w_shape, block_weight_cols);

        auto activate_node = dot_node->get_in_edge(0)->get_src();
        GNodeVector input_gv({activate_node, weight_values_node, weight_row_node, weight_col_node});

        m_graph->add_node(weight_values_node);
        m_graph->add_node(weight_row_node);
        m_graph->add_node(weight_col_node);

        // Handle the fuse option here
        if (has_bias)
        {
            auto bias_shape = nnfusion::Shape(vector<size_t>(
                {weight_count})); // TODO currently the memory space for bias is wasted
            // TODO also load the correct bias weights
            auto bias = std::make_shared<op::Constant>(
                from<float>(), bias_shape, static_cast<void*>(bias_data));
            if (this->bias_data_path[tesaid].size() > 0)
            {
                load_from_file(
                    (char*)bias_data, sizeof(float) * weight_count, this->bias_data_path[tesaid]);
            }
            auto bias_node = std::make_shared<GNode>(bias, GNodeVector({}));
            bias->revalidate_and_infer_types(bias_node->shared_from_this());
            bias_node->Set<NNFusion_DeviceType>("DeviceType", move(n_device_type));
            bias_node->Set<int>("DeviceID", move(ori_device_id));
            input_gv.push_back(bias_node);
            m_graph->add_node(bias_node);
        }
        GNodeVector empty_list;
        auto sparse_dot = std::make_shared<op::SparseDot>(dense_op);
        // auto quan_dot = std::make_shared<op::QuantizeDot>(dense_op, quantize_bit);
        auto sparse_dot_node = std::make_shared<GNode>(sparse_dot, empty_list);
        for (int i = 0; i < input_gv.size(); i++)
        {
            sparse_dot_node->set_input(
                i,
                std::make_shared<Input>(input_gv[i]->get_outputs().at(0)->get_element_type(),
                                        input_gv[i]->get_outputs().at(0)->get_partial_shape()));
        }
        sparse_dot_node->Set<NNFusion_DeviceType>("DeviceType", move(n_device_type));
        sparse_dot_node->Set<int>("DeviceID", move(ori_device_id));
        /// Remember after set the input node vector, we still need to set the edge manually!
        for (int i = 0; i < input_gv.size(); i++)
        {
            m_graph->add_edge(input_gv.at(i), 0, sparse_dot_node, i);
        }

        // replace node will revalidate and infer the output tensor
        auto last_node = dot_node;
        if (fusible_nodes.size())
            last_node = fusible_nodes[fusible_nodes.size() - 1];

        auto ori_outputs = last_node->get_outputs();
        //???
        for (int i = 0; i < ori_outputs.size(); i++)
        {
            sparse_dot_node->set_output(i, ori_outputs[i]);
        }

        m_graph->replace_node(last_node, sparse_dot_node, false);
        m_graph->remove_node(src_node);
        need_remove.push_back(dot_node);
        for (auto tmp_node : need_remove)
        {
            std::cout << " Removing " << tmp_node->get_name() << " " << tmp_node->get_op_type()
                      << std::endl;
        }
        for (auto tmp_node : need_remove)
        {
            if (tmp_node != last_node)
            {
                m_graph->remove_node(tmp_node);
            }
        }

        // Bind the fetched kernel here with the new kernel context
        std::shared_ptr<KernelContext> ctx(new KernelContext(sparse_dot_node));
        auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
        KernelEmitter::Pointer pkernel = kernel;

        // need to emit the source before bind the kernel
        kernel->get_or_emit_source();
        (*sparse_dot_node)["Kernel_Selection_Result"] = std::make_pair(n_device_type, pkernel);
        std::cout << "###############################" << std::endl;
        std::cout << kernel->get_or_emit_source()->body_unit->get_code() << std::endl;
        std::cout << kernel->get_or_emit_source()->signature_unit->get_code() << std::endl;
        //exit(-1);
        std::cout << "Bind the Quantized kernel!" << std::endl;
        has_constant = true;
    }
    vector<std::shared_ptr<GNode>> get_dot_fusible_nodes(std::shared_ptr<GNode> dot_node)
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

    vector<std::shared_ptr<GNode>> get_conv_fusible_nodes(std::shared_ptr<GNode> conv_node)
    {
        vector<std::shared_ptr<GNode>> fused_op;
        auto succs = find_successors(conv_node);
        if (succs.size() == 0)
        {
            return vector<std::shared_ptr<GNode>>();
        }
        auto son_node = succs[0];

        if (son_node->get_op_type().find("BatchNorm") != std::string::npos)
        {
            fused_op.push_back(son_node);
            auto grandsons = find_successors(son_node);
            if (grandsons.size() > 0)
            {
                for (auto tmp_node : grandsons)
                    std::cout << " ### " << tmp_node->get_op_type() << " ";
                std::cout << std::endl;
                if (grandsons[0]->get_op_type() == "Relu" ||
                    grandsons[0]->get_op_type() == "Swish" ||
                    grandsons[0]->get_op_type() == "Sigmoid")
                {
                    fused_op.push_back(grandsons[0]);
                }
            }
        }
        else if(son_node->get_op_type() == "Add"){
                auto add_node = son_node;
                for (auto in_edge : add_node->get_in_edges())
                {
                    auto src_node = in_edge->get_src();
                    if (src_node->is_constant())
                    {
                        auto ori_bias_weight = src_node;
                        auto bias_related = find_all_predecessors(src_node);
                        fused_op.push_back(ori_bias_weight);
                        fused_op.insert(
                            fused_op.end(), bias_related.begin(), bias_related.end());
                        fused_op.push_back(add_node);
                        
                    }
                    else if (src_node->get_op_type() == "Broadcast")
                    {
                        auto bias_broadcast = src_node;
                        auto bias_related = find_all_predecessors(bias_broadcast);
                        //ori_bias_weight = bias_broadcast->get_in_edge(0)->get_src();
                        fused_op.push_back(bias_broadcast);
                        fused_op.insert(
                            fused_op.end(), bias_related.begin(), bias_related.end());
                        fused_op.push_back(add_node);

                    }

                }
                auto grandsons = find_successors(son_node);
                if(grandsons.size()>0){
                    auto grandson = grandsons[0];
                    if(grandson->get_op_type() == "Relu" ||
                       grandson->get_op_type() == "Swish"||
                       grandson->get_op_type() == "Sigmoid"){
                           fused_op.push_back(grandson);
                       }
                }

        }
        else if (son_node->get_op_type() == "Relu" || son_node->get_op_type() == "Swish" ||
                 son_node->get_op_type() == "Sigmoid")
        {
            fused_op.push_back(son_node);
        }
        return fused_op;
    }
    vector<std::shared_ptr<GNode>> get_depth_conv_fusible_nodes(std::shared_ptr<GNode> conv_node)
    {
        vector<std::shared_ptr<GNode>> fused_op;
        auto succs = find_successors(conv_node);
        if (succs.size() == 0)
        {
            return vector<std::shared_ptr<GNode>>();
        }
        auto son_node = succs[0];

        if (son_node->get_op_type().find("BatchNorm") != std::string::npos)
        {
            fused_op.push_back(son_node);
            auto grandsons = find_successors(son_node);
            if (grandsons.size() > 0)
            {
                for (auto tmp_node : grandsons)
                    std::cout << " ### " << tmp_node->get_op_type() << " ";
                std::cout << std::endl;
                if (grandsons[0]->get_op_type() == "Relu" ||
                    grandsons[0]->get_op_type() == "Swish" ||
                    grandsons[0]->get_op_type() == "Sigmoid")
                {
                    fused_op.push_back(grandsons[0]);
                }
            }
        }
        else if(son_node->get_op_type() == "Add"){
                auto add_node = son_node;
                for (auto in_edge : add_node->get_in_edges())
                {
                    auto src_node = in_edge->get_src();
                    if (src_node->is_constant())
                    {
                        auto ori_bias_weight = src_node;
                        auto bias_related = find_all_predecessors(src_node);
                        fused_op.push_back(ori_bias_weight);
                        fused_op.insert(
                            fused_op.end(), bias_related.begin(), bias_related.end());
                        fused_op.push_back(add_node);

                    }
                    else if (src_node->get_op_type() == "Broadcast")
                    {
                        auto bias_broadcast = src_node;
                        auto bias_related = find_all_predecessors(bias_broadcast);
                        //ori_bias_weight = bias_broadcast->get_in_edge(0)->get_src();
                        fused_op.push_back(bias_broadcast);
                        fused_op.insert(
                            fused_op.end(), bias_related.begin(), bias_related.end());
                        fused_op.push_back(add_node);
                        
                    }

                }
                auto grandsons = find_successors(son_node);
                if(grandsons.size()>0){
                    auto grandson = grandsons[0];
                    if(grandson->get_op_type() == "Relu" ||
                       grandson->get_op_type() == "Swish"||
                       grandson->get_op_type() == "Sigmoid"){
                           fused_op.push_back(grandson);
                       }
                }

        }
        else if (son_node->get_op_type() == "Relu" || son_node->get_op_type() == "Swish" ||
                 son_node->get_op_type() == "Sigmoid")
        {
            fused_op.push_back(son_node);
        }
        return fused_op;
    }



    nnfusion::cache::KernelEntry_p
        fetch_kernel(std::shared_ptr<cache::KernelCacheManager> cache_manager,
                     string identifier,
                     NNFusion_DeviceType devtype)
    {
        std::cout << "Fetch Kernel by the Identifier: " << identifier << std::endl;
        const std::vector<std::string> SUPPORT_PLATFORM = {"CUDA_GPU", "CPU"};
        if (identifier != "" &&
            find(SUPPORT_PLATFORM.begin(), SUPPORT_PLATFORM.end(), get_device_str(devtype)) !=
                SUPPORT_PLATFORM.end())
        {
            auto fetched = cache_manager->fetch_all(identifier, get_device_str(devtype));
            nnfusion::cache::KernelEntry_p kernel_entry = nullptr;
            double kernel_time = 1000000000;
            std::cout << "Fetch " << fetched.size() << " Kernels from Kernel Cache!!!!!"
                      << std::endl;
            // Currently pick the first matched kernel
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

    std::shared_ptr<GNode>
        create_constant_node(NNFusion_DeviceType dt, int ori_device_id, int value = 0)
    {
        int* ptr = (int*)malloc(sizeof(int) * 2);
        *ptr = value;
        auto constant = std::make_shared<op::Constant>(
            from<float>(), nnfusion::Shape(vector<size_t>({1})), static_cast<void*>(ptr));
        auto constant_node = std::make_shared<GNode>(constant, GNodeVector({}));
        constant->revalidate_and_infer_types(constant_node->shared_from_this());
        constant_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
        constant_node->Set<int>("DeviceID", move(ori_device_id));
        return constant_node;
    }
    std::shared_ptr<GNode>
        create_constant_node(NNFusion_DeviceType dt, int ori_device_id, vector<size_t> shape)
    {
        int total_size = 1;
        for (int i : shape)
            total_size *= i;
        float* ptr = (float*)malloc(sizeof(float) * total_size);
        auto constant = std::make_shared<op::Constant>(
            from<float>(), nnfusion::Shape(shape), static_cast<void*>(ptr));
        auto constant_node = std::make_shared<GNode>(constant, GNodeVector({}));
        constant->revalidate_and_infer_types(constant_node->shared_from_this());
        constant_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
        constant_node->Set<int>("DeviceID", move(ori_device_id));
        return constant_node;
    }

    std::shared_ptr<GNode> create_constant_node(NNFusion_DeviceType dt,
                                                int ori_device_id,
                                                vector<size_t> shape,
                                                float* ptr)
    {
        auto constant = std::make_shared<op::Constant>(
            from<float>(), nnfusion::Shape(shape), static_cast<void*>(ptr));
        auto constant_node = std::make_shared<GNode>(constant, GNodeVector({}));
        constant->revalidate_and_infer_types(constant_node->shared_from_this());
        constant_node->Set<NNFusion_DeviceType>("DeviceType", move(dt));
        constant_node->Set<int>("DeviceID", move(ori_device_id));
        return constant_node;
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

    size_t load_from_file(char* ptr, size_t buff_size, string filepath)
    {
        std::ifstream fin(filepath, ios::in | ios::binary);
        size_t loaded_size = fin.read(ptr, buff_size).gcount();
        return loaded_size;
    }

    std::shared_ptr<Graph> m_graph;
    std::string cfg_path;
    std::shared_ptr<nnfusion::cache::KernelCacheManager> cache_manager;
    std::map<int, std::string> kernel_id;
    std::map<int, std::string> sparse_type;
    std::map<int, int> need_converter;
    std::map<int, std::string> convert_identifier;
    std::map<std::string, int> name2tesaid;
    std::map<int, std::string> tesaid2name;
    std::map<int, std::string> csr_rows;
    std::map<int, std::string> csr_cols;
    std::map<int, std::string> csr_values;
    std::map<int, std::string> weight_data_path;
    std::map<int, std::string> bias_data_path;
    std::map<int, std::string> scale_integer;
    std::map<int, std::string> scale_shift;
    std::map<int, int> in_quan_bit;
    std::map<int, int> out_quan_bit;
};

bool SparGenPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool enable_spargen = FLAGS_fspargen_cfg.size() > 0;
    if (!enable_spargen)
        return true;
    NNFUSION_LOG(INFO) << "Enable the BlockQuantized kernels";
    SparGenOptimizer optimizer(graph, FLAGS_fspargen_cfg);
    return optimizer.optimize();
}
