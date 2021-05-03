// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sparse_kernel_pass.hpp"
#include "util.hpp"
#include <queue>
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

#include "gflags/gflags.h"

DEFINE_bool(fsparse_kernel, false, "Sparse Kernel.");
DEFINE_string(fsparse_cfg, "", "Sparse cfg for the target model");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::element;

class SparseKernelOptimizer
{
public:
    SparseKernelOptimizer(std::shared_ptr<Graph> g, string sparse_cfg, float threshold)
        : m_graph(g)
        , cfg_path(sparse_cfg)
        , sparse_threshold(threshold)
    {
        
    }
    void load_sparse_cfg()
    {
        // Load the quantize config
        // pytorch name, onnx name, quantize bit
        ifstream cfg_file(cfg_path.c_str());
        assert(cfg_file.good());
        string line;
        while (std::getline(cfg_file, line))
        {
            std::istringstream iss(line);
            string name;
            string identifier;
            iss >> name >> identifier;
            sparse_cfg[name] = identifier;
        }
    }
    nnfusion::cache::KernelEntry_p
        fetch_sparse_kernel(shared_ptr<cache::KernelCacheManager> cache_manager,
                              shared_ptr<GNode> node,
                              NNFusion_DeviceType devtype,
                              std::string identifier,
                              vector<std::shared_ptr<GNode>> fused_op)
    {
        const std::vector<std::string> SUPPORT_PLATFORM = {"CUDA_GPU"};
        if (identifier != "" && cache_manager &&
            find(SUPPORT_PLATFORM.begin(), SUPPORT_PLATFORM.end(), get_device_str(devtype)) !=
                SUPPORT_PLATFORM.end())
        {

            std::cout << "New Identifier:" << identifier << std::endl;
            std::cout << "Device String: " << get_device_str(devtype) << std::endl;
            auto fetched = cache_manager->fetch_all(identifier, get_device_str(devtype));
            nnfusion::cache::KernelEntry_p kernel_entry = nullptr;
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
            if (kernel_entry)
                NNFUSION_CHECK(kernel_entry->tags.find("CudaEmitter") != kernel_entry->tags.end());
            return kernel_entry;
        }
        return nullptr;
    }
    bool CusparseOptimize()
    {
        auto gnodes = m_graph->get_ordered_ops();
        for (auto node : gnodes)
        {
            // traverse all nodes and check whether is valuable to
            // to use the sparse kernel
            if (node->get_op_type() == "Dot")
            {
                std::cout<<"Try to optimize "<<node->get_name()<<std::endl;
                DotSparseOptimize(node);
            }
            else if (node->get_op_type() == "Conv2d")
            {
                // ???
            }
        }
        return true;
    }
    bool CustomizedKernelOptimize()
    {
        load_sparse_cfg();
        cache_manager = std::make_shared<nnfusion::cache::KernelCacheManager>();
        if(!cache_manager->is_valid())
        {
            NNFUSION_LOG(INFO) << "No valid kernel cache, cannot find sparse kernel, use cusparse instead";
            cache_manager = nullptr;
        }
        auto gnodes = m_graph->get_ordered_ops();
        for (auto node : gnodes)
        {
            std::cout << "###### Node name:" << node->get_name() << " Unique name"
            << node->get_unique_name() << "  Type:" << node->get_op_type() << std::endl;
        }
        for (auto node : gnodes)
        {
            if((*node)["Kernel_Selection_Result"].is_valid())
                continue;
            if(!(*node)["DeviceType"].is_valid()){
                NNFUSION_CHECK_FAIL()
                    << "GNode DeviceType should be assigned before this pass " << node->get_name();
            }
            auto n_device_type = (*node)["DeviceType"].as<NNFusion_DeviceType>();
            NNFUSION_CHECK(n_device_type!=UNKNOWN);
            std::cout << "Sparse Node name: " << node->get_name()
                      << "Type: " << node->get_op_type() << std::endl;
            if(sparse_cfg.count(node->get_name())==0)
                continue;
            std::string identifier = sparse_cfg[node->get_name()];
            vector<std::shared_ptr<GNode>> fused_op;
            if (node->get_op_type() == "Dot")
            {
                fused_op = nnfusion::graph::get_dot_fuse_option(node);                
            }
            else if (node->get_op_type() == "Conv2d")
            {
                // ???
            }else{
                throw "Now supported OP";
            }
            auto kernel_entry = fetch_sparse_kernel(cache_manager, node, n_device_type, identifier, fused_op);
            if(kernel_entry==nullptr)
                continue;
            // Directly bind the sparse kernel here
            std::shared_ptr<KernelContext> ctx(new KernelContext(node));
            auto kernel = std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
            KernelEmitter::Pointer pkernel = kernel;
            kernel->get_or_emit_source();
            (*node)["Kernel_Selection_Result"] = std::make_pair(n_device_type, pkernel);
            std::cout << "###############################" << std::endl;
            std::cout << kernel->get_or_emit_source()->body_unit->get_code() << std::endl;
            std::cout << kernel->get_or_emit_source()->signature_unit->get_code() << std::endl;
            std::cout << "bind sparse kernel"<<std::endl;
            // DotSparseOptimize(node);
        }
        return true;
    }
    
private:
    void DotSparseOptimize(std::shared_ptr<GNode> cur_node)
    {
        bool has_constant = false;

        for (auto in_edge : cur_node->get_in_edges())
        {
            auto src_node = in_edge->get_src();
            std::cout<<"Dot src node: "<< src_node->get_op_type()<<" "<<src_node->get_name()<<std::endl;
            if (src_node->is_constant())
            {
                // Get the sparsity ratio, if sparsity ratio is larger than a threshold
                auto weight_constant =
                    std::dynamic_pointer_cast<nnfusion::op::Constant>(src_node->get_op_ptr());
                auto data_ptr = weight_constant->get_data_ptr();
                assert(data_ptr != nullptr);
                auto m_shape = weight_constant->get_shape();
                float sparsity_ratio =
                    get_sparsity_ratio<float>(static_cast<const float*>(data_ptr),
                                              nnfusion::shape_size(m_shape),
                                              sparse_threshold);
                // if sparsity ratio is too low then it's not worth
                std::cout<<" Sparsity Ratio: "<<sparsity_ratio<<std::endl;
                if (sparsity_ratio < 0.9)
                    continue;
                std::shared_ptr<vector<int32_t>> row_idx, col_idx;
                std::shared_ptr<vector<float>> values;
                std::tie(row_idx, col_idx, values) = convert_to_csr<float>(
                    static_cast<const float*>(data_ptr), m_shape, sparse_threshold);
                insert_sparse_dot(m_graph, in_edge, row_idx, col_idx, values);
                has_constant = true;
                break;
            } else if(src_node->get_op_type()=="Reshape"){
                auto weight_node = src_node->get_in_edge(0)->get_src();
                auto ori_weight_shape = weight_node->get_output_shape(0);
                std::cout<<" Ori weight outshape ";
                for(int i=0;i<ori_weight_shape.size();i++){
                    std::cout<<ori_weight_shape[i]<<" ";
                }       
                std::cout<<std::endl;
                if(!weight_node->is_constant())
                    continue;
                auto outputs = src_node->get_outputs();
                auto out_shape = src_node->get_output_shape(0);
                std::cout<<" Reshape outshape ";
                for(int i=0;i<out_shape.size();i++){
                    std::cout<<out_shape[i]<<" ";
                }
                std::cout<<std::endl;
                for(int i=0; i<outputs.size();i++)
                    weight_node->set_output(i, outputs[i]);
                auto new_weight_shape=weight_node->get_output_shape(0);
                std::cout<<" New weight outshape ";
                for(int i=0;i<new_weight_shape.size();i++){
                    std::cout<<new_weight_shape[i]<<" ";
                }     
                std::cout<<std::endl;
                int dst_input = in_edge->get_dst_input();
                m_graph->remove_node(src_node);
                m_graph->add_edge(weight_node, 0, cur_node, dst_input);
                auto weight_constant =
                    std::dynamic_pointer_cast<nnfusion::op::Constant>(weight_node->get_op_ptr());
                
                auto data_ptr = weight_constant->get_data_ptr();
                assert(data_ptr != nullptr);
                // auto m_shape = weight_constant->get_shape();
                float sparsity_ratio =
                    get_sparsity_ratio<float>(static_cast<const float*>(data_ptr),
                                              nnfusion::shape_size(out_shape),
                                              sparse_threshold);
                // if sparsity ratio is too low then it's not worth
                std::cout<<" Sparsity Ratio: "<<sparsity_ratio<<std::endl;
                if (sparsity_ratio < 0.9)
                    continue;
                std::shared_ptr<vector<int32_t>> row_idx, col_idx;
                std::shared_ptr<vector<float>> values;
                std::tie(row_idx, col_idx, values) = convert_to_csr<float>(
                    static_cast<const float*>(data_ptr), ori_weight_shape, sparse_threshold);
                insert_sparse_dot(m_graph, cur_node->get_in_edge(dst_input), row_idx, col_idx, values);
                has_constant = true;
                break;
            }
        }
        if (!has_constant)
            // on the fly convertion is to expensive
            return;
    }
    template <typename scalar_t>
    float get_sparsity_ratio(const scalar_t* data, size_t n, scalar_t threshold)
    {
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            if (data[i] <= threshold)
                count++;
        }
        return count * 1.0 / n;
    }

    template <typename scalar_t>
    tuple<std::shared_ptr<vector<int32_t>>,
          std::shared_ptr<vector<int32_t>>,
          std::shared_ptr<vector<scalar_t>>>
        convert_to_csr(const scalar_t* data, const nnfusion::Shape& m_shape, scalar_t threshold)
    {
        assert(m_shape.size() == 2);
        auto row_idx = std::make_shared<vector<int32_t>>();
        auto col_idx = std::make_shared<vector<int32_t>>();
        auto values = std::make_shared<vector<scalar_t>>();

        for (int i = 0; i < m_shape[0]; i++)
        {
            row_idx->push_back(values->size());
            for (int j = 0; j < m_shape[1]; j++)
            {
                size_t pos = i * m_shape[1] + j;
                if (data[pos] < threshold)
                {
                    // sparsity
                    continue;
                }
                else
                {
                    values->push_back(data[pos]);
                    col_idx->push_back(j);
                }
            }
        }
        row_idx->push_back(values->size());
        return std::make_tuple(row_idx, col_idx, values);
    }
    
    void insert_sparse_dot(std::shared_ptr<Graph> pgraph,
                           std::shared_ptr<Edge> in_edge,
                           std::shared_ptr<vector<int32_t>> row_idx,
                           std::shared_ptr<vector<int32_t>> col_idx,
                           std::shared_ptr<vector<float>> values)
    {
        std::shared_ptr<GNode> src_node = in_edge->get_src();
        std::shared_ptr<GNode> dst_node = in_edge->get_dst();
        auto n_device_type = (*dst_node)["DeviceType"].as<NNFusion_DeviceType>();
        NNFUSION_CHECK(n_device_type != UNKNOWN);
        int ori_device_id = (*dst_node)["DeviceID"];
        //create the constant nodes for the csr format weight
        auto row_idx_cons = std::make_shared<op::Constant>(
            from<int32_t>(), nnfusion::Shape({row_idx->size()}), (void*)row_idx->data());
        auto row_idx_node = std::make_shared<GNode>(row_idx_cons, GNodeVector({}));
        row_idx_node->get_op_ptr()->revalidate_and_infer_types(row_idx_node->shared_from_this());

        row_idx_node->Set<NNFusion_DeviceType>("DeviceType", move(n_device_type));
        row_idx_node->Set<int>("DeviceID", move(ori_device_id));

        auto col_idx_cons = std::make_shared<op::Constant>(
            from<int32_t>(), nnfusion::Shape({col_idx->size()}), (void*)col_idx->data());
        auto col_idx_node = std::make_shared<GNode>(col_idx_cons, GNodeVector({}));
        col_idx_node->get_op_ptr()->revalidate_and_infer_types(col_idx_node->shared_from_this());

        col_idx_node->Set<NNFusion_DeviceType>("DeviceType", move(n_device_type));
        col_idx_node->Set<int>("DeviceID", move(ori_device_id));

        auto values_cons = std::make_shared<op::Constant>(
            from<float>(), nnfusion::Shape({values->size()}), (void*)values->data());
        auto csr_values_node = std::make_shared<GNode>(values_cons, GNodeVector({}));
        csr_values_node->get_op_ptr()->revalidate_and_infer_types(csr_values_node->shared_from_this());

        csr_values_node->Set<NNFusion_DeviceType>("DeviceType", move(n_device_type));
        csr_values_node->Set<int>("DeviceID", move(ori_device_id));



        auto dense_op = std::dynamic_pointer_cast<op::Dot>(dst_node->get_op_ptr()); // The original dense gemm op

        pgraph->add_node(row_idx_node);
        pgraph->add_node(col_idx_node);
        pgraph->add_node(csr_values_node);

        auto dst_pos = in_edge->get_dst_input();
        size_t other_input = 1 - dst_pos;
        auto other_node = dst_node->get_in_edge(other_input)->get_src();
        GNodeVector input_gv({row_idx_node, col_idx_node, csr_values_node, other_node});
        auto ori_sparse_shape = src_node->get_output_shape(0);
        auto sparse_op =
            std::make_shared<op::SparseDot>(dense_op, dst_pos, values->size(), ori_sparse_shape);
        NNFUSION_LOG(INFO) << "Replace a Dot op with sparsity ratio:"
                           << 1.0 * values->size() / ori_sparse_shape[0] / ori_sparse_shape[1]
                           << "\n";
        auto sparse_node = std::make_shared<GNode>(sparse_op, input_gv);

        sparse_node->Set<NNFusion_DeviceType>("DeviceType", move(n_device_type));
        sparse_node->Set<int>("DeviceID", move(ori_device_id));
        
        for (int i = 0; i < input_gv.size(); i++)
        {
            pgraph->add_edge(input_gv.at(i), 0, sparse_node, i);
        }

        // pgraph->add_node_and_edge(sparse_node, GNodeVector({row_idx_node, col_idx_node, values_node}));
        auto ori_output = dst_node->get_outputs();
        // just copy the output from the original dense node
        for (int i = 0; i < ori_output.size(); i++)
            sparse_node->set_output(i, ori_output[i]);
        // insert the sparse node into the original graph
        pgraph->replace_node(dst_node, sparse_node, false);
        pgraph->remove_node(src_node);
    }

    std::shared_ptr<Graph> m_graph;
    std::map<std::string, std::string> sparse_cfg;
    string cfg_path;
    float sparse_threshold;
    std::shared_ptr<nnfusion::cache::KernelCacheManager> cache_manager;
};

bool SparseKernelPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool enable_sparse_kernel = FLAGS_fsparse_kernel || FLAGS_fsparse_cfg.size() > 0;
    if (!enable_sparse_kernel)
        return true;
    NNFUSION_LOG(INFO) << "Enable the Sparse kernels";
    SparseKernelOptimizer optimizer(graph, FLAGS_fsparse_cfg,1e-6);
    bool re=false;
    if(FLAGS_fsparse_cfg.size()>0){
        // Customized Kernel optimize is in hign priority
        re = optimizer.CustomizedKernelOptimize();
    }else{
        re = optimizer.CusparseOptimize();
    }
    return re;
}
