// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sparse_dot_transpose_pass.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/kernels/cuda_gpu/kernels/dot_transpose_placeholder.hpp"
#include "nnfusion/engine/cache/manager.hpp"
#include "nnfusion/engine/op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DEFINE_bool(fsparse_dot_transpose, false, "Transpose input and output of Sparse Dot");
DECLARE_int32(fblockfusion_level);

class SparseDotTransposeOptimizer
{
public:
    SparseDotTransposeOptimizer(std::shared_ptr<nnfusion::graph::Graph> g)
        : m_graph(g)
    {
        //
    }

    void transpose_dot_gnode(std::shared_ptr<GNode> dot_gnode)
    {
        auto input_gnode = dot_gnode->get_in_edge(0)->get_src();
        auto out_edges = dot_gnode->get_out_edges();
        std::shared_ptr<GNode> output_gnode = nullptr;
        if (out_edges.size() > 0)
        {
            output_gnode = out_edges[0]->get_dst();
        }

        // transpose input_gnode
        {
            NNFUSION_LOG(INFO) << "transpose input_gnode for dot " << dot_gnode->get_name();
            // auto trans_gnode = nnfusion::graph::numpy_transpose(
            //     input_gnode, nnfusion::AxisVector(), input1_index);
            AxisVector axis_order({1, 0});
            nnfusion::Shape input_shape, output_shape;
            if (dot_gnode->get_input_shape(0).size() == 3)
            {
                input_shape = nnfusion::Shape(
                    {dot_gnode->get_input_shape(0)[0] * dot_gnode->get_input_shape(0)[1],
                     dot_gnode->get_input_shape(0)[2]});
                output_shape = nnfusion::Shape(
                    {dot_gnode->get_input_shape(0)[2],
                     dot_gnode->get_input_shape(0)[0] * dot_gnode->get_input_shape(0)[1]});
            }
            else if (dot_gnode->get_input_shape(0).size() == 2)
            {
                input_shape = nnfusion::Shape(
                    {dot_gnode->get_input_shape(0)[0], dot_gnode->get_input_shape(0)[1]});
                output_shape = nnfusion::Shape(
                    {dot_gnode->get_input_shape(0)[1], dot_gnode->get_input_shape(0)[0]});
            }
            else
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "conflicted, transpose not applied, skip";
                return;
            }
            m_graph->remove_edge(dot_gnode->get_in_edge(0));
            // auto trans_gnode = nnfusion::graph::numpy_transpose(input_gnode, axis_order, 0);
            nnfusion::op::OpConfig::any op_config;
            auto dot_trans_op = std::make_shared<op::GenericOp>(
                dot_gnode->get_name() + "_input_trans", "DotTransposePlaceholder", op_config);
            auto trans_gnode = m_graph->add_node_and_edge(dot_trans_op, {input_gnode});
            (*trans_gnode)["DeviceType"] = (*dot_gnode)["DeviceType"].as<NNFusion_DeviceType>();
            (*trans_gnode)["DeviceID"] = (*dot_gnode)["DeviceID"].as<int>();
            // dot_gnode->get_in_edge(0)->m_dst_input = 0;
            // dot_gnode->get_in_edge(0)->m_dst = trans_gnode;
            // m_graph->add_node(trans_gnode);
            // m_graph->add_edge(input_gnode, 0, trans_gnode, 0);
            m_graph->add_edge(trans_gnode, 0, dot_gnode, 0);

            std::shared_ptr<nnfusion::kernels::KernelContext> ctx =
                std::make_shared<nnfusion::kernels::KernelContext>(trans_gnode);
            nnfusion::kernels::cuda::DotTransposePlaceholder trans_kernel(ctx);
            trans_kernel.init(input_shape, output_shape, axis_order);
            nnfusion::kernels::KernelEmitter::Pointer trans_kernel_p =
                std::make_shared<nnfusion::kernels::cuda::DotTransposePlaceholder>(trans_kernel);
            // (*trans_kernel_p).init(input_shape, output_shape, axis_order);
            trans_kernel_p->get_or_emit_source();
            (*trans_gnode)["Kernel_Selection_Result"] =
                std::make_pair(NNFusion_DeviceType::CUDA_GPU, trans_kernel_p);
            // dot_gnode->set_input(0, )

            // for (auto out_edge : input_gnode->get_output_users(0))
            // {
            //     auto dst_node = out_edge->get_dst();
            //     if (dst_node == trans_gnode)
            //     {
            //         continue;
            //     }
            //     graph->remove_edge(out_edge);
            //     auto new_input = make_shared<nnfusion::graph::Input>(
            //         dst_node->get_input_element_type(1), trans_gnode->get_shape());
            //     dst_node->set_input(1, new_input);
            //     graph->add_edge(trans_gnode, 0, dst_node, 1);
            //     auto dot = std::dynamic_pointer_cast<nnfusion::op::Dot>(dst_node->get_op_ptr());
            //     NNFUSION_CHECK(dot);
            //     dot->get_transpose_B() = true;

            //     auto func_p = generate_func_point(
            //         dst_node, std::make_shared<nnfusion::cache::KernelEntry>(transpose_dot_kernel));
            //     NNFUSION_CHECK(func_p);
            //     if (func_p)
            //         (*dst_node)["Kernel_Selection_Result"] =
            //             std::make_pair((*it)["DeviceType"].as<NNFusion_DeviceType>(), func_p);
            // }
        }

        // transpose output_gnode
        if (output_gnode != nullptr)
        {
            NNFUSION_LOG(INFO) << "transpose output_gnode for dot " << dot_gnode->get_name()
                               << ", num_out_edges: " << out_edges.size();

            AxisVector axis_order = AxisVector({1, 0});
            nnfusion::Shape input_shape, output_shape;
            if (dot_gnode->get_output_shape(0).size() == 3)
            {
                input_shape = nnfusion::Shape(
                    {dot_gnode->get_output_shape(0)[2],
                     dot_gnode->get_output_shape(0)[0] * dot_gnode->get_output_shape(0)[1]});
                output_shape = nnfusion::Shape(
                    {dot_gnode->get_output_shape(0)[0] * dot_gnode->get_output_shape(0)[1],
                     dot_gnode->get_output_shape(0)[2]});
            }
            else if (dot_gnode->get_output_shape(0).size() == 2)
            {
                input_shape = nnfusion::Shape(
                    {dot_gnode->get_output_shape(0)[1], dot_gnode->get_output_shape(0)[0]});
                output_shape = nnfusion::Shape(
                    {dot_gnode->get_output_shape(0)[0], dot_gnode->get_output_shape(0)[1]});
            }
            else
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "conflicted, transpose not applied, skip";
                return;
            }

            auto dst_id = dot_gnode->get_out_edges()[0]->get_dst_input();
            m_graph->remove_edge(dot_gnode->get_out_edges()[0]);

            nnfusion::op::OpConfig::any op_config;
            auto dot_trans_op = std::make_shared<op::GenericOp>(
                dot_gnode->get_name() + "_output_trans", "DotTransposePlaceholder", op_config);
            auto trans_gnode = m_graph->add_node_and_edge(dot_trans_op, {dot_gnode});
            (*trans_gnode)["DeviceType"] = (*dot_gnode)["DeviceType"].as<NNFusion_DeviceType>();
            (*trans_gnode)["DeviceID"] = (*dot_gnode)["DeviceID"].as<int>();
            m_graph->add_edge(trans_gnode, 0, output_gnode, dst_id);

            std::shared_ptr<nnfusion::kernels::KernelContext> ctx =
                std::make_shared<nnfusion::kernels::KernelContext>(trans_gnode);
            nnfusion::kernels::cuda::DotTransposePlaceholder trans_kernel(ctx);
            trans_kernel.init(input_shape, output_shape, axis_order);
            nnfusion::kernels::KernelEmitter::Pointer trans_kernel_p =
                std::make_shared<nnfusion::kernels::cuda::DotTransposePlaceholder>(trans_kernel);
            // (*trans_kernel_p).init(input_shape, output_shape, axis_order);
            trans_kernel_p->get_or_emit_source();
            (*trans_gnode)["Kernel_Selection_Result"] =
                std::make_pair(NNFusion_DeviceType::CUDA_GPU, trans_kernel_p);

            // auto trans_gnode = nnfusion::graph::numpy_transpose(dot_gnode, axis_order, 0);
            // (*trans_gnode)["DeviceType"] = (*dot_gnode)["DeviceType"].as<NNFusion_DeviceType>();
            // (*trans_gnode)["DeviceID"] = (*dot_gnode)["DeviceID"].as<int>();
            // m_graph->remove_edge(out_edges[0]);

            // m_graph->remove_edge(output_gnode->get_in_edge(0));
            // m_graph->add_node(trans_gnode);
            // m_graph->add_edge(dot_gnode, 0, trans_gnode, 0);
            // m_graph->add_edge(trans_gnode, 0, output_gnode, dst_id);

            // std::cout << dot_gnode->get_out_edges().size() << " " <<  << std::endl;
        }
    }

    bool Optimize()
    {
        std::vector<std::shared_ptr<GNode>> nodes = m_graph->get_nodes();
        for (auto it : nodes)
        {
            // if ((*it)["TESAID"].is_valid())
            // {
            //     if (it->get_input_shape(0).size() == 3) // bert
            //     {
            //         std::cout << "enable SparTA_Dot_Transpose for Dot " << it->get_name()
            //                   << std::endl;
            //         (*it)["Sparse_Dot_Transpose"] = true;
            //     }
            // }
            if ((*it)["Sparse_Dot_Transpose"].is_valid())
            {
                transpose_dot_gnode(it);
            }
        }

        return true;
    }

private:
    std::shared_ptr<nnfusion::graph::Graph> m_graph;
};

bool SparseDotTransposePass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (!FLAGS_fsparse_dot_transpose)
    {
        return true;
    }

    FLAGS_fblockfusion_level = 0; // disable blockfusion due to conflict with this hotfix pass

    SparseDotTransposeOptimizer sparse_dot_transpose_optimizer(graph);
    sparse_dot_transpose_optimizer.Optimize();

    return true;
}