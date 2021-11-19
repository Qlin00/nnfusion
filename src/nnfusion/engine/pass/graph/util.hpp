#include <map>
#include <queue>
#include <string>
#include <vector>
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace graph
    {
	    std::vector<std::shared_ptr<GNode>> find_successors(std::shared_ptr<GNode> gnode)
        {
		std::vector<std::shared_ptr<GNode>> successors;
            const std::vector<std::shared_ptr<nnfusion::graph::Edge>>& out_edges =
                gnode->get_out_edges();
            for (auto edge : out_edges)
            {
                successors.push_back(edge->get_dst());
            }
            return successors;
        }
	    std::vector<std::shared_ptr<GNode>> find_predecessors(std::shared_ptr<GNode> gnode)
        {
		std::vector<std::shared_ptr<GNode>> predecessors;
            const std::vector<std::shared_ptr<nnfusion::graph::Edge>>& in_edges =
                gnode->get_in_edges();
            for (auto edge : in_edges)
            {
                predecessors.push_back(edge->get_src());
            }
            return predecessors;
        }
	    std::vector<std::shared_ptr<GNode>> find_all_predecessors(std::shared_ptr<GNode> gnode)
        {
		std::vector<std::shared_ptr<GNode>> result;
            auto predecessors = find_predecessors(gnode);
            result.insert(result.end(), predecessors.begin(), predecessors.end());
            for (auto father : predecessors)
            {
                auto grandfathers = find_all_predecessors(father);
                result.insert(result.end(), grandfathers.begin(), grandfathers.end());
            }
            return result;
        }

	    std::vector<std::shared_ptr<GNode>> get_dot_fuse_option(std::shared_ptr<GNode> dot_node)
        {
		std::vector<std::shared_ptr<GNode>> fused_op;
            auto succs = find_successors(dot_node);
            if (succs.size() == 0)
                // return
                return std::vector<std::shared_ptr<GNode>>();
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
    } // namespace graph
} // namespace nnfusion
