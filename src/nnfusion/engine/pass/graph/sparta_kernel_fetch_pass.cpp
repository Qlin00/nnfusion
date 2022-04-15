// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/common.hpp"
#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/engine/cache/manager.hpp"
#include "nnfusion/engine/op.hpp"
#include "sparta_kernel_fetch_pass.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DEFINE_bool(fsparta, false, "Enable SparTA kernel fetch");

class SparTAKernelFetchOptimizer
{
public:
    SparTAKernelFetchOptimizer(std::shared_ptr<nnfusion::graph::Graph> g)
        : m_graph(g)
    {
        //
    }

    pair<NNFusion_DeviceType, kernels::KernelEmitter::Pointer>
        fetch_sparta_kernel(shared_ptr<cache::KernelCacheManager> cache_manager,
                            shared_ptr<GNode> gnode,
                            NNFusion_DeviceType devtype)
    {
        // std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        //     KernelRegistry::Global()->FindKernelRegistrations(
        //         gnode->get_op_type(), devtype, element::f32);
        shared_ptr<KernelContext> ctx(new KernelContext(gnode));
        // puts("22222");
        std::vector<nlohmann::json> functions;

        if (!(*gnode)["TESAID"].is_valid())
        {
            return std::make_pair(devtype, nullptr);
        }

        int TESAID = (*gnode)["TESAID"].as<int>();
        if (TESAID == -1)
        {
            return std::make_pair(devtype, nullptr);
        }
        std::cout << TESAID << " find" << std::endl;

        // std::string identifier = "kernel_" + std::to_string(TESAID); // kernel_{TESAID}
        // std::cout << identifier << std::endl;

        std::string identifier;
        if ((*gnode)["identifier"].is_valid())
        {
            identifier = (*gnode)["identifier"].as<std::string>();
        }
        else
        {
            identifier = gnode->get_op_type() + "_TESAID_" + std::to_string(TESAID);
        }
        std::cout << identifier << std::endl;
        // Todo: platform interface to be coordinated with nnfusion devtype
        const std::vector<std::string> SUPPORT_PLATFORM = {"CUDA_GPU", "ROCM_GPU"};

        if (identifier != "" &&
            find(SUPPORT_PLATFORM.begin(), SUPPORT_PLATFORM.end(), get_device_str(devtype)) !=
                SUPPORT_PLATFORM.end())
        {
            // fetch all available kernel entries from kernel cache DB
            auto fetched = cache_manager->fetch_all(identifier, "CUDA_GPU");

            {
                NNFUSION_CHECK(fetched.size() <= 1)
                    << "SparTA kernel key collided, please clear the kernel_cache.db and retry.";
                for (auto kernel_entry : fetched)
                {
                    // if (kernel_entry->source == "Compression" && kernel_entry->miscs["TESAID"] == TESAID)
                    // if (kernel_entry->source == "SparTA")
                    {
                        NNFUSION_CHECK(kernel_entry->tags.find("CudaEmitter") !=
                                       kernel_entry->tags.end());
                        auto kernel =
                            std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
                        if (kernel->get_or_emit_source())
                        {
                            std::cout << TESAID << " emitted!" << std::endl;
                            if (kernel_entry->miscs.find("SparTA") != kernel_entry->miscs.end() &&
                                kernel_entry->miscs["SparTA"].find("Weight_Dismantled") !=
                                    kernel_entry->miscs["SparTA"].end() &&
                                kernel_entry->miscs["SparTA"]["Weight_Dismantled"])
                            {
                                auto weight_gnode = gnode->get_in_edge(1)->get_src();
                                auto weight_op = static_pointer_cast<nnfusion::op::Constant>(
                                    weight_gnode->get_op_ptr());
                                if (weight_op != nullptr)
                                {
                                    // m_graph->remove_node(weight_gnode);
                                    (*weight_gnode)["SparTA_Value_Dismantled"] = true;
                                }
                                (*gnode)["Sparse_Dot_Transpose"] = true;
                            }
                            return std::make_pair(devtype, kernel);
                        }
                    }
                }
            }
        }
        return std::make_pair(devtype, nullptr);
    }

    bool Optimize()
    {
        auto cache_manager = std::make_shared<cache::KernelCacheManager>();
        if (!cache_manager->is_valid())
        {
            NNFUSION_LOG(INFO) << "No valid kernel cache, FetchBasedSelector will be skipped";
            return true;
        }
        // auto dev_name = FLAGS_fdefault_device.c_str();
        // NNFusion_DeviceType default_device = nnfusion::get_device_type(dev_name);

        std::vector<std::shared_ptr<GNode>> nodes = m_graph->get_nodes();
        for (auto it : nodes)
        {

            std::cout << it->get_name() << std::endl;
            if (!(*it)["Kernel_Selection_Result"].is_valid())
            {
                if (!(*it)["DeviceType"].is_valid())
                {
                    NNFUSION_CHECK_FAIL()
                        << "GNode DeviceType should be assigned before this pass: "
                        << it->get_name();
                }
                auto n_device_type = (*it)["DeviceType"].as<NNFusion_DeviceType>();
                NNFUSION_CHECK(n_device_type != UNKNOWN);
                auto ans = fetch_sparta_kernel(cache_manager, it, n_device_type);

                if (ans.second != nullptr)
                    (*it)["Kernel_Selection_Result"] = ans;
            }
        }

        return true;
    }

private:
    std::shared_ptr<nnfusion::graph::Graph> m_graph;
};

bool SparTAKernelFetchPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fsparta == false)
    {
        return true;
    }

    SparTAKernelFetchOptimizer sparta_kernel_fetch_optimizer(graph);
    sparta_kernel_fetch_optimizer.Optimize();

    return true;
}