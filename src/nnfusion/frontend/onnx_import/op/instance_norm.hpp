//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_6
            {
                NamedNodeVector
                    TranslateInstanceNormOp(const onnx::NodeProto& node_proto,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto x_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    auto scale_gnode = GetInputNode(all_ng_nodes, node_proto, 1);
                    auto bias_gnode = GetInputNode(all_ng_nodes, node_proto, 2);

                    std::shared_ptr<graph::GNode> mean_gnode{nullptr};
                    std::shared_ptr<graph::GNode> var_gnode{nullptr};

                    Node node(node_proto);
                    double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["epsilon"] = epsilon;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "InstanceNorm", myConfig);
                    auto generic_gnode =
                        m_graph->add_node_and_edge(generic_op, {x_gnode, scale_gnode, bias_gnode});

                    NamedNodeVector ret{{node_proto.output(0), generic_gnode}};
                    return ret;
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
