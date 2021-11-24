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

// Microsoft (c) 2019, NNFusion Team

#include <numeric>

#include "fused_convolution.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/util/validation_util.hpp"

using namespace std;
using namespace nnfusion::op;

FuseConvolution::FuseConvolution(const nnfusion::Strides& window_movement_strides,
                         const nnfusion::Strides& window_dilation_strides,
                         const nnfusion::CoordinateDiff& padding_below,
                         const nnfusion::CoordinateDiff& padding_above,
                         const nnfusion::Strides& data_dilation_strides,
                         std::string data_format)
    : Op("FuseConvolution")
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_data_format(data_format)
{
}

void FuseConvolution::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
}

nnfusion::Strides FuseConvolution::default_strides(const Op* op,
                                               const nnfusion::PartialShape& data_batch_shape,
                                               const nnfusion::PartialShape& filters_shape)
{
    size_t rank;

    if (data_batch_shape.rank().is_static() && static_cast<size_t>(data_batch_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(data_batch_shape.rank()) - 2;
    }
    else if (filters_shape.rank().is_static() && static_cast<size_t>(filters_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(filters_shape.rank()) - 2;
    }
    else
    {
        rank = 0;
    }

    return nnfusion::Strides(rank, 1);
}

FuseConvolution::FuseConvolution(const nnfusion::Strides& window_movement_strides,
                         const nnfusion::Strides& window_dilation_strides,
                         const nnfusion::CoordinateDiff& padding_below,
                         const nnfusion::CoordinateDiff& padding_above,
                         std::string data_format)
    : FuseConvolution(window_movement_strides,
                  window_dilation_strides,
                  padding_below,
                  padding_above,
                  nnfusion::Strides(),
                  data_format)
{
}

CoordinateDiff FuseConvolution::default_padding(const Op* op,
                                            const nnfusion::PartialShape& data_batch_shape,
                                            const nnfusion::PartialShape& filters_shape)
{
    size_t rank;

    if (data_batch_shape.rank().is_static() && static_cast<size_t>(data_batch_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(data_batch_shape.rank()) - 2;
    }
    else if (filters_shape.rank().is_static() && static_cast<size_t>(filters_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(filters_shape.rank()) - 2;
    }
    else
    {
        rank = 0;
    }

    return nnfusion::CoordinateDiff(rank, 0);
}

FuseConvolution::FuseConvolution(const nnfusion::Strides& window_movement_strides,
                         const nnfusion::Strides& window_dilation_strides)
    : FuseConvolution(window_movement_strides,
                  window_dilation_strides,
                  nnfusion::CoordinateDiff(),
                  nnfusion::CoordinateDiff())
{
}

FuseConvolution::FuseConvolution(const nnfusion::Strides& window_movement_strides)
    : FuseConvolution(window_movement_strides,
                  nnfusion::Strides(),
                  nnfusion::CoordinateDiff(),
                  nnfusion::CoordinateDiff())
{
}

FuseConvolution::FuseConvolution()
    : FuseConvolution(nnfusion::Strides(),
                  nnfusion::Strides(),
                  nnfusion::CoordinateDiff(),
                  nnfusion::CoordinateDiff())
{
}

void FuseConvolution::infer_shared_memory(std::shared_ptr<graph::GNode> gnode)
{
    for (auto s : get_window_movement_strides())
    {
        if (s != 1)
            return;
    }

    for (auto d : get_window_dilation_strides())
    {
        if (d != 1)
            return;
    }

    for (auto p : get_padding_below())
    {
        if (p != 0)
            return;
    }

    for (auto p : get_padding_above())
    {
        if (p != 0)
            return;
    }

    m_shared_memory.clear();
    const Shape& input_shape = gnode->get_input_shape(0);
    int channel = get_data_format() == "NCHW" ? 1 : 3;
    auto input_channel_count = input_shape[channel];

    for (size_t i = 0; i < gnode->get_output_shape(0).size(); i++)
    {
        if (i == channel)
            m_shared_memory.push_back(input_channel_count);
        else
            m_shared_memory.push_back(1);
    }
}
