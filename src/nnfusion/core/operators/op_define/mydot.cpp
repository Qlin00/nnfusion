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

#include <functional>
#include <memory>
#include <utility>

#include "mydot.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

MyDot::MyDot()
    : MyDot(0, false)
{
}

MyDot::MyDot(size_t reduction_axes_count, bool has_reduction_axes_count, bool trans_a, bool trans_b)
    : Op("MyDot")
    , m_reduction_axes_count(reduction_axes_count)
    , m_has_reduction_axes_count(has_reduction_axes_count)
    , m_transpose_A(trans_a)
    , m_transpose_B(trans_b)
    
{
}



void MyDot::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    
}
