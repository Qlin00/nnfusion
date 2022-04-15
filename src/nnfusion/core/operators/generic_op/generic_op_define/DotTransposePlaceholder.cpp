// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(DotTransposePlaceholder)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), input_shape_0);
    });