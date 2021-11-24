// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(FuseDepthwiseConv2dNative)
    .attr<nnfusion::op::OpConfig::any>("data_format")
    .attr<nnfusion::op::OpConfig::any>("padding_type")
    .attr<nnfusion::op::OpConfig::any>("strides")
    .attr<nnfusion::op::OpConfig::any>("dilations")
    .attr<nnfusion::op::OpConfig::any>("padding_before")
    .attr<nnfusion::op::OpConfig::any>("padding_after")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        // NNFUSION_CHECK(gnode->get_input_size() == 2);
        // auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

        // // [ batch, in_rows, in_cols, in_depth ]
        // const Shape& input_shape = gnode->get_input_shape(0);

        // // [ filter_rows, filter_cols, in_depth, depth_multiplier ]
        // const Shape& filter_shape = gnode->get_input_shape(1);

        // std::string data_format = op->localOpConfig.getRoot()["data_format"];
        // bool is_nhwc = (data_format == "NHWC");

        // const int64_t in_depth = is_nhwc ? input_shape[3] : input_shape[1];
        // NNFUSION_CHECK(in_depth == filter_shape[2]);
        // const int64_t depth_multiplier = filter_shape[3];
        // const int64_t out_depth = in_depth * depth_multiplier;
        // const int64_t input_rows = is_nhwc ? input_shape[1] : input_shape[2];
        // const int64_t input_cols = is_nhwc ? input_shape[2] : input_shape[3];
        // const int64_t filter_rows = filter_shape[0];
        // const int64_t filter_cols = filter_shape[1];
        // const int64_t batch = input_shape[0];
        // auto padding_before = op->localOpConfig.getRoot()["padding_before"];
        // auto padding_after = op->localOpConfig.getRoot()["padding_after"];
        // const int64_t padding_h = padding_before[0];
        // const int64_t padding_w = padding_before[1];
        // const int64_t dilation_h = op->localOpConfig.getRoot()["dilations"][0];
        // const int64_t dilation_w = op->localOpConfig.getRoot()["dilations"][1];

        // std::vector<int64_t> strides = op->localOpConfig.getRoot()["strides"];
        // NNFUSION_CHECK(strides.size() == 2);
        // const int64_t out_rows =
        //     (int64_t)((float)(input_rows + 2 * padding_h - dilation_h * (filter_rows - 1) - 1) /
        //                   (float)strides[0] +
        //               1);
        // const int64_t out_cols =
        //     (int64_t)((float)(input_cols + 2 * padding_w - dilation_w * (filter_cols - 1) - 1) /
        //                   (float)strides[1] +
        //               1);

        // Shape output_shape(
        //     is_nhwc
        //         ? Shape({(size_t)batch, (size_t)out_rows, (size_t)out_cols, (size_t)out_depth})
        //         : Shape({(size_t)batch, (size_t)out_depth, (size_t)out_rows, (size_t)out_cols}));

        // gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    });
    /*
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();
        // currently only support NHWC format
        if (op->localOpConfig.getRoot()["data_format"] != "NHWC")
            return "";
        const auto& padding_below = op->localOpConfig.getRoot()["padding_before"];
        const auto& padding_above = op->localOpConfig.getRoot()["padding_after"];
        uint64_t padding[] = {
            padding_below[1], padding_below[0], padding_above[1], padding_above[0]};

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); input("input1", @filter_shape@); output(@output_shape@, topi=topi.nn.depthwise_conv2d_nhwc(args("input0"), args("input1"), stride=@stride@, padding=@padding@, dilation=@dilation@)); )",
            {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
             {"filter_shape", vector_to_string(gnode->get_input_shape(1))},
             {"output_shape", vector_to_string(gnode->get_output_shape(0))},
             {"stride", vector_to_string(op->localOpConfig.getRoot()["strides"])},
             {"padding", vector_to_string(padding)},
             {"dilation", vector_to_string(op->localOpConfig.getRoot()["dilations"])}});
    })
    */
