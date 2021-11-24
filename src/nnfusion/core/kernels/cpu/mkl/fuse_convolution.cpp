// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "fuse_convolution.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::FuseConvolutionMkl::FuseConvolutionMkl(shared_ptr<KernelContext> ctx)
    : MklKernelEmitter(ctx)
{
    auto conv = static_pointer_cast<op::Convolution>(ctx->gnode->get_op_ptr());

    input_shape = ctx->inputs[0]->get_shape();
    filter_shape = ctx->inputs[1]->get_shape();
    output_shape = ctx->outputs[0]->get_shape();
    window_dilation_strides = conv->get_window_dilation_strides();
    window_movement_strides = conv->get_window_movement_strides();
    data_dilation_strides = conv->get_data_dilation_strides();
    padding_below_diff = conv->get_padding_below();
    padding_above_diff = conv->get_padding_above();
    data_format = conv->get_data_format();
    dtype = ctx->outputs[0]->get_element_type().c_type_string();

    std::stringstream tag;
    tag << "mkl_convolution_op_" << dtype << "_i" << join(input_shape, "_") << "_w"
        << join(filter_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
        << join(window_movement_strides, "_") << "_wd" << join(window_dilation_strides, "_")
        << "_pb" << join(padding_below_diff, "_") << "_pa" << join(padding_above_diff, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::FuseConvolutionMkl::emit_function_body()
{
    if (!(data_format == "NCW" || data_format == "NCHW"))
    {
        return nullptr;
    }

    bool is_deconvolution = false;
    for (auto a : data_dilation_strides)
    {
        if (a != 1)
        {
            is_deconvolution = true;
            break;
        }
    }
    if (is_deconvolution)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Deconvolution is not supported by now.";
        return nullptr;
    }

    // Conv1D: convert Conv1D to Conv2D
    if (data_format == "NCW")
    {
        input_shape = {input_shape[0], input_shape[1], 1, input_shape[2]};
        filter_shape = {filter_shape[0], filter_shape[1], 1, filter_shape[2]};
        output_shape = {output_shape[0], output_shape[1], 1, output_shape[2]};
        window_dilation_strides = {1, window_dilation_strides[0]};
        window_movement_strides = {1, window_movement_strides[0]};
        data_dilation_strides = {1, data_dilation_strides[0]};
        padding_below_diff = {0, padding_below_diff[0]};
        padding_above_diff = {0, padding_above_diff[0]};
    }

    // emit code
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    size_t batch_count = input_shape[0];
    size_t input_channels = input_shape[1];
    size_t input_height = input_shape[2];
    size_t input_width = input_shape[3];
    size_t filter_count = filter_shape[0];
    size_t kernel_height = filter_shape[2];
    size_t kernel_width = filter_shape[3];
    size_t padding_left_height = padding_below_diff[1];
    size_t padding_left_width = padding_below_diff[0];
    size_t padding_right_height = padding_above_diff[1];
    size_t padding_right_width = padding_above_diff[0];
    size_t dilation_height = window_dilation_strides[0];
    size_t dilation_width = window_dilation_strides[1];
    size_t stride_height = window_movement_strides[0];
    size_t stride_width = window_movement_strides[1];
    size_t output_height = output_shape[2];
    size_t output_width = output_shape[3];

    auto code = op::create_code_from_template(
        R"(



    // auto engine_kind = dnnl::engine::kind::cpu;
    // // Create execution dnnl::engine.
    // dnnl::engine my_engine(engine_kind, 0);
    // // Create dnnl::stream.
    // dnnl::stream engine_stream(my_engine);

    // auto& my_engine = * global_my_engine;
    // auto& engine_stream = *global_my_stream;

    const memory::dim N = @batch_count@, // batch size
            IC = @input_channels@, // input channels
            IH = @input_height@, // input height
            IW = @input_width@, // input width
            OC = @filter_count@, // output channels
            KH = @kernel_height@, // weights height
            KW = @kernel_width@, // weights width
            PH_L = @padding_left_height@, // height padding: left
            PH_R = @padding_right_height@, // height padding: right
            PW_L = @padding_left_width@, // width padding: left
            PW_R = @padding_right_width@, // width padding: right
            SH = @stride_height@, // height-wise stride
            SW = @stride_width@, // width-wise stride
            OH = @output_height@, // output height
            OW = @output_width@; // output width

        // Source (src), weights, bias, and destination (dst) tensors
        // dimensions.
        memory::dims src_dims = {N, IC, IH, IW};
        memory::dims weights_dims = {OC, IC, KH, KW};
        memory::dims bias_dims = {OC};
        memory::dims dst_dims = {N, OC, OH, OW};

        // Strides, padding dimensions.
        memory::dims strides_dims = {SH, SW};
        memory::dims padding_dims_l = {PH_L, PW_L};
        memory::dims padding_dims_r = {PH_R, PW_R};



        // Create memory objects for tensor data (src, weights, dst). In this
        // example, NCHW layout is assumed for src and dst, and OIHW for weights.
        auto user_src_mem = memory({src_dims, dt::f32, tag::nchw}, my_engine, (void*) input0);
        auto user_weights_mem = memory({weights_dims, dt::f32, tag::oihw}, my_engine, (void*) input1);
        auto user_dst_mem = memory({dst_dims, dt::f32, tag::nchw}, my_engine, (void*) output0);

        // Create memory descriptors with format_tag::any for the primitive. This
        // enables the convolution primitive to choose memory layouts for an
        // optimized primitive implementation, and these layouts may differ from the
        // ones provided by the user.
        auto conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
        auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
        auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);

        // Create memory descriptor and memory object for input bias.
        auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
        auto user_bias_mem = memory(user_bias_md, my_engine, (void*) input2);


        // Create operation descriptor.
        auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
                algorithm::convolution_auto , conv_src_md, conv_weights_md, user_bias_md,
                conv_dst_md, strides_dims, padding_dims_l,
                padding_dims_r);

        // // Create primitive post-ops (ReLU).
        const float scale = 1.f;
        const float alpha = 0.f;
        const float beta = 0.f;
        post_ops conv_ops;
        conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
        primitive_attr conv_attr;
        conv_attr.set_post_ops(conv_ops);

        // Create primitive descriptor.
        auto conv_pd
                = convolution_forward::primitive_desc(conv_desc, my_engine);

        // For now, assume that the src, weights, and dst memory layouts generated
        // by the primitive and the ones provided by the user are identical.
        auto conv_src_mem = user_src_mem;
        auto conv_weights_mem = user_weights_mem;
        auto conv_dst_mem = user_dst_mem;

        // Reorder the data in case the src and weights memory layouts generated by
        // the primitive and the ones provided by the user are different. In this
        // case, we create additional memory objects with internal buffers that will
        // contain the reordered data. The data in dst will be reordered after the
        // convolution computation has finalized.
        // if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        //     conv_src_mem = memory(conv_pd.src_desc(), my_engine);
        //     reorder(user_src_mem, conv_src_mem)
        //             .execute(engine_stream, user_src_mem, conv_src_mem);
        // }

        // if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        //     conv_weights_mem = memory(conv_pd.weights_desc(), my_engine);
        //     reorder(user_weights_mem, conv_weights_mem)
        //             .execute(engine_stream, user_weights_mem, conv_weights_mem);
        // }

        // if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        //     conv_dst_mem = memory(conv_pd.dst_desc(), my_engine);
        // }

        // Create the primitive.
        auto conv_prim = convolution_forward(conv_pd);

        // Primitive arguments.
        std::unordered_map<int, memory> conv_args;
        conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
        conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
        conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
        conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

        // Primitive execution: convolution with ReLU.
        conv_prim.execute(engine_stream, conv_args);

        // Reorder the data in case the dst memory descriptor generated by the
        // primitive and the one provided by the user are different.
        // if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        //     reorder(conv_dst_mem, user_dst_mem)
        //             .execute(engine_stream, conv_dst_mem, user_dst_mem);
        // } else
        //     user_dst_mem = conv_dst_mem;

        // Wait for the computation to finalize.
        engine_stream.wait();


)",
        {{"batch_count", batch_count},
         {"input_channels", input_channels},
         {"input_height", input_height},
         {"input_width", input_width},
         {"filter_count", filter_count},
         {"kernel_height", kernel_height},
         {"kernel_width", kernel_width},
         {"padding_left_height", padding_left_height},
         {"padding_left_width", padding_left_width},
         {"padding_right_height", padding_right_height},
         {"padding_right_width", padding_right_width},
         {"dilation_height", dilation_height},
         {"dilation_width", dilation_width},
         {"stride_height", stride_height},
         {"stride_width", stride_width},
         {"output_height", output_height},
         {"output_width", output_width}});

    lu << code;

    return _lu;
}

LanguageUnit_p cpu::FuseConvolutionMkl::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::dnnl);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "FuseConvolution",                                                            // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mkl").Priority(7), // attrs
    cpu::FuseConvolutionMkl)                                                     // constructor
