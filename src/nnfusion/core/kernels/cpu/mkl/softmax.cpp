// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "softmax.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::SoftmaxMkl::SoftmaxMkl(shared_ptr<KernelContext> ctx)
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

LanguageUnit_p cpu::SoftmaxMkl::emit_function_body()
{

    // emit code
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;



    auto code = op::create_code_from_template(
        R"(

    // Create execution dnnl::engine.
    // dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    // dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = @N@, // batch size
            IC = 1000; // channels

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC};

    // Allocate buffer.
    std::vector<float> src_data(product(src_dims));

    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Create src memory descriptor and memory object.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nc);
    auto src_mem = memory(src_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Softmax axis.
    const int axis = 1;

    // Create operation descriptor.
    auto softmax_d
            = softmax_forward::desc(prop_kind::forward_training, src_md, axis);

    // Create primitive descriptor.
    auto softmax_pd = softmax_forward::primitive_desc(softmax_d, engine);

    // Create the primitive.
    auto softmax_prim = softmax_forward(softmax_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, src_mem});

    // Primitive execution.
    softmax_prim.execute(engine_stream, softmax_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(src_data.data(), src_mem);

)",
        {{"batch_count", batch_count}});

    lu << code;

    return _lu;
}

LanguageUnit_p cpu::SoftmaxMkl::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::dnnl);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Softmax",                                                            // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mkl").Priority(7), // attrs
    cpu::SoftmaxMkl)                                                     // constructor
