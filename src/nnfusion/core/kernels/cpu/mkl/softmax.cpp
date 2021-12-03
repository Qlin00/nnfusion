// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "softmax.hpp"
#include "../cpu_kernel_emitter.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::SoftmaxMkl::SoftmaxMkl(shared_ptr<KernelContext> ctx)
    : MklKernelEmitter(ctx)
{
    auto pad = static_pointer_cast<nnfusion::op::Softmax>(ctx->gnode->get_op_ptr());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    axes = pad->get_axes();
    output_type = ctx->outputs[0]->get_element_type().c_type_string();

    rank = static_cast<uint32_t>(input_shape.size());

    std::stringstream tag;
    tag << rank << "softmax_i" << join(input_shape, "_") << "softmax_o"
        << join(output_shape, "_") << "_axes" << join(axes, "_");
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
    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {@in_shape@};



    // Create src memory descriptor and memory object.
    auto src_md = memory::desc(src_dims, dt::f32, tag::abcd);
    auto src_mem = memory(src_md, my_engine, (void*)input0);

    auto dst_md = memory::desc(src_dims, dt::f32, tag::abcd);
    auto dst_mem = memory(dst_md, my_engine, (void*)output0);

    // Softmax axis.
    const int axis = @r_axis@;

    // Create operation descriptor.
    auto softmax_d
            = softmax_forward::desc(prop_kind::forward_inference, src_md, axis);

    // Create primitive descriptor.
    auto softmax_pd = softmax_forward::primitive_desc(softmax_d, my_engine);

    // Create the primitive.
    auto softmax_prim = softmax_forward(softmax_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution.
    softmax_prim.execute(engine_stream, softmax_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

)",
        {{"r_axis", 3},
         {"in_shape", join(input_shape)}});

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
