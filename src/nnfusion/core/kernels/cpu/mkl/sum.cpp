// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sum.hpp"
#include "../cpu_kernel_emitter.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::SumMkl::SumMkl(shared_ptr<KernelContext> ctx)
    : MklKernelEmitter(ctx)
{
    auto pad = static_pointer_cast<nnfusion::op::Sum>(ctx->gnode->get_op_ptr());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

}

LanguageUnit_p cpu::SumMkl::emit_function_body()
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


    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {@in_shape@};
    memory::dims dst_dims = {@out_shape@};

    // Create src and dst memory descriptors and memory objects.
    auto src_md = memory::desc(src_dims, dt::f32, tag::any);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::any);

    auto src_mem = memory(src_md, my_engine, input0);
    auto dst_mem = memory(dst_md, my_engine, output0);


    // Create operation descriptor.
    auto reduction_d = reduction::desc(
            algorithm::reduction_sum, src_md, dst_md, 0.f, 0.f);

    // Create primitive descriptor.
    auto reduction_pd = reduction::primitive_desc(reduction_d, my_engine);

    // Create the primitive.
    auto reduction_prim = reduction(reduction_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> reduction_args;
    reduction_args.insert({DNNL_ARG_SRC, src_mem});
    reduction_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: Reduction (Sum).
    reduction_prim.execute(engine_stream, reduction_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    )",
        {{"out_shape", join(output_shape)},
         {"in_shape", join(input_shape)}});

    lu << code;

    return _lu;
}

LanguageUnit_p cpu::SumMkl::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::dnnl);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Sum",                                                            // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mkl").Priority(7), // attrs
    cpu::SumMkl)                                                     // constructor
