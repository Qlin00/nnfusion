// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "softmax.hpp"
#include "../cpu_kernel_emitter.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::ReshapeMkl::ReshapeMkl(shared_ptr<KernelContext> ctx)
    : MklKernelEmitter(ctx)
{
    auto pad = static_pointer_cast<nnfusion::op::Sum>(ctx->gnode->get_op_ptr());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    axes = pad->get_reduction_axes();
    input_type = ctx->inputs[0]->get_element_type().c_type_string();
    output_type = ctx->outputs[0]->get_element_type().c_type_string();

    std::stringstream tag;
    tag << rank << "sum_i" << join(input_shape, "_") << "sum_o"
        << join(output_shape, "_") << "_axes" << join(axes, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::ReshapeMkl::emit_function_body()
{

    // emit code
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;



    auto code = op::create_code_from_template(
        R"(

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);



    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {@in_shape@};
    memory::dims out_dims = {@in_shape@};
    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
 

    // Create memory descriptors and memory objects for src and dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::abcd);
    auto dst_md = memory::desc(src_dims, dt::f32, tag::acbd); 

    auto src_mem = memory(src_md, engine, (void*)input0);
    auto dst_mem = memory(dst_md, engine, (void*)output0);

    // Write data to memory object's handle.


    // Dimension of the dst tensor where the output scales will be applied
    
    // Create primitive post-ops (per-channel output scales)
    primitive_attr reorder_attr;
    
    // Create primitive descriptor.
    auto reorder_pd = reorder::primitive_desc(
            engine, src_md, engine, dst_md, reorder_attr);

    // Create the primitive.
    auto reorder_prim = reorder(reorder_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC, src_mem});
    reorder_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: reorder with scaled sum.
    reorder_prim.execute(engine_stream, reorder_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    )",
        {
         {'in_shape', join(input_shape)}});

    lu << code;

    return _lu;
}

LanguageUnit_p cpu::ReshapeMkl::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::dnnl);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Reshape",                                                            // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mkl").Priority(7), // attrs
    cpu::ReshapeMkl)                                                     // constructor
