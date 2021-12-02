// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sub.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::SubMkl::SubMkl(shared_ptr<KernelContext> ctx)
    : MklKernelEmitter(ctx)
{
    auto add_op = static_pointer_cast<nnfusion::op::Add>(ctx->gnode->get_op_ptr());

    arg0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    arg1_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());
    // assert(arg0_shape)
    std::stringstream tag;
    tag << "mklblas"
        << "_r_"  << "_i_" << join(arg0_shape, "_") << "_i_"
        << join(arg1_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::SubMkl::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    size_t out_count = 1;
    for(auto i:out_shape){
        out_count = out_count * i;
    }
    // function signature:
    // void kernel(mcontext->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu << "vsSub("<<out_count<<", input0, input1, output0);\n";

    return _lu;
}

LanguageUnit_p cpu::SubMkl::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cblas);



    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Subtract",                                                                   // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mkl").Priority(9), // attrs
    cpu::SubMkl)
