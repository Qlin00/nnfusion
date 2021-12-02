// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "erf.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::ErfMkl::ErfMkl(shared_ptr<KernelContext> ctx)
    : MklKernelEmitter(ctx)
{
    
    arg0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());
    // assert(arg0_shape)
    std::stringstream tag;
    tag << "mklblas"
        << "_r_"  << "_i_" << join(arg0_shape, "_") << "_i_"
        << join(arg1_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::ErfMkl::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    size_t out_count = 1;
    for(auto i:out_shape){
        out_count = out_count * i;
    }
    // function signature:
    // void kernel(mcontext->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu << "vserf("<<out_count<<", input0, output0);\n";

    return _lu;
}

LanguageUnit_p cpu::ErfMkl::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cblas);

    _lu->require(header::vector);


    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Erf",                                                                   // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mkl").Priority(9), // attrs
    cpu::ErfMkl)
