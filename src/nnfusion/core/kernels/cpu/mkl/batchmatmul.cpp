// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "batchmatmul.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::BatchMatMulMkl::BatchMatMulMkl(shared_ptr<KernelContext> ctx)
    : MklKernelEmitter(ctx)
{
    auto dot_op = static_pointer_cast<nnfusion::op::Dot>(ctx->gnode->get_op_ptr());

    reduction_axes = dot_op->get_reduction_axes_count();
    arg0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    arg1_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());

    std::stringstream tag;
    tag << "mklblas"
        << "_r_" << reduction_axes << "_i_" << join(arg0_shape, "_") << "_i_"
        << join(arg1_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::BatchMatMulMkl::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    size_t src_dim = arg0_shape.size();
    size_t weight_dim = arg1_shape.size();
    size_t batch_count = 1;
    size_t K = arg0_shape[src_dim-1];
    size_t M = arg0_shape[src_dim-2];
    size_t N = arg1_shape[src_dim-1];
    for(int i=0; i<src_dim-2;i++){
        batch_count *= arg0_shape[i];
    }

    // function signature:
    // void kernel(mcontext->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    auto code = op::create_code_from_template(
    R"(
    // dnnl::engine my_engine(engine_kind, 0);

    // Create dnnl::stream.
    //dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim MB = @batch_count@, // batch size
            M = @M@, K = @K@, N = @N@;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {MB, M, K};
    memory::dims weights_dims = {MB, K, N};
    // memory::dims bias_dims = {1, 1, N};
    memory::dims dst_dims = {MB, M, N};


    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::abc);
    auto weights_md = memory::desc(weights_dims, dt::f32, tag::abc);
    // auto bias_md = memory::desc(bias_dims, dt::f32, tag::abc);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::abc);

    auto src_mem = memory(src_md, my_engine, (void*)input0);
    auto weights_mem = memory(weights_md, my_engine, (void*)input1);
    // auto bias_mem = memory(bias_md, my_engine, (void*));
    auto dst_mem = memory(dst_md, my_engine, (void*)output0);

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, dst_md);

    // Create primitive post-ops (ReLU).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;
    // matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    primitive_attr matmul_attr;
    // matmul_attr.set_post_ops(matmul_ops);

    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, my_engine);

    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    // matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);

    // Wait for the computation to finalize.
    engine_stream.wait();
)",
        {{"batch_count", batch_count},
         {"M", M},
         {"K", K},
         {"N", N}});

    lu << code;


    return _lu;
}

LanguageUnit_p cpu::BatchMatMulMkl::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cblas);
    _lu->require(header::dnnl);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "BatchMatMul",                                                                   // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mkl").Priority(9), // attrs
    cpu::BatchMatMulMkl)
