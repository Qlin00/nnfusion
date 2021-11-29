// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "mklsparsedot.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::MklSparseDot::MklSparseDot(shared_ptr<KernelContext> ctx)
    : MklKernelEmitter(ctx)
{
    auto dot_op = static_pointer_cast<nnfusion::op::MyDot>(ctx->gnode->get_op_ptr());

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

LanguageUnit_p cpu::MklSparseDot::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto& ctx = m_context;
    auto sparsenode = ctx->gnode;
    auto sparsedot = static_pointer_cast<nnfusion::op::SputnikDot>(sparsenode->get_op_ptr());
    auto trans_A = sparsedot->get_transpose_A();
    auto trans_B = sparsedot->get_transpose_B();
    auto sparse_idx = sparsedot->get_sparse_index();

    int m = sparsedot->get_dim_m();
    int k = sparsedot->get_dim_k();
    int n = sparsedot->get_dim_n();
    int nnz = sparsedot->get_sparse_nnz();
    // function signature:
    // void kernel(mcontext->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    auto code = op::create_code_from_template(
        R"(
    float *dense_m = input0;
    float *values = input1;
    float *rowIndex = (MKL_INT *)input2;
    float *columns = (MKL_INT *)input3;
    
    // NxKxM mkl sparse support sparse_matrix * dense matrix
    status = mkl_sparse_s_create_csr(&SA, SPARSE_INDEX_BASE_ZERO, @M@, @K@, rowIndex, &(rowIndex[1]), columns, values);
    if (status != SPARSE_STATUS_SUCCESS)
    {
        printf("CSR Sparse matrix created failed.\n");
        return -2;
    }
    // Two Stage algorithms
    // (1) inspector
    // (2) executor

    int niter = 100;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_LOWER;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    status = mkl_sparse_set_mm_hint(SA, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_ROW_MAJOR, @K@, niter);

    if (status != SPARSE_STATUS_SUCCESS)
    {
        printf("Analysis failed!!\n");
        return -3;
    }
    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, SA, descr, SPARSE_LAYOUT_ROW_MAJOR, dense_m, @N@, @N@, beta, output0, @N@);

)",
        {{"K", k},
         {"N", n},
         {"M", m}});
    lu << code;

    return _lu;
}

LanguageUnit_p cpu::MklSparseDot::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cblas);

    if ((arg0_shape.size() == 3) && (arg1_shape.size() == 2) && reduction_axes == 1)
    {
        _lu->require(header::vector);
    }

    return _lu;
}


REGISTER_KERNEL_EMITTER(
    "MklSparseDot",                                                                   // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mkl").Priority(7), // attrs
    cpu::MklSparseDot)
