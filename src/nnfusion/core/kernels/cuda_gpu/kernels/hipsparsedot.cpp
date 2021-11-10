// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hipsparsedot.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::HipSparseDot::HipSparseDot(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto sparsenode = ctx->gnode;
    auto sparsedot = static_pointer_cast<nnfusion::op::HipSparseDot>(sparsenode->get_op_ptr());
    reduction_axes = sparsedot->get_reduction_axes_count();
    auto sparse_idx = sparsedot->get_sparse_index();
    auto dense_idx = 1-sparse_idx;
    // row_idx, col_idx, values, other input
    dense_shape = sparsenode->get_input_tensor_ptr(3)->get_shape();
    sparse_nnz = sparsedot->get_sparse_nnz();
    sparse_shape = sparsedot->get_sparse_shape();
    out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());

    std::stringstream tag;
    tag << "HipSparseDot initilization";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::HipSparseDot::emit_function_body()
{
    auto& ctx = m_context;
    auto sparsenode = ctx->gnode;
    auto sparsedot = static_pointer_cast<nnfusion::op::HipSparseDot>(sparsenode->get_op_ptr());
    auto trans_A = sparsedot->get_transpose_A();
    auto trans_B = sparsedot->get_transpose_B();
    auto sparse_idx = sparsedot->get_sparse_index();

    int m = sparsedot->get_dim_m();
    int k = sparsedot->get_dim_k();
    int n = sparsedot->get_dim_n();
    int nnz = sparsedot->get_sparse_nnz();
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    if(dtype == element::f32){
        lu<< "float * dense_m = input0;\n";
        lu<< "int m = " << m <<";\n";
        lu<< "int k = " << k <<";\n";
        lu<< "int n = " << n <<";\n";
        lu<< "int nnz = "<< nnz << ";\n";
        lu<< "float alpha=1.0;\n";
        lu<< "float beta = 0.0;\n";
        lu<< "float * values = input1;\n";
        lu<< "int * row_idx = (int *) input2;\n";
        lu<< "int * col_idx = (int *) input3;\n";
        lu<< "float * output_m = output0;\n";
        lu<< "hipsparseMatDescr_t descrA;\n";
        lu<< "hipsparseCreateMatDescr(&descrA);\n";
        lu<< "hipsparseScsrmm(hipsparse_handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, nnz, &alpha, descrA, values, row_idx, col_idx, dense_m, k, &beta, output0, m);";

    }
    return _lu;
}

LanguageUnit_p cuda::HipSparseDot::emit_comments()
{
    auto& ctx = m_context;


    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;


    lu<<"//HipSparseDot function commments here\n";
    //lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::HipSparseDot::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cublas);
    _lu->require(header::stdexcept);
    _lu->require(header::sstream);
    _lu->require(header::hipsparse);
    _lu->require(macro::CUSPARSE_SAFE_CALL);
    _lu->require(macro::CUDA_SAFE_CALL);

    // _lu->require(declaration::cuda_fp16_scale);
    //_lu->require(declaration::cublas_handle);
    return _lu;
}

LanguageUnit_p cuda::HipSparseDot::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }
    lu << "void "
    << "(hipsparseHandle_t hipsparse_handle, " << join(params, ", ") << ")";
    return _lu;

}

REGISTER_KERNEL_EMITTER(
    "HipSparseDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cusparse").Priority(2), // attrs
    cuda::HipSparseDot)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "HipSparseDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cusparse").Priority(2), // attrs
    cuda::HipSparseDot)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "HipSparseDot",                                                                   // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cusparse").Priority(2), // attrs
    cuda::HipSparseDot)                                                               // constructor
