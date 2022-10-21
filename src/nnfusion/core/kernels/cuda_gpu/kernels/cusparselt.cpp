// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cusparselt.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::CusparseLT::CusparseLT(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto sparsenode = ctx->gnode;
    auto sparsedot = static_pointer_cast<nnfusion::op::SparseDot>(sparsenode->get_op_ptr());
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
    tag << "SparseDot initilization";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::CusparseLT::emit_function_body()
{
    static int call_count = 1;
    call_count += 1;
    auto& ctx = m_context;
    auto sparsenode = ctx->gnode;
    auto sparsedot = static_pointer_cast<nnfusion::op::SparseDot>(sparsenode->get_op_ptr());
    auto trans_A = sparsedot->get_transpose_A();
    auto trans_B = sparsedot->get_transpose_B();
    auto sparse_idx = sparsedot->get_sparse_index();
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    std::map<bool, string> trans_string = {{true, "CUSPARSE_OPERATION_TRANSPOSE"}, {false, "CUSPARSE_OPERATION_NON_TRANSPOSE"}};
    if(dtype==element::f32 && sparse_idx==1){
        int m, k, n;
        std::cout<<"Dense Shape";
        for (size_t i = 0; i < dense_shape.size(); i++)
        {
            /* code */
            std::cout<< dense_shape[i]<<" ";

        }
        std::cout<<std::endl;
        m = 1;
        for(int i=0;i<dense_shape.size()-1;i++)
            m = m* dense_shape[i];
        // m = dense_shape[0];
        k = dense_shape[dense_shape.size()-1];
        n = trans_B? sparse_shape[0]: sparse_shape[1];
        // currently only support sparse index equals to 1
        lu << "const float alpha = 1.0;\n const float beta = 0;\n";
        lu << "// static variables\n";
        lu << "static __init = 0;\n";
        lu << "static cusparseLtMatDescriptor_t matA, matB, matC;\n";
        lu << "static cusparseLtMatmulDescriptor_t matmul;\n";
        lu << "static cusparseLtMatmulAlgSelection_t alg_sel;\n";
        lu << "static cusparseLtMatmulPlan_t plan;\n";
        lu << "static cudaStream_t stream = nullptr;\n";
        lu << "static int alg=0;\n";
        lu << "static size_t workspace_size, compressed_size;\n";
        lu << "int callId = " << call_count<<";\n";
        lu << "if(__init==0){\n";
        lu << "\t __init=1;\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(&cusparselt_handle, &matA, num_A_rows, num_A_cols, lda, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT) )\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(&handle, &matB, num_B_rows, num_B_cols, ldb, alignment, type, order) )\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(&handle, &matC, num_C_rows, num_C_cols, ldc, alignment, type, order) )\n"; 
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type) )\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size) )\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA1, d_valid, stream) )\n";
        lu << "\t int is_valid;\n";
        lu << "\t CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream) )\n";
        lu << "\t CHECK_CUDA( cudaStreamSynchronize(stream) ) \n";
        lu << "\t assert(is_valid == 0);\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size) )\n";
        lu << "\t CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA1, dA_compressed, stream) )\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC,dD, d_workspace, streams, num_streams) )\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)) )\n"; 
        lu << "}\n";
        lu << "CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams, num_streams) )\n";
    }

    return _lu;
}



LanguageUnit_p cuda::CusparseLT::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cublas);
    _lu->require(header::stdexcept);
    _lu->require(header::sstream);
    _lu->require(header::cusparse);
    _lu->require(macro::CUSPARSE_SAFE_CALL);
    _lu->require(macro::CUDA_SAFE_CALL);

    // _lu->require(declaration::cuda_fp16_scale);
    //_lu->require(declaration::cublas_handle);
    return _lu;
}

LanguageUnit_p cuda::CusparseLT::emit_function_signature()
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
    << "(cusparseHandle_t cusparse_handle, " << join(params, ", ") << ")";
    return _lu;

}

REGISTER_KERNEL_EMITTER(
    "SparseDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cusparselt").Priority(3), // attrs
    cuda::CusparseLT)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "SparseDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cusparselt").Priority(3), // attrs
    cuda::CusparseLT)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "SparseDot",                                                                   // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cusparselt").Priority(3), // attrs
    cuda::CusparseLT)                                                               // constructor
