// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cusparselt_int8.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::CusparseLTInt8::CusparseLTInt8(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto sparsenode = ctx->gnode;
    auto sparsedot = static_pointer_cast<nnfusion::op::Dot>(sparsenode->get_op_ptr());
    reduction_axes = sparsedot->get_reduction_axes_count();
    // auto sparse_idx = sparsedot->get_sparse_index();
    // auto dense_idx = 1-sparse_idx;
    // row_idx, col_idx, values, other input
    dense_shape = sparsenode->get_input_tensor_ptr(0)->get_shape();
    // sparse_nnz = sparsedot->get_sparse_nnz();
    weight_shape = sparsenode->get_input_tensor_ptr(1)->get_shape();
    sparse_nnz = 1;
    for(int i=0;i<weight_shape.size();i++)
        sparse_nnz *= weight_shape[i];
    sparse_nnz /= 2; // cusparselt only support 50% sparsity
    out_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());

    std::stringstream tag;
    tag << "CusparseLTInt8 emitter initilization";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::CusparseLTInt8::emit_function_body()
{
    static int call_count = 1;
    call_count += 1;
    auto& ctx = m_context;
    auto sparsenode = ctx->gnode;
    auto sparsedot = static_pointer_cast<nnfusion::op::Dot>(sparsenode->get_op_ptr());
    auto trans_A = sparsedot->get_transpose_A();
    auto trans_B = sparsedot->get_transpose_B();
    // auto sparse_idx = sparsedot->get_sparse_index();
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    std::map<bool, string> trans_string = {{true, "CUSPARSE_OPERATION_TRANSPOSE"}, {false, "CUSPARSE_OPERATION_NON_TRANSPOSE"}};
    if(dtype==element::f32){
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
        
        int weight_numel = 1;
        for(int i=0;i<weight_shape.size();i++){
            weight_numel *= weight_shape[i];
        }
        n = weight_numel / k;
        // currently only support sparse index equals to 1
        lu << "const float alpha = 1.0;\n const float beta = 0;\n";
        lu << "//M:"<<m<<" K:"<<k<<" N:"<<n<<"\n";
        lu << "// static variables\n";
        lu << "static int __init = 0;\n";
        lu << "static cusparseLtMatDescriptor_t matA, matB, matC;\n";
        lu << "static cusparseLtMatmulDescriptor_t matmul;\n";
        lu << "static cusparseLtMatmulAlgSelection_t alg_sel;\n";
        lu << "static cusparseLtMatmulPlan_t plan;\n";
        lu << "static cudaStream_t stream = nullptr;\n";
        lu << "static int alg=0;\n";
        lu << "static size_t workspace_size, compressed_size;\n";
        lu << "auto num_A_rows=" << m <<";\n";
        lu << "auto num_A_cols=" << k <<";\n";
        lu << "auto num_B_rows=" << n <<";\n";
        lu << "auto num_B_cols=" << k <<";\n";
        lu << "auto num_C_rows=" << m <<";\n";
        lu << "auto num_C_cols=" << n <<";\n";
        lu << "unsigned alignment = 16;\n";
        lu << "auto order = CUSPARSE_ORDER_ROW;\n";
        lu << "auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;\n";
        lu << "auto opB = CUSPARSE_OPERATION_TRANSPOSE;\n";
        lu << "auto type  = CUDA_R_8I;\n";
        lu << "auto compute_type = CUSPARSE_COMPUTE_32I;\n";
        lu << "bool is_rowmajor = (order == CUSPARSE_ORDER_ROW);\n";
        lu << "bool isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);\nbool isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);\n";
        lu << R"(auto lda = (is_rowmajor) ? num_A_cols : num_A_rows;
auto ldb = (is_rowmajor) ? num_B_cols : num_B_rows;
auto ldc = (is_rowmajor) ? num_C_cols : num_C_rows;
auto A_height = (is_rowmajor) ? num_A_rows : num_A_cols;
auto B_height = (is_rowmajor) ? num_B_rows : num_B_cols;
auto C_height = (is_rowmajor) ? num_C_rows : num_C_cols;
auto A_size = A_height * lda * sizeof(float);
auto B_size = B_height * ldb * sizeof(float);
auto C_size = C_height * ldc * sizeof(float);
// static float *dA, *dB, *dC, *dD, *dA_compressed;
static int8_t * dA, *dB, * dC, *d_compressed;
dA = (int8_t*) input0;
dB = (int8_t*) input1;
// static float* d_compressed;
static int *d_valid;
static int is_valid;
static void* d_workspace = nullptr;
static int num_streams = 0;
static cudaStream_t* streams = nullptr;
static int alg_id;

        )";
        lu << "int callId = " << call_count<<";\n";
        lu << "if(__init==0){\n";
        lu << "\t __init=1;\n";
        lu << R"(
        // CUDA_SAFE_CALL( cudaMalloc((void**) &dA, A_size) );
        // CUDA_SAFE_CALL( cudaMalloc((void**) &dB, B_size) );
        // CUDA_SAFE_CALL( cudaMalloc((void**) &dC, C_size) );
        CUDA_SAFE_CALL( cudaMalloc((void**) &d_valid, sizeof(d_valid)) );
        )";
        lu << "\t CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(&cusparselt_handle, &matA, num_A_rows, num_A_cols, lda, alignment, type, order) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(&cusparselt_handle, &matB, num_B_rows, num_B_cols, ldb, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT) );\n";
        // lu << "\t CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(&cusparselt_handle, &matA, num_A_rows, num_A_cols, lda, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT) );\n";
        // lu << "\t CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(&cusparselt_handle, &matB, num_B_rows, num_B_cols, ldb, alignment, type, order) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(&cusparselt_handle, &matC, num_C_rows, num_C_cols, ldc, alignment, type, order) );\n"; 
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(&cusparselt_handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(&cusparselt_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(&cusparselt_handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&cusparselt_handle, &alg_sel, &workspace_size) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&cusparselt_handle, &plan, &matmul, &alg_sel, workspace_size) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtSpMMAPrune(&cusparselt_handle, &matmul, dB, dB, CUSPARSELT_PRUNE_SPMMA_TILE, stream) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&cusparselt_handle, &matmul, dB, d_valid, stream) );\n";
        lu << "\t CUDA_SAFE_CALL( cudaMemcpy(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost) );\n";
        lu << "\t CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );\n";
        lu << "\t assert(is_valid == 0);\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&cusparselt_handle, &plan, &compressed_size) );\n";
        lu << "\t CUDA_SAFE_CALL( cudaMalloc((void**) &d_compressed, compressed_size) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtSpMMACompress(&cusparselt_handle, &plan, dB, d_compressed, stream) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulSearch(&cusparselt_handle, &plan, &alpha, dA, d_compressed, &beta, output0, output0, d_workspace, streams, num_streams) );\n";
        lu << "\t CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&cusparselt_handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)) );\n"; 
        lu << "}\n";
        lu << "CHECK_CUSPARSE( cusparseLtMatmul(&cusparselt_handle, &plan, &alpha, dA, d_compressed, &beta, output0, output0, d_workspace, streams, num_streams) );\n";
    }

    return _lu;
}



LanguageUnit_p cuda::CusparseLTInt8::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cublas);
    _lu->require(header::stdexcept);
    _lu->require(header::sstream);
    _lu->require(header::cusparselt);
    _lu->require(macro::CUSPARSE_SAFE_CALL);
    _lu->require(macro::CHECK_CUSPARSE);
    _lu->require(macro::CUDA_SAFE_CALL);

    // _lu->require(declaration::cuda_fp16_scale);
    //_lu->require(declaration::cublas_handle);
    return _lu;
}

LanguageUnit_p cuda::CusparseLTInt8::emit_function_signature()
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
    << "(cusparseLtHandle_t cusparselt_handle, " << join(params, ", ") << ")";
    return _lu;

}

// REGISTER_KERNEL_EMITTER(
//     "SparseDot",                                                                   // op_name
//     Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cusparselt").Priority(3), // attrs
//     cuda::CusparseLTInt8)                                                               // constructor

// REGISTER_KERNEL_EMITTER(
//     "SparseDot",                                                                   // op_name
//     Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cusparselt").Priority(3), // attrs
//     cuda::CusparseLTInt8)                                                               // constructor

// REGISTER_KERNEL_EMITTER(
//     "SparseDot",                                                                   // op_name
//     Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cusparselt").Priority(3), // attrs
//     cuda::CusparseLTInt8)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "Dot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cusparselt").Priority(1), // attrs
    cuda::CusparseLTInt8)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "Dot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cusparselt").Priority(1), // attrs
    cuda::CusparseLTInt8)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "Dot",                                                                   // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cusparselt").Priority(1), // attrs
    cuda::CusparseLTInt8)                                                               // constructor
