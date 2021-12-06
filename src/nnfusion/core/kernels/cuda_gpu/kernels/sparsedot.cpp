// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sparsedot.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::SparseDot::SparseDot(shared_ptr<KernelContext> ctx)
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

LanguageUnit_p cuda::SparseDot::emit_function_body()
{
    auto& ctx = m_context;
    auto sparsenode = ctx->gnode;
    auto sparsedot = static_pointer_cast<nnfusion::op::SparseDot>(sparsenode->get_op_ptr());
    auto trans_A = sparsedot->get_transpose_A();
    auto trans_B = sparsedot->get_transpose_B();
    auto sparse_idx = sparsedot->get_sparse_index();
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    std::map<bool, string> trans_string = {{true, "CUSPARSE_OPERATION_TRANSPOSE"}, {false, "CUSPARSE_OPERATION_NON_TRANSPOSE"}};
    if(dtype == element::f32){
        lu << "const float alpha = 1.0;\n const float beta = 0;\n";

        lu<< "//Create the sparse matrix description\n";
        lu<< "cusparseMatDescr_t descrA = NULL;\n";
        lu<< "CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&descrA));\n";
        lu<< "cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);\n";
        lu<< "cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );\n";
        if(sparse_idx == 0){
            // calculate in row-major
            lu<< "CUSPARSE_SAFE_CALL()";
        }else if(sparse_idx == 1){
            // calculate in col-major
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
            std::cout<< "Cusparse "<<" M:"<<m<<"  K:"<<k<< " N:"<<n<<std::endl;
            std::cout<< "Trans_B: "<< trans_B<<std::endl;
            if(trans_B){
                std::cout<<k<<" "<<sparse_shape[0]<<" "<<sparse_shape[1]<<std::endl;
                assert(k == sparse_shape[1]);
            }else{
                assert(k == sparse_shape[0]);
            }
            lu << "CUSPARSE_SAFE_CALL(cusparseScsrmm("
               << "cusparse_handle"\
               << ", "<<trans_string[!trans_B]\
               << ", "<<n //M
               << ", "<<m //N
               << ", "<<k //K
               << ", "<<sparse_nnz
               << ", &alpha"
               << ", descrA"
               << ", input2"
               << ", input0"
               << ", input1"
               << ", input3"
               << ", "<<k  //LDB
               << ",&beta"
               << ",output0"
               << ","<<n<<"));"; //LDC
        }else{
            throw "Invalid sparse index for the SparseDot operation!";
        }
    }
    // cuda 11.1
    // if(dtype == element::f32){
    //     lu << "const float alpha = 1.0;\n const float beta = 0;\n";

    //     lu<< "//Create the sparse matrix description\n";
    //     lu<< "cusparseSpMatDescr_t* spMatDescr;\n";
    //     lu<< "CUSPARSE_SAFE_CALL(cusparseCreateCsr(spMatDescr"<<\
    //         sparse_shape[0]<<", "<<sparse_shape[1]<<", "<<sparse_nnz<<\ 
    //         "static_cast<void*>(input1),"<<\
    //         "static_cast<void*>(input2), "<<
    //         "static_cast<void*>(input3)," <<
    //         "CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,"<<
    //         "CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)";
        
    //     lu<< "// Call the CuSparse SPMM here\n";
    //     std::map<bool, string> trans_string = {{true, "CUSPARSE_OPERATION_TRANSPOSE"}, {false, "CUSPARSE_OPERATION_NON_TRANSPOSE"}};
    //     if(sparse_idx==0){
    //         // sparse matrix * dense matrix
    //         // calculate in the row-major
    //     }else{
    //         lu<< "// Create the dense matrix description\n";
    //         lu<< "cusparseDnMatDescr_t* dnMatDescr;\n";
    //         lu<< "CUSPARSE_SAFE_CALL(cusparseCreateDnMat(dnMatDescr \\ \n";
    //         // need transpose the dense matrix due to the col_major;
    //         lu<< ","<< dense_shape[1];
    //         lu<< ","<< dense_shape[2];
    //         lu<< ",ld// to be done";
    //         lu<< ",(void*) input4";
    //         lu<< ",   CUDA_R_32F, CUSPARSE_ORDER_COL))";

    //         lu<< "// Create the dense matrix description\n";
    //         lu<< "cusparseDnMatDescr_t* dnMatDescr;\n";
    //         lu<< "CUSPARSE_SAFE_CALL(cusparseCreateDnMat(dnMatDescr \\ \n";
    //         // need transpose the dense matrix due to the col_major;
    //         lu<< ","<< dense_shape[1];
    //         lu<< ","<< dense_shape[2];
    //         lu<< ",ld// to be done";
    //         lu<< ",(void*) input4";
    //         lu<< ",   CUDA_R_32F, CUSPARSE_ORDER_COL))";
    //         // dense matrix * sparse matrix
    //         //calculate in the column major
    //         lu<<"cusparseSpMM(cusparse_handle, "<<trans_string[!trans_A]<<", "<< \
    //         trans_string[trans_B]<< ",&alpha"<<",*spMatDescr"<<",*dnMatDescr"<< \
    //         ",&beta"<<",*dnMatDescr_out"<<",CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, externalBuffer);";
    //     }
    // }
    return _lu;
}

// LanguageUnit_p cuda::SparseDot::emit_comments()
// {
//     auto& ctx = m_context;


//     LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
//     auto& lu = *_lu;


//     lu<<"//SparseDot function commments here\n";
//     //lu.block_end();
//     return _lu;
// }

LanguageUnit_p cuda::SparseDot::emit_dependency()
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

LanguageUnit_p cuda::SparseDot::emit_function_signature()
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
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cusparse").Priority(2), // attrs
    cuda::SparseDot)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "SparseDot",                                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cusparse").Priority(2), // attrs
    cuda::SparseDot)                                                               // constructor

REGISTER_KERNEL_EMITTER(
    "SparseDot",                                                                   // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cusparse").Priority(2), // attrs
    cuda::SparseDot)                                                               // constructor
