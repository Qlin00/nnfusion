#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class SparseDot : public CudaLibEmitter
            {
            public:
                SparseDot(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                // LanguageUnit_p emit_comments() override;
                bool require_cusparse_handle() override { return true; }
            private:
                shared_ptr<KernelContext> kernel_ctx;
                size_t reduction_axes;
                size_t sparse_nnz;
                nnfusion::Shape dense_shape, sparse_shape;
                nnfusion::Shape out_shape;
                nnfusion::element::Type dtype;
                
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
