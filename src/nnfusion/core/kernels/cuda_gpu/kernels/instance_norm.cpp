// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../cuda_cudnn.hpp"
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class InstanceNorm : public CudaLibEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                InstanceNorm(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << "\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("InstanceNorm",                                               // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudalib"), // attrs
                        cuda::InstanceNorm)                                           // constructor
