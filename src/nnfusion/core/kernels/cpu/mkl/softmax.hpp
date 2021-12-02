// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cpu_kernel_emitter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class SoftmaxMkl : public MklKernelEmitter
            {
            public:
                SoftmaxMkl(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;


            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape input_shape, output_shape;
                nnfusion::AxisSet axes;
                uint32_t rank;
                string output_type;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion