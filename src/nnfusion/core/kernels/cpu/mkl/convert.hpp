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
            class ConvertMkl : public MklKernelEmitter
            {
            public:
                ConvertMkl(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;


            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape input_shape, output_shape;
                nnfusion::AxisSet axes;
                string output_type, input_type;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion