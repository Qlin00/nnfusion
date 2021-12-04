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
            class ReshapeMkl : public MklKernelEmitter
            {
            public:
                ReshapeMkl(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;


            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape input_shape, output_shape;
                nnfusion::AxisSet axes;
                string output_type, input_type;
                string in_layout, out_layout;
                bool is_copy;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion