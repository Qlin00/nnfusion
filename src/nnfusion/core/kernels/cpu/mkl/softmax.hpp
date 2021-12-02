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
                nnfusion::Shape input_shape, filter_shape, output_shape, padding;

                std::string dtype, data_format;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion