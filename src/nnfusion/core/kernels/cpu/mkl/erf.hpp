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
            class ErfMkl : public MklKernelEmitter
            {
            public:
                ErfMkl(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape arg0_shape, arg1_shape;
                nnfusion::Shape out_shape;
                nnfusion::element::Type dtype;

            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion