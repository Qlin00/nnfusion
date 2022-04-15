// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class DotTransposePlaceholder : public CudaEmitter
            {
            public:
                DotTransposePlaceholder(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

                void init(const nnfusion::Shape& input_shape,
                          const nnfusion::Shape& output_shape,
                          const nnfusion::AxisVector& input_order);

            protected:
                nnfusion::Shape arg_shape;
                size_t arg_rank;
                nnfusion::Shape result_shape;
                nnfusion::AxisVector input_order;
                shared_ptr<nnfusion::op::Reshape> reshape;
                bool is_memcpy = false;
                bool is_noop;
                uint32_t block_size;
                NVShape input_strides;
                NVShape output_strides;
                NVShape trans_strides;
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion