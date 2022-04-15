// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dot_transpose_placeholder.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::DotTransposePlaceholder::DotTransposePlaceholder(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    //
}

void cuda::DotTransposePlaceholder::init(const nnfusion::Shape& input_shape,
                                         const nnfusion::Shape& output_shape,
                                         const nnfusion::AxisVector& _input_order)
{
    arg_shape = input_shape;
    arg_rank = arg_shape.size();
    result_shape = output_shape;
    input_order = _input_order;
    size_t result_shape_product = shape_size(result_shape);

    //combine inordered dimensons after reorder in shape, update output shape and input order
    Shape in_order_map(arg_rank, 0);
    for (int i = 0; i < arg_rank - 1; i++)
    {
        if (static_cast<int64_t>(input_order[i + 1]) - static_cast<int64_t>(input_order[i]) == 1)
        {
            in_order_map[input_order[i]] = 1;
        }
    }

    Shape combine_arg_shape;
    Shape combine_idx_map(arg_rank, 0);
    Shape combine_input_order;
    size_t shape_i = 1;
    size_t combine_rank = 0;
    for (int i = 0; i < arg_rank; i++)
    {
        if (in_order_map[i] == 1)
        {
            shape_i *= arg_shape[i];
        }
        else
        {
            combine_arg_shape.push_back(shape_i * arg_shape[i]);
            shape_i = 1;
            combine_idx_map[i] = combine_rank++;
        }
    }

    for (int i = 0; i < arg_rank; i++)
    {
        if (in_order_map[input_order[i]] == 0)
        {
            combine_input_order.push_back(combine_idx_map[input_order[i]]);
        }
    }

    //eleminate dimenson size = 1, update input order and output shape
    Shape new_arg_shape;
    Shape new_result_shape;
    Shape new_idx_map(combine_rank, 0);
    Shape new_input_order;
    size_t new_rank = 0;

    for (int i = 0; i < combine_rank; i++)
    {
        if (combine_arg_shape[i] != 1)
        {
            new_arg_shape.push_back(combine_arg_shape[i]);
            new_idx_map[i] = new_rank++;
        }
    }
    for (int i = 0; i < combine_rank; i++)
    {
        if (combine_arg_shape[combine_input_order[i]] != 1)
        {
            new_input_order.push_back(new_idx_map[combine_input_order[i]]);
        }
    }
    for (int i = 0; i < new_rank; i++)
    {
        new_result_shape.push_back(new_arg_shape[new_input_order[i]]);
    }

    arg_shape = new_arg_shape;
    arg_rank = arg_shape.size();
    result_shape = new_result_shape;
    input_order = new_input_order;

    // <TODO> currently we set it to 16, will add tuning method later
    block_size = 16;
    input_strides = row_major_strides(arg_shape);
    output_strides = nnfusion::NVShape(arg_rank);
    trans_strides = nnfusion::NVShape(arg_rank);
    int stride = 1;
    for (int64_t i = arg_rank - 1; i >= 0; i--)
    {
        output_strides[i] = stride;
        stride *= arg_shape[input_order[i]];
    }
    for (int64_t i = 0; i < arg_rank; i++)
    {
        trans_strides[input_order[i]] = output_strides[i];
    }

    std::stringstream tag;
    tag << "cuda_reshape_2D"
        << "_i_" << join(arg_shape, "_") << "_o_" << join(input_order, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::DotTransposePlaceholder::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

LanguageUnit_p cuda::DotTransposePlaceholder::emit_function_body()
{
    if (is_noop || is_memcpy || arg_rank != 2)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "no kernel for DotTransposePlaceholder: " << is_noop
                                       << " " << is_memcpy << " " << arg_rank;
        return nullptr;
    }

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto& data_type = m_context->dtypes[1];
    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[2]* output0)
    //lu.block_begin();
    {
        // Common data area starts
        auto expand_vector_uint32 = [](string name, vector<uint32_t>& d)
        {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        lu << expand_vector_uint32("input_strides", input_strides);
        lu << expand_vector_uint32("trans_strides", trans_strides);
        lu << "size_t nx = " << arg_shape[1] << ";\n";
        lu << "size_t ny = " << arg_shape[0] << ";\n";
        // Common data area ends

        lu << "__shared__ " << data_type << " tile[" << block_size << "][" << block_size + 1
           << "];\n";
        lu << "uint32_t base1 = blockIdx.x * blockDim.x;\n";
        lu << "uint32_t base0 = blockIdx.y * blockDim.y;\n";
        lu << "uint32_t tid1 = threadIdx.x;\n";
        lu << "uint32_t tid0 = threadIdx.y;\n";
        lu << "uint32_t idx1 = base1 + tid1;\n";
        lu << "uint32_t idx0 = base0 + tid0;\n";

        lu << "if (idx1 < nx && idx0 < ny)\n";
        lu.block_begin();
        {
            lu << "uint32_t input_idx = 0;\n";
            for (int i = 0; i < 2; i++)
            {
                lu << "input_idx += input_strides" << i << "* idx" << i << ";\n";
            }
            lu << "tile[tid0][tid1] = input0[input_idx];\n";
        }
        lu.block_end();

        lu << "idx1 = base1 + tid0;\n";
        lu << "idx0 = base0 + tid1;\n";
        lu << "__syncthreads();\n";

        lu << "if (idx1 < nx && idx0 < ny)\n";
        lu.block_begin();
        {
            lu << "uint32_t output_idx = 0;\n";
            for (int i = 0; i < 2; i++)
            {
                lu << "output_idx += trans_strides" << i << "* idx" << i << ";\n";
            }
            lu << "output0[output_idx] = tile[tid1][tid0];\n";
        }
        lu.block_end();
    }
    //lu.block_end();

    return _lu;
}

void cuda::DotTransposePlaceholder::set_launch_config()
{
    uint32_t aligned_grid_size_x = align_to_block_size(arg_shape[1], block_size);
    uint32_t aligned_grid_size_y = align_to_block_size(arg_shape[0], block_size);

    m_gridDim = dim3(aligned_grid_size_x, aligned_grid_size_y, 1);
    m_blockDim = dim3(block_size, block_size, 1);
}

// Register Reshape kernel emitter

REGISTER_KERNEL_EMITTER(
    "DotTransposePlaceholder",                                             // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda").Priority(2), // attrs
    cuda::DotTransposePlaceholder)                                         // constructor
