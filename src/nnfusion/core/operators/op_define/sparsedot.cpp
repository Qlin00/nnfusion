// Microsoft (c) 2019, NNFusion Team

#include <functional>
#include <memory>
#include <utility>

#include "nnfusion/core/graph/gnode.hpp"
#include "sparsedot.hpp"

using namespace std;
using namespace nnfusion::op;

SparseDot::SparseDot(size_t reduction_axes_count,
                     bool has_reduction_axes_count,
                     bool trans_a,
                     bool trans_b,
                     size_t sparse_index,
                     size_t sparse_nnz,
                     Shape ori_sparse_shape)
    : Op("SparseDot")
    , m_reduction_axes_count(reduction_axes_count)
    , m_has_reduction_axes_count(has_reduction_axes_count)
    , m_transpose_A(trans_a)
    , m_transpose_B(trans_b)
    , m_sparse_index(sparse_index)
    , m_sparse_nnz(sparse_nnz)
    , m_sparse_shape(ori_sparse_shape)
{
}

SparseDot::SparseDot(shared_ptr<Dot> ori_dot,
                     size_t sparse_index,
                     size_t sparse_nnz,
                     Shape ori_sparse_shape)
    : Op("SparseDot")
    , m_sparse_index(sparse_index)
    , m_sparse_nnz(sparse_nnz)
    , m_sparse_shape(ori_sparse_shape)
// Initialize the SparseDot Op according to the original Dot Op
{
    m_reduction_axes_count = ori_dot->get_reduction_axes_count();
    m_transpose_A = ori_dot->get_transpose_A();
    m_transpose_B = ori_dot->get_transpose_B();
}
SparseDot::SparseDot(shared_ptr<Dot> ori_dot
)
    : Op("SparseDot")
// Initialize the SparseDot Op according to the original Dot Op
{
    m_reduction_axes_count = ori_dot->get_reduction_axes_count();
    m_transpose_A = ori_dot->get_transpose_A();
    m_transpose_B = ori_dot->get_transpose_B();
}

void SparseDot::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // No need to infer the output shape and type again, should
    // throw "Shape inference of sparsedot is not implemented yet";
    return;
}
