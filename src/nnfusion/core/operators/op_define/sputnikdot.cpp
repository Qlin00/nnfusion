// Microsoft (c) 2019, NNFusion Team

#include <functional>
#include <memory>
#include <utility>

#include "nnfusion/core/graph/gnode.hpp"
#include "sputnikdot.hpp"

using namespace std;
using namespace nnfusion::op;

SputnikDot::SputnikDot(size_t reduction_axes_count,
                     bool has_reduction_axes_count,
                     bool trans_a,
                     bool trans_b,
                     size_t sparse_index,
                     size_t sparse_nnz,
                     Shape ori_sparse_shape)
    : Op("SputnikDot")
    , m_reduction_axes_count(reduction_axes_count)
    , m_has_reduction_axes_count(has_reduction_axes_count)
    , m_transpose_A(trans_a)
    , m_transpose_B(trans_b)
    , m_sparse_index(sparse_index)
    , m_sparse_nnz(sparse_nnz)
    , m_sparse_shape(ori_sparse_shape)
{
}

SputnikDot::SputnikDot(shared_ptr<Dot> ori_dot,
                     int dim_m,
                     int dim_k,
                     int dim_n,
                     int sparse_nnz)
    : Op("SputnikDot")
    , m_dim_m(dim_m)
    , m_dim_k(dim_k)
    , m_dim_n(dim_n)
    , m_sparse_nnz(sparse_nnz)
// Initialize the SputnikDot Op according to the original Dot Op
{
    m_reduction_axes_count = ori_dot->get_reduction_axes_count();
    m_transpose_A = ori_dot->get_transpose_A();
    m_transpose_B = ori_dot->get_transpose_B();
}
SputnikDot::SputnikDot(shared_ptr<Dot> ori_dot
)
    : Op("SputnikDot")
// Initialize the SparseDot Op according to the original Dot Op
{
    m_reduction_axes_count = ori_dot->get_reduction_axes_count();
    m_transpose_A = ori_dot->get_transpose_A();
    m_transpose_B = ori_dot->get_transpose_B();
}

void SputnikDot::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // No need to infer the output shape and type again, should
    // throw "Shape inference of sparsedot is not implemented yet";
    return;
}
