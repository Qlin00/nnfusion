
// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"

namespace nnfusion{
    namespace op{
        class SparseDot: public Op
        {
        public:
            SparseDot(size_t reduction_axes_count,
            bool has_reduction_axes_count = true,
            bool trans_a = false,
            bool trans_b = false,
            size_t sparse_index=0,
            size_t sparse_nnz=0);
            SparseDot();
            SparseDot(std::shared_ptr<Dot> ori_dot, size_t sparse_index, size_t sparse_nnz);
    
            void validate_and_infer_types(std::shared_ptr<graph::GNode>) override;
            size_t get_reduction_axes_count() const { return m_reduction_axes_count; }
            void set_transpose(bool trans_a, bool trans_b)
            {
                m_transpose_A = trans_a;
                m_transpose_B = trans_b;
            }

            bool& get_transpose_A() { return m_transpose_A; }
            bool& get_transpose_B() { return m_transpose_B; }
            size_t& get_sparse_index() { return m_sparse_index; }
            size_t& get_sparse_nnz() { return m_sparse_nnz;}
        protected:
            size_t m_reduction_axes_count;
            bool m_has_reduction_axes_count;
            bool m_transpose_A = false;
            bool m_transpose_B = false;
            // indicate the index of the sparse matrix 0
            size_t m_sparse_index;
            size_t m_sparse_nnz;
        };
    }
}