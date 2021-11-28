

void MatrixMulCUDA_8bit_bias(float *input0, float *input1, float *input2, float *input3, float *input4, float *input5, float * input6, float *input7, float *output0) 
{
    float * A = reinterpret_cast<float*>(input0);
    float * val = reinterpret_cast<float*>(input1);
    int * row_ptr = reinterpret_cast<int*>(input2);
    int * col_idx = reinterpret_cast<int*>(input3);
    const int alpha = (int)(*input4);
    const int integer = (int)(*input5);
    int * C = reinterpret_cast< int *>(input6);             // new add
    float *a_buffer_whole = GLOBAL_MEMORY;   // new add
    uint8_t * C_int8 = reinterpret_cast<uint8_t*>(output0);

    const int M=M_VALUE;
    const int N=N_VALUE;
    const int K=K_VALUE;
    constexpr int M_BLOCKING = BLOCK_SIZE_M_VALUE;
    constexpr int N_BLOCKING = BLOCK_SIZE_N_VALUE;
    constexpr int K_BLOCKING = BLOCK_SIZE_K_VALUE;
    constexpr int M_THREAD_TILE = THREAD_SIZE_M_VALUE;
    constexpr int N_THREAD_TILE = THREAD_SIZE_N_VALUE;
    constexpr int M_ITER = (M_THREAD_TILE/16);
    constexpr int N_ITER = (N_THREAD_TILE/2);

    
    int m_inc = M_BLOCKING, n_inc = N_BLOCKING, k_inc = K_BLOCKING;

    //uint8_t *a_buffer_whole = (uint8_t *)aligned_alloc(4096, M * K * sizeof(uint8_t));
    uint8_t *a_buffer_whole = (uint8_t *)malloc(M * K * sizeof(uint8_t));
    uint8_t *b_buffer_whole = val;

#pragma omp parallel for num_threads(4) collapse(2)
    for(int m_count = 0; m_count < M; m_count += m_inc){
        for(int k_count = 0; k_count < K; k_count += k_inc){
            int block_id = (m_count / M_BLOCKING) * K / K_BLOCKING + k_count / K_BLOCKING;
            packing_a_k11(A+m_count+k_count*LDA, a_buffer_whole + block_id * m_inc * k_inc, LDA, m_inc, k_inc, M_THREAD_TILE);
        }
    }

#pragma omp parallel for num_threads(4) collapse(2)
    for (int n_count=0;n_count<N;n_count+=n_inc){
        for(int m_count=0; m_count<M; m_count+=m_inc){
            int start_idx = row_ptr[n_count / N_BLOCKING], end_idx = row_ptr[(n_count / N_BLOCKING) + 1];
            for(int idx = start_idx; idx < end_idx; idx += 1){
                int k_count = col_idx[idx] * K_BLOCKING;
                int block_id_a = (m_count / M_BLOCKING) * K / K_BLOCKING + k_count / K_BLOCKING;
                //packing_b_k11(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc);
                //packing_a_k11(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc);
                macro_kernel_k11(a_buffer_whole + block_id_a * m_inc * k_inc, b_buffer_whole + idx * n_inc * k_inc, m_inc, n_inc, k_inc, &C(m_count, n_count), LDC, alpha, M_THREAD_TILE, N_THREAD_TILE, M_ITER, N_ITER);
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i += 1){
        for(int j = 0; j < M; j += 1){
            C_int8[i * M + j] = (uint8_t)C[i * M + j];
        }
    }

}