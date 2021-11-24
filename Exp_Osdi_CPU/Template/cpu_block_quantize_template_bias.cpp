void packing_a_k11(uint8_t *src, uint8_t *dst, int leading_dim, int dim_first, int dim_second, int M_THREAD_TILE){
    //dim_first: M, dim_second: K
    uint8_t *tosrc,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_first;
    for (count_first=0;count_sub>M_THREAD_TILE-1;count_first+=M_THREAD_TILE,count_sub-=M_THREAD_TILE){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second+=4){
            #pragma unroll
            /*
            for(int i = 0; i < M_THREAD_TILE; i+=1){
                *(todst+i) = *(tosrc+i);
            }
            */
            for(int i = 0; i < M_THREAD_TILE; i+=1){
                *(todst+i*4) = *(tosrc+i);
                *(todst+i*4+1) = *(tosrc+i+leading_dim);
                *(todst+i*4+2) = *(tosrc+i+2*leading_dim);
                *(todst+i*4+3) = *(tosrc+i+3*leading_dim);
            }
            tosrc+=4*leading_dim;
            todst+=4*M_THREAD_TILE;
        }
    }
    // edge case

}


void kernel_n_n_v2_k11(uint8_t *a_buffer,uint8_t *b_buffer,int *c_ptr,int m,int K,int LDC,int alpha, int M_THREAD_TILE, int N_THREAD_TILE, int M_ITER, int N_ITER){
    int m_count,m_count_sub;
    int i,j,k;
    int *C=c_ptr;
    __m512i b0,b1;
    int k_start,k_end,K4;
    K4=K&-4;k_end=K;k_start=0;
    // printf("*****\n");
    // print_matrix(C,m,8);
    // printf("*****\n");
    __m512i a_reg[M_ITER];
    __m512i c_reg[N_THREAD_TILE][M_ITER];
    int *ptr_packing_a;
    int *ptr_packing_b[N_ITER];
    for (m_count_sub=m, m_count=0; m_count_sub>M_THREAD_TILE-1;m_count_sub-=M_THREAD_TILE,m_count+=M_THREAD_TILE){
        i=m_count;j=0;ptr_packing_a=(int*)(a_buffer+m_count*K);
        for(int i_n = 0; i_n < N_ITER; i_n += 1){
            ptr_packing_b[i_n] = (int*)(b_buffer+ i_n * 2 * K);
        }

        for(int i_n = 0; i_n < N_THREAD_TILE; i_n += 1){
            for(int j_m = 0; j_m < M_ITER; j_m += 1){
                c_reg[i_n][j_m] = _mm512_setzero_epi32();
            }
        }
        for(k = k_start; k<K4;){
            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_loadu_si512(ptr_packing_a+m_iter*16);\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_epi32(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_epi32(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2][m_iter], a_reg[m_iter], b0);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2 + 1][m_iter], a_reg[m_iter], b1);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k+=4;

            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_loadu_si512(ptr_packing_a+m_iter*16);\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_epi32(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_epi32(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2][m_iter], a_reg[m_iter], b0);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2 + 1][m_iter], a_reg[m_iter], b1);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k+=4;

            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_loadu_si512(ptr_packing_a+m_iter*16);\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_epi32(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_epi32(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2][m_iter], a_reg[m_iter], b0);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2 + 1][m_iter], a_reg[m_iter], b1);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k+=4;

            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_loadu_si512(ptr_packing_a+m_iter*16);\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_epi32(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_epi32(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2][m_iter], a_reg[m_iter], b0);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2 + 1][m_iter], a_reg[m_iter], b1);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k+=4;
        }
        for(k = K4; k < k_end;){
            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_loadu_si512(ptr_packing_a+m_iter*16);\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_epi32(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_epi32(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2][m_iter], a_reg[m_iter], b0);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_dpbusds_epi32(c_reg[n_iter * 2 + 1][m_iter], a_reg[m_iter], b1);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k+=4;
        }
        for(int i_n = 0; i_n < N_THREAD_TILE; i_n += 1){
            for(int j_m = 0; j_m < M_ITER; j_m += 1){
                _mm512_storeu_si512(&C(i+j_m*16, j+i_n), _mm512_add_epi32(c_reg[i_n][j_m],_mm512_loadu_si512(&C(i+j_m*16, j+i_n))));
            }
        }
    }

}


void macro_kernel_k11(uint8_t *a_buffer,uint8_t *b_buffer,int m,int n,int k,int *C, int LDC,int alpha, int M_THREAD_TILE, int N_THREAD_TILE, int M_ITER, int N_ITER){
    int m_count,n_count,m_count_sub,n_count_sub;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);

    for (n_count_sub=n,n_count=0;n_count_sub>N_THREAD_TILE-1;n_count_sub-=N_THREAD_TILE,n_count+=N_THREAD_TILE){
        kernel_n_n_v2_k11(a_buffer,b_buffer+n_count*k,C+n_count*LDC,m,k,LDC,alpha, M_THREAD_TILE, N_THREAD_TILE, M_ITER, N_ITER);
    }

}

void MatrixMulCUDA_8bit_bias(float *input0, float *input1, float *input2, float *input3, float *input4, float *input5, float * input6, float *input7, float *output0) 
{
    float * A = reinterpret_cast<float*>(input0);
    float * val = reinterpret_cast<float*>(input1);
    int * row_ptr = reinterpret_cast<int*>(input2);
    int * col_idx = reinterpret_cast<int*>(input3);
    const int alpha = (int)(*input4);
    const int integer = (int)(*input5);
    int * C = reinterpret_cast< int *>(input6);             // new add
    float *a_buffer_whole = reinterpret_cast<float*>(input7);   // new add
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