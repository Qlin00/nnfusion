#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

void packing_a_k11(float *src, float *dst, int leading_dim, int dim_first, int dim_second, int M_THREAD_TILE){
    //dim_first: M, dim_second: K
    float *tosrc,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_first;
    int inner_iter = int(M_THREAD_TILE / 16);
    for (count_first=0;count_sub>M_THREAD_TILE-1;count_first+=M_THREAD_TILE,count_sub-=M_THREAD_TILE){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            #pragma unroll
            for(int i = 0; i < inner_iter; i += 1){
                _mm512_store_ps(todst+16*i,_mm512_loadu_ps(tosrc+16*i));
            }
            tosrc+=leading_dim;
            todst+=M_THREAD_TILE;
        }
    }
    // edge case

}


void kernel_n_n_v2_k11(float *a_buffer,float *b_buffer,float *c_ptr,int m,int K,int LDC,float alpha, int M_THREAD_TILE, int N_THREAD_TILE, int N_ITER, int M_ITER){
    int m_count,m_count_sub;
    int i,j,k;
    float *C=c_ptr;
    __m512 valpha = _mm512_set1_ps(alpha);//broadcast alpha to a 512-bit vector
    __m128 dvalpha = _mm_set1_ps(alpha);//broadcast alpha to a 128-bit vector
    __m512 b0,b1;
    int k_start,k_end,K4;
    K4=K&-4;k_end=K;k_start=0;
    // printf("*****\n");
    // print_matrix(C,m,8);
    // printf("*****\n");
    __m512 a_reg[M_ITER];
    __m512 c_reg[N_THREAD_TILE][M_ITER];
    float *ptr_packing_a;
    float *ptr_packing_b[N_ITER];
    for (m_count_sub=m, m_count=0; m_count_sub>M_THREAD_TILE-1;m_count_sub-=M_THREAD_TILE,m_count+=M_THREAD_TILE){
        i=m_count;j=0;ptr_packing_a=a_buffer+m_count*K;
        for(int i_n = 0; i_n < N_ITER; i_n += 1){
            ptr_packing_b[i_n] = b_buffer+i_n * 2 * K;
        }
        
        for(int i_n = 0; i_n < N_THREAD_TILE; i_n += 1){
            for(int j_m = 0; j_m < M_ITER; j_m += 1){
                c_reg[i_n][j_m] = _mm512_setzero_ps();
            }
        }
        for(k = k_start; k<K4;){
            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_mul_ps(valpha, _mm512_load_ps(ptr_packing_a+m_iter*16));\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_ps(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_ps(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b0, c_reg[n_iter * 2][m_iter]);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b1, c_reg[n_iter * 2 + 1][m_iter]);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k++;

            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_mul_ps(valpha, _mm512_load_ps(ptr_packing_a+m_iter*16));\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_ps(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_ps(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b0, c_reg[n_iter * 2][m_iter]);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b1, c_reg[n_iter * 2 + 1][m_iter]);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k++;

            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_mul_ps(valpha, _mm512_load_ps(ptr_packing_a+m_iter*16));\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_ps(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_ps(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b0, c_reg[n_iter * 2][m_iter]);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b1, c_reg[n_iter * 2 + 1][m_iter]);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k++;

            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_mul_ps(valpha, _mm512_load_ps(ptr_packing_a+m_iter*16));\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_ps(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_ps(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b0, c_reg[n_iter * 2][m_iter]);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b1, c_reg[n_iter * 2 + 1][m_iter]);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k++;
        }
        for(k = K4; k < k_end;){
            for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                a_reg[m_iter] = _mm512_mul_ps(valpha, _mm512_load_ps(ptr_packing_a+m_iter*16));\
            }\
            for(int n_iter = 0; n_iter < N_ITER; n_iter += 1){\
                b0 = _mm512_set1_ps(*ptr_packing_b[n_iter]);\
                b1 = _mm512_set1_ps(*(ptr_packing_b[n_iter]+1));\
                for(int m_iter = 0; m_iter < M_ITER; m_iter += 1){\
                    c_reg[n_iter * 2][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b0, c_reg[n_iter * 2][m_iter]);\
                    c_reg[n_iter * 2 + 1][m_iter] = _mm512_fmadd_ps(a_reg[m_iter], b1, c_reg[n_iter * 2 + 1][m_iter]);\
                }\
                ptr_packing_b[n_iter] += 2;\
            }\
            ptr_packing_a += M_THREAD_TILE; k++;
        }

        for(int i_n = 0; i_n < N_THREAD_TILE; i_n += 1){
            for(int j_m = 0; j_m < M_ITER; j_m += 1){
                _mm512_storeu_ps(&C(i+j_m*16, j+i_n), _mm512_add_ps(c_reg[i_n][j_m],_mm512_loadu_ps(&C(i+j_m*16, j+i_n))));
            }
        }
    }

}

void macro_kernel_k11(float *a_buffer,float *b_buffer,int m,int n,int k,float *C, int LDC,float alpha, int M_THREAD_TILE, int N_THREAD_TILE, int N_ITER, int M_ITER){
    int m_count,n_count,m_count_sub,n_count_sub;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);

    for (n_count_sub=n,n_count=0;n_count_sub>N_THREAD_TILE-1;n_count_sub-=N_THREAD_TILE,n_count+=N_THREAD_TILE){
        kernel_n_n_v2_k11(a_buffer,b_buffer+n_count*k,C+n_count*LDC,m,k,LDC,alpha, M_THREAD_TILE, N_THREAD_TILE, N_ITER, M_ITER);
    }

}

void BLOCK_SPARSE_MATMUL(float* input0, float* input1,float* input2, float* input3, float* input4, float* input5, float *output0){
    const int M=M_VALUE;
    const int N=N_VALUE;
    const int K=K_VALUE;
    float * A = reinterpret_cast<float*>(input0);
    float * val = reinterpret_cast<float*>(input1);
    int * row_ptr = reinterpret_cast<int*>(input2);
    int * col_idx = reinterpret_cast<int*>(input3);
    float * bias = reinterpret_cast<float*>(input4);
    float *a_buffer_whole = reinterpret_cast<float*>(input5);
    float * C = reinterpret_cast<float*>(output0);

    constexpr int M_BLOCKING = BLOCK_SIZE_M_VALUE;
    constexpr int N_BLOCKING = BLOCK_SIZE_N_VALUE;
    constexpr int K_BLOCKING = BLOCK_SIZE_K_VALUE;
    constexpr int M_THREAD_TILE = THREAD_SIZE_M_VALUE;
    constexpr int N_THREAD_TILE = THREAD_SIZE_N_VALUE;
    constexpr int M_ITER = (M_THREAD_TILE/16);
    constexpr int N_ITER = (N_THREAD_TILE/2);

    int m_inc = M_BLOCKING, n_inc = N_BLOCKING, k_inc = K_BLOCKING;

    //float *a_buffer_whole = (float *)aligned_alloc(4096, M * K * sizeof(float));
    float *b_buffer_whole = val;
    
#pragma omp parallel for collapse(2)
    for(int m_count = 0; m_count < M; m_count += m_inc){
        for(int k_count = 0; k_count < K; k_count += k_inc){
            int block_id = (m_count / M_BLOCKING) * K / K_BLOCKING + k_count / K_BLOCKING;
            packing_a_k11(A+m_count+k_count*LDA, a_buffer_whole + block_id * m_inc * k_inc, LDA, m_inc, k_inc, M_THREAD_TILE);
        }
    }

#pragma omp parallel for collapse(2)
    for (int n_count=0;n_count<N;n_count+=n_inc){
        for(int m_count=0; m_count<M; m_count+=m_inc){
            int start_idx = row_ptr[n_count / N_BLOCKING], end_idx = row_ptr[(n_count / N_BLOCKING) + 1];

            for(int idx = start_idx; idx < end_idx; idx += 1){
                int k_count = col_idx[idx] * K_BLOCKING;
                int block_id_a = (m_count / M_BLOCKING) * K / K_BLOCKING + k_count / K_BLOCKING;
                int block_id_b = (n_count / N_BLOCKING) * K / K_BLOCKING + k_count / K_BLOCKING;
                //packing_b_k11(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc);
                //packing_a_k11(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc);
                macro_kernel_k11(a_buffer_whole + block_id_a * m_inc * k_inc, b_buffer_whole + idx * n_inc * k_inc, m_inc, n_inc, k_inc, &C(m_count, n_count), LDC, alpha, M_THREAD_TILE, N_THREAD_TILE, N_ITER, M_ITER);
            }
        }
    }

#pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i += 1){
        for(int j = 0; j < M; j += 1){
            C[i * M + j] = max(C[i * M + j], float(0));
        }
    }
}
