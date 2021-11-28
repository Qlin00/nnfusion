
#include <algorithm>
using namespace std;
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
float * GLOBAL_MEMORY = (float*)malloc(sizeof(float) *1024*4096*32);
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