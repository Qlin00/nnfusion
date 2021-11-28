#include <algorithm>
using namespace std;
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
float * GLOBAL_MEMORY = (float*)malloc(sizeof(float) *1024*4096*32);
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