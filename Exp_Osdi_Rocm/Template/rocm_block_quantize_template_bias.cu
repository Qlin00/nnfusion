extern "C" __global__ void MatrixMulCUDA_8bit_bias(float *input0, float *input1, float *input2, float *input3, float *input4, float *input5, float * input6, float *output0) 
{
  {
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    const int BLOCK_SIZE_M = BLOCK_SIZE_M_VALUE;
    const int BLOCK_SIZE_K = BLOCK_SIZE_K_VALUE;
    const int BLOCK_SIZE_N = BLOCK_SIZE_N_VALUE;
    const int THREAD_SIZE_M = THREAD_SIZE_M_VALUE;
    const int THREAD_SIZE_K = THREAD_SIZE_K_VALUE;
    const int THREAD_SIZE_N = THREAD_SIZE_N_VALUE;
    /*
    COMMENT_TAG
    */
    const int M = GLOBAL_M_VALUE;
    const int K = GLOBAL_K_VALUE;
    const int N = GLOBAL_N_VALUE;

    int8_t * A = reinterpret_cast<int8_t*>(input0);
    int8_t * W_val = reinterpret_cast<int8_t*>(input1);
    int8_t * W_row = reinterpret_cast<int8_t*>(input2);
    int8_t * W_col = reinterpret_cast<int8_t*>(input3);
    const int Integer = (int)(*input4);
    const int Shift_val = (int)(*input5);
    int * C = reinterpret_cast< int *>(input6);
    int8_t * D = reinterpret_cast<int8_t*>(output0);

    __shared__ int8_t As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ int8_t Bs[BLOCK_SIZE_N * BLOCK_SIZE_K];

    int accum[THREAD_SIZE_M][THREAD_SIZE_N] = {0};
    int8_t a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
    int8_t b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

    int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    int B_THREAD_PER_ROW = BLOCK_SIZE_K / 4;

    int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;

    int index_start = W_row[bx], index_end = W_row[bx+1];
    for(int tile_idx = index_start; tile_idx < index_end; tile_idx += 1){
        int k_idx = W_col[tile_idx] * BLOCK_SIZE_K;
        for(int m = 0; m < BLOCK_SIZE_M; m += A_TILE_ROW_STRIDE){
            reinterpret_cast<unsigned int*>(&(As[(m+A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START]))[0] =
            reinterpret_cast<unsigned int*>(&(A[(by*BLOCK_SIZE_M+m+A_BLOCK_ROW_START) * K + k_idx + A_BLOCK_COL_START]))[0];
        }
        for(int n = 0; n < BLOCK_SIZE_N; n += B_TILE_ROW_STRIDE){
            reinterpret_cast<unsigned int*>(&(Bs[(n+B_BLOCK_ROW_START) * BLOCK_SIZE_K + B_BLOCK_COL_START]))[0] =
            reinterpret_cast<unsigned int*>(&(W_val[tile_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (n+B_BLOCK_ROW_START)*BLOCK_SIZE_K + B_BLOCK_COL_START]))[0];
        }
        __syncthreads();
        for(int k = 0; k < BLOCK_SIZE_K; k += THREAD_SIZE_K){
            for(int i = 0; i < THREAD_SIZE_M; i ++){
                for(int j = 0; j < THREAD_SIZE_K; j+=4){
                    reinterpret_cast<unsigned int*>(&(a_frag[i][j]))[0] = 
                    reinterpret_cast<unsigned int*>(&(As[(ty*THREAD_SIZE_M+i)*BLOCK_SIZE_K+k+j]))[0];
                }
            }
            for(int i = 0; i < THREAD_SIZE_N; i ++){
                for(int j = 0; j < THREAD_SIZE_K; j+=4){
                    reinterpret_cast<unsigned int*>(&(b_frag[i][j]))[0] = 
                    reinterpret_cast<unsigned int*>(&(Bs[(tx*THREAD_SIZE_N+i)*BLOCK_SIZE_K+k+j]))[0];
                }
            }

            for(int i = 0; i < THREAD_SIZE_M; i++){
                for(int j = 0; j < THREAD_SIZE_N; j++){
                    for(int c_in = 0; c_in < THREAD_SIZE_K; c_in+=4){
                        int pack_val1 = reinterpret_cast<unsigned int*>(&(a_frag[i][c_in]))[0];
                        int pack_val2 = reinterpret_cast<unsigned int*>(&(b_frag[j][c_in]))[0];
                        accum[i][j] = amd_mixed_dot(pack_val1, pack_val2, accum[i][j], true);
                    }
                }
            }
        }
    }

    for(int i = 0; i < THREAD_SIZE_M; i++){
        for(int j = 0; j < THREAD_SIZE_N; j++){
            D[(by * BLOCK_SIZE_M + ty * THREAD_SIZE_M + i) * N + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_N + j] =
            (int8_t)(((accum[i][j] + C[(by * BLOCK_SIZE_M + ty * THREAD_SIZE_M + i) * N + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_N + j]) * Integer) >> Shift_val);
        }
    }
  }
}