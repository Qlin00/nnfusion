extern "C" __global__ void MatrixMulCUDA_8bit_bias(float *input0, float *input1, float *input2, float *input3, float *input4, float *input5, float * input6,float  *input7, float *output0) 
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
  int8_t * W = reinterpret_cast<int8_t*>(input1);
  int * C = reinterpret_cast<int *>(input7);
  int8_t * D = reinterpret_cast<int8_t*>(output0);
  const int Integer = (int)(*input5);
  const int Shift_val = (int)(*input6);

  __shared__ int8_t As[BLOCK_SIZE_M * BLOCK_SIZE_K];
  __shared__ int8_t Bs[BLOCK_SIZE_N * BLOCK_SIZE_K];

  int integer = Integer;
  int shift_val = Shift_val;
  int accum[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
  int8_t a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
  int8_t b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

  int A_THREAD_PER_ROW = BLOCK_SIZE_M / 4;
  int B_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

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

  for(int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K){
  #pragma unroll
      for(int k = 0; k < BLOCK_SIZE_K; k += A_TILE_ROW_STRIDE){
          reinterpret_cast<unsigned int*>(&(As[(k+A_BLOCK_ROW_START) * BLOCK_SIZE_M + A_BLOCK_COL_START]))[0] =
          reinterpret_cast<unsigned int*>(&(A[(tile_idx+k+A_BLOCK_ROW_START) * M + by*BLOCK_SIZE_M+A_BLOCK_COL_START]))[0];
      }

  #pragma unroll
      for(int k = 0; k < BLOCK_SIZE_K; k += B_TILE_ROW_STRIDE){
          reinterpret_cast<unsigned int*>(&(Bs[(k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]))[0] =
          reinterpret_cast<unsigned int*>(&(W[(tile_idx+k+B_BLOCK_ROW_START) * N + bx*BLOCK_SIZE_N+B_BLOCK_COL_START]))[0];
      }

      __syncthreads();

  #pragma unroll
      for(int k = 0; k < BLOCK_SIZE_K; k += THREAD_SIZE_K){
      #pragma unroll
          for(int i = 0; i < THREAD_SIZE_K; i ++){
          #pragma unroll
              for(int j = 0; j < THREAD_SIZE_M; j += 1){
                  a_frag[j][i] = As[(k+i) * BLOCK_SIZE_M + ty * THREAD_SIZE_M+j];
              }
          }
      #pragma unroll
          for(int i = 0; i < THREAD_SIZE_K; i ++){
          #pragma unroll
              for(int j = 0; j < THREAD_SIZE_N; j += 1){
                  b_frag[j][i] = Bs[(k+i) * BLOCK_SIZE_N + tx * THREAD_SIZE_N+j];
              }
          }

      #pragma unroll
          for(int i = 0; i < THREAD_SIZE_N; i++){
          #pragma unroll
              for(int j = 0; j < THREAD_SIZE_M; j++){
              #pragma unroll
                  for(int k_in = 0; k_in < THREAD_SIZE_K; k_in += 4){
                      int pack_val1 = reinterpret_cast<unsigned int*>(&(a_frag[j][k_in]))[0];
                      int pack_val2 = reinterpret_cast<unsigned int*>(&(b_frag[i][k_in]))[0];
                      accum[i][j] = amd_mixed_dot(pack_val1, pack_val2, accum[i][j], true);
                  }
              }
          }
      }
      __syncthreads();
  }

#pragma unroll
  for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
  #pragma unroll
      for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y++){
          (
              D[(BLOCK_SIZE_N * bx + tx * THREAD_SIZE_N + thread_x) * M +
              BLOCK_SIZE_M * by + ty * THREAD_SIZE_M + thread_y
          ]) = (int8_t)(((accum[thread_x][thread_y] + C[(BLOCK_SIZE_N * bx + tx * THREAD_SIZE_N + thread_x)]) * integer) >> shift_val);
      }
  }

}
