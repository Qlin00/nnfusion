

extern "C" __global__ void MatrixMulCUDA_8bit_bias(float *input0, float *input1, float *input2, float *input3, float *input4, float *input5, float * input6,float  *input7, float *output0) 
{



  /*
  COMMENT_TAG
  */
  const int M = GLOBAL_M_VALUE;
  const int K = GLOBAL_K_VALUE;
  const int N = GLOBAL_N_VALUE;
  /*
    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    */
  int8_t * A = reinterpret_cast<int8_t*>(input0);
  int8_t * B = reinterpret_cast<int8_t*>(input1);
  int * bias = reinterpret_cast<int *>(input7);
  int8_t * C = reinterpret_cast<int8_t*>(output0);
  const int integer = (int)(*input5);
  const int shift = (int)(*input6);

}
