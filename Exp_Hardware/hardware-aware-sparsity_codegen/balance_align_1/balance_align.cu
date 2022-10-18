// dim3 dimBlock((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N));
// dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
__global__ void MatMul_TILE_THREAD_GENERAL_NO_SHARED(float *input0, float *input1, float *input2, float *input3, float *output0) {
	float *g_vec = input0;
	float *g_mat_data = input1;
	int *g_mat_index = (int*)input2;
	float *bias = input3;
	float *g_data = output0;

	const float SPARSITY = SPARSITY_VALUE;
    const int M = M_GLOBAL_VALUE;
    const int K = K_GLOBAL_VALUE;
    const int N = N_GLOBAL_VALUE;
    const int K_sparse = int(K * SPARSITY);
    const int BLOCK_SIZE_M = BLOCK_SIZE_M_VALUE;
    const int BLOCK_SIZE_K = BLOCK_SIZE_K_VALUE;
    const int BLOCK_SIZE_N = BLOCK_SIZE_N_VALUE;
    const int THREAD_SIZE_M = THREAD_SIZE_M_VALUE;
    const int THREAD_SIZE_N = THREAD_SIZE_N_VALUE;

    const int BANK_VAL = BANK_VAL_VALUE;
    const int NUM_BANK = K / BANK_VAL;

    const int BANK_NUM_PER_BLOCK = BLOCK_SIZE_K / BANK_VAL;
    const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1-SPARSITY));
    const int LEN_OF_BANK_PER_SPARSE_BLOCK = BLOCK_SIZE_K_SPARSE / BANK_NUM_PER_BLOCK;

	int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
	int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

	const int A_THREADS_PER_ROW = BLOCK_SIZE_K / 4;
	const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

	const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

	const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
	const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

	__shared__ float A_shared[BLOCK_SIZE_M * BLOCK_SIZE_K];

	float B_reg[THREAD_SIZE_N];
	int B_reg_index[THREAD_SIZE_N];
	float C_reg[THREAD_SIZE_M][THREAD_SIZE_N] = {0};

	int tid = threadIdx.x;

	int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
	int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

	int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;

	int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;

	for(int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE){
		float *A_global_ptr = g_vec + M_BLOCK_START * K + K_BLOCK_START;

		__syncthreads();

		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE_M; i += A_STRIDES){
			*(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START) = 
				*(float4 *)(A_global_ptr + (i + A_BLOCK_ROW_START) * K + A_BLOCK_COL_START);
		}

		__syncthreads();

		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE_K_SPARSE;i += 1){
			#pragma unroll
			for(int k = 0; k < THREAD_SIZE_N; k += 1){
				B_reg[k] = g_mat_data[(K_SPARSE_BLOCK_START + i) * N + N_BLOCK_START + t_N * THREAD_SIZE_N + k];
				B_reg_index[k] = g_mat_index[(K_SPARSE_BLOCK_START + i) * N + N_BLOCK_START + t_N * THREAD_SIZE_N + k];
			}
			#pragma unroll
			for(int k = 0; k < THREAD_SIZE_N; k += 1){
				int bank_idx = i / LEN_OF_BANK_PER_SPARSE_BLOCK;
				int B_index = B_reg_index[k] % BANK_VAL+bank_idx * BANK_VAL;
				#pragma unroll
				for(int j = 0; j < THREAD_SIZE_M; j += 1){
					C_reg[j][k] += B_reg[k] * A_shared[(t_M * THREAD_SIZE_M+j) * BLOCK_SIZE_K + B_index];
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < THREAD_SIZE_M; i += 1){
		#pragma unroll
		for(int j = 0; j < THREAD_SIZE_N; j += 1){
			g_data[(BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M + i) * N + BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N + j] = C_reg[i][j];
		}
	}
}