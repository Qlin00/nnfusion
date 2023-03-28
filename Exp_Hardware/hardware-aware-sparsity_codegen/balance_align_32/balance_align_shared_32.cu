__global__ void MatMul_TILE_BLOCK_GENERAL(float *input0, float *input1, float *input2, float *input3, float *output0){
	float *g_vec = input0;
	float *g_mat_data = input1;
	int *g_mat_index = (int*)input2;
	float *bias = input3;
	float *g_data = output0;
	const float SPARSITY = SPARSITY_VALUE;
    const int M = M_GLOBAL_VALUE;
    const int K = K_GLOBAL_VALUE;
    const int N = N_GLOBAL_VALUE;
    const int K_sparse = int(K * (1-SPARSITY));

    const int BLOCK_SIZE_M = BLOCK_SIZE_M_VALUE;
    const int BLOCK_SIZE_N = BLOCK_SIZE_N_VALUE;
    // BLOCK_SIZE_K should > NUM_BANK
    const int BLOCK_SIZE_K = BLOCK_SIZE_K_VALUE;
    const int THREAD_SIZE_M = THREAD_SIZE_M_VALUE;
    const int THREAD_SIZE_N = THREAD_SIZE_N_VALUE;

    const int ALIGN_N = BLOCK_SIZE_N;

    const int BANK_VAL = BANK_VAL_VALUE;
    const int NUM_BANK = K / BANK_VAL;

    const int BANK_NUM_PER_BLOCK = BLOCK_SIZE_K / BANK_VAL;
    const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1-SPARSITY));
    const int LEN_OF_BANK_PER_SPARSE_BLOCK = BLOCK_SIZE_K_SPARSE / BANK_NUM_PER_BLOCK;

	int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
	int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

	

	const int A_THREADS_PER_ROW = BLOCK_SIZE_M / 4;
	const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

	const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

	const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
	const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

	__shared__ float A_shared[BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE];
	__shared__ float B_shared[BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE];

	float A_reg[THREAD_SIZE_M];
	float B_reg[THREAD_SIZE_N];
	float C_reg[THREAD_SIZE_N][THREAD_SIZE_M] = {0};

	int tid = threadIdx.x;

	int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
	int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

	int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
	int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

	int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
	int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

	for(int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE){
		float *A_global_ptr = g_vec + M_BLOCK_START;
		float *B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
		int *B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

		__syncthreads();

		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE_K_SPARSE; i += A_STRIDES){
			int idx = *(B_index_global_ptr + (i + A_BLOCK_ROW_START) * N);
			*(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_M + A_BLOCK_COL_START) = 
				*(float4 *)(A_global_ptr + idx * M + A_BLOCK_COL_START);
		}

		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES){
			*(float4 *)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
				*(float4 *)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
		}

		__syncthreads();

		#pragma unroll
		for(int i = 0; i < BLOCK_SIZE_K_SPARSE; i += 1){
			#pragma unroll
			for(int k = 0; k < THREAD_SIZE_M; k += 1){
				A_reg[k] = A_shared[i * BLOCK_SIZE_M + t_M * THREAD_SIZE_M + k];
			}
			#pragma unroll
			for(int k = 0; k < THREAD_SIZE_N; k += 1){
				B_reg[k] = B_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
			}
			#pragma unroll
			for(int k = 0; k < THREAD_SIZE_N; k += 1){
				#pragma unroll
				for(int j = 0; j < THREAD_SIZE_M; j += 1){
					C_reg[k][j] += B_reg[k] * A_reg[j];
				}
			}
		}
	}
	
	#pragma unroll
	for(int i = 0; i < THREAD_SIZE_N; i += 1){
		#pragma unroll
		for(int j = 0; j < THREAD_SIZE_M; j += 1){
			g_data[(BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N + i) * M + (BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M + j)] =
				C_reg[i][j];
		}
	}
}