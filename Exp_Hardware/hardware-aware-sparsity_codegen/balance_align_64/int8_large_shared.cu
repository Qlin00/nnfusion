
__global__ void compute_gemm_imma(float *input0, float *input1, float *input2, float* input3, float* input4, float*input5, float* input6,
                                  float *output0) {
    // keys need to be replaced M_GLOBAL_VALUE, N_GLOBAL_VALUE, K_GLOBA_VALUEL, SPARSITY_VALUE, CHUNK_K_VALUE

	//extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];
    const uint8_t * A = reinterpret_cast<uint8_t*>(input0); // activation
    const uint8_t * B = reinterpret_cast<uint8_t*>(input1); // weight
    const int * B_index = reinterpret_cast<int *>(input2);
    const int * B_col = reinterpret_cast< int *>(input3);
    const int alpha = (int)(*input4);
    const int integer = (int)(*input5);
    int * bias = reinterpret_cast< int *>(input6);
    uint8_t * D = reinterpret_cast<uint8_t*>(output0);

	const int WARP_SIZE = 32;
    const int M = 16;
    const int K = 16;
    const int N = 16;
    const int WMMA_M = 16;
    const int WMMA_K = 16;
    const int WMMA_N = 16;
    const int M_GLOBAL = M_GLOBAL_VALUE;
    const int K_GLOBAL = K_GLOBAL_VALUE;
    const int N_GLOBAL = N_GLOBAL_VALUE;
    const float SPARSITY = SPARSITY_VALUE;
    const int K_GLOBAL_SPARSE = (int(K_GLOBAL * (1-SPARSITY)));
    const int M_TILES = (M_GLOBAL / M);
    const int N_TILES = (N_GLOBAL / N);
    const int K_TILES = (K_GLOBAL / K);
    const int BLOCK_ROW_WARPS = 2;
    const int BLOCK_COL_WARPS = 4;
    const int WARP_ROW_TILES = 4;
    const int WARP_COL_TILES = 2;
    const int SKEW_UINT8 = 16;
    const int BANK_VAL = 32;
    const int NUM_BANK = (K_GLOBAL / BANK_VAL);
    /////////// BLOCK_ROW_TILES <= N_TILES ////////////
    const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS);
    ////////// BLOCK_COL_TILES <= M_TILES ////////////
    const int BLOCK_COL_TILES = (WARP_COL_TILES * BLOCK_COL_WARPS);
    auto C_LAYOUT = wmma::mem_col_major;
    const int CHUNK_K = CHUNK_K_VALUE;
    const int CHUNK_K_SPARSE = (int(CHUNK_K * (1-SPARSITY)));
    const int BLOCK_SIZE_K = (int(CHUNK_K * K));
    const int BLOCK_SIZE_K_SPARSE = (int((CHUNK_K * K) * (1 - SPARSITY)));
    const int WARP_COPY_BYTES = (WARP_SIZE * sizeof(int4));
    const int CHUNK_LINE_BYTES_A (BLOCK_COL_TILES * M * sizeof(uint8_t));
    const int CHUNK_COPY_LINES_PER_WARP_A = (WARP_COPY_BYTES / CHUNK_LINE_BYTES_A);
    const int CHUNK_COPY_LINE_LANES_A = (CHUNK_LINE_BYTES_A / sizeof(int4));
    const int SHARED_OFFSET_A = (BLOCK_COL_TILES * M + SKEW_UINT8);

    const int CHUNK_LINE_BYTES_B = (BLOCK_SIZE_K_SPARSE * sizeof(uint8_t));
    const int CHUNK_COPY_LINES_PER_WARP_B = (WARP_COPY_BYTES / CHUNK_LINE_BYTES_B);
    const int CHUNK_COPY_LINE_LANES_B = (CHUNK_LINE_BYTES_B / sizeof(int4));
    const int SHARED_OFFSET_B = (BLOCK_SIZE_K_SPARSE + SKEW_UINT8);
    const int SHARED_TO_GLOBAL_BYTES_PER_LINE = ((WARP_COL_TILES * M) * sizeof(int));
    const int SHARED_TO_GLOBAL_BYTES_PER_WARP = (WARP_SIZE * sizeof(int));
    const int SHARED_TO_GLOBAL_LINES_PER_WARP = (SHARED_TO_GLOBAL_BYTES_PER_WARP / SHARED_TO_GLOBAL_BYTES_PER_LINE);
    const int SHARED_TO_GLOBAL_LANES_PER_LINE = (WARP_SIZE / SHARED_TO_GLOBAL_LINES_PER_WARP);
    const int SHARED_TO_GLOBAL_ITERS = ((WARP_ROW_TILES * N) / SHARED_TO_GLOBAL_LINES_PER_WARP);

    const int LANE_ROW_STRIDE = (WARP_ROW_TILES * N / 8);
    const int LANE_COL_STRIDE = (WARP_COL_TILES * M / 4);
    const int WARP_STRIDE = (WARP_COL_TILES * M);
    const int WARPS_PER_BLOCK = (BLOCK_ROW_WARPS * BLOCK_COL_WARPS);
    const int THREADS_PER_BLOCK = (WARP_SIZE * WARPS_PER_BLOCK);
    // const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS);
    // const int BLOCK_COL_TILES = (WARP_COL_TILES * BLOCK_COL_WARPS);
    const int GLOBAL_MEM_STRIDE = M_GLOBAL;
    const int SHMEM_STRIDE = (M * BLOCK_COL_TILES);
    const int SHMEM_OFFSET = (M * WARP_COL_TILES);
    const int BLOCK_SIZE_M = (M * BLOCK_COL_TILES);
    const int BLOCK_SIZE_N = (N * BLOCK_ROW_TILES);

   extern __shared__ uint8_t shmem[];

	// Warp and lane identification.
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;


	// Offset in shared memory from which the B matrix is stored.
	// const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;       // BLOCK_COL_TILES * M is shared_A row numbers in one block
	const size_t shmem_idx_b_off = BLOCK_SIZE_K_SPARSE * SHARED_OFFSET_A;


	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.

    unsigned int block_pos = blockIdx.x;
	const unsigned int block_tile_i =
		((block_pos * BLOCK_COL_TILES) / M_TILES) * (BLOCK_ROW_TILES);
	const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % M_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.


    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.

    //__syncthreads();
    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
	wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_ROW_TILES]
													 [WARP_COL_TILES];

    // Load the C matrix tiles into fragments from shared memory.

#pragma unroll
	for(int i = 0; i < WARP_ROW_TILES; i += 1){
	#pragma unroll
		for(int j = 0; j < WARP_COL_TILES; j += 1){
			wmma::fill_fragment(c[i][j], 0);
		}
	}

    __syncthreads();

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.

    // int start_tile = B_row[block_tile_j / WARP_COL_TILES + (warpId % BLOCK_ROW_WARPS)];
    // int end_tile = B_row[block_tile_j / WARP_COL_TILES + (warpId % BLOCK_ROW_WARPS) + 1];

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    //for(int tile_k_idx = start_tile; tile_k_idx < end_tile; tile_k_idx += 1){
    for(int tile_k_idx_sparse = 0, tile_k_idx = 0; tile_k_idx_sparse < K_GLOBAL_SPARSE; tile_k_idx_sparse += BLOCK_SIZE_K_SPARSE, tile_k_idx += BLOCK_SIZE_K){

		size_t shmem_idx = 
		warpId < (WARPS_PER_BLOCK / 2)
			? (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_A * SHARED_OFFSET_A
			: (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_B * SHARED_OFFSET_B + shmem_idx_b_off;

		int4 *lane_ptr = NULL;
		int *lane_ptr_index = NULL;
		const uint8_t *warp_ptr = NULL;


		if(warpId < (WARPS_PER_BLOCK / 2)){
			//warp_ptr = &A[block_tile_j * M] +
			//	(warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_A * M_GLOBAL;
			warp_ptr = &A[block_tile_j * M];
			
			const int *warp_ptr_index = &B_index[block_tile_i * N * K_GLOBAL_SPARSE] +
									((warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_A);

			lane_ptr_index = (int *)(warp_ptr_index + tile_k_idx_sparse + (laneId / CHUNK_COPY_LINE_LANES_A));

			shmem_idx += (laneId / CHUNK_COPY_LINE_LANES_A) * SHARED_OFFSET_A;
		}else{
			warp_ptr = &B[block_tile_i * N * K_GLOBAL_SPARSE] +
				(warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_B * K_GLOBAL_SPARSE;
			lane_ptr = (int4 *)(warp_ptr + tile_k_idx_sparse +
								(laneId / CHUNK_COPY_LINE_LANES_B) * K_GLOBAL_SPARSE) +
								(laneId % CHUNK_COPY_LINE_LANES_B);
			shmem_idx += (laneId / CHUNK_COPY_LINE_LANES_B) * SHARED_OFFSET_B;
		}


      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      // shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

	  int iter_index = warpId < (WARPS_PER_BLOCK / 2)
	  	? BLOCK_SIZE_K_SPARSE / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_A)
		: BLOCK_SIZE_N / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_B);

	  /*
      int iter_index = warpId < (WARPS_PER_BLOCK / 2)
        ? (BLOCK_COL_TILES * M) / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP)
        : (BLOCK_ROW_TILES * N) / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP);
	  */

	  /*
      int tile_k_idx_A;
      if(warpId < (WARPS_PER_BLOCK / 2)){
          tile_k_idx_A = *(lane_ptr_index);
      }
	  */

	  #pragma unroll
	  for(int i = 0; i < iter_index; i += 1){
		  if(warpId < (WARPS_PER_BLOCK / 2)){
			int tile_k_idx_A = *(lane_ptr_index);
			lane_ptr = (int4 *)(warp_ptr + tile_k_idx_A * M_GLOBAL) + (laneId % CHUNK_COPY_LINE_LANES_A);
			*((int4 *)&shmem[shmem_idx] + (laneId % CHUNK_COPY_LINE_LANES_A)) =
				*lane_ptr;
			//warp_ptr = (uint8_t *)((uint8_t *)warp_ptr + M_GLOBAL * (WARPS_PER_BLOCK / 2) *CHUNK_COPY_LINES_PER_WARP_A);
			lane_ptr_index = (int *)((int *)lane_ptr_index +  (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_A);
			shmem_idx += (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_A * SHARED_OFFSET_A;
		  }else{
			*((int4 *)&shmem[shmem_idx] + (laneId % CHUNK_COPY_LINE_LANES_B)) =
				*lane_ptr;
			lane_ptr = (int4 *)((uint8_t *)lane_ptr + K_GLOBAL_SPARSE * (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_B);
			shmem_idx += (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_B * SHARED_OFFSET_B;
		  }
	  }

      __syncthreads();

	#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K_SPARSE; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::col_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major>
            b[WARP_ROW_TILES];

	#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i += 1) {
			size_t shmem_idx_a = (warpId % BLOCK_COL_WARPS) * M * WARP_COL_TILES + (i * M);
			const uint8_t *tile_ptr = shmem + shmem_idx_a + k_step * K * SHARED_OFFSET_A;

			wmma::load_matrix_sync(a[i], tile_ptr, SHARED_OFFSET_A);
		#pragma unroll
			for(int j = 0; j < WARP_ROW_TILES; j += 1){
				if(i == 0){
					size_t shmem_idx_b = shmem_idx_b_off +
											(warpId / BLOCK_COL_WARPS) * (WARP_ROW_TILES * N) * SHARED_OFFSET_B +
											(j * N) * SHARED_OFFSET_B;
					const uint8_t *tile_ptr = shmem + shmem_idx_b + k_step * K;
					wmma::load_matrix_sync(b[j], tile_ptr, SHARED_OFFSET_B);
				}
				wmma::mma_sync(c[j][i], a[i], b[j], c[j][i]);
			}

        }
      }

      __syncthreads();
    }

    // This pointer is used to access the C and D matrix tiles this warp computes.
	int *shmem_warp_tile_ptr = (int *)shmem + (warpId / BLOCK_COL_WARPS) * N * WARP_ROW_TILES * SHMEM_STRIDE +
	(warpId % BLOCK_COL_WARPS) * SHMEM_OFFSET;

      // Store the D fragments to shared memory.
#pragma unroll
	for(int i = 0; i < WARP_ROW_TILES; i += 1){
	#pragma unroll
		for(int j = 0; j < WARP_COL_TILES; j += 1){
		#pragma unroll
			for(int t = 0; t < c[i][j].num_elements; t += 1){
				c[i][j].x[t] = ((c[i][j].x[t] * alpha) >> integer);
			}
			int *tile_ptr = shmem_warp_tile_ptr + i * N * SHMEM_STRIDE + j * M;
			wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
		}
	}

    __syncthreads();

	int *shmem_warp_stream_ptr = (int *)shmem + (warpId / BLOCK_COL_WARPS) * WARP_ROW_TILES * N * SHMEM_STRIDE
									+ (warpId % BLOCK_COL_WARPS) * WARP_COL_TILES * M;
	const size_t gmem_idx =
		(block_tile_i * N + (warpId / BLOCK_COL_WARPS) * WARP_ROW_TILES * N) * GLOBAL_MEM_STRIDE +
		block_tile_j * M + (warpId % BLOCK_COL_WARPS) * WARP_COL_TILES * M;
	uint8_t *dst_gmem_warp_stream_ptr = &D[gmem_idx];

	int *shmem_lane_stream_ptr =
		shmem_warp_stream_ptr +
		(laneId / SHARED_TO_GLOBAL_LANES_PER_LINE) * SHMEM_STRIDE +
		(laneId % SHARED_TO_GLOBAL_LANES_PER_LINE);
	
	uint8_t *dst_gmem_lane_stream_ptr =
		dst_gmem_warp_stream_ptr +
		(laneId / SHARED_TO_GLOBAL_LANES_PER_LINE) * GLOBAL_MEM_STRIDE +
		(laneId % SHARED_TO_GLOBAL_LANES_PER_LINE);

	for(int i = 0; i < WARP_ROW_TILES * N; i += SHARED_TO_GLOBAL_LINES_PER_WARP){
		*(dst_gmem_lane_stream_ptr + i * GLOBAL_MEM_STRIDE) = (uint8_t)(*(shmem_lane_stream_ptr + i * SHMEM_STRIDE));
	}

	__syncthreads();
}
