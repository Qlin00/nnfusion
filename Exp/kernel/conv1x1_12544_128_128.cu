// This kernel's input datatype and output datatype are all 8bit, using cuda core.

extern "C" __global__ void MatrixMulCUDA_8bit_bias(float *input0, float *input1, float *input2, float *input3, float *input4, float *input5, float * input6,float  *input7, float *output0) 
{

    const unsigned int M_GLOBAL=12544;
    const unsigned int K_GLOBAL=128;
    const unsigned int N_GLOBAL=128;
    // const parameters
    const unsigned int  WARP_SIZE=32;
    const unsigned int  M=16;
    const unsigned int  N=16;
    const unsigned int  K=16;
    const unsigned int  WMMA_M=16;
    const unsigned int  WMMA_N=16;
    const unsigned int  WMMA_K=16;

    const unsigned int  M_TILES=784;
    const unsigned int  K_TILES=8;
    const unsigned int  N_TILES=8;
    


    // typedef C_LAYOUT wmma::mem_row_major;

    const unsigned int WARPS_PER_BLOCK=8;
    const unsigned int THREADS_PER_BLOCK= (WARP_SIZE * WARPS_PER_BLOCK);
    const unsigned int CHUNK_K=8;
    const unsigned int CHUNK_LINE_BYTES=(CHUNK_K * K * sizeof(uint8_t));
    const unsigned int WARP_COPY_BYTES=(WARP_SIZE * sizeof(int4));
    const unsigned int CHUNK_COPY_LINES_PER_WARP=(WARP_COPY_BYTES / CHUNK_LINE_BYTES);
    const unsigned int CHUNK_COPY_LINE_LANES=(WARP_SIZE / CHUNK_COPY_LINES_PER_WARP);
    
    const unsigned int BLOCK_ROW_WARPS=2;
    const unsigned int BLOCK_COL_WARPS=4;
    
    const unsigned int WARP_ROW_TILES =4;
    const unsigned int WARP_COL_TILES =2;
    
    const unsigned int BLOCK_ROW_TILES =(WARP_ROW_TILES * BLOCK_ROW_WARPS);
    const unsigned int BLOCK_COL_TILES =(WARP_COL_TILES * BLOCK_COL_WARPS);
    
    const unsigned int GLOBAL_MEM_STRIDE =N_GLOBAL;
    
    const unsigned int SHMEM_STRIDE=(N * BLOCK_ROW_TILES);
    const unsigned int SHMEM_OFFSET=(N * WARP_ROW_TILES);
    
    const unsigned int SKEW_UINT8=32;
    

    // Convert the input pointers
    const uint8_t * A = reinterpret_cast<uint8_t*>(input0); // activation
    const uint8_t * B =  reinterpret_cast<uint8_t*>(input1); // weight
    const int * C = reinterpret_cast< int *>(input7);
    uint8_t * D = reinterpret_cast<uint8_t*>(output0);
    const int alpha = (int)(*input5);
    const int integer = (int)(*input6);



extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];

// Warp and lane identification.
const unsigned int warpId = threadIdx.x / WARP_SIZE;
const unsigned int laneId = threadIdx.x % WARP_SIZE;

// Offset in shared memory from which the B matrix is stored.
const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;       // BLOCK_COL_TILES is shared_A row numbers in one block

// This pointer is used to access the C and D matrix tiles this warp computes.
int *shmem_warp_tile_ptr = (int *)&shmem[0][0] +
(warpId / 2) * SHMEM_STRIDE * K * 2 +    // K * 2 is because one warp calculate k * 2 rows.
(warpId % 2) * SHMEM_OFFSET;

// This pointer is used to stream the C and D matrices block-wide tile to and
// from shared memory.
int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;   // confuse, may be used to read from global memory?

// Each CTA slides along the 128 x 128 tiles from the top left corner of the
// matrix to the right and down, and selects the next tile to compute. Once
// there's no such tile, all warps in this CTA exit.
for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
const unsigned int block_tile_i =                                   // get the i (row) index of all tiles
((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

// Stop when there are no more D matrix tiles to compute in this CTA.
if (block_tile_i >= M_TILES) {
break;
}

// This warp's pointer to the C matrix data to copy memory from to shared
// memory.
const size_t gmem_idx =
(block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
const int *src_gmem_warp_stream_ptr = &C[gmem_idx];

// Stream multiple C tiles to shared memory.
#pragma unroll
for (int i = 0; i < K; i++) {
typedef int4 copy_t;

*((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
*((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
laneId);
}

__syncthreads();

// These fragments will accumulate the result of A and B matrix fragment
// multiplications along the K_GLOBAL dimension.
wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
                       [WARP_ROW_TILES];

// Load the C matrix tiles into fragments from shared memory.
#pragma unroll
for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
for (int j = 0; j < WARP_ROW_TILES; j++) {
const int *tile_ptr =
shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, wmma::mem_row_major);
}
}

__syncthreads();

// Select what warp copies what matrix to shared memory.
// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
const uint8_t *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                M * K_GLOBAL * (warpId % 4) * 2)
             : (&B[block_tile_j * N * K_GLOBAL] +
                N * K_GLOBAL * (warpId % 4) * 2);

// Go through the global K dimension by a fixed step at a time.
#pragma unroll
for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
// Copy slices of the A and B matrices to shared memory.
// The first half of the warps in the CTA copy the A matrix, the rest copy
// the B matrix.
size_t shmem_idx =
warpId < (WARPS_PER_BLOCK / 2)
? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
: (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

// First half of the warp copies the first row / column of the matrix,
// the second half of the warp copies the next.
int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
  (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
(laneId % CHUNK_COPY_LINE_LANES);

// Shift the second half of the warp to the next row / column in the
// shared memory.
shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
i++) {
// Copy 16 bytes at once in each lane.
*((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
*lane_ptr;

// Advance the global memory pointer and the shared memory index.
lane_ptr = (int4 *)((uint8_t *)lane_ptr +
K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
shmem_idx += CHUNK_COPY_LINES_PER_WARP;
}

__syncthreads();

// Compute a grid of C matrix tiles in each warp.
#pragma unroll
for (int k_step = 0; k_step < CHUNK_K; k_step++) {
wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major>
a[WARP_COL_TILES];
wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major>
b[WARP_ROW_TILES];

#pragma unroll
for (int i = 0; i < WARP_COL_TILES; i++) {
size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
const uint8_t *tile_ptr = &shmem[shmem_idx_a][k_step * K];

wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);

#pragma unroll
for (int j = 0; j < WARP_ROW_TILES; j++) {
if (i == 0) {
// Load the B matrix fragment once, because it is going to be
// reused against the other A matrix fragments.
size_t shmem_idx_b = shmem_idx_b_off +
     (WARP_ROW_TILES * N) * (warpId % 2) +
     (j * N);
const uint8_t *tile_ptr = &shmem[shmem_idx_b][k_step * K];

wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
}

wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
}
}
}

__syncthreads();
}

// Store the D fragments to shared memory.
#pragma unroll
for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
// Uniform, point-wise transformations of ALL fragment elements by ALL
// threads in the warp are well-defined even though element indices
// within fragment storage are not defined.
for (int t = 0; t < c[i][j].num_elements; t++) {
c[i][j].x[t] = ((c[i][j].x[t] * alpha) >> integer);
}

int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE,wmma::mem_row_major);
}
}

__syncthreads();

// Now that shared memory contains all the D tiles, stream them to global
// memory.
uint8_t *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
for (int i = 0; i < K; i++) {
for(int k = 0; k < 4; k++){
uint8_t val_tmp = (uint8_t)*((int *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i + laneId * 4 + k));
*(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i + laneId * 4 + k) = 
val_tmp > 0 ? val_tmp:0;
}
}
__syncthreads();
}
}