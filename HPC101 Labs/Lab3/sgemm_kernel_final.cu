#include "gemm.cuh"

// __global__ void sgemm_kernel(const float* __restrict__ A,
// 							 const float* __restrict__ B, float* __restrict__ C,
// 							 const int m, const int n, const int k) {
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;

// 	if (row < m && col < n) {
// 		float sum = 0.0f;
// 		for (int i = 0; i < k; i++) {
// 			sum += A[row * k + i] * B[i + col * k];
// 		}
// 		C[row + col * m] = sum;
// 	}
// }

__global__ void sgemm_kernel(float* __restrict__ A,
							 float* __restrict__ B, float* __restrict__ C,
							 const int m, const int n, const int k) {
	// int row = blockIdx.y * blockDim.y + threadIdx.y;
	// int col = blockIdx.x * blockDim.x + threadIdx.x;

	// 设置小 block 参数，也是开辟的 shared memory 空间
	const int BM = 128;
    const int BN = 128;
	const int BK = 8;

	// 每个 thread 处理结果矩阵中 8 * 8 的数据
	const int TM = 8;
	const int TN = 8;

	const int tid = threadIdx.y * blockDim.x + threadIdx.x; // thread 在对应 block 中的 id

	__shared__ float sub_a[BM][BK];
	__shared__ float sub_b[BK][BN];

	float r_c[TM][TN] = {0.0};

	int a_smem_row = tid >> 1;
	int a_smem_col = (tid & 1) << 2;
	int b_smem_row = (tid & 1) << 2;
	int b_smem_col = tid >> 1;

	int a_gmem_row = blockIdx.y * BM + a_smem_row;
	int b_gmem_col = blockIdx.x * BN + b_smem_col;

	for (int bk = 0; bk < (k + BK - 1) / BK; bk++){
		int a_gmem_col = bk * BK + a_smem_col;
		int b_gmem_row = bk * BK + b_smem_row;

		float4 tmpA = *reinterpret_cast<float4 *>(&A[a_gmem_row * k + a_gmem_col]);
		sub_a[a_smem_row][a_smem_col] = tmpA.x;
		sub_a[a_smem_row][a_smem_col + 1] = tmpA.y;
		sub_a[a_smem_row][a_smem_col + 2] = tmpA.z;
		sub_a[a_smem_row][a_smem_col + 3] = tmpA.w;

		float4 tmpB = *reinterpret_cast<float4 *>(&B[b_gmem_col * k + b_gmem_row]);
		sub_b[b_smem_row][b_smem_col] = tmpB.x;
		sub_b[b_smem_row + 1][b_smem_col] = tmpB.y;
		sub_b[b_smem_row + 2][b_smem_col] = tmpB.z;
		sub_b[b_smem_row + 3][b_smem_col] = tmpB.w;

		__syncthreads();

		#pragma unroll
		for (int K = 0; K < BK; K++){
			#pragma unroll
			for(int M = 0; M < TM; M++){
				#pragma unroll
				for(int N = 0; N < TN; N++){
					r_c[M][N] += sub_a[threadIdx.y * TM + M][K] * sub_b[K][threadIdx.x * TN + N];
				}
			}
		}
		__syncthreads();
	}

	#pragma unroll
	for(int j = 0; j < TN; j++){
		int c_gmem_col = blockIdx.x * BN + threadIdx.x * TN + j;
		#pragma unroll
		for(int i = 0; i < TM; i+= 4){
			int c_gmem_row = blockIdx.y * BM + threadIdx.y * TM + i;

			float4* C_float4 = reinterpret_cast<float4*>(&C[c_gmem_row + c_gmem_col * m]);
			float4 r_c_vec = make_float4(r_c[i][j], r_c[i + 1][j], r_c[i + 2][j], r_c[i + 3][j]);
			C_float4[0] = r_c_vec;
		}
	}
}

template <>
void run_custom<float>(thrust::device_vector<float>& d_A,
					   thrust::device_vector<float>& d_B,
					   thrust::device_vector<float>& d_C, const int m,
					   const int n, const int k) {
	
	dim3 block(16, 16);  // 一个block里有 16 * 16 个线程，每个线程计算 C 矩阵中 8 * 8 的小矩阵值
	dim3 grid((n + block.x - 1) / (block.x * 8), (m + block.y - 1) / (block.y * 8));
	sgemm_kernel<<<grid, block>>>(d_A.data().get(),
								  d_B.data().get(),
								  d_C.data().get(), m, n, k);
}