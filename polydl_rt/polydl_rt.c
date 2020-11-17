#include <stdio.h>
#include <immintrin.h>



void print_f32_polydl(
	long long int rank, long long int offset,
	long long int size1, long long int size2,
	long long int stride1, long long int stride2,
	void *base) {
	int i, j;
	printf("rank = %ld, offset = %ld, size1 = %ld, size2 = %ld, stride1 = %ld, stride2 = %ld",
		rank, offset, size1, size2, stride1, stride2);
	float *ptr = (float*)base;
	for (i = 0; i < size1; i++) {
		for (j = 0; j < size2; j++) {
			printf("%f ", ptr[i*stride1 + j * stride2 + offset]);
		}
	}

}

#ifndef USE_AVX512
void polydl_lib_matmul_f32(
	long long int M, long long int N, long long int K,
	long long int A_stride, long long int B_stride, long long int C_stride,
	float *A, float *B, float *C) {

	printf("In polydl_lib_matmul_f32 function\n");
	printf("M = %ld, N = %ld, K = %ld, A_stride = %ld, B_stride = %ld, C_stride = %ld\n",
		M, N, K, A_stride, B_stride, C_stride);
	printf("A = %f, B = %f \n", A[0],B[0]);

	int i, j, k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < K; k++) {
				C[i*C_stride + j] +=
					A[i*A_stride + k] * B[k*B_stride + j];
			}
		}
	}
}
#endif

#ifdef USE_AVX512
void polydl_lib_matmul_f32(
	long long int M, long long int N, long long int K,
	long long int A_stride, long long int B_stride, long long int C_stride,
	float *A, float *B, float *C) {

	__m512 vec_C = _mm512_setzero_ps();
	__m512 vec_A = _mm512_setzero_ps();
	__m512 vec_B = _mm512_setzero_ps();


	int i, j, k;
	for (i = 0; i < M; i++) {
		for (k = 0; k < K; k++) {
			vec_A = _mm512_set1_ps(A[i*A_stride + k]);

			for (j = 0; j < N; j += 16) {

				/* Equivalent to:
				C[i][j] += A[i][k]* B[k][j];
				C[i][j+1] += A[i][k]* B[k][j+1];
				C[i][j+2] += A[i][k]* B[k][j+2];
				...*/

				vec_B = _mm512_load_ps((__m512*)&B[k*B_stride + j]);

				vec_C = _mm512_load_ps((__m512*)&C[i*C_stride + j]);
				vec_C = _mm512_add_ps(vec_C, _mm512_mul_ps(vec_A, vec_B));
				_mm512_store_ps((__m512*)&C[i*C_stride + j], vec_C);
			}
		}
	}
}
#endif
