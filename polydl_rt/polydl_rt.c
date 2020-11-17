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
void polydl_lib_matmul_f32_ver2(
	long long int M, long long int N, long long int K,
	long long int A_stride, long long int B_stride, long long int C_stride,
	float *A, float *B, float *C) {

	__m512 vec_C = _mm512_setzero_ps();
	__m512 vec_A;
	__m512 vec_B;

	__m512 vec_C_1 = _mm512_setzero_ps();
	__m512 vec_C_2 = _mm512_setzero_ps();
	__m512 vec_C_3 = _mm512_setzero_ps();

	__m512 vec_A_1, vec_A_2, vec_A_3;
	__m512 vec_A_k_1, vec_A_1_k_1, vec_A_2_k_1, vec_A_3_k_1;
	__m512 vec_B_k_1;

	int i, j, k;
	for (i = 0; i < M; i += 4) {
		for (j = 0; j < N; j += 16) {
			vec_C = _mm512_load_ps((__m512*)&C[i*C_stride + j]);
			vec_C_1 = _mm512_load_ps((__m512*)&C[(i + 1)*C_stride + j]);
			vec_C_2 = _mm512_load_ps((__m512*)&C[(i + 2)*C_stride + j]);
			vec_C_3 = _mm512_load_ps((__m512*)&C[(i + 3)*C_stride + j]);


			for (k = 0; k < K; k += 2) {
				/* Equivalent to:
				C[i][j] += A[i][k]* B[k][j];
				C[i][j+1] += A[i][k]* B[k][j+1];
				C[i][j+2] += A[i][k]* B[k][j+2];

				C[i+1][j] += A[i+1][k]* B[k][j];
				C[i+1][j+1] += A[i+1][k]* B[k][j+1];
				C[i+1][j+2] += A[i+1][k]* B[k][j+2];

				C[i][j] += A[i][k+1]* B[k+1][j];
				C[i][j+1] += A[i][k+1]* B[k+1][j+1];
				C[i][j+2] += A[i][k+1]* B[k+1][j+2];


				...*/
				vec_A = _mm512_set1_ps(A[i*A_stride + k]);
				vec_A_1 = _mm512_set1_ps(A[(i + 1)*A_stride + k]);
				vec_A_2 = _mm512_set1_ps(A[(i + 2)*A_stride + k]);
				vec_A_3 = _mm512_set1_ps(A[(i + 3)*A_stride + k]);
				vec_B = _mm512_load_ps((__m512*)&B[k*B_stride + j]);

				vec_A_k_1 = _mm512_set1_ps(A[i*A_stride + (k + 1)]);
				vec_A_1_k_1 = _mm512_set1_ps(A[(i + 1)*A_stride + (k + 1)]);
				vec_A_2_k_1 = _mm512_set1_ps(A[(i + 2)*A_stride + (k + 1)]);
				vec_A_3_k_1 = _mm512_set1_ps(A[(i + 3)*A_stride + (k + 1)]);
				vec_B_k_1 = _mm512_load_ps((__m512*)&B[(k + 1)*B_stride + j]);

				vec_C = _mm512_add_ps(vec_C, _mm512_mul_ps(vec_A, vec_B));
				vec_C_1 = _mm512_add_ps(vec_C_1, _mm512_mul_ps(vec_A_1, vec_B));
				vec_C_2 = _mm512_add_ps(vec_C_2, _mm512_mul_ps(vec_A_2, vec_B));
				vec_C_3 = _mm512_add_ps(vec_C_3, _mm512_mul_ps(vec_A_3, vec_B));

				vec_C = _mm512_add_ps(vec_C, _mm512_mul_ps(vec_A_k_1, vec_B_k_1));
				vec_C_1 = _mm512_add_ps(vec_C_1, _mm512_mul_ps(vec_A_1_k_1, vec_B_k_1));
				vec_C_2 = _mm512_add_ps(vec_C_2, _mm512_mul_ps(vec_A_2_k_1, vec_B_k_1));
				vec_C_3 = _mm512_add_ps(vec_C_3, _mm512_mul_ps(vec_A_3_k_1, vec_B_k_1));
			}

			_mm512_store_ps((__m512*)&C[i*C_stride + j], vec_C);
			_mm512_store_ps((__m512*)&C[(i + 1)*C_stride + j], vec_C_1);
			_mm512_store_ps((__m512*)&C[(i + 2)*C_stride + j], vec_C_2);
			_mm512_store_ps((__m512*)&C[(i + 3)*C_stride + j], vec_C_3);
		}
	}
}


void polydl_lib_matmul_f32(
	long long int M, long long int N, long long int K,
	long long int A_stride, long long int B_stride, long long int C_stride,
	float *A, float *B, float *C) {

	__m512 vec_C;
	__m512 vec_A;
	__m512 vec_B;

	__m512 vec_C_1, vec_C_2, vec_C_3, vec_C_4, vec_C_5, vec_C_6, vec_C_7;
	__m512 vec_A_1, vec_A_2, vec_A_3, vec_A_4, vec_A_5, vec_A_6, vec_A_7;

	int i, j, k;
	for (i = 0; i < M; i += 8) {
		for (j = 0; j < N; j += 16) {
			vec_C = _mm512_load_ps((__m512*)&C[i*C_stride + j]);
			vec_C_1 = _mm512_load_ps((__m512*)&C[(i + 1)*C_stride + j]);
			vec_C_2 = _mm512_load_ps((__m512*)&C[(i + 2)*C_stride + j]);
			vec_C_3 = _mm512_load_ps((__m512*)&C[(i + 3)*C_stride + j]);
			vec_C_4 = _mm512_load_ps((__m512*)&C[(i + 4)*C_stride + j]);
			vec_C_5 = _mm512_load_ps((__m512*)&C[(i + 5)*C_stride + j]);
			vec_C_6 = _mm512_load_ps((__m512*)&C[(i + 6)*C_stride + j]);
			vec_C_7 = _mm512_load_ps((__m512*)&C[(i + 7)*C_stride + j]);


			for (k = 0; k < K; k++) {
				vec_B = _mm512_load_ps((__m512*)&B[k*B_stride + j]);

				vec_A = _mm512_set1_ps(A[i*A_stride + k]);
				vec_A_1 = _mm512_set1_ps(A[(i + 1)*A_stride + k]);
				vec_A_2 = _mm512_set1_ps(A[(i + 2)*A_stride + k]);
				vec_A_3 = _mm512_set1_ps(A[(i + 3)*A_stride + k]);
				vec_A_4 = _mm512_set1_ps(A[(i + 4)*A_stride + k]);
				vec_A_5 = _mm512_set1_ps(A[(i + 5)*A_stride + k]);
				vec_A_6 = _mm512_set1_ps(A[(i + 6)*A_stride + k]);
				vec_A_7 = _mm512_set1_ps(A[(i + 7)*A_stride + k]);


				vec_C = _mm512_add_ps(vec_C, _mm512_mul_ps(vec_A, vec_B));
				vec_C_1 = _mm512_add_ps(vec_C_1, _mm512_mul_ps(vec_A_1, vec_B));
				vec_C_2 = _mm512_add_ps(vec_C_2, _mm512_mul_ps(vec_A_2, vec_B));
				vec_C_3 = _mm512_add_ps(vec_C_3, _mm512_mul_ps(vec_A_3, vec_B));
				vec_C_4 = _mm512_add_ps(vec_C_4, _mm512_mul_ps(vec_A_4, vec_B));
				vec_C_5 = _mm512_add_ps(vec_C_5, _mm512_mul_ps(vec_A_5, vec_B));
				vec_C_6 = _mm512_add_ps(vec_C_6, _mm512_mul_ps(vec_A_6, vec_B));
				vec_C_7 = _mm512_add_ps(vec_C_7, _mm512_mul_ps(vec_A_7, vec_B));
			}

			_mm512_store_ps((__m512*)&C[i*C_stride + j], vec_C);
			_mm512_store_ps((__m512*)&C[(i + 1)*C_stride + j], vec_C_1);
			_mm512_store_ps((__m512*)&C[(i + 2)*C_stride + j], vec_C_2);
			_mm512_store_ps((__m512*)&C[(i + 3)*C_stride + j], vec_C_3);
			_mm512_store_ps((__m512*)&C[(i + 4)*C_stride + j], vec_C_4);
			_mm512_store_ps((__m512*)&C[(i + 5)*C_stride + j], vec_C_5);
			_mm512_store_ps((__m512*)&C[(i + 6)*C_stride + j], vec_C_6);
			_mm512_store_ps((__m512*)&C[(i + 7)*C_stride + j], vec_C_7);
		}
	}
}

void polydl_lib_matmul_f32_ver1(
	long long int M, long long int N, long long int K,
	long long int A_stride, long long int B_stride, long long int C_stride,
	float *A, float *B, float *C) {

	__m512 vec_C = _mm512_setzero_ps();
	__m512 vec_A = _mm512_setzero_ps();
	__m512 vec_B = _mm512_setzero_ps();

	__m512 vec_C_1 = _mm512_setzero_ps();
	__m512 vec_C_2 = _mm512_setzero_ps();
	__m512 vec_C_3 = _mm512_setzero_ps();

	__m512 vec_A_1, vec_A_2, vec_A_3;

	int i, j, k;
	for (i = 0; i < M; i += 4) {
		for (j = 0; j < N; j += 16) {
			vec_C = _mm512_load_ps((__m512*)&C[i*C_stride + j]);
			vec_C_1 = _mm512_load_ps((__m512*)&C[(i + 1)*C_stride + j]);
			vec_C_2 = _mm512_load_ps((__m512*)&C[(i + 2)*C_stride + j]);
			vec_C_3 = _mm512_load_ps((__m512*)&C[(i + 3)*C_stride + j]);


			for (k = 0; k < K; k++) {

				vec_A = _mm512_set1_ps(A[i*A_stride + k]);
				vec_A_1 = _mm512_set1_ps(A[(i + 1)*A_stride + k]);
				vec_A_2 = _mm512_set1_ps(A[(i + 2)*A_stride + k]);
				vec_A_3 = _mm512_set1_ps(A[(i + 3)*A_stride + k]);
				vec_B = _mm512_load_ps((__m512*)&B[k*B_stride + j]);

				vec_C = _mm512_add_ps(vec_C, _mm512_mul_ps(vec_A, vec_B));
				vec_C_1 = _mm512_add_ps(vec_C_1, _mm512_mul_ps(vec_A_1, vec_B));
				vec_C_2 = _mm512_add_ps(vec_C_2, _mm512_mul_ps(vec_A_2, vec_B));
				vec_C_3 = _mm512_add_ps(vec_C_3, _mm512_mul_ps(vec_A_3, vec_B));
			}

			_mm512_store_ps((__m512*)&C[i*C_stride + j], vec_C);
			_mm512_store_ps((__m512*)&C[(i + 1)*C_stride + j], vec_C_1);
			_mm512_store_ps((__m512*)&C[(i + 2)*C_stride + j], vec_C_2);
			_mm512_store_ps((__m512*)&C[(i + 3)*C_stride + j], vec_C_3);
		}
	}
}


void polydl_lib_matmul_f32_base(
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
