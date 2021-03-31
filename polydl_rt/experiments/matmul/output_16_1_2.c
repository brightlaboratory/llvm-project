void polydl_lib_matmul_f32_fma( long long int M, long long int N, long long int K,long long int A_stride, long long int B_stride, long long int C_stride, float *A, float *B, float *C) {
	__m512 vec_C;
	__m512 vec_B, vec_B_1;
	__m512 vec_A, vec_A_1;
	int i, j, k;
	int i_aux, j_aux;
	long long M_full, N_full, K_full;
	M_full = (M / 16) * 16 ;
	N_full = (N / 1) * 1 ;
	K_full = (K / 2) * 2 ;
	for (i = 0; i < M_full; i += 16) {
		for (j = 0; j < N_full; j += 1) {

			float C_aux[16];
			for (int i2 = 0; i2 < 16; i2++) {
				C_aux[i2] = C[(i+i2)*C_stride+j];
			}

			vec_C = _mm512_load_ps((__m512*)C_aux);
			for (k = 0; k < K_full; k += 2) {
				// C[i][j] += A[i][k] * B[k][j];
				// C[i+1][j] += A[i+1][k] * B[k][j];
				// C[i+2][j] + A[i+2][k] * B[k][j];

				// C[i][j] += A[i][k+1] * B[k+1][j];
				// C[i+1][j] += A[i+1][k+1] * B[k+1][j];
				// C[i+2][j] + A[i+2][k+1] * B[k+1][j];

				vec_B = _mm512_set1_ps(B[k*B_stride + j]);
				vec_B_1 = _mm512_set1_ps(B[(k+1)*B_stride + j]);

				float A_aux[16], A_aux_1[16];
				for (int i2 = 0; i2 < 16; i2++) {
					A_aux[i2] = A[(i+i2)*A_stride+k];
					A_aux_1[i2] = A[(i+i2)*A_stride+k+1];
				}

				vec_A = _mm512_load_ps((__m512*)A_aux);
				vec_A_1 = _mm512_load_ps((__m512*)A_aux_1);
				vec_C = _mm512_fmadd_ps(vec_A, vec_B, vec_C);
				vec_C = _mm512_fmadd_ps(vec_A_1, vec_B_1, vec_C);
			}
			_mm512_store_ps((__m512*)C_aux, vec_C);

			for (int i2 = 0; i2 < 16; i2++) {
				C[(i+i2)*C_stride+j] = C_aux[i2];
			}

		}
	}
} 
