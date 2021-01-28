#ifndef Ti 
#define Ti 64 
#endif
#ifndef Tj 
#define Tj 64 
#endif
#ifndef Tk 
#define Tk 64 
#endif
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))
void polydl_lib_matmul_f32_i_8_j_16_k_1_fma( long long int M, long long int N, long long int K,long long int A_stride, long long int B_stride, long long int C_stride, float *A, float *B, float *C) {

        __m512 vec_A, vec_A1, vec_A2, vec_A3, vec_A4, vec_A5, vec_A6, vec_A7;
        __m512 vec_A8, vec_A9, vec_A10, vec_A11, vec_A12, vec_A13, vec_A14, vec_A15;
        __m512 vec_B, vec_B1, vec_B2, vec_B3, vec_B4, vec_B5, vec_B6, vec_B7;
        __m512 vec_C, vec_C1, vec_C2, vec_C3, vec_C4, vec_C5, vec_C6, vec_C7;

	int i, j, k;
	long long M_full, N_full, K_full;
	M_full = (M / Ti) * Ti ;
	N_full = (N / Tj) * Tj ;
	K_full = (K / Tk) * Tk ;
	int it,jt,kt;
	for (it = 0; it < M_full; it+=Ti) {
		for (jt = 0; jt < N_full; jt+=Tj) {
			for (kt = 0; kt < K_full; kt+=Tk) {
				for (i = it; i < min(M_full, it+Ti); i += 4) {
					for (j = jt; j < min(N_full, jt+Tj); j += 32) {
						vec_C = _mm512_load_ps((__m512*)&C[i*C_stride + j]);
						vec_C1 = _mm512_load_ps((__m512*)&C[(i + 1)*C_stride + j]);
						vec_C2 = _mm512_load_ps((__m512*)&C[(i + 2)*C_stride + j]);
						vec_C3 = _mm512_load_ps((__m512*)&C[(i + 3)*C_stride + j]);

                                                vec_C4 = _mm512_load_ps((__m512*)&C[i*C_stride + j+16]);
                                                vec_C5 = _mm512_load_ps((__m512*)&C[(i + 1)*C_stride + j+16]);
                                                vec_C6 = _mm512_load_ps((__m512*)&C[(i + 2)*C_stride + j+16]);
                                                vec_C7 = _mm512_load_ps((__m512*)&C[(i + 3)*C_stride + j+16]);

						for (k = kt; k < min(K_full, kt+Tk); k += 4) {

                                                        vec_B = _mm512_load_ps((__m512*)&B[k*B_stride + j]);
                                                        vec_B1 = _mm512_load_ps((__m512*)&B[(k+1)*B_stride + j]);  
                                                        vec_B2 = _mm512_load_ps((__m512*)&B[(k+2)*B_stride + j]);
                                                        vec_B3 = _mm512_load_ps((__m512*)&B[(k+3)*B_stride + j]);  
                                                                                    
                                                        vec_B4 = _mm512_load_ps((__m512*)&B[k*B_stride + j+16]);
                                                        vec_B5 = _mm512_load_ps((__m512*)&B[(k+1)*B_stride + j+16]);
                                                        vec_B6 = _mm512_load_ps((__m512*)&B[(k+2)*B_stride + j+16]);
                                                        vec_B7 = _mm512_load_ps((__m512*)&B[(k+3)*B_stride + j+16]);
                          
                                                        vec_A = _mm512_set1_ps(A[i*A_stride + k]);
                                                        vec_A1 = _mm512_set1_ps(A[(i + 1)*A_stride + k]);
                                                        vec_A2 = _mm512_set1_ps(A[(i + 2)*A_stride + k]);
                                                        vec_A3 = _mm512_set1_ps(A[(i + 3)*A_stride + k]);
                                                        vec_A4 = _mm512_set1_ps(A[i*A_stride + (k+1)]);
                                                        vec_A5 = _mm512_set1_ps(A[(i + 1)*A_stride + (k+1)]);
                                                        vec_A6 = _mm512_set1_ps(A[(i + 2)*A_stride + (k+1)]);
                                                        vec_A7 = _mm512_set1_ps(A[(i + 3)*A_stride + (k+1)]);
                                                        vec_A8 = _mm512_set1_ps(A[i*A_stride + (k+2)]);
                                                        vec_A9 = _mm512_set1_ps(A[(i + 1)*A_stride + (k+2)]);
                                                        vec_A10 = _mm512_set1_ps(A[(i + 2)*A_stride + (k+2)]);
                                                        vec_A11 = _mm512_set1_ps(A[(i + 3)*A_stride + (k+2)]);                    
                                                        vec_A12 = _mm512_set1_ps(A[i*A_stride + (k+3)]);
                                                        vec_A13 = _mm512_set1_ps(A[(i + 1)*A_stride + (k+3)]);
                                                        vec_A14 = _mm512_set1_ps(A[(i + 2)*A_stride + (k+3)]);
                                                        vec_A15 = _mm512_set1_ps(A[(i + 3)*A_stride + (k+3)]);

                                                        vec_C = _mm512_fmadd_ps(vec_A, vec_B, vec_C);
                                                        vec_C1 = _mm512_fmadd_ps(vec_A1, vec_B, vec_C1);
                                                        vec_C2 = _mm512_fmadd_ps(vec_A2, vec_B, vec_C2);
                                                        vec_C3 = _mm512_fmadd_ps(vec_A3, vec_B, vec_C3);
                                                        vec_C = _mm512_fmadd_ps(vec_A4, vec_B1, vec_C);
                                                        vec_C1 = _mm512_fmadd_ps(vec_A5, vec_B1, vec_C1);
                                                        vec_C2 = _mm512_fmadd_ps(vec_A6, vec_B1, vec_C2);
                                                        vec_C3 = _mm512_fmadd_ps(vec_A7, vec_B1, vec_C3);  

                                                        vec_C = _mm512_fmadd_ps(vec_A8, vec_B2, vec_C);
                                                        vec_C1 = _mm512_fmadd_ps(vec_A9, vec_B2, vec_C1);
                                                        vec_C2 = _mm512_fmadd_ps(vec_A10, vec_B2, vec_C2);
                                                        vec_C3 = _mm512_fmadd_ps(vec_A11, vec_B2, vec_C3);
                                                        vec_C = _mm512_fmadd_ps(vec_A12, vec_B3, vec_C);
                                                        vec_C1 = _mm512_fmadd_ps(vec_A13, vec_B3, vec_C1);
                                                        vec_C2 = _mm512_fmadd_ps(vec_A14, vec_B3, vec_C2);
                                                        vec_C3 = _mm512_fmadd_ps(vec_A15, vec_B3, vec_C3);  

                                                        vec_C4 = _mm512_fmadd_ps(vec_A, vec_B4, vec_C4);
                                                        vec_C5 = _mm512_fmadd_ps(vec_A1, vec_B5, vec_C5);
                                                        vec_C6 = _mm512_fmadd_ps(vec_A2, vec_B6, vec_C6);
                                                        vec_C7 = _mm512_fmadd_ps(vec_A3, vec_B7, vec_C7);
                                                        vec_C4 = _mm512_fmadd_ps(vec_A4, vec_B4, vec_C4);
                                                        vec_C5 = _mm512_fmadd_ps(vec_A5, vec_B5, vec_C5);
                                                        vec_C6 = _mm512_fmadd_ps(vec_A6, vec_B6, vec_C6);
                                                        vec_C7 = _mm512_fmadd_ps(vec_A7, vec_B7, vec_C7);

                                                        vec_C4 = _mm512_fmadd_ps(vec_A8, vec_B4, vec_C4);
                                                        vec_C5 = _mm512_fmadd_ps(vec_A9, vec_B5, vec_C5);
                                                        vec_C6 = _mm512_fmadd_ps(vec_A10, vec_B6, vec_C6);
                                                        vec_C7 = _mm512_fmadd_ps(vec_A11, vec_B7, vec_C7);
                                                        vec_C4 = _mm512_fmadd_ps(vec_A12, vec_B4, vec_C4);
                                                        vec_C5 = _mm512_fmadd_ps(vec_A13, vec_B5, vec_C5);
                                                        vec_C6 = _mm512_fmadd_ps(vec_A14, vec_B6, vec_C6);
                                                        vec_C7 = _mm512_fmadd_ps(vec_A15, vec_B7, vec_C7);


						}
						_mm512_store_ps((__m512*)&C[i*C_stride + j], vec_C);
						_mm512_store_ps((__m512*)&C[(i + 1)*C_stride + j], vec_C1);
						_mm512_store_ps((__m512*)&C[(i + 2)*C_stride + j], vec_C2);
						_mm512_store_ps((__m512*)&C[(i + 3)*C_stride + j], vec_C3);
                                                _mm512_store_ps((__m512*)&C[i*C_stride + j+16], vec_C4);
                                                _mm512_store_ps((__m512*)&C[(i + 1)*C_stride + j+16], vec_C5);
                                                _mm512_store_ps((__m512*)&C[(i + 2)*C_stride + j+16], vec_C6);
                                                _mm512_store_ps((__m512*)&C[(i + 3)*C_stride + j+16], vec_C7);

					}
				}
			}
		}
	}
} 
