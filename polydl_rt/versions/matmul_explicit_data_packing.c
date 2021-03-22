#include <immintrin.h>  
#include "output.c" 

#ifndef M1
#define M1 4096
#endif // !M1

#ifndef N1
#define N1 8192
#endif // !N1

#ifndef K1
#define K1 16384
#endif // !K1

#ifndef M2_Tile
#define M2_Tile 512
#endif // !M2_Tile

#ifndef N2_Tile
#define N2_Tile 1024
#endif // !N2_Tile

#ifndef K2_Tile
#define K2_Tile 2048
#endif // !K2_Tile

#ifndef M1_Tile
#define M1_Tile 64
#endif // !M1_Tile

#ifndef N1_Tile
#define N1_Tile 64
#endif // !N1_Tile

#ifndef K1_Tile
#define K1_Tile 64
#endif // !K1_Tile

#define M_pad ((M1%2)? (2-(M1%2)) : (0))
#define N_pad ((N1%16)? (16-(N1%16)) : (0))
#define K_pad ((K1%2)? (2-(K1%2)) : (0))

// #define M_pad 0
// #define N_pad 0
// #define K_pad 0

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

double matmul_high_performance_scop(float A[M1][K1], float B[K1][N1], float C[M1][N1], int iters)
{
	int it2, jt2, kt2, it1, jt1, kt1, i, j, k;
	printf("In matmul3 matmul_high_performance_scop\n");
#pragma scop
	// First level of tiling
#pragma omp parallel for private(jt2, kt2, it1, jt1, kt1, i, j, k)
	for (it2 = 0; it2 < M1; it2 += M2_Tile) {
		for (jt2 = 0; jt2 < N1; jt2 += N2_Tile) {
			for (kt2 = 0; kt2 < K1; kt2 += K2_Tile) {

				// Second level of tiling
				for (it1 = it2; it1 < min(M1, it2 + M2_Tile); it1 += M1_Tile) {
					for (jt1 = jt2; jt1 < min(N1, jt2 + N2_Tile); jt1 += N1_Tile) {
						for (kt1 = kt2; kt1 < min(K1, kt2 + K2_Tile); kt1 += K1_Tile) {

							// Inner most intra-tile loops
							for (i = it1; i < min(M1, it1 + M1_Tile); i++) {
								for (j = jt1; j < min(N1, jt1 + N1_Tile); j++) {
									for (k = kt1; k < min(K1, kt1 + K1_Tile); k++) {
										C[i][j] = C[i][j] + A[i][k] * B[k][j];
									}
								}
							}
						}
					}
				}
			}
		}
	}
#pragma endscop

	return 1;
}

#ifdef USE_LIBXSMM
#include <libxsmm.h>
extern libxsmm_smmfunction fwd_gemm;


void copyToTiledArray(int SIZE1, int SIZE2, int T1, int T2, int pad1, int pad2,
		float A[SIZE1-pad1][SIZE2-pad2], float A_Tiled[SIZE1 / T1][SIZE2 / T2][T1][T2]) {
	int it, jt, i, j;
	int realSize1 = (SIZE1 - pad1);
	int realSize2 = (SIZE2 - pad2);

	int paddedTileDimSize1 = SIZE1 / T1;
	int paddedTileDimSize2 = SIZE2 / T2;

#pragma omp parallel for private(jt, i, j)
	for (it = 0; it < paddedTileDimSize1; it++) {
		int itT1 = it*T1;
		for (jt = 0; jt < paddedTileDimSize2; jt++) {
			int jtT2 = jt*T2;
			int max_i = (itT1 + T1) > realSize1 ? T1 - (itT1 + T1 - realSize1) : T1;
			for (i = 0; i < max_i; i++) {
				int max_j = (jtT2 + T2) > realSize2 ? T2 - (jtT2 + T2 - realSize2) : T2;
				for (j = 0; j < max_j; j++) {
					A_Tiled[it][jt][i][j] = A[itT1 + i][jtT2 + j];
				}

				for (j = max_j; j < T2; j++) {
					A_Tiled[it][jt][i][j] = 0;

				}
			}

			for (i = max_i; i < T1; i++) {
				for (j = 0; j < T2; j++) {
					A_Tiled[it][jt][i][j] = 0;
				}
			}
		}
	}
}



void copyFromTiledArray(int SIZE1, int SIZE2, int T1, int T2, int pad1, int pad2,
		float A[SIZE1-pad1][SIZE2-pad2], float A_Tiled[SIZE1 / T1][SIZE2 / T2][T1][T2]) {

	int it, jt, i, j;
	int realSize1 = (SIZE1 - pad1);
	int realSize2 = (SIZE2 - pad2);

	int paddedTileDimSize1 = SIZE1 / T1;
	int paddedTileDimSize2 = SIZE2 / T2;

#pragma omp parallel for private(jt, i, j)
	for (it = 0; it < paddedTileDimSize1; it++) {
		int itT1 = it*T1;
		for (jt = 0; jt < paddedTileDimSize2; jt++) {
			int jtT2 = jt*T2;
			int max_i = (itT1 + T1) > realSize1 ? T1 - (itT1 + T1 - realSize1) : T1;
			for (i = 0; i < max_i; i++) {
				int max_j = (jtT2 + T2) > realSize2 ? T2 - (jtT2 + T2 - realSize2) : T2;
				for (j = 0; j < max_j; j++) {
					A[itT1 + i][jtT2 + j] = A_Tiled[it][jt][i][j];
				}

			}

		}
	}

}

double matmul_high_performance(float A[M1][K1], float B[K1][N1], float C[M1][N1], int iters) {
	unsigned long long l_start, l_end;
	double l_total = 0.0;
	int i;
	printf("In matmul_explicit_data_packing.c\n");
	printf("M1_Tile = %d, N1_Tile = %d, K1_Tile = %d\n", M1_Tile, N1_Tile, K1_Tile);
	printf("M2_Tile = %d, N2_Tile = %d, K2_Tile = %d\n", M2_Tile, N2_Tile, K2_Tile);

#ifdef PARALLEL_it2
	printf("it2 loop is parallel\n");
#endif

#ifdef PARALLEL_jt2
	printf("jt2 loop is parallel\n");
#endif

	printf("M_pad = %d,  N_pad = %d, K_pad = %d\n", M_pad, N_pad, K_pad);
	float(*A_Tiled)[(K1+K_pad) / K1_Tile][M1_Tile][K1_Tile] =
		(float*)libxsmm_aligned_malloc((M1+M_pad)*(K1+K_pad) * sizeof(float), 2097152);
	float(*B_Tiled)[(N1+N_pad) / N1_Tile][K1_Tile][N1_Tile] =
		(float*)libxsmm_aligned_malloc((N1+N_pad)*(K1+K_pad) * sizeof(float), 2097152);
	float(*C_Tiled)[(N1+N_pad) / N1_Tile][M1_Tile][N1_Tile] =
		(float*)libxsmm_aligned_malloc((M1+M_pad)*(N1+N_pad) * sizeof(float), 2097152);

	l_start = libxsmm_timer_tick();

	for (i = 0; i < iters; i++) {
		copyToTiledArray(M1+M_pad, K1+K_pad, M1_Tile, K1_Tile, M_pad, K_pad, A, A_Tiled);
		copyToTiledArray(K1+K_pad, N1+N_pad, K1_Tile, N1_Tile, K_pad, N_pad, B, B_Tiled);
		copyToTiledArray(M1+M_pad, N1+N_pad, M1_Tile, N1_Tile, M_pad, N_pad, C, C_Tiled);

		matmul_high_performance_core(A_Tiled, B_Tiled, C_Tiled);

		copyFromTiledArray(M1+M_pad, N1+N_pad, M1_Tile, N1_Tile, M_pad, N_pad, C, C_Tiled);
	}

	l_end = libxsmm_timer_tick();
	l_total = libxsmm_timer_duration(l_start, l_end);

	libxsmm_free(A_Tiled);
	libxsmm_free(B_Tiled);
	libxsmm_free(C_Tiled);

	return l_total;
}

	void matmul_high_performance_core(
			float A[(M1+M_pad) / M1_Tile][(K1+K_pad) / K1_Tile][M1_Tile][K1_Tile],
			float B[(K1+K_pad) / K1_Tile][(N1+N_pad) / N1_Tile][K1_Tile][N1_Tile],
			float C[(M1+M_pad) / M1_Tile][(N1+N_pad) / N1_Tile][M1_Tile][N1_Tile])
{
	int it2, jt2, kt2, it1, jt1, kt1;
	int it2_start = 0;
	int it2_end = M1+M_pad;
	int jt2_start = 0;
	int jt2_end = N1+N_pad;
	int k2_start = 0;
	int kt2_end = K1+K_pad;

	// First level of tiling
#ifdef PARALLEL_it2
#pragma omp parallel for private(jt2, kt2, it1, jt1, kt1)
#endif
	for (it2 = it2_start; it2 < it2_end; it2 += M2_Tile) {
		int it1_start = it2;
		int it1_end = min(M1+M_pad, it2 + M2_Tile);

#ifdef PARALLEL_jt2
#pragma omp parallel for private(kt2, it1, jt1, kt1)
#endif
		for (jt2 = jt2_start; jt2 < jt2_end; jt2 += N2_Tile) {
			printf("thread_id = %d\n", omp_get_thread_num());
			int jt1_start = jt2;
			int jt1_end = min(N1+N_pad, jt2 + N2_Tile);

			for (kt2 = k2_start; kt2 < kt2_end; kt2 += K2_Tile) {
				int kt1_start = kt2;
				int kt1_end = min(K1+K_pad, kt2 + K2_Tile);
				// Second level of tiling
				for (it1 = it1_start; it1 < it1_end; it1 += M1_Tile) {
					for (jt1 = jt1_start; jt1 < jt1_end; jt1 += N1_Tile) {
						for (kt1 = kt1_start; kt1 < kt1_end; kt1 += K1_Tile) {
#ifdef jit_variant
							polydl_lib_matmul_f32_fma(M1_Tile,N1_Tile,K1_Tile,K1_Tile,N1_Tile,N1_Tile, 
									&A[it1 / M1_Tile][kt1 / K1_Tile][0][0], 
									&B[kt1 / K1_Tile][jt1 / N1_Tile][0][0], 
									&C[it1 / M1_Tile][jt1 / N1_Tile][0][0]);
#endif

#ifndef jit_variant
							fwd_gemm(&B[kt1 / K1_Tile][jt1 / N1_Tile][0][0],
									&A[it1 / M1_Tile][kt1 / K1_Tile][0][0],
									&C[it1 / M1_Tile][jt1 / N1_Tile][0][0]);
#endif

						}
					}
				}
			}
		}
	}


}

#endif

