#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>
#include <libxsmm.h>


#ifndef M1
#define M1 32
#endif // !M1

#ifndef N1
#define N1 32
#endif // !N1

#ifndef K1
#define K1 32
#endif // !K1


#ifndef M2_Tile
#define M2_Tile M1
#endif // !M2_Tile

#ifndef N2_Tile
#define N2_Tile N1
#endif // !N2_Tile

#ifndef K2_Tile
#define K2_Tile K1
#endif // !K2_Tile

#ifndef M1_Tile
#define M1_Tile M2_Tile
#endif // !M1_Tile

#ifndef N1_Tile
#define N1_Tile N2_Tile
#endif // !N1_Tile

#ifndef K1_Tile
#define K1_Tile K2_Tile
#endif // !K1_Tile


#ifndef alpha
#define alpha 1
#endif // !alpha

#ifndef beta
#define beta 1
#endif // !beta

#ifndef NUM_ITERS
#define NUM_ITERS 1000
#endif // !NUM_ITERS



#define TIME
#ifdef TIME
#define IF_TIME(foo) foo;
#else
#define IF_TIME(foo)
#endif


/* function-pointer to LIBXSMM kernel */

void init_array(float A[M1][K1], float B[K1][N1], float C[M1][N1], float C_ref[M1][N1]) {
	int i, j;

	for (i = 0; i < M1; i++) {
		for (j = 0; j < K1; j++) {
			A[i][j] = ((float)i + (float)j) / (float)(M1 + K1);
		}
	}

	for (i = 0; i < K1; i++) {
		for (j = 0; j < N1; j++) {
			B[i][j] = ((float)i * (float)j) / (float)(K1 + N1);
		}
	}

	for (i = 0; i < M1; i++) {
		for (j = 0; j < N1; j++) {
			C[i][j] = 0.0;
			C_ref[i][j] = 0.0;
		}
	}
}


void matmul_ref(float A[M1][K1], float B[K1][N1], float C[M1][N1]) {
	int i, j, k;
#pragma omp parallel for private(j, k)
	for (i = 0; i < M1; i++)
		for (j = 0; j < N1; j++)
			for (k = 0; k < K1; k++)
				C[i][j] = C[i][j] + A[i][k] * B[k][j];

}


double rtclock() {
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday(&Tp, &Tzp);
	if (stat != 0)
		printf("Error return from gettimeofday: %d", stat);
	return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
double t_start, t_end;


int main() {
	int i, j, k, t;

	double l_total;

	// C[M][N] = A[M][K] * B[K][N];
	float(*A)[K1] = (float*)libxsmm_aligned_malloc(M1*K1 * sizeof(float), 2097152);
	float(*B)[N1] = (float*)libxsmm_aligned_malloc(K1*N1 * sizeof(float), 2097152);
	float(*C)[N1] = (float*)libxsmm_aligned_malloc(M1*N1 * sizeof(float), 2097152);
	float(*C_ref)[N1] = (float*)libxsmm_aligned_malloc(M1*N1 * sizeof(float), 2097152);

	int N1_val = N1;
	int M1_val = M1;
	int K1_val = K1;

	printf("M1 = %d, N1 = %d, K1 = %d\n", M1, N1, K1);
	printf("SIZE A  (MB): %10.2f MB\n", (double)(M1*K1 * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE B  (MB): %10.2f MB\n", (double)(K1*N1 * sizeof(float)) / (1024.0*1024.0));
	printf("SIZE C  (MB): %10.2f MB\n", (double)(M1*N1 * sizeof(float)) / (1024.0*1024.0));

	init_array(A, B, C, C_ref);

	unsigned long long l_start, l_end;
	l_start = libxsmm_timer_tick();
	for (int iter = 0; iter < NUM_ITERS; iter++) {
		matmul_ref(A, B, C);
	}

	l_end = libxsmm_timer_tick();
	l_total = libxsmm_timer_duration(l_start, l_end);

	double flops = NUM_ITERS * 2.0 * M1 * N1 * K1;
	printf("Real_GFLOPS =%0.2lf\n",
			(flops*1e-9) / l_total);

	printf("A: %f, %f\n", A[0][0], A[M1 - 1][K1 - 1]);
	printf("B: %f, %f\n", B[0][0], B[K1 - 1][N1 - 1]);
	printf("C_ref: %f, %f, %f\n", C_ref[0][0], C_ref[M1 / 2][N1 / 2], C_ref[M1 - 1][N1 - 1]);
	printf("C    : %f, %f, %f\n", C[0][0], C[M1 / 2][N1 / 2], C[M1 - 1][N1 - 1]);

	libxsmm_free(A);
	libxsmm_free(B);
	libxsmm_free(C);
	libxsmm_free(C_ref);

	return 0;
}
