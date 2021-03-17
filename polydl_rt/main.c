
#include <stdio.h>
#include <sys/time.h>

// export LD_LIBRARY_PATH=${PWD}/../polydl_rt:$LD_LIBRARY_PATH
// gcc -O3 main.c -L . -lpolydl_rt -lm

extern void polydl_lib_matmul_f32(long long int M, long long int N, long long int K,
	long long int A_stride, long long int B_stride, long long int C_stride,
	float *A, float *B, float *C);

void init_array(long long int M, long long int N, long long int K,
	float A[M][K], float B[K][N], float C[M][N], float C_ref[M][N]) {

	int i, j;
	for (i = 0; i < M; i++) {
		for (j = 0; j < K; j++) {
			A[i][j] = (i + j);
		}
	}

	for (i = 0; i < K; i++) {
		for (j = 0; j < N; j++) {
			B[i][j] = (float)(i * j);
		}
	}

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			C[i][j] = 0.0;
			C_ref[i][j] = 0.0;
		}
	}
}

void matmul_ref(long long int M, long long int N, long long int K,
	float A[M][K], float B[K][N], float C[M][N]) {
	int i, j, k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < K; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void print_array(long long int M, long long int N, float C[M][N]) {
	int i, j;

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			fprintf(stderr, "%lf ", C[i][j]);
			if (j % 80 == 79)
				fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
	}
}

typedef struct {
	double max_rel_err;
	double max_abs_err;
	double l2_rel_err;
	double one_norm_ref;
	double one_norm_test;
} correctness_t;

void compare_buf(float* ref, float* test, long size, correctness_t* norms)
{
	int i;
	double diff, rel_err;

	norms->max_rel_err = 0.;
	norms->max_abs_err = 0.;
	norms->l2_rel_err = 0.;
	norms->one_norm_ref = 0.;
	norms->one_norm_test = 0.;

	for (i = 0; i < size; ++i) {
		norms->one_norm_ref += (double)ref[i];
		norms->one_norm_test += (double)test[i];
		diff = fabs((double)ref[i] - (double)test[i]);
		norms->l2_rel_err += (diff*diff);
		rel_err = 0.0;
		if (diff > 0.0) {
			rel_err = diff / fabs((double)ref[i]);
		}
		if (rel_err > norms->max_rel_err) {
			norms->max_rel_err = rel_err;
#if 0
			printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e) (R:%12.4e)\n", i, ref[i], test[i], diff, rel_err);
#endif
		}
		if (diff > norms->max_abs_err) {
			norms->max_abs_err = diff;
		}
#if 0
		if (diff > 1.0) {
			printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e)\n", i, ref[i], test[i], diff);
		}
#endif

	}
	norms->l2_rel_err = sqrt(norms->l2_rel_err);
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

int main(int argc, char** argv) {
	long long int M = 32, N = 32, K = 32;

	int iters = 1000;
	int i = 1;
	if (argc > i) M = atoi(argv[i++]);
	if (argc > i) N = atoi(argv[i++]);
	if (argc > i) K = atoi(argv[i++]);
	if (argc > i) iters = atoi(argv[i++]);

	float *A = (float*)malloc(sizeof(float)*M*K);
	float *B = (float*)malloc(sizeof(float)*K*N);
	float *C = (float*)malloc(sizeof(float)*M*N);
	float *C_ref = (float*)malloc(sizeof(float)*M*N);

	init_array(M, N, K, A, B, C, C_ref);
	matmul_ref(M, N, K, A, B, C_ref);
	double t_start, t_end;

	polydl_lib_matmul_f32(M, N, K, K, N, N, A, B, C);
	correctness_t norms_fwd;
	memset(&norms_fwd, 0, sizeof(norms_fwd));
	/* compare */
	compare_buf(C_ref, C, M*N, &norms_fwd);
	printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
	printf("             1-norm of GEMM-code: %f\n", norms_fwd.one_norm_test);
	printf("      L2-error-norm of GEMM-code: %f\n", norms_fwd.l2_rel_err);
	printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
	printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);

	t_start = rtclock();
	for (i = 0; i < iters; i++) {
		polydl_lib_matmul_f32(M, N, K, K, N, N, A, B, C);
	}

	t_end = rtclock();

	printf("%0.2lf GFLOPS\n",
		(iters * 2.0 * M * N * K) / (t_end - t_start) / 1E9);

	return 0;
}
