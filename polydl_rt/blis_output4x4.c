#define alpha( i,j ) A[ (i)*ldA + (j) ]   // map alpha( i,j ) to array A
#define beta( i,j )  B[ (i)*ldB + (j) ]   // map beta( i,j ) to array B
#define gamma( i,j ) C[ (i)*ldC + (j) ]   // map gamma( i,j ) to array C

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))
#define NC 1024
#define KC 128
#define MC 180
#define MR 4
#define NR 4
void LoopFive( int, int, int, float *, int, float *, int, float *, int );
void LoopFour( int, int, int, float *, int, float *, int, float *, int );
void LoopThree( int, int, int, float *, int, float *, int, float *, int );
void LoopTwo( int, int, int, float *, int, float *, int, float *, int );
void LoopOne( int, int, int, float *, int, float *, int, float *, int );
void Gemm_4x4Kernel( int, float *, int, float *, int, float *, int );

void LoopFive( int m, int n, int k, float *A, int ldA,
		   float *B, int ldB, float *C, int ldC )
{   int j;
  for ( j=0; j<n; j+=NC ) {
    int jb = min( NC, n-j );    /* Last loop may not involve a full block */
    LoopFour( m, jb, k, A, ldA, &beta( 0,j ), ldB, &gamma( 0,j ), ldC );
  }
}

void LoopFour( int m, int n, int k, float *A, int ldA, 
	       float *B, int ldB, float *C, int ldC )
{   int p;
  for ( p=0; p<k; p+=KC ) {
    int pb = min( KC, k-p );    /* Last loop may not involve a full block */
    LoopThree( m, n, pb, &alpha( 0, p ), ldA, &beta( p, 0 ), ldB, C, ldC );
  }
}

void LoopThree( int m, int n, int k, float *A, int ldA, 
		float *B, int ldB, float *C, int ldC )
{ int i;
  for ( i=0; i<m; i+=MC ) {
    int ib = min( MC, m-i );    /* Last loop may not involve a full block */
    LoopTwo( ib, n, k, &alpha( i, 0), ldA, B, ldB, &gamma( i,0 ), ldC );
  }
}

void LoopTwo( int m, int n, int k, float *A, int ldA,
	      float *B, int ldB, float *C, int ldC )
{ int j;
  for ( j=0; j<n; j+=NR ) {
    int jb = min( NR, n-j );
    LoopOne( m, jb, k, A, ldA, &beta( 0,j ), ldB, &gamma( 0,j ), ldC );
  }
}

void LoopOne( int m, int n, int k, float *A, int ldA,
	      float *B, int ldB, float *C, int ldC )
{ int i;
  for ( i=0; i<m; i+=MR ) {
    int ib = min( MR, m-i );
    Gemm_4x4Kernel( k, &alpha( i, 0 ), ldA, B, ldB, &gamma( i,0 ), ldC );
  }
}

void Gemm_4x4Kernel( int k, float *A, int ldA, float *B, int ldB, float *C, int ldC ){
    int p;
    for ( p=0; p<k; p++ ){
        gamma(0,0) += alpha( 0,p ) * beta( p, 0) ;
        gamma(1,0) += alpha( 1,p ) * beta( p, 0) ;
        gamma(2,0) += alpha( 2,p ) * beta( p, 0) ;
        gamma(3,0) += alpha( 3,p ) * beta( p, 0) ;

        gamma(0,1) += alpha( 0,p ) * beta( p, 1) ;
        gamma(1,1) += alpha( 1,p ) * beta( p, 1) ;
        gamma(2,1) += alpha( 2,p ) * beta( p, 1) ;
        gamma(3,1) += alpha( 3,p ) * beta( p, 1) ;

        gamma(0,2) += alpha( 0,p ) * beta( p, 2) ;
        gamma(1,2) += alpha( 1,p ) * beta( p, 2) ;
        gamma(2,2) += alpha( 2,p ) * beta( p, 2) ;
        gamma(3,2) += alpha( 3,p ) * beta( p, 2) ;

        gamma(0,3) += alpha( 0,p ) * beta( p, 3) ;
        gamma(1,3) += alpha( 1,p ) * beta( p, 3) ;
        gamma(2,3) += alpha( 2,p ) * beta( p, 3) ;
        gamma(3,3) += alpha( 3,p ) * beta( p, 3) ;
    }
}

void polydl_lib_matmul_f32_i_8_j_16_k_1_fma( long long int M, long long int N, long long int K,long long int A_stride, long long int B_stride, long long int C_stride, float *A, float *B, float *C) {
    
    LoopFive( M, N, K, A, A_stride, B, B_stride, C, C_stride );
} 
