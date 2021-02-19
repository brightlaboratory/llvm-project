#define alpha( i,j ) A[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A
#define beta( i,j )  B[ (j)*ldB + (i) ]   // map beta( i,j ) to array B
#define gamma( i,j ) C[ (j)*ldC + (i) ]   // map gamma( i,j ) to array C

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))
#define NC 1024
#define KC 128
#define MC 180
#define MR 4
#define NR 4
void LoopFive( int, int, int, float *, int, float *, int, float *, int );
void LoopFour( int, int, int, float *, int, float *, int,  float *, int );
void LoopThree( int, int, int, float *, int, float *, float *, int );
void LoopTwo( int, int, int, float *, float *, float *, int );
void LoopOne( int, int, int, float *, float *, float *, int );
void Gemm_4x4Kernel_Packed( int, float *, float *, float *, int );
void PackBlockA_MCxKC( int, int, float *, int, float * );
void PackPanelB_KCxNC( int, int, float *, int, float * );

void LoopFive( int m, int n, int k, float *A, int ldA,
		   float *B, int ldB, float *C, int ldC )
{int j;
  for (  j=0; j<n; j+=NC ) {
    int jb = min( NC, n-j );    /* Last loop may not involve a full block */
    LoopFour( m, jb, k, A, ldA, &beta( 0,j ), ldB, &gamma( 0,j ), ldC );
  } 
}

void LoopFour( int m, int n, int k, float *A, int ldA, float *B, int ldB,
	       float *C, int ldC )
{
  float *Btilde = ( float * ) malloc( KC * NC * sizeof( float ) );
  int p;
  for ( p=0; p<k; p+=KC ) {
    int pb = min( KC, k-p );    /* Last loop may not involve a full block */
    PackPanelB_KCxNC( pb, n, &beta( p, 0 ), ldB, Btilde );
    LoopThree( m, n, pb, &alpha( 0, p ), ldA, Btilde, C, ldC );
  }

  free( Btilde); 
}

void LoopThree( int m, int n, int k, float *A, int ldA, float *Btilde, float *C, int ldC )
{
  float *Atilde = ( float * ) malloc( MC * KC * sizeof( float ) );
       int i;
  for ( i=0; i<m; i+=MC ) {
    int ib = min( MC, m-i );    /* Last loop may not involve a full block */
    PackBlockA_MCxKC( ib, k, &alpha( i, 0 ), ldA, Atilde );
    LoopTwo( ib, n, k, Atilde, Btilde, &gamma( i,0 ), ldC );
  }

  free( Atilde);
}

void LoopTwo( int m, int n, int k, float *Atilde, float *Btilde, float *C, int ldC )
{int j;
  for ( j=0; j<n; j+=NR ) {
    int jb = min( NR, n-j );
    LoopOne( m, jb, k, Atilde, &Btilde[ j*k ], &gamma( 0,j ), ldC );
  }
}

void LoopOne( int m, int n, int k, float *Atilde, float *MicroPanelB, float *C, int ldC )
{int i;
  for ( i=0; i<m; i+=MR ) {
    int ib = min( MR, m-i );
    Gemm_4x4Kernel_Packed( k, &Atilde[ i*k ], MicroPanelB, &gamma( i,0 ), ldC );
  }
}

void PackMicroPanelA_MRxKC( int m, int k, float *A, int ldA, float *Atilde )
/* Pack a micro-panel of A into buffer pointed to by Atilde. 
   This is an unoptimized implementation for general MR and KC. */
{
  /* March through A in column-major order, packing into Atilde as we go. */
int p,i;
  if ( m == MR )   /* Full row size micro-panel.*/
    for ( p=0; p<k; p++ ) 
      for (  i=0; i<MR; i++ )
	*Atilde++ = alpha( i, p );
  else /* Not a full row size micro-panel.  We pad with zeroes. */
    for (  p=0; p<k; p++ ) {
      for ( i=0; i<m; i++ )
	*Atilde++ = alpha( i, p );
      for ( i=m; i<MR; i++ )
	*Atilde++ = 0.0;
    }
}

void PackBlockA_MCxKC( int m, int k, float *A, int ldA, float *Atilde )
/* Pack a MC x KC block of A.  MC is assumed to be a multiple of MR.  The block is 
   packed into Atilde a micro-panel at a time. If necessary, the last micro-panel 
   is padded with rows of zeroes. */
{int i;
  for ( i=0; i<m; i+= MR ){
    int ib = min( MR, m-i );
    PackMicroPanelA_MRxKC( ib, k, &alpha( i, 0 ), ldA, Atilde );
    Atilde += ib * k;
  }
}

void PackMicroPanelB_KCxNR( int k, int n, float *B, int ldB, float *Btilde )
/* Pack a micro-panel of B into buffer pointed to by Btilde. 
   This is an unoptimized implementation for general KC and NR. */
{
  /* March through B in row-major order, packing into Btilde as we go. */
int p,j;
  if ( n == NR ) /* Full column width micro-panel.*/
    for ( p=0; p<k; p++ ) 
      for ( j=0; j<NR; j++ )
	*Btilde++ = beta( p, j );
  else /* Not a full row size micro-panel.  We pad with zeroes. */
    for (  p=0; p<k; p++ ) {
      for ( j=0; j<n; j++ )
	*Btilde++ = beta( p, j );
      for ( j=n; j<NR; j++ )
	*Btilde++ = 0.0;
    }
}

void PackPanelB_KCxNC( int k, int n, float *B, int ldB, float *Btilde )
/* Pack a KC x NC panel of B.  NC is assumed to be a multiple of NR.  The block is 
   packed into Btilde a micro-panel at a time. If necessary, the last micro-panel 
   is padded with columns of zeroes. */
{ int j;
  for (  j=0; j<n; j+= NR ){
    int jb = min( NR, n-j );
    
    PackMicroPanelB_KCxNR( k, jb, &beta( 0, j ), ldB, Btilde );
    Btilde += k * jb;
  }
}


void Gemm_4x4Kernel_Packed( int k, float *MicroPanelA, float *MicroPanelB, float *C, int ldC ){
    int p;
    for ( p=0; p<k; p++ ){
        gamma(0,0) += MicroPanelA[0] * MicroPanelB[0] ;
        gamma(1,0) += MicroPanelA[1] * MicroPanelB[0] ;
        gamma(2,0) += MicroPanelA[2] * MicroPanelB[0] ;
        gamma(3,0) += MicroPanelA[3] * MicroPanelB[0] ;

        gamma(0,1) += MicroPanelA[0] * MicroPanelB[1] ;
        gamma(1,1) += MicroPanelA[1] * MicroPanelB[1] ;
        gamma(2,1) += MicroPanelA[2] * MicroPanelB[1] ;
        gamma(3,1) += MicroPanelA[3] * MicroPanelB[1] ;

        gamma(0,2) += MicroPanelA[0] * MicroPanelB[2] ;
        gamma(1,2) += MicroPanelA[1] * MicroPanelB[2] ;
        gamma(2,2) += MicroPanelA[2] * MicroPanelB[2] ;
        gamma(3,2) += MicroPanelA[3] * MicroPanelB[2] ;

        gamma(0,3) += MicroPanelA[0] * MicroPanelB[3] ;
        gamma(1,3) += MicroPanelA[1] * MicroPanelB[3] ;
        gamma(2,3) += MicroPanelA[2] * MicroPanelB[3] ;
        gamma(3,3) += MicroPanelA[3] * MicroPanelB[3] ;

        MicroPanelA += MR;
        MicroPanelB += NR;
    }
}

void polydl_lib_matmul_f32_i_8_j_16_k_1_fma( long long int M, long long int N, long long int K,long long int A_stride, long long int B_stride, long long int C_stride, float *A, float *B, float *C) {
    
    LoopFive( M, N, K, A, A_stride, B, B_stride, C, C_stride );
} 
