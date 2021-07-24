import sys

output = "output.c"

# New File

def reset():
    f = open(output, "w")
    f.write('')
    f.close()

# Open / Close function

def open_function(step_M,step_N,step_K):
    f = open(output, "a+") 
    # f.write('void polydl_lib_matmul_f32_i_%d_j_%d_k_%d_fma( long long int M, long long int N, long long int K,long long int A_stride, long long int B_stride, long long int C_stride, float *A, float *B, float *C) {\n' %(step_M,step_N,step_K))
    f.write('void polydl_lib_matmul_f32_fma( long long int M, long long int N, long long int K,long long int A_stride, long long int B_stride, long long int C_stride, float *A, float *B, float *C) {\n')
    f.close()
    
def close_function():
    f = open(output, "a+") 
    f.write('} \n')
    f.close()    

# Declarations
def declaration(step_M,step_N,step_K):
    f = open(output, "a+")
    
    #  Declarations for step_M variable
    for variable in ['A','B','acc']:
        f.write('__m512 ')
        for i in range (0,int(step_K/16)):
            f.write('vec_' + variable +str(i))
            if( i != int(step_K/16) - 1) :
                f.write(', ')
        f.write(';\n')
    
    
    # Others
    f.write('int i, j, k;\n')
    f.write('int i_aux, j_aux;\n')
    
    f.write('long long M_full, N_full, K_full;\n')
    
    f.write('M_full = (M / %d) * %d ;\n' %(step_M,step_M))
    f.write('N_full = (N / %d) * %d ;\n' %(step_N,step_N))
    f.write('K_full = (K / %d) * %d ;\n' %(step_K,step_K))
    
    f.close()



def loopClosingbracket():
    f = open(output, "a+")
    f.write('}\n')
    f.close()

def loopOver(step_M,step_N,step_K):
    f = open(output, "a+")
    
    # Tiled Loops
    M_Conditional = 'M_full'
    N_Conditional = 'N_full'
    K_Conditional = 'K_full'
    i_start = '0'
    j_start = '0'
    k_start = '0'
        
    # i Loop
    f.write('for (i = %s; i < %s; i += %d) {\n' % (i_start,M_Conditional,step_M))
        
    # j Loop
    f.write('for (j = %s; j < %s; j += %d) {\n' % (j_start,N_Conditional, step_N))    
    
    for i in range(0, int(step_K/16)):
        f.write('vec_acc%d = _mm512_setzero_ps();\n' % (i))


    # k Loop
    f.write('for (k = %s; k < %s; k += %d) {\n' % (k_start,K_Conditional,step_K))
    
    # step_k for A and B
    for i in range(0, int(step_K/16)):
        f.write('vec_A%d = _mm512_load_ps((__m512*)&A[i*A_stride + (k+%d)]);\n'%(i,i*16))
        f.write('vec_B%d = _mm512_load_ps((__m512*)&B[j*B_stride + (k+%d)]);\n'%(i,i*16))
        f.write('vec_acc%d = _mm512_fmadd_ps(vec_A%d, vec_B%d,vec_acc%d);\n'%(i,i,i,i))
    
    f.close()
    
    # k Loop
    loopClosingbracket()
    # C[i*C_stride + j] += _mm512_reduce_add_ps (zero_vec);
    
    f = open(output, "a+")
    for i in range(0, int(step_K/16)):
        f.write('C[i*C_stride + j] += _mm512_reduce_add_ps (vec_acc%d);\n'%(i))

    f.close()
    
    # j Loop
    loopClosingbracket()

    
    # i Loop
    loopClosingbracket()


# Main Program 

if len(sys.argv):
    step_M=int(sys.argv[1])
    step_N=int(sys.argv[2])
    step_K=int(sys.argv[3])
else:
    step_M=1
    step_N=1
    step_K=16

reset()


open_function(step_M,step_N,step_K)

declaration(step_M,step_N,step_K)

loopOver(step_M,step_N,step_K)


close_function()