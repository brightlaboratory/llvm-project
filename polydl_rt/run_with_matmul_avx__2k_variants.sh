#!/bin/bash

set +x
M=$1
N=$2
K=$3

Step_M=4
Step_N=32
Step_K=1

OUTPUT_FILE=perf_$1_$2_$3.csv

export OMP_NUM_THREADS=1

echo M: $M N: $N K: $K

for (( Outer_Mj = (($M<128 ? $M : 128)); Outer_Mj<=(($M<256 ? $M : 256)); Outer_Mj=Outer_Mj*2))
do  
for (( Outer_Nj =(($N<128 ? $N : 128)); Outer_Nj<=(($N<256 ? $N : 256)); Outer_Nj=Outer_Nj*2))
do  
for (( Outer_Kj =(($K<128 ? $K : 128)); Outer_Kj<=(($K<256 ? $K : 256)); Outer_Kj=Outer_Kj*2))
do  


for (( Outer_Mi =(($Outer_Mj<32 ? $Outer_Mj : 32)); Outer_Mi<=(($Outer_Mj<64 ? $Outer_Mj : 64)); Outer_Mi=Outer_Mi*2))
do 
for (( Outer_Ni =(($Outer_Nj<32 ? $Outer_Nj : 32)); Outer_Ni<=(($Outer_Nj<64 ? $Outer_Nj : 64)); Outer_Ni=Outer_Ni*2))
do  
for (( Outer_Ki =(($Outer_Kj<32 ? $Outer_Kj : 32)); Outer_Ki<=(($Outer_Kj<64 ? $Outer_Kj : 64)); Outer_Ki=Outer_Ki*2))
do  
 


for (( Step_M_i=4; Step_M_i<=$Step_M; Step_M_i=Step_M_i*2))
do  
   if [ `expr $Step_M % ${Step_M_i}` -eq 0 ]
   then

   for (( Step_N_j=32; Step_N_j<=$Step_N; Step_N_j=Step_N_j*2))
   do  
      if [ `expr $Step_N % ${Step_N_j}` -eq 0 ]
      then

      for (( Step_K_k=1; Step_K_k<=$Step_K; Step_K_k=Step_K_k*2))
      do  
         if [ `expr $Step_K % ${Step_K_k}` -eq 0 ]
         then




python microkernel_codeGenerator.py ${Step_M_i} ${Step_N_j} ${Step_K_k} 0 0 0
make clean
make version_file=versions/matmul_explicit_data_packing.c MACROFLAGS="-Djit_variant -DM1=${M} -DN1=${N} -DK1=${K} -DNUM_ITERS=100 -DM2_Tile=${Outer_Mj} -DN2_Tile=${Outer_Nj} -DK2_Tile=${Outer_Kj} -DM1_Tile=${Outer_Mi} -DN1_Tile=${Outer_Ni} -DK1_Tile=${Outer_Ki}"
./matmul &> run_output
GFLOPS=`cat run_output | grep GFLOPS | cut -d"=" -f 2`
ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`
RELERROR=`cat run_output | grep "inf-norm of comp. rel. error" | cut -d: -f 2`
echo ${Outer_Mj},${Outer_Nj},${Outer_Kj},${Outer_Mi},${Outer_Ni},${Outer_Ki},${Step_M_i},${Step_N_j},${Step_K_k},${GFLOPS}
echo ${Outer_Mj},${Outer_Nj},${Outer_Kj},${Outer_Mi},${Outer_Ni},${Outer_Ki},${Step_M_i},${Step_N_j},${Step_K_k},${ERROR},${RELERROR},${GFLOPS} >> ${OUTPUT_FILE}
# exit
          	 fi
        	done
          fi
        done
   fi
done



done
done
done
done
done
done
