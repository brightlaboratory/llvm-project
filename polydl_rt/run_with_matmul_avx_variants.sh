#!/bin/bash

set +x
M=1024
N=1024
K=1024

Step_M=64
Step_N=64
Step_K=64

OUTPUT_FILE=perf.csv

export OMP_NUM_THREADS=1

echo M: $M N: $N K: $K

for (( Outer_Mj =128; Outer_Mj<=$M; Outer_Mj=Outer_Mj*2))
do  
for (( Outer_Nj =128; Outer_Nj<=$N; Outer_Nj=Outer_Nj*2))
do  
for (( Outer_Kj =128; Outer_Kj<=$K; Outer_Kj=Outer_Kj*2))
do  


for (( Outer_Mi =64; Outer_Mi<=$Outer_Mj; Outer_Mi=Outer_Mi*2))
do 
for (( Outer_Ni =64; Outer_Ni<=$Outer_Nj; Outer_Ni=Outer_Ni*2))
do  
for (( Outer_Ki =64; Outer_Ki<=$Outer_Kj; Outer_Ki=Outer_Ki*2))
do  
 


for (( Step_M_i=2; Step_M_i<=$Step_M; Step_M_i=Step_M_i*2))
do  
   if [ `expr $Step_M % ${Step_M_i}` -eq 0 ]
   then

   for (( Step_N_j=16; Step_N_j<=$Step_N; Step_N_j=Step_N_j*2))
   do  
      if [ `expr $Step_N % ${Step_N_j}` -eq 0 ]
      then

      for (( Step_K_k=1; Step_K_k<=$Step_K; Step_K_k=Step_K_k*2))
      do  
         if [ `expr $Step_K % ${Step_K_k}` -eq 0 ]
         then




python microkernel_codeGenerator.py ${Step_M_i} ${Step_N_j} ${Step_K_k} 0 0 0
make clean
make version_file=versions/matmul_explicit_data_packing.c MACROFLAGS="-DM1=1024 -DN1=1024 -DK1=1024 -DNUM_ITERS=100 -DM2_Tile=${Outer_Mj} -DN2_Tile=${Outer_Nj} -DK2_Tile=${Outer_Kj} -DM1_Tile=${Outer_Mi} -DN1_Tile=${Outer_Ni} -DK1_Tile=${Outer_Ki}"
./matmul &> run_output
GFLOPS=`cat run_output | grep GFLOPS | cut -d"=" -f 2`
echo ${Outer_Mj},${Outer_Nj},${Outer_Kj},${Outer_Mi},${Outer_Ni},${Outer_Ki},${Step_M_i},${Step_N_j},${Step_K_k},${GFLOPS}
echo ${Outer_Mj},${Outer_Nj},${Outer_Kj},${Outer_Mi},${Outer_Ni},${Outer_Ki},${Step_M_i},${Step_N_j},${Step_K_k},${GFLOPS} >> ${OUTPUT_FILE}
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