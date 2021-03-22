#!/bin/bash

set +x
M=$1
N=$2
K=$3
PARALLEL_LOOP=$4


ITERS=100
THREADS=28

Step_M=32
Step_N=64
Step_K=32

echo M: $M N: $N K: $K

for (( Outer_Mj =32; Outer_Mj<=128; Outer_Mj=Outer_Mj*2))
do  
for (( Outer_Nj =32; Outer_Nj<=128; Outer_Nj=Outer_Nj*2))
do  
for (( Outer_Kj =32; Outer_Kj<=128; Outer_Kj=Outer_Kj*2))
do  


for (( Outer_Mi =32; Outer_Mi<=${Outer_Mj}; Outer_Mi=Outer_Mi*2))
do 
for (( Outer_Ni =32; Outer_Ni<=${Outer_Nj}; Outer_Ni=Outer_Ni*2))
do  
for (( Outer_Ki =32; Outer_Ki<=${Outer_Kj}; Outer_Ki=Outer_Ki*2))
do  
 


for (( Step_M_i=1; Step_M_i<=$Step_M; Step_M_i=Step_M_i*2))
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


sh run_matmul.sh ${ITERS} $M $N $K ${Outer_Mj} ${Outer_Nj} ${Outer_Kj} ${Outer_Mi} ${Outer_Ni} ${Outer_Ki} ${Step_M_i} ${Step_N_j} ${Step_K_k} ${PARALLEL_LOOP} ${THREADS} 

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
