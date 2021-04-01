#!/bin/bash

set +x
M=$1
N=$2
K=$3
PARALLEL_LOOP=$4


ITERS=10
THREADS=1

Step_M=8
Step_N=32
Step_K=8

echo M: $M N: $N K: $K

for (( Outer_Mj =32; Outer_Mj<=1024; Outer_Mj=Outer_Mj*4))
do  
for (( Outer_Nj =32; Outer_Nj<=1024; Outer_Nj=Outer_Nj*4))
do  
for (( Outer_Kj =32; Outer_Kj<=1024; Outer_Kj=Outer_Kj*4))
do  

   if [ ${Outer_Mj} -le ${M} -a ${Outer_Nj} -le ${N} -a ${Outer_Kj} -le ${K} ]
   then


for (( Outer_Mi =32; Outer_Mi<=64; Outer_Mi=Outer_Mi*2))
do 
for (( Outer_Ni =32; Outer_Ni<=64; Outer_Ni=Outer_Ni*2))
do  
for (( Outer_Ki =32; Outer_Ki<=64; Outer_Ki=Outer_Ki*2))
do  
 
   if [ ${Outer_Mi} -le ${Outer_Mj} -a ${Outer_Ni} -le ${Outer_Nj} -a ${Outer_Ki} -le ${Outer_Kj} ]
   then


for (( Step_M_i=1; Step_M_i<=$Step_M; Step_M_i=Step_M_i*4))
do  
   for (( Step_N_j=16; Step_N_j<=$Step_N; Step_N_j=Step_N_j*2))
   do  
      for (( Step_K_k=1; Step_K_k<=$Step_K; Step_K_k=Step_K_k*4))
      do  

if [ ${Step_M_i} -le ${Outer_Mi} -a ${Step_N_j} -le ${Outer_Ni} -a ${Step_K_k} -le ${Outer_Ki} ] 
then

sh run_matmul.sh ${ITERS} $M $N $K ${Outer_Mj} ${Outer_Nj} ${Outer_Kj} ${Outer_Mi} ${Outer_Ni} ${Outer_Ki} ${Step_M_i} ${Step_N_j} ${Step_K_k} ${PARALLEL_LOOP} ${THREADS} 

fi
        	done
        done
done

fi

done
done
done

fi

done
done
done
