
#!/bin/bash

M=$1
N=$2
K=$3

Step_M=64
Step_N=64
Step_K=64


for (( Step_M_i=1; Step_M_i<=$Step_M; Step_M_i=Step_M_i*2))
do
   if [ ${Step_M_i} -le ${M} ]
   then

   for (( Step_N_j=16; Step_N_j<=$Step_N; Step_N_j=Step_N_j*2))
   do

   if [ ${Step_N_j} -le ${N} ]
   then
   
      for (( Step_K_k=1; Step_K_k<=$Step_K; Step_K_k=Step_K_k*2))
      do

     if [ ${Step_K_k} -le ${K} ]
      then

	sh run_with_jit_compiler_only_avx.sh $M $N $K ${Step_M_i} ${Step_N_j} ${Step_K_k}
     fi
      done
     fi
   done
    fi
done
