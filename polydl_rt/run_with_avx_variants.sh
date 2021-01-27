#!/bin/bash

set +x
M=1024
N=1024
K=1024

Step_M=64
Step_N=128
Step_K=1

OUTPUT_FILE=perf.csv

echo M: $M N: $N K: $K

for (( Step_M_i=4; Step_M_i<=$Step_M; Step_M_i=Step_M_i*2))
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

for (( M2_Tile=16; M2_Tile<=$M; M2_Tile=M2_Tile*2 ))
do  
   if [[ `expr $M % ${M2_Tile}` -eq 0 ]] && [[ `expr $M2_Tile % ${Step_M_i}` -eq 0 ]] 
   then

        for (( N2_Tile=16; N2_Tile<=$N; N2_Tile=N2_Tile*2 ))
        do
          if [[ `expr $N % ${N2_Tile}` -eq 0 ]] && [[ `expr $N2_Tile % ${Step_N_j}` -eq 0 ]]
          then

        	for (( K2_Tile=16; K2_Tile<=$K; K2_Tile=K2_Tile*2 ))
        	do
          	 if [[ `expr $K % ${K2_Tile}` -eq 0 ]] && [[ `expr $K2_Tile % ${Step_K_k}` -eq 0 ]]
         	 then

python microkernel_codeGenerator.py ${Step_M_i} ${Step_N_j} ${Step_K_k} ${M2_Tile} ${N2_Tile} ${K2_Tile}
sh all_steps.sh &> run_output
GFLOPS=`cat run_output | grep GFLOPS | cut -d" " -f 1`
echo ${Step_M_i},${Step_N_j},${Step_K_k},${M2_Tile},${N2_Tile},${K2_Tile},${GFLOPS}
echo ${Step_M_i},${Step_N_j},${Step_K_k},${M2_Tile},${N2_Tile},${K2_Tile},${GFLOPS} >> ${OUTPUT_FILE}
# exit
          	 fi
        	done
          fi
        done
   fi
done

          	 fi
        	done
          fi
        done
   fi
done