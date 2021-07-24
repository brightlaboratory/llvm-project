#!/bin/bash

# export OMP_NUM_THREADS=1

set +x
M=32
N=4096
K=32

Step_M=16
Step_N=32
Step_K=16

Step_M_i=4
Step_N_j=32
Step_K_k=1

OUTPUT_FILE=perf_exhaust.csv

echo M: $M N: $N K: $K



for (( M2_Tile=4; M2_Tile<=$M; M2_Tile=M2_Tile*2 ))
do  
   if [[ `expr $M % ${M2_Tile}` -eq 0 ]] && [[ `expr $M2_Tile % ${Step_M_i}` -eq 0 ]] 
   then

        for (( N2_Tile=64; N2_Tile<=$N; N2_Tile=N2_Tile*2 ))
        do
          if [[ `expr $N % ${N2_Tile}` -eq 0 ]] && [[ `expr $N2_Tile % ${Step_N_j}` -eq 0 ]]
          then

        	for (( K2_Tile=32; K2_Tile<=$K; K2_Tile=K2_Tile*2 ))
        	do
          	 if [[ `expr $K % ${K2_Tile}` -eq 0 ]] && [[ `expr $K2_Tile % ${Step_K_k}` -eq 0 ]]
         	 then


for (( M1_Tile=4; M1_Tile<=$M2_Tile; M1_Tile=M1_Tile*2 ))
do  
   if [[ `expr $M % ${M2_Tile}` -eq 0 ]] && [[ `expr $M2_Tile % ${M1_Tile}` -eq 0 ]] 
   then

        for (( N1_Tile=16; N1_Tile<=$N2_Tile; N1_Tile=N1_Tile*2 ))
        do
          if [[ `expr $N % ${N2_Tile}` -eq 0 ]] && [[ `expr $N2_Tile % ${N1_Tile}` -eq 0 ]]
          then

        	for (( K1_Tile=16; K1_Tile<=$K2_Tile; K1_Tile=K1_Tile*2 ))
        	do
          	 if [[ `expr $K % ${K2_Tile}` -eq 0 ]] && [[ `expr $K2_Tile % ${K1_Tile}` -eq 0 ]]
         	 then




    # for (( Step_M_i=1; Step_M_i<=$Step_M; Step_M_i=Step_M_i*2))
    # do  
    # if [ `expr $M1_Tile % ${Step_M_i}` -eq 0 ]
    # then

    # for (( Step_N_j=16; Step_N_j<=$Step_N; Step_N_j=Step_N_j*2))
    # do  
    # if [ `expr $N1_Tile % ${Step_N_j}` -eq 0 ]
    # then

    # for (( Step_K_k=1; Step_K_k<=$Step_K; Step_K_k=Step_K_k*2))
    # do  
    # if [ `expr $K1_Tile % ${Step_K_k}` -eq 0 ]
    # then


echo ${M2_Tile},${N2_Tile},${K2_Tile},${M1_Tile},${N1_Tile},${K1_Tile},${Step_M_i},${Step_N_j},${Step_K_k},
sh run_with_jit_compiler.sh ${M2_Tile} ${N2_Tile} ${K2_Tile} ${M1_Tile} ${N1_Tile} ${K1_Tile} ${Step_M_i} ${Step_N_j} ${Step_K_k}

# exit

#           	 fi
#         	done
#           fi
#         done
#    fi
# done


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