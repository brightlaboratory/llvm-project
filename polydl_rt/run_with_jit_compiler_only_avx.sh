#!/bin/bash

export OMP_NUM_THREADS=1
set +x
M=64
N=64
K=64

OUTPUT=perf_${M}_${N}_${K}.csv

Outer_Mj=$1
Outer_Nj=$2
Outer_Kj=$3
Outer_Mi=$4
Outer_Ni=$5
Outer_Ki=$6

Step_M=$7
Step_N=$8
Step_K=$9

python microkernel_codeGenerator.py ${Step_M} ${Step_N} ${Step_K} 0 0 0
sh delete_bins.sh 
sh create_lib.sh &>temp
sh compile_main.sh &>temp
./a.out $M $N $K 1000 &> run_output
GFLOPS=`cat run_output | grep GFLOPS | cut -d" " -f 1`
ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`
echo "GFLOPS="${GFLOPS}
echo  "${Step_M}_${Step_N}_${Step_K},${GFLOPS},${ERROR}" >> ${OUTPUT}
# exit
