#!/bin/bash

export OMP_NUM_THREADS=1
set +x

Step_M=$1
Step_N=$2
Step_K=$3

M=32
N=32
K=32

M=$4
N=$5
K=$6

OUTPUT=perf_${M}_${N}_${K}.csv


python microkernel_codeGenerator.py ${Step_M} ${Step_N} ${Step_K} 0 0 0
sh delete_bins.sh 
sh create_lib.sh &>temp
sh compile_main.sh &>temp
./a.out $M $N $K 100000 &> run_output
GFLOPS=`cat run_output | grep GFLOPS | cut -d" " -f 1`
ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`
RELERROR=`cat run_output | grep "inf-norm of comp. rel. error" | cut -d: -f 2`
echo "GFLOPS="${GFLOPS}
echo  "${Step_M}_${Step_N}_${Step_K}_${M}_${N}_${K},${GFLOPS},${ERROR},${RELERROR}" >> ${OUTPUT}
# exit
