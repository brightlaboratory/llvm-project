#!/bin/bash

export OMP_NUM_THREADS=1
set +x
M=$1
N=$2
K=$3

OUTPUT=perf_${M}_${N}_${K}.csv

Step_M=$4
Step_N=$5
Step_K=$6

python ../../microkernel_codeGenerator.py ${Step_M} ${Step_N} ${Step_K} 0 0 0
sh delete_bins.sh 
sh create_lib.sh &>temp
sh compile_main.sh &>temp
./a.out $M $N $K 1000 &> run_output
GFLOPS=`cat run_output | grep GFLOPS | cut -d" " -f 1`
ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`
RELERROR=`cat run_output | grep "inf-norm of comp. rel. error" | cut -d: -f 2`
echo "GFLOPS="${GFLOPS}
echo  "${Step_M}_${Step_N}_${Step_K},${GFLOPS},${ERROR},${RELERROR}" >> ${OUTPUT}
# exit
