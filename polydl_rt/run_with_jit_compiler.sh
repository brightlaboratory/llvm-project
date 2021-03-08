#!/bin/bash

export OMP_NUM_THREADS=1

set +x
M=1024
N=1024
K=1024

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
make clean
make version_file=versions/matmul_explicit_data_packing.c MACROFLAGS="-Djit_variant -DM1=$M -DN1=$N -DK1=$K -DNUM_ITERS=100 -DM2_Tile=${Outer_Mj} -DN2_Tile=${Outer_Nj} -DK2_Tile=${Outer_Kj} -DM1_Tile=${Outer_Mi} -DN1_Tile=${Outer_Ni} -DK1_Tile=${Outer_Ki}"
./matmul &> run_output
GFLOPS=`cat run_output | grep GFLOPS | cut -d"=" -f 2`
ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`
RELERROR=`cat run_output | grep "inf-norm of comp. rel. error" | cut -d: -f 2`
echo "GFLOPS="${GFLOPS}
echo  "${Outer_Mj}_${Outer_Nj}_${Outer_Kj}_${Outer_Mi}_${Outer_Ni}_${Outer_Ki}_${Step_M}_${Step_N}_${Step_K},${GFLOPS},${ERROR},${RELERROR}" >> ${OUTPUT}
# exit
