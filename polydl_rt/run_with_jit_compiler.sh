#!/bin/bash

export OMP_NUM_THREADS=28
# export KMP_AFFINITY=verbose,granularity=fine,compact,1,0
# export KMP_AFFINITY=granularity=fine,compact,1,0
# export KMP_AFFINITY="verbose,none"
# export KMP_AFFINITY="explicit,proclist=[7],verbose"

# export KMP_AFFINITY=disabled
# export KMP_AFFINITY="explicit,proclist=[18],verbose"
export KMP_AFFINITY=granularity=fine,compact,28,0


set +x
M=2048
N=4096
K=32
# K=256

OUTPUT=new_perf_exhaustive__wo_UF_only_i_${M}_${N}_${K}.csv

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
make version_file=versions/matmul_explicit_data_packing_experiments.c MACROFLAGS=" -DPARALLEL_it2  -Djit_variant -DNO_DATA_PACKING -DM1=$M -DN1=$N -DK1=$K -DNUM_ITERS=300 -DM2_Tile=${Outer_Mj} -DN2_Tile=${Outer_Nj} -DK2_Tile=${Outer_Kj} -DM1_Tile=${Outer_Mi} -DN1_Tile=${Outer_Ni} -DK1_Tile=${Outer_Ki}"
./matmul &> run_output
GFLOPS=`cat run_output | grep GFLOPS | cut -d"=" -f 2`
ERROR=`cat run_output | grep "inf-norm of comp. abs. error" | cut -d: -f 2`
RELERROR=`cat run_output | grep "inf-norm of comp. rel. error" | cut -d: -f 2`
echo "GFLOPS="${GFLOPS}
echo  "${Outer_Mj}_${Outer_Nj}_${Outer_Kj}_${Outer_Mi}_${Outer_Ni}_${Outer_Ki}_${Step_M}_${Step_N}_${Step_K},${GFLOPS},${ERROR},${RELERROR}" >> ${OUTPUT}
# exit
