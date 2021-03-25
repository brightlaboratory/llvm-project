
set -x
export KMP_AFFINITY=granularity=fine,compact,1,28


OUT=baseline_perf.csv
PERF_DIR=perf_data

mkdir ${PERF_DIR}

iters=$1
M1=$2
N1=$3
K1=$4
NUM_THREADS=${5}

config=${iters}_${M1}_${N1}_${K1}__${NUM_THREADS}
echo config: $config

#rm ${CONFIG_OUT}
#rm ${META_CONFIG_OUT}

export OMP_NUM_THREADS=${NUM_THREADS} #FIXME

make clean && make MACROFLAGS="-DM1=$M1 -DN1=$N1 -DK1=$K1 -DNUM_ITERS=$iters"

./matmul &> run_output
GFLOPS=`cat run_output |  grep Real_GFLOPS |  cut -d= -f2`

echo  "${config},${GFLOPS}" >> ${OUT}


