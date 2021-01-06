#!/bin/bash

set +x
M=128
N=2048
K=4096

M=$1
N=$2
K=$3

OUTPUT_FILE=perf.csv
OUTPUT_FILE=perf_withlibxsmm_$1_$2_$3.csv

llvm_project_path=/homes/gaganiith/work/mlir/mlirNov2020/llvm-project
test_file_path=${llvm_project_path}/polydl_tests

echo M: $M N: $N K: $K

for (( M2_Tile=16; M2_Tile<=$M; M2_Tile=M2_Tile*2 ))
do  
   if [ `expr $M % ${M2_Tile}` -eq 0 ]
   then

        for (( N2_Tile=16; N2_Tile<=$N; N2_Tile=N2_Tile*2 ))
        do
          if [ `expr $N % ${N2_Tile}` -eq 0 ]
          then

        	for (( K2_Tile=16; K2_Tile<=$K; K2_Tile=K2_Tile*2 ))
        	do
          	 if [ `expr $K % ${K2_Tile}` -eq 0 ]
         	 then

${llvm_project_path}/build/bin/mlir-opt --affine-polydl="tile-sizes=${M2_Tile} tile-sizes=${N2_Tile} tile-sizes=${K2_Tile} pMaps=0 pMaps=1 pMaps=2 cacheSizes=32768 cacheSizes=1048576 cacheSizes=40370176"  ${test_file_path}/test6.mlir &> run_IR_output
${llvm_project_path}/build/bin/mlir-opt --affine-polydl="tile-sizes=${M2_Tile} tile-sizes=${N2_Tile} tile-sizes=${K2_Tile} pMaps=0 pMaps=1 pMaps=2 cacheSizes=32768 cacheSizes=1048576 cacheSizes=40370176" --convert-linalg-to-affine-loops -affine-gemm-recognizer --lower-affine --convert-scf-to-std ${test_file_path}/test6.mlir | ${llvm_project_path}/build/bin/mlir-opt --convert-std-to-llvm | ${llvm_project_path}/build/bin/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=${llvm_project_path}/build/lib/libmlir_runner_utils.so,${llvm_project_path}/build/lib/libmlir_c_runner_utils.so,${llvm_project_path}/polydl_rt/oneDNN/oneDNN/install/lib64/libmkldnn.so  &> run_output
sleep 1
L1_WSS=`cat run_IR_output | grep L1_WSS | cut -d" " -f 2`
L2_WSS=`cat run_IR_output | grep L2_WSS | cut -d" " -f 2`
L3_WSS=`cat run_IR_output | grep L3_WSS | cut -d" " -f 2`
Mem_WSS=`cat run_IR_output | grep Mem_WSS | cut -d" " -f 2`
GFLOPS=`cat run_output | grep GFLOPS | cut -d" " -f 1`
echo ${M2_Tile},${N2_Tile},${K2_Tile},$L1_WSS,$L2_WSS,$L3_WSS,$Mem_WSS,$GFLOPS
echo ${M2_Tile}_${N2_Tile}_${K2_Tile},$L1_WSS,$L2_WSS,$L3_WSS,$Mem_WSS,$GFLOPS >> ${OUTPUT_FILE}
# echo ${M2_Tile},${N2_Tile},${K2_Tile},$GFLOPS >> ${OUTPUT_FILE}
# exit
          	 fi
        	done
          fi
        done
   fi
done