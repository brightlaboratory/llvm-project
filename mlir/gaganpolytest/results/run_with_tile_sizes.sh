#!/bin/bash

set +x
M=2048
N=2048
K=2048
OUTPUT_FILE=perf.csv
llvm_project_path=/home/gagandeep/Desktop/work/mlir_poly/mlirOct2020_gitFolk/llvm-project
test_file_path=${llvm_project_path}/mlir/gaganpolytest

echo M: $M N: $N K: $K

for (( M2_Tile=4; M2_Tile<=$M; M2_Tile=M2_Tile*2 ))
do  
   if [ `expr $M % ${M2_Tile}` -eq 0 ]
   then

        for (( N2_Tile=4; N2_Tile<=$N; N2_Tile=N2_Tile*2 ))
        do
          if [ `expr $N % ${N2_Tile}` -eq 0 ]
          then

        	for (( K2_Tile=4; K2_Tile<=$K; K2_Tile=K2_Tile*2 ))
        	do
          	 if [ `expr $K % ${K2_Tile}` -eq 0 ]
         	 then

${llvm_project_path}/build/bin/mlir-opt --affine-polydl="tile-sizes=${M2_Tile} tile-sizes=${N2_Tile} tile-sizes=${K2_Tile} pMaps=0 pMaps=1 pMaps=2 cacheSizes=32768 cacheSizes=262144 cacheSizes=15728640" --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-std --convert-std-to-llvm  ${test_file_path}/test3.mlir &> run_IR_output
${llvm_project_path}/build/bin/mlir-opt --affine-polydl="tile-sizes=${M2_Tile} tile-sizes=${N2_Tile} tile-sizes=${K2_Tile} pMaps=0 pMaps=1 pMaps=2 cacheSizes=32768 cacheSizes=262144 cacheSizes=15728640" --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-std --convert-std-to-llvm  ${test_file_path}/test3.mlir | ${llvm_project_path}/build/bin/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=${llvm_project_path}/build/lib/libmlir_runner_utils.so,${llvm_project_path}/build/lib/libmlir_c_runner_utils.so  &> run_output
sleep 1
L1_WSS=`cat run_IR_output | grep L1_WSS | cut -d" " -f 2`
L2_WSS=`cat run_IR_output | grep L2_WSS | cut -d" " -f 2`
L3_WSS=`cat run_IR_output | grep L3_WSS | cut -d" " -f 2`
Mem_WSS=`cat run_IR_output | grep Mem_WSS | cut -d" " -f 2`
GFLOPS=`cat run_output | grep GFLOPS | cut -d" " -f 1`
echo ${M2_Tile},${N2_Tile},${K2_Tile},$L1_WSS,$L2_WSS,$L3_WSS,$Mem_WSS,$GFLOPS
echo $L1_WSS,$L2_WSS,$L3_WSS,$Mem_WSS,$GFLOPS >> ${OUTPUT_FILE}
# echo ${M2_Tile},${N2_Tile},${K2_Tile},$GFLOPS >> ${OUTPUT_FILE}
# exit
          	 fi
        	done
          fi
        done
   fi
done