#! /bin/bash 

set +x
OUTPUT_FILE=perf.csv

llvm_project_path=/homes/gaganiith/work/mlir/mlirNov2020/llvm-project
test_file_path=${llvm_project_path}/polydl_tests

arr_csv=() 
while IFS= read -r line 
do
    arr_csv+=("$line")
done < code_variants.log

echo "Displaying the contents of array mapped from csv file:"
index=0
for record in "${arr_csv[@]}"
do
    echo "Record at index-${index} : $record"
    my_array=($(echo $record | tr "," "\n"))
    echo " can i print : $my_array[0]"

    ${llvm_project_path}/build/bin/mlir-opt --affine-polydl="tile-sizes=${my_array[0]} tile-sizes=${my_array[1]} tile-sizes=${my_array[2]} pMaps=${my_array[3]} pMaps=${my_array[4]} pMaps=${my_array[5]} cacheSizes=32768 cacheSizes=1048576 cacheSizes=40370176"  ${test_file_path}/test6.mlir &> run_IR_output
    ${llvm_project_path}/build/bin/mlir-opt --affine-polydl="tile-sizes=${my_array[0]} tile-sizes=${my_array[1]} tile-sizes=${my_array[2]} pMaps=${my_array[3]} pMaps=${my_array[4]} pMaps=${my_array[5]} cacheSizes=32768 cacheSizes=1048576 cacheSizes=40370176" --convert-linalg-to-affine-loops -affine-gemm-recognizer --lower-affine --convert-scf-to-std ${test_file_path}/test6.mlir | ${llvm_project_path}/build/bin/mlir-opt --convert-std-to-llvm | ${llvm_project_path}/build/bin/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=${llvm_project_path}/build/lib/libmlir_runner_utils.so,${llvm_project_path}/build/lib/libmlir_c_runner_utils.so,${llvm_project_path}/polydl_rt/oneDNN/oneDNN/install/lib64/libmkldnn.so  &> run_output
    sleep 1
    L1_WSS=`cat run_IR_output | grep L1_WSS | cut -d" " -f 2`
    L2_WSS=`cat run_IR_output | grep L2_WSS | cut -d" " -f 2`
    L3_WSS=`cat run_IR_output | grep L3_WSS | cut -d" " -f 2`
    Mem_WSS=`cat run_IR_output | grep Mem_WSS | cut -d" " -f 2`
    GFLOPS=`cat run_output | grep GFLOPS | cut -d" " -f 1`
    # echo ${M2_Tile},${N2_Tile},${K2_Tile},$L1_WSS,$L2_WSS,$L3_WSS,$Mem_WSS,$GFLOPS
    echo ${my_array[0]}_${my_array[1]}_${my_array[2]}_${my_array[3]}_${my_array[4]}_${my_array[5]},$L1_WSS,$L2_WSS,$L3_WSS,$Mem_WSS,$GFLOPS >> ${OUTPUT_FILE}


	((index++))
done