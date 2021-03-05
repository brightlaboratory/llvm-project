Install the Intel oneDNN library using the script: oneDNN/buildoneDNN.sh
sh create_lib.sh

Set the number of threads to use to 1 because the oneDNN library will be used for microkernel implementations. We will have parallelism in the MLIR code and we do not want to use oneDNN's parallelization. Instead we would like the microkernels to be run on individual cores.

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=${PWD}/oneDNN/oneDNN/install/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PWD}/../polydl_rt:$LD_LIBRARY_PATH

In the build directory:

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_PREFIX_PATH=../polydl_rt/


cmake --build . --target check-mlir


File Description

1. run_with_jit_compiler.sh
     
     * ``This file Takes the outer and Inner tile sizes with Unrolling factors. Generate the Code variants using JIT compiler and execute via tha explicit data packing file route.``

2. run_with_jit_compiler_only_avx.sh
     
     * ``This file takes only Unroll factor and generate code variant using JIT compiler and execute the file.This file takes only Unroll factor and generate code variant using JIT compiler and execute the file.``

3. run_with_matmul_avx_variants.sh
     
     * ``This file generates various combinations of Outer and Inner tile sizes along with unrolling factors and run the JIT compiler via executing the explicit data packing file route.``

