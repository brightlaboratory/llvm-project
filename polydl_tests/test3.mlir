
// export LD_LIBRARY_PATH=${PWD}/../polydl_rt:$LD_LIBRARY_PATH

// ../build/bin/mlir-opt  -convert-linalg-to-loops -affine-gemm-recognizer -lower-affine -convert-scf-to-std  test1.mlir > test1_intermediate.mlir

// ../build/bin/mlir-opt  -convert-std-to-llvm test1_intermediate.mlir | ../build/bin/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=../build/lib/libmlir_runner_utils.so,../build/lib/libmlir_c_runner_utils.so

func @main() {
  %A = alloc() : memref<2048x2048xf32>
  %B = alloc() : memref<2048x2048xf32>
  %C = alloc() : memref<2048x2048xf32>
  %cf1 = constant 1.00000e+00 : f32

  %ci0 = constant 0 : index
  %ci1 = constant 1 : index
  %ci2 = constant 2 : index

  linalg.fill(%A, %cf1) : memref<2048x2048xf32>, f32
  linalg.fill(%B, %cf1) : memref<2048x2048xf32>, f32
  linalg.fill(%C, %cf1) : memref<2048x2048xf32>, f32

  %pA = memref_cast %A : memref<2048x2048xf32> to memref<*xf32>
  %pB = memref_cast %B : memref<2048x2048xf32> to memref<*xf32>
  %pC = memref_cast %C : memref<2048x2048xf32> to memref<*xf32>
  %reps = constant 1 : index

  %M = dim %C, %ci0 : memref<2048x2048xf32>
  %N = dim %C, %ci1 : memref<2048x2048xf32>
  %K = dim %A, %ci1 : memref<2048x2048xf32>

  %t_start = call @rtclock() : () -> f64
  // affine.for %arg0 = 0 to 1 {
  call @sgemm_naive(%A, %B, %C) : (memref<2048x2048xf32>, memref<2048x2048xf32>, memref<2048x2048xf32>) -> ()
  // call @polydl_matmul_f32(%pA, %pB, %pC, %M, %N, %K) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index) -> ()
 // }
  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64

  // call @print_memref_f32(%pC) : (memref<*xf32>) -> ()
  call @print_memref_f32_polydl(%pC) : (memref<*xf32>) -> ()

  %f1 = muli %M, %N : index
  %f2 = muli %f1, %K : index

  // 2*M*N*K.
  %c2 = constant 2 : index
  %f3 = muli %c2, %f2 : index
  %num_flops = muli %reps, %f3 : index
  %num_flops_i = index_cast %num_flops : index to i64
  %num_flops_f = sitofp %num_flops_i : i64 to f64
  %flops = divf %num_flops_f, %t : f64
  call @print_flops(%flops) : (f64) -> ()

  return
}
// CHECK: 2049, 2049, 2049,

func @sgemm_naive(%arg0: memref<2048x2048xf32>, %arg1: memref<2048x2048xf32>, %arg2: memref<2048x2048xf32>) {
   affine.for %arg3 = 0 to 2048 step 32 {
      affine.for %arg4 = 0 to 2048 step 64 {
        affine.for %arg5 = 0 to 2048 step 16 {
          %ar4 = index_cast %arg3 : index to i32
          %ar5 = sitofp %ar4 : i32 to f32
	call @printF32(%ar5): (f32) -> ()
	call @printComma() : () -> ()
          %ar6 = index_cast %arg4 : index to i32
          %ar7 = sitofp %ar6 : i32 to f32
	call @printF32(%ar7): (f32) -> ()
	call @printComma() : () -> ()
          %ar8 = index_cast %arg5 : index to i32
          %ar9 = sitofp %ar8 : i32 to f32
	call @printF32(%ar9): (f32) -> ()
	call @printComma() : () -> ()
          %M = constant 32 : index
          %N = constant 64 : index
          %K = constant 16 : index
          %stride = constant 2048 : index
          %c1 = constant 1 : index
          %arg0_subview = subview %arg0[%arg3, %arg5][32, 16][1, 1] : memref<2048x2048xf32> to memref<32x16xf32, offset: ?, strides: [2048, 1]>
          %arg1_subview = subview %arg1[%arg5, %arg4][16, 64][1, 1] : memref<2048x2048xf32> to memref<16x64xf32, offset: ?, strides: [2048, 1]>
          %arg2_subview = subview %arg2[%arg3, %arg4][32, 64][1, 1] : memref<2048x2048xf32> to memref<32x64xf32, offset: ?, strides: [2048, 1]>
          %0 = memref_cast %arg0_subview : memref<32x16xf32, offset: ?, strides: [2048, 1]> to memref<*xf32>
          %1 = memref_cast %arg1_subview : memref<16x64xf32, offset: ?, strides: [2048, 1]> to memref<*xf32>
          %2 = memref_cast %arg2_subview : memref<32x64xf32, offset: ?, strides: [2048, 1]> to memref<*xf32>
          call @polydl_matmul_f32( %0 , %1 , %2,%M,%N,%K) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index) -> ()

        }
      }
    }
  return
}

func @printF32(f32)
func @print_flops(f64)
func @rtclock() -> f64
func @print_memref_f32_polydl(memref<*xf32>)
func @print_memref_f32(memref<*xf32>)
func @print_open()
func @print_close() 
func @polydl_matmul_f32(memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index)
func @printComma()
