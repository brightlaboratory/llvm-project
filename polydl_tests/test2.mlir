// export LD_LIBRARY_PATH=${PWD}/../polydl_rt:$LD_LIBRARY_PATH

// ../build/bin/mlir-opt  -convert-linalg-to-loops -affine-gemm-recognizer -lower-affine -convert-scf-to-std  test2.mlir > test2_intermediate.mlir

// ../build/bin/mlir-opt  -convert-std-to-llvm test2_intermediate.mlir | ../build/bin/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=../build/lib/libmlir_runner_utils.so,../build/lib/libmlir_c_runner_utils.so


#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 + 16)>
#map3 = affine_map<(d0) -> (d0 + 64)>
#map4 = affine_map<(d0) -> (d0 + 32)>
#map5 = affine_map<() -> (0)>
#map6 = affine_map<() -> (2048)>


module {
  func @main() {
    %0 = alloc() : memref<2048x2048xf32>
    %1 = alloc() : memref<2048x2048xf32>
    %2 = alloc() : memref<2048x2048xf32>
    %cst = constant 1.000000e+00 : f32
    linalg.fill(%0, %cst) : memref<2048x2048xf32>, f32
    linalg.fill(%1, %cst) : memref<2048x2048xf32>, f32
    linalg.fill(%2, %cst) : memref<2048x2048xf32>, f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %3 = call @rtclock() : () -> f64
    call @sgemm_naive(%0, %1, %2) : (memref<2048x2048xf32>, memref<2048x2048xf32>, memref<2048x2048xf32>) -> ()
    %4 = call @rtclock() : () -> f64
    %5 = subf %4, %3 : f64
    %6 = memref_cast %2 : memref<2048x2048xf32> to memref<*xf32>
    call @print_memref_f32(%6) : (memref<*xf32>) -> ()
    %7 = dim %2, %c0 : memref<2048x2048xf32>
    %8 = dim %2, %c1 : memref<2048x2048xf32>
    %9 = dim %0, %c1 : memref<2048x2048xf32>
    %10 = muli %7, %8 : index
    %11 = muli %10, %9 : index
    %c2 = constant 2 : index
    %12 = muli %c2, %11 : index
    %13 = muli %c1, %12 : index
    %14 = index_cast %13 : index to i64
    %15 = sitofp %14 : i64 to f64
    %16 = divf %15, %5 : f64
    call @print_flops(%16) : (f64) -> ()
    return
  }
  func @sgemm_naive(%arg0: memref<2048x2048xf32>, %arg1: memref<2048x2048xf32>, %arg2: memref<2048x2048xf32>) {
    %c0 = constant 0 : index
    affine.for %arg3 = 0 to 2048 step 32 {
      affine.for %arg4 = 0 to 2048 step 64 {
        affine.for %arg5 = 0 to 2048 step 16 {
          affine.for %arg6 = #map1(%arg3) to #map4(%arg3) {
            affine.for %arg7 = #map1(%arg4) to #map3(%arg4) {
              affine.for %arg8 = #map1(%arg5) to #map2(%arg5) {
                %0 = affine.load %arg0[%arg6, %arg8] : memref<2048x2048xf32>
                %1 = affine.load %arg1[%arg8, %arg7] : memref<2048x2048xf32>
                %2 = affine.load %arg2[%arg6, %arg7] : memref<2048x2048xf32>
                %3 = mulf %0, %1 : f32
                %4 = addf %3, %2 : f32
                affine.store %4, %arg2[%arg6, %arg7] : memref<2048x2048xf32>
              }
            }
          }
        }
      }
    }
    return
  }
  func @print_flops(f64)
  func @rtclock() -> f64
  func @print_memref_f32(memref<*xf32>)
  func @polydl_matmul_f32(memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index)
}
