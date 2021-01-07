// Command line:
// ../build/bin/mlir-opt --convert-linalg-to-affine-loops --lower-affine -convert-scf-to-std --convert-std-to-llvm conv2d_padded_GEMM.mlir | ../build/bin/mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=../build/lib/libmlir_runner_utils.so,../build/lib/libmlir_c_runner_utils.so,../polydl_rt/oneDNN/oneDNN/install/lib64/libmkldnn.so


#map1 = affine_map<(d0,d1) -> (d0 + 2*d1)>
#map2 = affine_map<(d0,d1,d2) -> ((d0 + 2*d1 - d2))>
#map3 = affine_map<(d0)[d1] -> ((d0 floordiv d1))>
#map4 = affine_map<(d0,d1) -> (d0 + d1)>
#map5 = affine_map<(d0)[d1] -> (d0 * d1)>
#map6 = affine_map<(d0) -> (d0+1)>
#map10 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

func @cnn(%nImg: index,%nIfm: index,%nOfm: index,%ifhp: index,%ifwp: index,%ofhp: index,%ofwp: index,%ifh: index,%ifw: index,%ofh: index,%ofw: index,%pad_h: index,%pad_w: index,%pad_h_in: index,
  %pad_w_in: index,%pad_h_out: index,%pad_w_out: index,%kh: index,%kw: index,%stride_h: index,%stride_w: index,
  %input: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?x?xf32>) {
  %c0 = constant 0 : index
  %c5 = constant 5 : index
  %GEMM_BLOCK = constant 64 : index
  %STRIDE_H = constant 1 : index
  %STRIDE_W = constant 1 : index

  // Applying convolution

  affine.for %img = %c0 to %nImg {

    %C = alloc(%GEMM_BLOCK,%ofw) : memref<?x?xf32>
    %B = alloc(%GEMM_BLOCK,%ofw) : memref<?x?xf32>
    %A = alloc(%GEMM_BLOCK,%GEMM_BLOCK) : memref<?x?xf32>

    %nofm_GEMM = affine.apply #map3 (%nOfm )[%GEMM_BLOCK]
    affine.for %ofm_tile = 0 to %nofm_GEMM {
      %nIfm_GEMM = affine.apply #map3 (%nIfm )[%GEMM_BLOCK]
      affine.for %ifm_tile = 0 to %nIfm_GEMM {
        affine.for %oj = 0 to %ofh {
          %ij =  affine.apply #map5 (%oj )[%STRIDE_H]
        affine.for %kj = 0 to %kh {
          affine.for %ki = 0 to %kw {

            //Pack
            //Packing Code for C
            affine.for %oi = 0 to %ofw  {
                affine.for %ofm = 0 to %GEMM_BLOCK {
                    %store_output = affine.load %output[%img, %ofm_tile, %oj, %oi,%ofm] : memref<?x?x?x?x?xf32>
                    affine.store %store_output, %C[%ofm,%oi]  : memref<?x?xf32>                 
                }
            }

            //Packing Code for B
            affine.for %oi = 0 to %ofw  {
                %ii =  affine.apply #map5 (%oi )[%STRIDE_W]
                %ij_kj =  affine.apply #map4 (%ij ,%kj)
                %ii_ki =  affine.apply #map4 (%ii ,%ki) 
                affine.for %ifm = 0 to %GEMM_BLOCK {
                    %store_input = affine.load %input[%img, %ifm_tile, %ij_kj, %ii_ki,%ifm] : memref<?x?x?x?x?xf32>
                    affine.store %store_input, %B[%ifm,%oi]  : memref<?x?xf32>                 
                }
            }

            //Packing Code for A
            affine.for %ofm = 0 to %GEMM_BLOCK {
                affine.for %ifm = 0 to %GEMM_BLOCK {
                    %store_filter = affine.load %filter[%ofm_tile, %ifm_tile, %kj, %ki,%ifm,%ofm] : memref<?x?x?x?x?x?xf32>
                    affine.store %store_filter, %A[%ofm,%ifm]  : memref<?x?xf32>                 
                }
            }

            //%arg0_subview = std.subview %A[%c0, %c0][%GEMM_BLOCK, %GEMM_BLOCK][1, 1] : memref<?x?xf32> to memref<*xf32>	
            //%arg1_subview = std.subview %B[%c0, %c0][%GEMM_BLOCK, %ofw][1, 1] : memref<?x?xf32> to memref<*xf32>		
            //%arg2_subview = std.subview %C[%c0, %c0][%GEMM_BLOCK, %ofw][1, 1] : memref<?x?xf32> to memref<*xf32>	
            %0 = memref_cast %A : memref<?x?xf32> to memref<*xf32>	
            %1 = memref_cast %B : memref<?x?xf32> to memref<*xf32>	
            %2 = memref_cast %C : memref<?x?xf32> to memref<*xf32>
            //%M =  affine.apply #map4 (%ij ,%kj)
            //%N =  affine.apply #map4 (%ij ,%kj)
            //%K =  affine.apply #map4 (%ij ,%kj)

            call @polydl_matmul_f32( %0 , %1 , %2,%GEMM_BLOCK,%ofw,%GEMM_BLOCK) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index) -> ()

            //%3 = memref_cast %2 :  memref<*xf32> to memref<?x?xf32>
            //%t = affine.load %3[%c5,%c5] : memref<?x?xf32>
            //call @printF32(%t): (f32) -> ()
            // Gemm

            affine.for %oi = 0 to %ofw {
              %ii =  affine.apply #map5 (%oi )[%STRIDE_W]
              affine.for %ofm = 0 to %GEMM_BLOCK {
                affine.for %ifm = 0 to %GEMM_BLOCK {
                    %ij_kj =  affine.apply #map4 (%ij ,%kj)
                    %ii_ki =  affine.apply #map4 (%ii ,%ki) 

//                    %temp_input = affine.load %input[%img, %ifm_tile, %ij_kj, %ii_ki,%ifm] : memref<?x?x?x?x?xf32>
//                    %temp_filter = affine.load %filter[%ofm_tile, %ifm_tile, %kj, %ki,%ifm,%ofm] : memref<?x?x?x?x?x?xf32>
//                    %temp_output = affine.load %output[%img, %ofm_tile, %oj, %oi,%ofm] : memref<?x?x?x?x?xf32>
                    //
//                    %temp_mul = mulf %temp_input, %temp_filter : f32
//                    %temp_add = addf %temp_output, %temp_mul : f32
//
                    ////call @print_f32(%temp_add): (f32) -> ()
//
//                    affine.store %temp_add, %output[%img, %ofm_tile, %oj, %oi,%ofm] : memref<?x?x?x?x?xf32>

//                    %temp_C = affine.load %C[%ofm,%oi] : memref<?x?xf32>
//                    %temp_B = affine.load %B[%ifm,%oi] : memref<?x?xf32>
//                    %temp_A = affine.load %A[%ofm,%ifm] : memref<?x?xf32>
                    //
//                    %temp_mul = mulf %temp_A, %temp_B : f32
//                    %temp_add = addf %temp_C, %temp_mul : f32

//                    affine.store %temp_add, %C[%ofm,%oi] : memref<?x?xf32>

                  }
                }
              }
              

              //Unpack
              affine.for %oi = 0 to %ofw  {
                affine.for %ofm = 0 to %GEMM_BLOCK {
                    %unpack_output = affine.load %C[%ofm,%oi] : memref<?x?xf32>
                    affine.store %unpack_output, %output[%img, %ofm_tile, %oj, %oi,%ofm] : memref<?x?x?x?x?xf32>                 
                }
            }

            }
          }
        }
      }
    }
  }

  return
}

func @printoutput_cnn(%nImg: index,%nIfm: index,%nOfm: index,%ifhp: index,%ifwp: index,%ofhp: index,%ofwp: index,%ifh: index,%ifw: index,%ofh: index,%ofw: index,%pad_h: index,%pad_w: index,%pad_h_in: index,
  %pad_w_in: index,%pad_h_out: index,%pad_w_out: index,%kh: index,%kw: index,%stride_h: index,%stride_w: index,
  %input: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?x?xf32>) {

  %GEMM_BLOCK = constant 64 : index
  %STRIDE_H = constant 1 : index
  %STRIDE_W = constant 1 : index

  // Applying convolution

  affine.for %img = 0 to %nImg {
    %nofm_GEMM = affine.apply #map3 (%nOfm )[%GEMM_BLOCK]
    affine.for %ofm_tile = 0 to %nofm_GEMM {
    affine.for %oj = 0 to %ofh {
        affine.for %oi = 0 to %ofw {
          affine.for %ofm = 0 to %GEMM_BLOCK {

            %temp_output = affine.load %output[%img, %ofm_tile, %oj, %oi,%ofm] : memref<?x?x?x?x?xf32>
          
            call @printF32(%temp_output): (f32) -> ()
            call @printNewline() : () -> ()
          }
        }
      }
    }
  }

  return
}


func @main() {

  //Declaraing constants with 0 and 1
  
  %cf0 = constant 0.00000e+00 : f32
  %cf1 = constant 1.00000e+00 : f32

  //Declaraing index variables

  %ifw = constant 56 : index
  %ifh = constant 56 : index 
  %nIfm = constant 64 : index
  %nOfm = constant 256 : index

  %kw = constant 1 : index
  %kh = constant 1 : index
  %pad_w = constant 0 : index
  %pad_h = constant 0 : index
  %nImg = constant 1 : index

  %stride_w = constant 1 : index
  %stride_h = constant 1 : index
  
  %pad_h_in = constant 0 : index
  %pad_w_in = constant 0 : index
  %pad_h_out = constant 0 : index
  %pad_w_out = constant 0 : index

  %GEMM_BLOCK_MAIN = constant 64: index
  
  %ofh1 = affine.apply #map2 (%ifh ,%pad_h , %kh)
  %ofh2 = affine.apply #map3 (%ofh1)[%stride_h]
  %ofh = affine.apply #map6  (%ofh2)
  %ofw1 = affine.apply #map2 (%ifw ,%pad_w , %kw)
  %ofw2 = affine.apply #map3 (%ofw1)[%stride_w]
  %ofw = affine.apply #map6  (%ofw2)

  %ifhp =  affine.apply #map1 (%ifh ,%pad_h_in)
  %ifwp =  affine.apply #map1 (%ifw ,%pad_w_in)
  %ofhp =  affine.apply #map1 (%ofh ,%pad_h_out)
  %ofwp =  affine.apply #map1 (%ofw ,%pad_w_out)


  // Test code to cast index values to integers and back to index.
  // index2index, index2float , float2index are all invalid operations.

  //%gagan = index_cast %ofw : index to i32

  // Creating Indices for Input, Output and Filter array.
  %Ip2 = affine.apply #map3 (%nIfm)[%GEMM_BLOCK_MAIN]
  %Ip3 = affine.apply #map1 (%ifhp, %pad_h)
  %Ip4 = affine.apply #map1 (%ifwp ,%pad_w)

  %Op2 = affine.apply #map3 (%nOfm)[%GEMM_BLOCK_MAIN]


  //Declaraing input/output images and filter for CNN-Convolution

  %input = alloc(%nImg,%Ip2,%Ip3,%Ip4,%GEMM_BLOCK_MAIN) : memref<?x?x?x?x?xf32>
  %output = alloc(%nImg,%Op2,%ofhp,%ofwp,%GEMM_BLOCK_MAIN) : memref<?x?x?x?x?xf32>
  %filter = alloc(%Op2,%Ip2,%kh,%kw,%GEMM_BLOCK_MAIN,%GEMM_BLOCK_MAIN) : memref<?x?x?x?x?x?xf32>

  // Using Linear algebra Dialect to fill these matrices.

  //linalg.fill(%input, %cf1) : memref<?x?x?x?x?xf32>, f32
  linalg.fill(%output, %cf0) : memref<?x?x?x?x?xf32>, f32
  //linalg.fill(%filter, %cf1) : memref<?x?x?x?x?x?xf32>, f32


  // Instead of filling the whole with one constant we are filling it with the sequence of natural numbers.

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  
  %f1 = alloc(%c1) : memref<?xf32>
  affine.store %cf1, %f1[%c0] : memref<?xf32>

  //Filling Input array
  affine.for %arg0 = %c0 to %nImg {
    affine.for %arg1 = %c0 to %Ip2 {
      affine.for %arg2 = %c0 to %Ip3 {
        affine.for %arg3 = %c0 to %Ip4 {
            %temp_input = affine.load %f1[%c0] : memref<?xf32>
            %temp_input1 = addf %temp_input ,%cf1 : f32
          affine.for %arg4 = %c0 to %GEMM_BLOCK_MAIN {


            store %temp_input, %input[%arg0, %arg1,%arg2, %arg3,%arg4] : memref<?x?x?x?x?xf32>
          }
            affine.store %temp_input1, %f1[%c0] : memref<?xf32>
        }
      }
      affine.store %cf1, %f1[%c0] : memref<?xf32>
    }
  }

 //Filling Filter array.
  affine.store %cf1, %f1[%c0] : memref<?xf32>
  affine.for %arg0 = %c0 to %Op2 {
    affine.for %arg1 = %c0 to %Ip2 {
      affine.for %arg2 = %c0 to %kh {
        affine.for %arg3 = %c0 to %kw {
          affine.for %arg4 = %c0 to %GEMM_BLOCK_MAIN {
            %temp_input = affine.load %f1[%c0] : memref<?xf32>
            %temp_input1 = addf %temp_input ,%cf1 : f32
            affine.for %arg5 = %c0 to %GEMM_BLOCK_MAIN {


              store %temp_input, %filter[%arg0, %arg1,%arg2, %arg3,%arg4,%arg5] : memref<?x?x?x?x?x?xf32>
            }
              affine.store %temp_input1, %f1[%c0] : memref<?xf32>
          }
        }
        affine.store %cf1, %f1[%c0] : memref<?xf32>
      }
    }
  }




  // Applying Convolution Function.
  call @cnn(%nImg,%nIfm,%nOfm,%ifhp,%ifwp,%ofhp,%ofwp,%ifh,%ifw,%ofh,%ofw,%pad_h,%pad_w,%pad_h_in,%pad_w_in,%pad_h_out,%pad_w_out,%kh,%kw,%stride_h,%stride_w,
  %input, %output, %filter) : (index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,
  memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?x?xf32>) -> ()

  // Printing Convolution Results.
  call @printoutput_cnn(%nImg,%nIfm,%nOfm,%ifhp,%ifwp,%ofhp,%ofwp,%ifh,%ifw,%ofh,%ofw,%pad_h,%pad_w,%pad_h_in,%pad_w_in,%pad_h_out,%pad_w_out,%kh,%kw,%stride_h,%stride_w,
  %input, %output, %filter) : (index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,index,
  memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?x?xf32>) -> ()

  return
}

func @printNewline() 
func @printF32(f32)
func @polydl_matmul_f32(memref<*xf32>, memref<*xf32>, memref<*xf32>, index, index, index)
