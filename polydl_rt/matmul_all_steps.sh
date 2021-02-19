make clean
make version_file=versions/matmul_explicit_data_packing.c MACROFLAGS="-DM1=1024 -DN1=1024 -DK1=1024 -DNUM_ITERS=100 -DM2_Tile=256 -DN2_Tile=256 -DK2_Tile=256 -DM1_Tile=64 -DN1_Tile=64 -DK1_Tile=64"
./matmul