rm a.out
# export OMP_NUM_THREADS=1
# export LD_LIBRARY_PATH=${PWD}/oneDNN/oneDNN/install/lib64/:$LD_LIBRARY_PATH
icc -O3 ../../main.c -L . -lpolydl_rt -lm 
