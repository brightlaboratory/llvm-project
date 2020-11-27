FILE=polydl_rt
CC=icc
FLAGS="-DUSE_AVX512 -qopenmp"
#CC=gcc
#FLAGS=-fopenmp
set -x
$CC -O3 -I ./oneDNN/oneDNN/install/include/ $FLAGS -c -fPIC ${FILE}.c -o ${FILE}.o  -lmkldnn
$CC ${FILE}.o -shared -o lib${FILE}.so

