FILE=../../polydl_rt
CC=icc
FLAGS="-DUSE_AVX512 -qopenmp"
#CC=gcc
#FLAGS=-fopenmp
set -x
$CC -O3 $FLAGS -c -fPIC ${FILE}.c -o ${FILE}.o 
$CC ${FILE}.o -shared -o libpolydl_rt.so

