FILE=polydl_rt
#CC=icc
#FLAGS=-DUSE_AVX512
CC=gcc
FLAGS=
$CC -O3 $FLAGS -c -fPIC ${FILE}.c -o ${FILE}.o
$CC ${FILE}.o -shared -o lib${FILE}.so

