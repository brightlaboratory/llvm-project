FILE=polydl_rt
gcc  -O3 -c -fPIC ${FILE}.c -o ${FILE}.o
gcc ${FILE}.o -shared -o lib${FILE}.so

