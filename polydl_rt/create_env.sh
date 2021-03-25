cp ../polydl_rt/matmul.c .
cp ../polydl_rt/microkernel_codeGenerator.py .
cp ../polydl_rt/Makefile .
mkdir versions
cp ../polydl_rt/versions/matmul_explicit_data_packing.c versions/
mkdir -p experiments/matmul
cd experiments/matmul/
cp ../../../polydl_rt/experiments/matmul/run_matmul.sh .
cp ../../../polydl_rt/experiments/matmul/run_variants_data_packing_matmul.sh .
cp ../../../polydl_rt/experiments/matmul/submit_matmul_experiments.sh .
cp ../../../polydl_rt/experiments/matmul/run_matmul_experiments.batch .
