
Step_M=32
Step_N=32
Step_K=32

for (( Step_M_i=2; Step_M_i<=$Step_M; Step_M_i=Step_M_i*2))
do  
   if [ `expr $Step_M % ${Step_M_i}` -eq 0 ]
   then

   for (( Step_N_j=16; Step_N_j<=$Step_N; Step_N_j=Step_N_j*2))
   do  
      if [ `expr $Step_N % ${Step_N_j}` -eq 0 ]
      then

      for (( Step_K_k=1; Step_K_k<=$Step_K; Step_K_k=Step_K_k*2))
      do  
         if [ `expr $Step_K % ${Step_K_k}` -eq 0 ]
         then
echo ${Step_M_i},${Step_N_j},${Step_K_k},
sh run_with_jit_compiler_only_avx.sh 32 32 32 32 32 32 ${Step_M_i} ${Step_N_j} ${Step_K_k}

            fi
        done
        fi
    done
fi
done
