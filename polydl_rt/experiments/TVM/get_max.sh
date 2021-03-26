
for filename in `ls inputs_*_perf.csv`; do
  echo -n $filename,
  cut -d, -f2,2 < $filename | sort -nr | head -1
done
