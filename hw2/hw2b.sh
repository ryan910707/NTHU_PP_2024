#!/bin/bash

for p in {1,2,4,6}; do 
    for c in {1,3,6}; do
        echo "$c thread, $p process"
        IPM_REPORT=full IPM_REPORT_MEM=yes IPM_LOG=full LD_PRELOAD=/opt/ipm/lib/libipm.so srun -N3 -n$p -c$c ./hw2b ./exp.png 174170376 -0.7894722222222 -0.7825277777777 0.1450468 0.1489531 2549 1439
    done
done


# compile script
# mpicxx hw2b.cc -o hw2b -pthread -lpng -lm -O3 -fopenmp