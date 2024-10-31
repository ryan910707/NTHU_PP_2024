#!/bin/bash

for c in 1 4 8 12 16 20 24; do
    echo "$c thread"
    srun -c$c time ./hw2a ./exp01.png 174170376 -0.7894722222222222 -0.7825277777777778 0.145046875 0.148953125 2549 1439
done

# compile:
# g++ hw2a.cc -o hw2a -pthread -lpng -lm -O3 