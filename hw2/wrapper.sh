#! /bin/bash

mkdir -p nsys_reports

# Output to ./nsys_reports/rank_$N.nsys-rep
nsys profile \
    -o "./nsys_reports/r.nsys-rep" \
    --trace mpi,ucx,osrt,nvtx \
    $@
