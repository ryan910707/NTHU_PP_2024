#!/bin/bash

# Define the combinations of TL and devices
combinations=(
    "TL=ud_verbs NET_DEVICE=ibp3s0:1"
    "TL=rc_verbs NET_DEVICE=ibp3s0:1"
    "TL=tcp NET_DEVICE=enp4s0f0"
    "TL=tcp NET_DEVICE=lo"
    "TL=tcp NET_DEVICE=ibp3s0"
    "TL=sysv"
    "TL=posix"
    "TL=cma"
)

# Paths to the osu_latency and osu_bw executables
osu_latency="$HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_latency"
osu_bw="$HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_bw"

# Create a directory for logs
log_dir="ucx_experiment_logs"
mkdir -p $log_dir

# Iterate through the combinations
for combination in "${combinations[@]}"; do
    # Extract the TL and device from the combination string
    TL=$(echo $combination | awk -F' ' '{print $1}' | cut -d= -f2)
    DEVICE=$(echo $combination | awk -F' ' '{print $2}' | cut -d= -f2)

    # Set the UCX environment variables
    export UCX_TLS=$TL
    export UCX_NET_DEVICES=$DEVICE

    # Create log files for each combination
    latency_log="${log_dir}/osu_latency_${TL}_${DEVICE}.log"
    bw_log="${log_dir}/osu_bw_${TL}_${DEVICE}.log"

    echo "Running osu_latency with $TL and $DEVICE..."
    mpiucx -n 2 $osu_latency > $latency_log 2>&1
    echo "osu_latency completed. Results saved to $latency_log"

    echo "Running osu_bw with $TL and $DEVICE..."
    mpiucx -n 2 $osu_bw > $bw_log 2>&1
    echo "osu_bw completed. Results saved to $bw_log"
done

echo "All experiments completed. Logs are in $log_dir."
