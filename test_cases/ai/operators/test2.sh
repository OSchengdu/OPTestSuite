#!/bin/bash

output_file="perf_data.log"
echo "Timestamp: $(date)" > "$output_file"

# Function to run the performance test
run_perf_test() {
    local file="$1"
    timeout 900s perf stat -o "${file}_perf.log" -e duration_time,task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses,LLC-loads python "$file" > "${file}_output.log" 2> "${file}_error.log"

    if [ $? -eq 0 ]; then
        echo "$file completed successfully."
    else
        echo "$file encountered an error or timed out. Check ${file}_error.log for details."
    fi

    # Debugging: Print the contents of the perf.log file
    echo "Contents of ${file}_perf.log:"
    cat "${file}_perf.log"

    duration_time=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '1p')
    task_clock=$(grep -oP '^\s*\K\d+\.\d+' "${file}_perf.log" | sed -n '2p')
    cycles=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '3p')
    instructions=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '4p')
    cache_references=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '5p')
    cache_misses=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '6p')
    branches=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '7p')
    branch_misses=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '8p')
    L1_dcache_loads=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '9p')
    L1_dcache_load_misses=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '10p')
    LLC_load_misses=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '11p')
    LLC_loads=$(grep -oP '^\s*\K\d+' "${file}_perf.log" | sed -n '12p')

    # Debugging: Print the extracted values
    echo "$duration_time"
    echo "$task_clock"
    echo "$cycles"
    echo "$instructions"
    echo "$cache_references"
    echo "$cache_misses"
    echo "$branches"
    echo "$branch_misses"
    echo "$L1_dcache_loads"
    echo "$L1_dcache_load_misses"
    echo "$LLC_load_misses"
    echo "$LLC_loads"

    echo "$duration_time" >> "$output_file"
    echo "$task_clock" >> "$output_file"
    echo "$cycles" >> "$output_file"
    echo "$instructions" >> "$output_file"
    echo "$cache_references" >> "$output_file"
    echo "$cache_misses" >> "$output_file"
    echo "$branches" >> "$output_file"
    echo "$branch_misses" >> "$output_file"
    echo "$L1_dcache_loads" >> "$output_file"
    echo "$L1_dcache_load_misses" >> "$output_file"
    echo "$LLC_load_misses" >> "$output_file"
    echo "$LLC_loads" >> "$output_file"

    if [ -n "$cycles" ] && [ -n "$instructions" ] && [ "$cycles" -ne 0 ]; then
        ipc=$(echo "scale=2; $instructions / $cycles" | bc)
        echo "$ipc" >> "$output_file"
    fi
}

export -f run_perf_test

# Function to clean up background processes
cleanup() {
    echo "Cleaning up background processes..."
    pkill -P $$  # Kill all child processes of the current script
}

# Set up the trap to call the cleanup function on script exit or interrupt
trap cleanup EXIT
trap cleanup SIGINT

# Use xargs for parallel processing
ls *.py | xargs -I {} -P 4 bash -c 'run_perf_test "$@"' _ {}

wait  # Wait for all background jobs to finish

rm *.json
echo "All data has been saved to $output_file"
