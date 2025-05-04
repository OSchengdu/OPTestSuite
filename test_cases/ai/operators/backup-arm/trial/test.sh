#pp.pyf bin/sh
if [ $# -ne 2 ]; then
    echo "Usage: ./performance_counter.sh parameter1 parameter2"
    exit 1
fi

echo "parameter1=$1"

# Extract base command name and remove spaces
base_command=$(basename "$1")
file_name=$(echo "$base_command" | tr -d ' ')

rec_tmp="$file_name.tmp.txt"
perf_file_name="$file_name.perf.txt"

# Clean up previous temporary files
rm -f "$rec_tmp" "$perf_file_name"

# Warm up
$1

# Perf stat collection
perf stat --sync -e duration_time,task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses,LLC-loads -r 10 -o "$rec_tmp" $1

# Extract and format metrics
awk '/^(duration_time|task-clock|cycles|instructions|cache-references|cache-misses|branches|branch-misses|L1-dcache-loads|L1-dcache-load-misses|LLC-load-misses|LLC-loads)/ {gsub(/,/, "", $1); print $1}' "$rec_tmp" > "$perf_file_name"

# Calculate IPC
instructions=$(awk '/instructions/ {print $1}' "$perf_file_name")
cycles=$(awk '/cycles/ {print $1}' "$perf_file_name")
IPC=$(echo "scale=3; $instructions / $cycles" | bc)
echo "$IPC" >> "$perf_file_name"

# Move final file to specified directory
mv "$perf_file_name" "$2"

# Clean up temporary file
rm -f "$rec_tmp"

# Error handling function
check_variable() {
    local var_name=$1
    local var_value=${!var_name}

    if [ -z "$var_value" ] || [ "$var_value" -eq 0 ]; then
        echo "Error: Variable '$var_name' is not set or zero"
        exit 1
    fi
}

# Check if file is writable
check_file_writable() {
    if [ ! -w "$1" ]; then
        echo "Error: File '$1' is not writable"
        exit 1
    fi
}

