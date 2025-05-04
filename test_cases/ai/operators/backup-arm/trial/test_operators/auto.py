# commandr = f"perf stat duration_times task-clock cycles instructions cache-references cache-missed branches cache-misses L1-dcache-loads L1-dcache-load-misses LLC-loads python3.11.6 {}.py"
# parse use grep  to get the key word
# get perf.txt
# write = with open txt w, xxx.write(content of txt \n)
#  
import subprocess
from auto import . # waiting for fill this up
import pandas as pd
from tqdm import tqdm

def could_it_run(path):
    c = f"grep import subprocess {path}/*"

def read_first18lines(file, path, n):
    with open(perf.out.txt) as ei:
        for i, line in enumerate(ei):
            if i < n:
                print(line.strip(), "succed")
            else:
                # need to call a sh script to restart this again
                print("sorry {file} test failed, need to retry?")
                continue

def catch_result(file, file, grep_target):
    command1 = f"grep sorry {file} | "
    command2 = f"count succed of perf.out.txt"
    res1 = subprocess(command1)
    res2 = subprocess(command2)
    return res1, res2

def write_in():
    command

# two option, get file, use it in global, or get res_file, use in some specific command
def get_file_name(file):
    res_file = get some file name
    return {res_file}.py

def main():
    path = ["~/trial/MXnet/benchmark/opperf/nd_operations","~/trial/MXnet/benchmark/opperf/custom_operations","~/trial/MXnet/tests", "~/trial/MXnet/example"]
    files = [some file// maybe I should use some more automatical commands or scripts]
    get_file_name()
    catch_result(res_file, path, gre)

    
if __name__ == "__name__":
    main()
