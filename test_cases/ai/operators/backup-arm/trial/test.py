"""
duration_time(ns) 
task-clock(ms)
cycle
instructions
cache-references 
cache-misses 
branches 
branch-misses 
L1-dcache-loads 
L1-dcache-load-misses 
LLC-load-misses 
LLC-loads 
IPC
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import mxnet as mx
import openpyxl
import pandas as pd
import time
import random
import subprocess
from tqdm import tqdm
from mxnet import nd

# trial version below, I will rewrite variables and functions when develop formal version

def quantum_inspired_optimization(operators):
    optimized_operators = []
    for op in operators:
        # Simulate quantum-inspired optimization by randomly adjusting input shapes
        op['input1'] = nd.random.uniform(shape=(random.randint(500, 1500), random.randint(500, 1500)))
        op['input2'] = nd.random.uniform(shape=(random.randint(500, 1500), random.randint(500, 1500)))
        optimized_operators.append(op)
    return optimized_operators

# Function to run perf and collect detailed metrics
def run_perf(command):
    result = subprocess.run(['perf', 'stat', '-x,', '-o', '/tmp/perf.out', '--'] + command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open('/tmp/perf.out', 'r') as f:
        lines = f.readlines()
    metrics = {}
    for line in lines:
        if ',' in line:
            parts = line.strip().split(',')
            if len(parts) == 3:
                metrics[parts[2]] = float(parts[0])
    return metrics

def measure_single_performance(op):
    command = ['python3', '-c', f'from mxnet import nd; nd.dot({op["input1"].shape}, {op["input2"].shape})']
    metrics = run_perf(command)
    metrics['Operator'] = op['name']
    return metrics

# Main function to run the performance testing framework
def main():
    #NOTE: I will replace it when develop formal version
    operators = [
        {'name': 'matmul_1', 'input1': nd.random.uniform(shape=(1000, 1000)), 'input2': nd.random.uniform(shape=(1000, 1000))},
    ]

    # Apply quantum-inspired optimization (simulated)
    optimized_operators = quantum_inspired_optimization(operators)

    # Measure performance of the optimized operators with detailed metrics in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(measure_single_performance, op): op for op in optimized_operators}
        for future in as_completed(futures):
            results.append(future.result())

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Write results to an Excel file
    df.to_excel('performance_results.xlsx', index=False)

"""
___________________________________________________________
"""
# simple sample for instrument and perf every single operators
def profile_with_perf(operation, *args, **kwargs):
    # Prepare the command to run perf stat
    cmd = f"perf stat -o perf_output.log {operation.__name__} {args} {kwargs}"
    
    # Run the command
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Process the output if needed
    with open('perf_output.log', 'r') as f:
        perf_output = f.read()
    
    return perf_output

# Example operators
def add_op(a, b):
    return nd.add(a, b)

def mul_op(a, b):
    return nd.multiply(a, b)

if __name__ == "__main__":
    main()
