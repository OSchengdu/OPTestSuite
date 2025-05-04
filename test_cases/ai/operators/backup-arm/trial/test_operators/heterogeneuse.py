import subprocess

def check_gpu_support():
    try:
        result = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        return "NVIDIA GPU detected and supported."
    except subprocess.CalledProcessError:
        return "No NVIDIA GPU detected or supported."

def check_npu_support(npu_type):
    if npu_type == 'MLU':
        try:
            result = subprocess.check_output(['cnmon'], stderr=subprocess.STDOUT)
            return "Cambricon MLU detected and supported."
        except subprocess.CalledProcessError:
            return "No Cambricon MLU detected or supported."
    elif npu_type == 'TPU':
        try:
            result = subprocess.check_output(['tpu-smi'], stderr=subprocess.STDOUT)
            return "Google TPU detected and supported."
        except subprocess.CalledProcessError:
            return "No Google TPU detected or supported."

def check_simd_support():
    try:
        result = subprocess.check_output(['lscpu'], stderr=subprocess.STDOUT)
        if 'asimd' in result.decode('utf-8'):
            return "ARM NEON supported."
        if 'rvv' in result.decode('utf-8'):
            return "RISC-V Vector (RVV) supported."
        return "No SIMD support detected."
    except subprocess.CalledProcessError:
        return "Error checking SIMD support."

def analyze_heterogeneous_support():
    print("Analyzing heterogeneous computing support...")
    print("GPU Support:", check_gpu_support())
    print("NPU (MLU) Support:", check_npu_support('MLU'))
    print("NPU (TPU) Support:", check_npu_support('TPU'))
    print("SIMD Support:", check_simd_support())

if __name__ == "__main__":
    analyze_heterogeneous_support()
