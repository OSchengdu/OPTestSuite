#!/bin/bash

# Function to check GPU support
check_gpu_support() {
    local gpu_support=()
    if command -v nvidia-smi &> /dev/null; then
        gpu_support+=("NVIDIA GPU")
    fi
    if command -v amdconfig &> /dev/null; then
        gpu_support+=("AMD GPU")
    fi
    if command -v intel_gpu_top &> /dev/null; then
        gpu_support+=("Intel GPU")
    fi
    if command -v rocm-smi &> /dev/null; then
        gpu_support+=("AMD ROCm GPU")
    fi
    if command -v glxinfo &> /dev/null; then
        if glxinfo | grep -q 'NVIDIA'; then
            gpu_support+=("NVIDIA GPU (via glxinfo)")
        fi
        if glxinfo | grep -q 'AMD'; then
            gpu_support+=("AMD GPU (via glxinfo)")
        fi
        if glxinfo | grep -q 'Intel'; then
            gpu_support+=("Intel GPU (via glxinfo)")
        fi
    fi

    if [ ${#gpu_support[@]} -eq 0 ]; then
        echo "No GPU detected or supported."
    else
        echo "Supported GPUs: ${gpu_support[*]}"
    fi
}

# Function to check NPU support
check_npu_support() {
    local npu_type=$1
    local npu_support=()
    if [ "$npu_type" == "MLU" ]; then
        if command -v cnmon &> /dev/null; then
            npu_support+=("Cambricon MLU")
        fi
    elif [ "$npu_type" == "TPU" ]; then
        if command -v tpu-smi &> /dev/null; then
            npu_support+=("Google TPU")
        fi
    elif [ "$npu_type" == "Intel_NPU" ]; then
        if command -v intel_npu_info &> /dev/null; then
            npu_support+=("Intel NPU")
        fi
    elif [ "$npu_type" == "Habana" ]; then
        if command -v habana_smi &> /dev/null; then
            npu_support+=("Habana Gaudi NPU")
        fi
    fi

    if [ ${#npu_support[@]} -eq 0 ]; then
        echo "No $npu_type detected or supported."
    else
        echo "Supported $npu_type: ${npu_support[*]}"
    fi
}

# Function to check SIMD support
check_simd_support() {
    local simd_support=()
    if lscpu | grep -q 'asimd'; then
        simd_support+=("ARM NEON")
    fi
    if lscpu | grep -q 'rvv'; then
        simd_support+=("RISC-V Vector (RVV)")
    fi
    if lscpu | grep -q 'avx'; then
        simd_support+=("Intel AVX")
    fi
    if lscpu | grep -q 'sse'; then
        simd_support+=("Intel SSE")
    fi
    if lscpu | grep -q 'asimd'; then
        simd_support+=("AMD SIMD")
    fi

    if [ ${#simd_support[@]} -eq 0 ]; then
        echo "No SIMD support detected."
    else
        echo "Supported SIMD: ${simd_support[*]}"
    fi
}

# Function to check other hardware accelerators
check_other_accelerators() {
    local other_support=()
    if command -v dmidecode &> /dev/null; then
        if dmidecode -t 39 | grep -q 'Intel'; then
            other_support+=("Intel QuickAssist Technology (QAT)")
        fi
        if dmidecode -t 39 | grep -q 'AMD'; then
            other_support+=("AMD Infinity Fabric")
        fi
    fi

    if [ ${#other_support[@]} -eq 0 ]; then
        echo "No other hardware accelerators detected."
    else
        echo "Supported Other Accelerators: ${other_support[*]}"
    fi
}

# Function to check FPGA support
check_fpga_support() {
    local fpga_support=()
    if command -v fpga-list-cards &> /dev/null; then
        fpga_support+=("FPGA")
    fi

    if [ ${#fpga_support[@]} -eq 0 ]; then
        echo "No FPGA detected or supported."
    else
        echo "Supported FPGA: ${fpga_support[*]}"
    fi
}

# Function to check Intel DL Boost
check_intel_dl_boost() {
    local dl_boost_support=()
    if lscpu | grep -q 'avx512_vnni'; then
        dl_boost_support+=("Intel DL Boost (AVX-512 VNNI)")
    fi

    if [ ${#dl_boost_support[@]} -eq 0 ]; then
        echo "No Intel DL Boost support detected."
    else
        echo "Supported Intel DL Boost: ${dl_boost_support[*]}"
    fi
}

# Analyze heterogeneous computing support
analyze_heterogeneous_support() {
    echo "Analyzing heterogeneous computing support..."
    echo "GPU Support: $(check_gpu_support)"
    echo "NPU (MLU) Support: $(check_npu_support 'MLU')"
    echo "NPU (TPU) Support: $(check_npu_support 'TPU')"
    echo "NPU (Intel) Support: $(check_npu_support 'Intel_NPU')"
    echo "NPU (Habana) Support: $(check_npu_support 'Habana')"
    echo "SIMD Support: $(check_simd_support)"
    echo "Other Accelerators: $(check_other_accelerators)"
    echo "FPGA Support: $(check_fpga_support)"
    echo "Intel DL Boost Support: $(check_intel_dl_boost)"
}

# Main function
main() {
    analyze_heterogeneous_support
}

main
