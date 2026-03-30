#!/bin/bash
set -e

echo "=== Installing MILo (SIGGRAPH Asia 2025) ==="

# Install miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda
    rm /tmp/miniconda.sh
    export PATH="/opt/miniconda/bin:$PATH"
    conda init bash 2>/dev/null || true
    # Accept TOS (required since conda 25.x)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
fi

# Ensure conda is on PATH
export PATH="/opt/miniconda/bin:$PATH"

# Create MILo conda environment
echo "Creating conda environment 'milo' with Python 3.9..."
conda create -y -n milo python=3.9
conda run -n milo conda install -y -c conda-forge cgal gmp cmake

# Install PyTorch 2.3.1 + CUDA 11.8
echo "Installing PyTorch 2.3.1 + CUDA 11.8..."
conda run -n milo pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Clone MILo
echo "Cloning MILo..."
if [ ! -d "/opt/milo" ]; then
    git clone --recursive https://github.com/Anttwo/MILo.git /opt/milo
fi

# Build submodules
echo "Building CUDA extensions..."
cd /opt/milo

# Set CUDA paths for 11.8 (even if system has 12.x)
export CUDA_HOME=/usr/local/cuda-11.8
# If CUDA 11.8 not available, try system CUDA
if [ ! -d "$CUDA_HOME" ]; then
    export CUDA_HOME=/usr/local/cuda
fi

for submod in diff-gaussian-rasterization simple-knn fused-ssim; do
    if [ -d "submodules/$submod" ]; then
        echo "Building $submod..."
        conda run -n milo pip install "submodules/$submod/" || echo "WARNING: $submod build failed"
    fi
done

# tetra_triangulation needs cmake
if [ -d "submodules/tetra_triangulation" ]; then
    echo "Building tetra_triangulation..."
    cd submodules/tetra_triangulation
    conda run -n milo cmake .
    conda run -n milo make -j"$(nproc)"
    conda run -n milo pip install -e .
    cd /opt/milo
fi

# nvdiffrast
if [ -d "submodules/nvdiffrast" ]; then
    echo "Building nvdiffrast..."
    conda run -n milo pip install "submodules/nvdiffrast/"
fi

# Install Python requirements
echo "Installing Python requirements..."
conda run -n milo pip install open3d==0.19.0 trimesh scikit-image opencv-python plyfile tqdm

# Verify
echo "Verifying MILo installation..."
conda run -n milo python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import open3d
print(f'Open3D {open3d.__version__}')
print('MILo installation verified OK')
"

echo "=== MILo installation complete ==="
echo "Usage: conda run -n milo python /opt/milo/train.py -s <COLMAP_DATASET> -m <OUTPUT_DIR> --imp_metric indoor"
