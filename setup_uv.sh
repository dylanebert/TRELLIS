#!/bin/bash

# TRELLIS Setup Script - UV Adaptation
# This script uses UV instead of pip/conda for faster package management

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Initialize failed components tracking
FAILED_COMPONENTS=()

# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,train,xformers,flash-attn,diffoctreerast,vox2seq,spconv,mipgaussian,kaolin,nvdiffrast,demo -n 'setup_uv.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
TRAIN=false
XFORMERS=false
FLASHATTN=false
DIFFOCTREERAST=false
VOX2SEQ=false
LINEAR_ASSIGNMENT=false
SPCONV=false
ERROR=false
MIPGAUSSIAN=false
KAOLIN=false
NVDIFFRAST=false
DEMO=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --train) TRAIN=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --diffoctreerast) DIFFOCTREERAST=true ; shift ;;
        --vox2seq) VOX2SEQ=true ; shift ;;
        --spconv) SPCONV=true ; shift ;;
        --mipgaussian) MIPGAUSSIAN=true ; shift ;;
        --kaolin) KAOLIN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --demo) DEMO=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup_uv.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new uv virtual environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --train                 Install training dependencies"
    echo "  --xformers              Install xformers"
    echo "  --flash-attn            Install flash-attn"
    echo "  --diffoctreerast        Install diffoctreerast"
    echo "  --vox2seq               Install vox2seq"
    echo "  --spconv                Install spconv"
    echo "  --mipgaussian           Install mip-splatting"
    echo "  --kaolin                Install kaolin"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --demo                  Install all dependencies for demo"
    echo ""
    echo "Note: This script uses UV for faster package management."
    echo "Make sure UV is installed: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 0
fi

if [ "$NEW_ENV" = true ] ; then
    echo "[ENV] Creating new UV virtual environment 'trellis_env'..."
    if ! uv venv trellis_env --python 3.10; then
        FAILED_COMPONENTS+=("Environment Creation")
    fi
    echo "[ENV] Activating virtual environment..."
    source trellis_env/bin/activate
    
    # Detect CUDA version for PyTorch installation
    echo "[ENV] Detecting CUDA version..."
    if command -v nvcc &> /dev/null; then
        SYSTEM_CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        echo "[ENV] Detected CUDA version: $SYSTEM_CUDA_VERSION"
    else
        echo "[ENV] CUDA not detected, defaulting to CUDA 11.8"
        SYSTEM_CUDA_VERSION="11.8"
    fi
    
    # Install appropriate PyTorch version based on CUDA version
    case $SYSTEM_CUDA_VERSION in
        11.8)
            echo "[ENV] Installing PyTorch 2.4.0 with CUDA 11.8 support..."
            if ! uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118; then
                FAILED_COMPONENTS+=("PyTorch Installation")
            fi
            ;;
        12.1)
            echo "[ENV] Installing PyTorch 2.4.0 with CUDA 12.1 support..."
            if ! uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121; then
                FAILED_COMPONENTS+=("PyTorch Installation")
            fi
            ;;
        12.4)
            echo "[ENV] Installing PyTorch 2.5.0 with CUDA 12.4 support..."
            if ! uv pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124; then
                FAILED_COMPONENTS+=("PyTorch Installation")
            fi
            ;;
        *)
            echo "[ENV] Unsupported or undetected CUDA version: $SYSTEM_CUDA_VERSION, defaulting to CUDA 11.8"
            if ! uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118; then
                FAILED_COMPONENTS+=("PyTorch Installation")
            fi
            ;;
    esac
fi

# Get system information
WORKDIR=$(pwd)
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not_installed")
if [ "$PYTORCH_VERSION" = "not_installed" ]; then
    echo "[WARNING] PyTorch not found. Please install PyTorch first or use --new-env flag."
    exit 1
fi

PLATFORM=$(python -c "import torch; print(('cuda' if torch.version.cuda else ('hip' if torch.version.hip else 'unknown')) if torch.cuda.is_available() else 'cpu')")
case $PLATFORM in
    cuda)
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
        CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f2)
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
        ;;
    hip)
        HIP_VERSION=$(python -c "import torch; print(torch.version.hip)")
        HIP_MAJOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f1)
        HIP_MINOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f2)
        # Install pytorch 2.4.1 for hip
        if [ "$PYTORCH_VERSION" != "2.4.1+rocm6.1" ] ; then
        echo "[SYSTEM] Installing PyTorch 2.4.1 for HIP ($PYTORCH_VERSION -> 2.4.1+rocm6.1)"
            if ! uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/rocm6.1; then
                FAILED_COMPONENTS+=("PyTorch HIP Installation")
            fi
            mkdir -p /tmp/extensions
            sudo cp /opt/rocm/share/amd_smi /tmp/extensions/amd_smi -r
            cd /tmp/extensions/amd_smi
            sudo chmod -R 777 .
            if ! uv pip install .; then
                FAILED_COMPONENTS+=("AMD SMI Installation")
            fi
            cd $WORKDIR
            PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        fi
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, HIP Version: $HIP_VERSION"
        ;;
    *)
        ;;
esac

if [ "$BASIC" = true ] ; then
    echo "[BASIC] Installing basic dependencies with UV..."
    if ! uv pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers; then
        FAILED_COMPONENTS+=("Basic Dependencies")
    fi
    if ! uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8; then
        FAILED_COMPONENTS+=("Utils3D")
    fi
fi

if [ "$TRAIN" = true ] ; then
    echo "[TRAIN] Installing training dependencies with UV..."
    if ! uv pip install tensorboard pandas lpips; then
        FAILED_COMPONENTS+=("Training Dependencies")
    fi
    uv pip uninstall -y pillow
    sudo apt install -y libjpeg-dev
    if ! uv pip install pillow-simd; then
        FAILED_COMPONENTS+=("Pillow-SIMD")
    fi
fi

if [ "$XFORMERS" = true ] ; then
    echo "[XFORMERS] Installing xformers with UV..."
    # install xformers
    if [ "$PLATFORM" = "cuda" ] ; then
        if [ "$CUDA_VERSION" = "11.8" ] ; then
            case $PYTORCH_VERSION in
                2.0.1*) 
                    if ! uv pip install https://files.pythonhosted.org/packages/52/ca/82aeee5dcc24a3429ff5de65cc58ae9695f90f49fbba71755e7fab69a706/xformers-0.0.22-cp310-cp310-manylinux2014_x86_64.whl; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.1.0*) 
                    if ! uv pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.1.1*) 
                    if ! uv pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.1.2*) 
                    if ! uv pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.2.0*) 
                    if ! uv pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.2.1*) 
                    if ! uv pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.2.2*) 
                    if ! uv pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.3.0*) 
                    if ! uv pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.4.0*) 
                    if ! uv pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.4.1*) 
                    if ! uv pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.5.0*) 
                    if ! uv pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu118; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION"
                   FAILED_COMPONENTS+=("Xformers - Unsupported Version") ;;
            esac
        elif [ "$CUDA_VERSION" = "12.1" ] ; then
            case $PYTORCH_VERSION in
                2.1.0*) 
                    if ! uv pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.1.1*) 
                    if ! uv pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.1.2*) 
                    if ! uv pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.2.0*) 
                    if ! uv pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.2.1*) 
                    if ! uv pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.2.2*) 
                    if ! uv pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.3.0*) 
                    if ! uv pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.4.0*) 
                    if ! uv pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.4.1*) 
                    if ! uv pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                2.5.0*) 
                    if ! uv pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION"
                   FAILED_COMPONENTS+=("Xformers - Unsupported Version") ;;
            esac
        elif [ "$CUDA_VERSION" = "12.4" ] ; then
            case $PYTORCH_VERSION in
                2.5.0*) 
                    if ! uv pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124; then
                        FAILED_COMPONENTS+=("Xformers")
                    fi ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION"
                   FAILED_COMPONENTS+=("Xformers - Unsupported Version") ;;
            esac
        else
            echo "[XFORMERS] Unsupported CUDA version: $CUDA_MAJOR_VERSION"
            FAILED_COMPONENTS+=("Xformers - Unsupported CUDA")
        fi
    elif [ "$PLATFORM" = "hip" ] ; then
        case $PYTORCH_VERSION in
            2.4.1\+rocm6.1) 
                if ! uv pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/rocm6.1; then
                    FAILED_COMPONENTS+=("Xformers")
                fi ;;
            *) echo "[XFORMERS] Unsupported PyTorch version: $PYTORCH_VERSION"
               FAILED_COMPONENTS+=("Xformers - Unsupported PyTorch") ;;
        esac
    else
        echo "[XFORMERS] Unsupported platform: $PLATFORM"
        FAILED_COMPONENTS+=("Xformers - Unsupported Platform")
    fi
fi

if [ "$FLASHATTN" = true ] ; then
    echo "[FLASHATTN] Installing flash-attn with UV..."
    if [ "$PLATFORM" = "cuda" ] ; then
        echo "[FLASHATTN] Installing build dependencies..."
        if ! uv pip install psutil packaging ninja; then
            FAILED_COMPONENTS+=("Flash Attention Build Dependencies")
        fi
        echo "[FLASHATTN] Installing flash-attn..."
        if ! uv pip install flash-attn --no-build-isolation; then
            FAILED_COMPONENTS+=("Flash Attention")
        fi
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.6.3-cktile
        if ! GPU_ARCHS=gfx942 python setup.py install; then  #MI300 series
            FAILED_COMPONENTS+=("Flash Attention HIP")
        fi
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
        FAILED_COMPONENTS+=("Flash Attention - Unsupported Platform")
    fi
fi

if [ "$KAOLIN" = true ] ; then
    echo "[KAOLIN] Installing kaolin with UV..."
    # install kaolin
    if [ "$PLATFORM" = "cuda" ] ; then
        case $PYTORCH_VERSION in
            2.0.1*) 
                if ! uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html; then
                    FAILED_COMPONENTS+=("Kaolin")
                fi ;;
            2.1.0*) 
                if ! uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html; then
                    FAILED_COMPONENTS+=("Kaolin")
                fi ;;
            2.1.1*) 
                if ! uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html; then
                    FAILED_COMPONENTS+=("Kaolin")
                fi ;;
            2.2.0*) 
                if ! uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.0_cu118.html; then
                    FAILED_COMPONENTS+=("Kaolin")
                fi ;;
            2.2.1*) 
                if ! uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.1_cu118.html; then
                    FAILED_COMPONENTS+=("Kaolin")
                fi ;;
            2.2.2*) 
                if ! uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu118.html; then
                    FAILED_COMPONENTS+=("Kaolin")
                fi ;;
            2.4.0*) 
                if ! uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html; then
                    FAILED_COMPONENTS+=("Kaolin")
                fi ;;
            2.5.0*) 
                if ! uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html; then
                    FAILED_COMPONENTS+=("Kaolin")
                fi ;;
            *) echo "[KAOLIN] Unsupported PyTorch version: $PYTORCH_VERSION"
               FAILED_COMPONENTS+=("Kaolin - Unsupported PyTorch") ;;
        esac
    else
        echo "[KAOLIN] Unsupported platform: $PLATFORM"
        FAILED_COMPONENTS+=("Kaolin - Unsupported Platform")
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    echo "[NVDIFFRAST] Installing nvdiffrast with UV..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        rm -rf /tmp/extensions/nvdiffrast
        git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        if ! uv pip install /tmp/extensions/nvdiffrast --no-build-isolation; then
            FAILED_COMPONENTS+=("NVDiffRast")
        fi
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
        FAILED_COMPONENTS+=("NVDiffRast - Unsupported Platform")
    fi
fi

if [ "$DIFFOCTREERAST" = true ] ; then
    echo "[DIFFOCTREERAST] Installing diffoctreerast with UV..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        rm -rf /tmp/extensions/diffoctreerast
        git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
        # These packages have improperly declared build dependencies, so we use --no-build-isolation
        # as the documented workaround for packages that don't follow PEP 518 properly
        if ! uv pip install /tmp/extensions/diffoctreerast --no-build-isolation; then
            FAILED_COMPONENTS+=("DiffOctreeRast")
        fi
    else
        echo "[DIFFOCTREERAST] Unsupported platform: $PLATFORM"
        FAILED_COMPONENTS+=("DiffOctreeRast - Unsupported Platform")
    fi
fi

if [ "$MIPGAUSSIAN" = true ] ; then
    echo "[MIPGAUSSIAN] Installing mip-splatting with UV..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        rm -rf /tmp/extensions/mip-splatting
        git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
        if ! uv pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation; then
            FAILED_COMPONENTS+=("Mip-Splatting")
        fi
    else
        echo "[MIPGAUSSIAN] Unsupported platform: $PLATFORM"
        FAILED_COMPONENTS+=("Mip-Splatting - Unsupported Platform")
    fi
fi

if [ "$VOX2SEQ" = true ] ; then
    echo "[VOX2SEQ] Installing vox2seq with UV..."
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        rm -rf /tmp/extensions/vox2seq
        cp -r extensions/vox2seq /tmp/extensions/vox2seq
        if ! uv pip install /tmp/extensions/vox2seq --no-build-isolation; then
            FAILED_COMPONENTS+=("Vox2Seq")
        fi
    else
        echo "[VOX2SEQ] Unsupported platform: $PLATFORM"
        FAILED_COMPONENTS+=("Vox2Seq - Unsupported Platform")
    fi
fi

if [ "$SPCONV" = true ] ; then
    echo "[SPCONV] Installing spconv with UV..."
    # install spconv
    if [ "$PLATFORM" = "cuda" ] ; then
        case $CUDA_MAJOR_VERSION in
            11) 
                if ! uv pip install spconv-cu118; then
                    FAILED_COMPONENTS+=("SPConv")
                fi ;;
            12) 
                if ! uv pip install spconv-cu120; then
                    FAILED_COMPONENTS+=("SPConv")
                fi ;;
            *) echo "[SPCONV] Unsupported PyTorch CUDA version: $CUDA_MAJOR_VERSION"
               FAILED_COMPONENTS+=("SPConv - Unsupported CUDA") ;;
        esac
    else
        echo "[SPCONV] Unsupported platform: $PLATFORM"
        FAILED_COMPONENTS+=("SPConv - Unsupported Platform")
    fi
fi

if [ "$DEMO" = true ] ; then
    echo "[DEMO] Installing demo dependencies with UV..."
    # Fix Pydantic compatibility issue with Gradio 4.44.1
    if ! uv pip install "pydantic<2.10,>=2.0"; then
        FAILED_COMPONENTS+=("Pydantic")
    fi
    if ! uv pip install gradio==4.44.1 gradio_litmodel3d==0.0.1; then
        FAILED_COMPONENTS+=("Demo Dependencies")
    fi
fi

# Final status report
echo ""
echo "========================================="
if [ ${#FAILED_COMPONENTS[@]} -eq 0 ]; then
    echo "[SETUP] Installation completed successfully!"
else
    echo "[SETUP] Installation completed with failures!"
    echo "Failed components:"
    for component in "${FAILED_COMPONENTS[@]}"; do
        echo "  - $component"
    done
    echo ""
    echo "You may want to manually install the failed components or check the error messages above."
fi

if [ "$NEW_ENV" = true ] ; then
    echo "[SETUP] To activate the environment in the future, run:"
    echo "source trellis_env/bin/activate"
fi 
