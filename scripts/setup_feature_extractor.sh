#!/bin/bash
# Build the bottom-up feature extractor against a modern PyTorch.
#
#   bash scripts/setup_feature_extractor.sh [install_dir]
#
# The extractor that produced FGA's image features is Faster R-CNN with a
# ResNeXt-101 backbone fine-tuned on Visual Genome (Anderson et al.'s bottom-up
# attention), and it has not been maintained since 2019. Every published copy of
# the extracted features is now offline, so regenerating them is the only route --
# and that means getting the extractor to build against a current torch.
#
# This does the four things needed:
#   1. clone vqa-maskrcnn-benchmark
#   2. port its CUDA/C++/Python sources off APIs torch and NumPy removed
#   3. compile the extension
#   4. fetch MMF's extraction script and detach it from the rest of MMF
#
# Requirements: a CUDA GPU, nvcc, and a torch built against the *same* CUDA major
# version as that nvcc. Mismatched versions fail at the extension build with a
# clear message; install torch from the matching index, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu128

set -euo pipefail

INSTALL_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
BENCHMARK_DIR="$INSTALL_DIR/vqa-maskrcnn-benchmark"

mkdir -p "$INSTALL_DIR"

echo "== 1/4 python dependencies"
DEPS='yacs cython opencv-python-headless ninja tqdm numpy<2 pillow requests'
if $PYTHON -m pip --version >/dev/null 2>&1; then
    $PYTHON -m pip install -q $DEPS
elif command -v uv >/dev/null 2>&1; then
    # uv-created venvs have no pip of their own.
    VIRTUAL_ENV="$($PYTHON -c 'import sys; print(sys.prefix)')" uv pip install -q $DEPS
else
    echo "   neither pip nor uv is available for $PYTHON" >&2
    exit 1
fi

echo "== 2/4 clone vqa-maskrcnn-benchmark"
if [ ! -d "$BENCHMARK_DIR" ]; then
    git clone -q https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git "$BENCHMARK_DIR"
else
    echo "   already cloned"
fi

echo "== 3/4 port it to a modern torch, then build"
$PYTHON "$REPO_ROOT/scripts/patch_maskrcnn_benchmark.py" --repo "$BENCHMARK_DIR"

# Ada (L40S, 4090) is 8.9; Ampere (A100) 8.0; Turing (T4, 2080) 7.5.
# Detected automatically, with a broad fallback.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$($PYTHON - <<'EOF' 2>/dev/null || echo "7.5;8.0;8.6;8.9"
import torch
caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
print(";".join(f"{a}.{b}" for a, b in sorted(caps)) or "7.5;8.0;8.6;8.9")
EOF
)"
fi
export TORCH_CUDA_ARCH_LIST
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
echo "   TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  CUDA_HOME=$CUDA_HOME"

(cd "$BENCHMARK_DIR" && $PYTHON setup.py -q build develop)

$PYTHON - <<'EOF'
import torch  # must precede the extension so its shared libraries are loaded
from maskrcnn_benchmark import _C

boxes = torch.tensor(
    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]], device="cuda"
)
scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
kept = _C.nms(boxes, scores, 0.5).tolist()
assert kept == [0, 2], f"NMS returned {kept}, expected [0, 2]"
print("   extension builds and NMS is correct")
EOF

echo "== 4/4 fetch MMF's extraction script"
cd "$INSTALL_DIR"
for name in extract_features_vmb.py extraction_utils.py; do
    if [ ! -s "$name" ]; then
        curl -fsSL -o "$name" \
            "https://raw.githubusercontent.com/facebookresearch/mmf/main/tools/scripts/features/$name"
    fi
done

# Detach it from the rest of MMF: the only uses are an auto-download helper we do
# not need (the model is passed explicitly) and a package-relative import.
$PYTHON - <<EOF
import pathlib

path = pathlib.Path("$INSTALL_DIR/extract_features_vmb.py")
source = path.read_text()
source = source.replace(
    "from mmf.utils.download import download",
    'def download(*args, **kwargs):\n'
    '    raise SystemExit("auto-download disabled; pass --model_file and --config_file")',
)
source = source.replace(
    "from tools.scripts.features.extraction_utils import chunks, get_image_files",
    "from extraction_utils import chunks, get_image_files",
)
path.write_text(source)
print("   extractor ready:", path)
EOF

echo
echo "Done. Extractor at $INSTALL_DIR/extract_features_vmb.py"
echo "Next: bash scripts/download_extraction_assets.sh detector"
