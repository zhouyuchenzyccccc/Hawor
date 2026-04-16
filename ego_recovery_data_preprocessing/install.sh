#!/bin/bash
# install.sh - One-shot setup for HaWoR ego recovery data preprocessing
set -e

HAWOR_DIR="/home/ubuntu/WorkSpace/ZYC/hawor"
HAMER_VENV="/home/ubuntu/WorkSpace/ZYC/hamer/.hamer/bin/activate"
HAMER_DATA="/home/ubuntu/WorkSpace/ZYC/hamer/_DATA"

echo "=== Activating HaMeR venv ==="
source "$HAMER_VENV"

echo "=== Installing Python dependencies ==="
cd "$HAWOR_DIR"
pip install natsort ultralytics joblib smplx==0.1.28 mmcv==1.3.9 --quiet

# pytorch3d (optional, for visualization only)
pip install pytorch3d \
    --index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt220/ \
    --quiet 2>/dev/null || echo "[WARN] pytorch3d not installed (optional, pose extraction works without it)"

echo "=== Setting up MANO models ==="
mkdir -p "$HAWOR_DIR/_DATA/data/mano"
mkdir -p "$HAWOR_DIR/_DATA/data_left/mano_left"

ln -sf "$HAMER_DATA/data/mano/MANO_RIGHT.pkl" \
       "$HAWOR_DIR/_DATA/data/mano/MANO_RIGHT.pkl"

python "$HAWOR_DIR/ego_recovery_data_preprocessing/make_mano_left.py"

echo "=== Downloading model weights (via hf-mirror.com) ==="
mkdir -p "$HAWOR_DIR/weights/hawor/checkpoints"
mkdir -p "$HAWOR_DIR/weights/external"

download_if_missing() {
    local url="$1"
    local dst="$2"
    if [ -f "$dst" ] && [ -s "$dst" ]; then
        echo "  [SKIP] $(basename $dst) already exists"
    else
        echo "  Downloading $(basename $dst) ..."
        wget -q --show-progress "$url" -O "$dst"
    fi
}

download_if_missing \
    "https://hf-mirror.com/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt" \
    "$HAWOR_DIR/weights/hawor/checkpoints/hawor.ckpt"

download_if_missing \
    "https://hf-mirror.com/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt" \
    "$HAWOR_DIR/weights/hawor/checkpoints/infiller.pt"

download_if_missing \
    "https://hf-mirror.com/ThunderVVV/HaWoR/resolve/main/model_config.yaml" \
    "$HAWOR_DIR/weights/hawor/model_config.yaml"

download_if_missing \
    "https://hf-mirror.com/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt" \
    "$HAWOR_DIR/weights/external/detector.pt"

echo ""
echo "=== Setup complete! ==="
echo "Run:"
echo "  source $HAMER_VENV"
echo "  cd $HAWOR_DIR/ego_recovery_data_preprocessing"
echo "  python hawor_extract_poses.py --data_dir /path/to/sequence --output wrist_poses.npz"
