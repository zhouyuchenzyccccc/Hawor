#!/bin/bash

# ==============================================================================
# Configuration Section: Modify these variables to change inputs/outputs
# 使用方法：
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh [BASE_DATASET_DIR] [OPTIONS]
#
# 示例：
#   ./run_pipeline.sh /path/to/dataset
#   ./run_pipeline.sh /path/to/dataset --ego_cam_id 08 --task "my task"
#   ./run_pipeline.sh --ego_cam_id 09 --jump_thresh_mm 50
# ==============================================================================

# 1. Environment Setup (一般不需要改)
CONDA_ENV_PATH="/home/ubuntu/WorkSpace/ZYC/hamer/.hamer"
PROJECT_DIR="/home/ubuntu/WorkSpace/ZYC/hawor/ego_recovery_data_preprocessing"

# ==============================================================================
# 默认配置（可被命令行参数覆盖）
# ==============================================================================

# Dataset Paths (Base Directory)
BASE_DATASET_DIR="/home/ubuntu/orbbec/src/sync/test/hand_insertion_new_success"

# Processing Parameters
EGO_CAM_ID="06"
LEFT_WRIST_CAM_ID="07"
RIGHT_WRIST_CAM_ID="08"
TASK_NAME="hand insertion recovery"

# Post-processing Thresholds
JUMP_THRESH_MM=30
RPY_JUMP_THRESH_DEG=20

# ==============================================================================
# 解析命令行参数
# ==============================================================================

# 第一个位置参数：BASE_DATASET_DIR（可选）
if [ -n "$1" ] && [ "${1:0:2}" != "--" ]; then
    BASE_DATASET_DIR="$1"
    shift
fi

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ego_cam_id)
            EGO_CAM_ID="$2"
            shift 2
            ;;
        --left_wrist_cam_id)
            LEFT_WRIST_CAM_ID="$2"
            shift 2
            ;;
        --right_wrist_cam_id)
            RIGHT_WRIST_CAM_ID="$2"
            shift 2
            ;;
        --task|--task_name)
            TASK_NAME="$2"
            shift 2
            ;;
        --jump_thresh_mm)
            JUMP_THRESH_MM="$2"
            shift 2
            ;;
        --rpy_jump_thresh_deg)
            RPY_JUMP_THRESH_DEG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [BASE_DATASET_DIR] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  BASE_DATASET_DIR          Base directory of dataset (optional)"
            echo ""
            echo "Options:"
            echo "  --ego_cam_id ID           Ego camera ID (default: 06)"
            echo "  --left_wrist_cam_id ID    Left wrist camera ID (default: 07)"
            echo "  --right_wrist_cam_id ID   Right wrist camera ID (default: 08)"
            echo "  --task_name NAME          Task name (default: 'hand insertion recovery')"
            echo "  --jump_thresh_mm MM       Jump threshold in mm (default: 30)"
            echo "  --rpy_jump_thresh_deg DEG Jump threshold in degrees (default: 20)"
            echo "  --help, -h                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

# ==============================================================================
# 基于 BASE_DATASET_DIR 派生所有路径
# ==============================================================================
INPUT_DATA_DIR="${BASE_DATASET_DIR}/sweep success"
PADDED_DATA_DIR="${INPUT_DATA_DIR}_padded"
HAWOR_POSES_DIR="${INPUT_DATA_DIR}_padded_hawor"
CLEAN_POSES_DIR="${INPUT_DATA_DIR}_padded_hawor_clean"
FINAL_LEROBOT_DIR="${INPUT_DATA_DIR}_padded_hawor_clean_lerobot"

# ==============================================================================
# Execution Logic
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status.

echo "=================================================="
echo "Starting Ego Recovery Data Preprocessing Pipeline"
echo "=================================================="
echo "Configuration:"
echo "  BASE_DATASET_DIR       = ${BASE_DATASET_DIR}"
echo "  INPUT_DATA_DIR         = ${INPUT_DATA_DIR}"
echo "  PADDED_DATA_DIR        = ${PADDED_DATA_DIR}"
echo "  HAWOR_POSES_DIR        = ${HAWOR_POSES_DIR}"
echo "  CLEAN_POSES_DIR        = ${CLEAN_POSES_DIR}"
echo "  FINAL_LEROBOT_DIR      = ${FINAL_LEROBOT_DIR}"
echo "  EGO_CAM_ID             = ${EGO_CAM_ID}"
echo "  LEFT_WRIST_CAM_ID      = ${LEFT_WRIST_CAM_ID}"
echo "  RIGHT_WRIST_CAM_ID     = ${RIGHT_WRIST_CAM_ID}"
echo "  TASK_NAME              = ${TASK_NAME}"
echo "  JUMP_THRESH_MM         = ${JUMP_THRESH_MM}"
echo "  RPY_JUMP_THRESH_DEG    = ${RPY_JUMP_THRESH_DEG}"
echo "=================================================="

# Step 0: Activate Environment and Change Directory
echo "[Step 0] Activating conda environment and changing directory..."
source "${CONDA_ENV_PATH}/bin/activate"
cd "${PROJECT_DIR}"

if [ ! -f "pad_episodes_to_same_length.py" ]; then
    echo "Error: Could not find processing scripts in ${PROJECT_DIR}"
    exit 1
fi

# Step 1: Pad Episodes to Same Length
echo "[Step 1] Padding episodes to same length..."
echo "  Input:  ${INPUT_DATA_DIR}"
echo "  Output: ${PADDED_DATA_DIR}"

python pad_episodes_to_same_length.py \
    --data_dir "${INPUT_DATA_DIR}" \
    --output_dir "${PADDED_DATA_DIR}"

echo "[Step 1] Completed successfully."

# Step 2: Extract HaWoR Poses
echo "[Step 2] Extracting HaWoR poses..."
echo "  Input:  ${PADDED_DATA_DIR}"
echo "  Output: ${HAWOR_POSES_DIR}"

python hawor_extract_poses.py \
    --data_dir "${PADDED_DATA_DIR}" \
    --output "${HAWOR_POSES_DIR}" \
    --ego_cam_id "${EGO_CAM_ID}"

echo "[Step 2] Completed successfully."

# Step 3: Post-process Poses (Cleaning)
echo "[Step 3] Post-processing poses (cleaning jumps)..."
echo "  Input:  ${HAWOR_POSES_DIR}"
echo "  Output: ${CLEAN_POSES_DIR}"

python postprocess_poses.py \
    --input_dir "${HAWOR_POSES_DIR}" \
    --output_dir "${CLEAN_POSES_DIR}" \
    --jump_thresh_mm "${JUMP_THRESH_MM}" \
    --rpy_jump_thresh_deg "${RPY_JUMP_THRESH_DEG}"

echo "[Step 3] Completed successfully."

# Step 4: Convert to LeRobot Format
echo "[Step 4] Converting to LeRobot format..."
echo "  Data Dir:   ${PADDED_DATA_DIR}"
echo "  Poses Dir:  ${CLEAN_POSES_DIR}"
echo "  Output:     ${FINAL_LEROBOT_DIR}"

python convert_to_lerobot.py \
    --data_dir "${PADDED_DATA_DIR}" \
    --poses "${CLEAN_POSES_DIR}" \
    --output "${FINAL_LEROBOT_DIR}" \
    --task "${TASK_NAME}" \
    --ego_cam_id "${EGO_CAM_ID}" \
    --left_wrist_cam_id "${LEFT_WRIST_CAM_ID}" \
    --right_wrist_cam_id "${RIGHT_WRIST_CAM_ID}"

echo "[Step 4] Completed successfully."

echo "=================================================="
echo "Pipeline Finished Successfully!"
echo "Final Output located at: ${FINAL_LEROBOT_DIR}"
echo "=================================================="