# HaWoR 手部位姿提取工具

## 项目简介

本项目基于 **HaWoR**（CVPR 2025 Highlight）提取时序一致的 3D 手腕位姿，替代原有 MediaPipe 方案。

**为什么换 HaWoR？**

| 对比项 | MediaPipe | HaWoR |
|---|---|---|
| 时序一致性 | 无，逐帧独立 | 内置 Transformer 时序先验 |
| 手部模型 | 21 关键点 | MANO 参数化网格（778顶点，21关节） |
| 遮挡处理 | 无效帧插值 | Infiller 网络自动填充 |
| 精度 | 中等 | CVPR 2025 SOTA |
| 输出格式 | 自定义 | 与原 wrist_poses.npz 完全兼容 |

输出格式与原 `extract_wrist_pose.py` 完全兼容，可直接接入 `convert_to_lerobot.py`。

---

## 目录结构

```
hawor/
├── (HaWoR 原始代码)
├── weights/
│   ├── hawor/checkpoints/
│   │   ├── hawor.ckpt          ← HaWoR 主模型
│   │   └── infiller.pt         ← 时序填充网络
│   └── external/
│       └── detector.pt         ← WiLoR 手部检测器
├── _DATA/
│   ├── data/mano/
│   │   └── MANO_RIGHT.pkl      ← symlink from hamer
│   └── data_left/mano_left/
│       └── MANO_LEFT.pkl       ← 从 RIGHT 镜像生成
└── ego_recovery_data_preprocessing/
    ├── hawor_extract_poses.py      ← 主提取脚本（本项目）
    ├── postprocess_poses.py        ← 位姿后处理：去除大跳变
    ├── evaluate_pose_quality.py    ← 位姿质量评估
    ├── visualize_hawor_poses.py
    ├── convert_to_lerobot.py
    └── README.md
```

---

## 安装

### 一键安装

```bash
bash /home/ubuntu/WorkSpace/ZYC/hawor/ego_recovery_data_preprocessing/install.sh
```

### 手动安装

```bash
# 激活 HaMeR venv（已有 PyTorch 2.2.0+cu118）
source /home/ubuntu/WorkSpace/ZYC/hamer/.hamer/bin/activate

# 安装 HaWoR 依赖
cd /home/ubuntu/WorkSpace/ZYC/hawor
pip install natsort ultralytics joblib smplx==0.1.28 mmcv==1.3.9

# pytorch3d（可选，仅用于可视化；位姿提取不需要）
pip install pytorch3d --index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt220/

# torch-scatter（可选）
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

### 下载模型权重

```bash
source /home/ubuntu/WorkSpace/ZYC/hamer/.hamer/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /home/ubuntu/WorkSpace/ZYC/hawor/weights/hawor/checkpoints
mkdir -p /home/ubuntu/WorkSpace/ZYC/hawor/weights/external

# HaWoR 主模型
wget "https://hf-mirror.com/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt" \
     -O /home/ubuntu/WorkSpace/ZYC/hawor/weights/hawor/checkpoints/hawor.ckpt

# 时序填充网络
wget "https://hf-mirror.com/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt" \
     -O /home/ubuntu/WorkSpace/ZYC/hawor/weights/hawor/checkpoints/infiller.pt

# 模型配置
wget "https://hf-mirror.com/ThunderVVV/HaWoR/resolve/main/model_config.yaml" \
     -O /home/ubuntu/WorkSpace/ZYC/hawor/weights/hawor/model_config.yaml

# WiLoR 检测器
wget "https://hf-mirror.com/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt" \
     -O /home/ubuntu/WorkSpace/ZYC/hawor/weights/external/detector.pt
```

### 设置 MANO 模型

```bash
# MANO_RIGHT：从 HaMeR 复用
mkdir -p /home/ubuntu/WorkSpace/ZYC/hawor/_DATA/data/mano
ln -sf /home/ubuntu/WorkSpace/ZYC/hamer/_DATA/data/mano/MANO_RIGHT.pkl \
       /home/ubuntu/WorkSpace/ZYC/hawor/_DATA/data/mano/MANO_RIGHT.pkl

# MANO_LEFT：从 RIGHT 镜像生成
mkdir -p /home/ubuntu/WorkSpace/ZYC/hawor/_DATA/data_left/mano_left
source /home/ubuntu/WorkSpace/ZYC/hamer/.hamer/bin/activate
python /home/ubuntu/WorkSpace/ZYC/hawor/ego_recovery_data_preprocessing/make_mano_left.py
```

---

## 使用步骤
```bash
  # 原地修改（直接覆盖原始 episode 目录）：
  python pad_episodes_to_same_length.py --data_dir /path/to/dataset

  # 写到新目录（保留原始数据）：
  python pad_episodes_to_same_length.py --data_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_new_success/pick_and_place --output_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_new_success/pick_and_place_padded

  # 预览，不写入任何文件：
  python pad_episodes_to_same_length.py --data_dir /path/to/dataset --dry_run

```


### 步骤 1：提取手腕位姿

```bash
source /home/ubuntu/WorkSpace/ZYC/hamer/.hamer/bin/activate
cd /home/ubuntu/WorkSpace/ZYC/hawor/ego_recovery_data_preprocessing

python hawor_extract_poses.py \
    --data_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_new_recovery/pick_and_place_padded \
    --output /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_new_recovery/hand_insertion_new_recovery_hawor \
    --ego_cam_id 06
```

**批量处理所有序列（0~100）：**

```bash
python hawor_extract_poses.py \
    --data_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_new_success/pick_and_place \
    --output /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/poses/wrist_poses.npz \
    --ego_cam_id 07


    nohup python ego_recovery_data_preprocessing/hawor_extract_poses.py       --data_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/test       --output /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/wrist_poses_hawor       --ego_cam_id 07       > /tmp/hawor_extract_insertion.log 2>&1 &


    python ego_recovery_data_preprocessing/hawor_extract_poses.py       --data_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_416/pick_and_place/       --output /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_416/wrist_poses_hawor       --ego_cam_id 07 
```

批量模式下输出文件命名为 `0_wrist_poses.npz`、`1_wrist_poses.npz` 等。

**主要参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | 必填 | 单序列目录或包含多序列的父目录 |
| `--output` | 序列目录内 | 输出 .npz 路径 |
| `--ego_cam_id` | `07` | ego 相机文件夹 ID |
| `--hawor_dir` | `/home/ubuntu/WorkSpace/ZYC/hawor` | HaWoR 代码目录 |
| `--fps` | `30.0` | 视频帧率 |
| `--no_depth_anchor` | 关闭 | 跳过 Orbbec 深度 metric 校正 |
| `--smooth_method` | `none` | 额外平滑（HaWoR 已内置时序平滑） |

### 步骤 1.5（可选）：位姿后处理——去除大跳变

HaWoR 在部分帧可能出现 tracking failure，导致 XYZ 瞬间跳变 100mm 以上。
`postprocess_poses.py` 对已生成的 `wrist_poses.npz` 做**异常帧检测 → 插值 → 平滑**，无需重跑 HaWoR。

**先 dry_run 查看会修复多少帧（不写文件）：**

```bash
python postprocess_poses.py \
    --input_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/wrist_poses_hawor \
    --dry_run
```

**输出到新目录（保留原文件）：**

```bash
python postprocess_poses.py \
    --input_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/wrist_poses_hawor \
    --output_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/wrist_poses_clean \
    --jump_thresh_mm 50
```

**直接覆盖原文件：**

```bash
python postprocess_poses.py \
    --input_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/wrist_poses_hawor \
    --inplace \
    --jump_thresh_mm 50
```

**主要参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--jump_thresh_mm` | `50` | XYZ 帧间跳变阈值（mm），超过则视为异常帧 |
| `--context` | `1` | 异常帧前后额外标记的帧数，避免插值锚点也是坏帧 |
| `--smooth_method` | `median_then_savgol` | 平滑方法：`savgol` / `median_then_savgol` / `none` |
| `--smooth_window` | `11` | Savitzky-Golay 窗口大小（奇数） |
| `--dry_run` | 关闭 | 只统计不写文件 |

> **调参建议：** 若处理后仍有较多 >10cm 跳变，可将 `--jump_thresh_mm` 降至 30，或将 `--context` 加到 2。

---

### 步骤 2：可视化验证

```bash
python visualize_hawor_poses.py \
    --data_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/test/0 \
    --poses /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/test/0/wrist_poses.npz \
    --output /tmp/hawor_viz.mp4 \
    --ego_cam_id 07
```

可视化内容：
- 3D 坐标轴（X=红色, Y=绿色, Z=蓝色）
- 位置坐标文本（米）
- 夹爪开合距离（拇指-食指距离）
- 检测状态（DET=直接检测, INTERP=插值填充）

### 步骤 3：转换为 LeRobot 数据集


**批量转换多个序列：**

```bash
source /home/ubuntu/WorkSpace/ZYC/hamer/.hamer/bin/activate
  cd /home/ubuntu/WorkSpace/ZYC/hawor

  # hand_insertion_recovery1
  python convert_to_lerobot.py \
      --data_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_recovery2/test \
      --poses /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_recovery2/wrist_poses_hawor_clean \
      --output /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion_recovery2/lerobot_hawor_clean \
      --task "hand insertion recovery" \
      --ego_cam_id 07 --left_wrist_cam_id 06 --right_wrist_cam_id 08
```

### 步骤 4（可选）：评估位姿质量

对转换后的 LeRobot 数据集计算位姿质量指标，用于对比后处理前后的效果。

```bash

python evaluate_pose_quality.py \
    --dataset /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/lerobot_hawor

# 只评估左手
python evaluate_pose_quality.py \
    --dataset /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/lerobot_hawor \
    --side left

# 打印每个 episode 的大跳变帧位置
python evaluate_pose_quality.py \
    --dataset /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/lerobot_hawor \
    --verbose
```

**输出指标：**

| 指标 | 说明 |
|---|---|
| RPY 平均帧间跳变 | 旋转平均变化量（deg/帧） |
| RPY P99 跳变 | 99 分位跳变，反映极端抖动 |
| RPY 大跳变 (>46°) | 明显旋转突变帧数 |
| RPY 自相关 (lag-1) | 越接近 1 越平滑 |
| XYZ 平均帧间跳变 | 位置平均变化量（mm/帧） |
| XYZ P99 跳变 | 99 分位跳变 |
| XYZ 大跳变 (>20mm) | 明显位置突变帧数 |
| XYZ >10cm 跳变 episodes | 存在严重 tracking failure 的 episode 列表 |
| XYZ 自相关 (lag-1) | 越接近 1 越平滑 |

---

## 输出格式

`wrist_poses.npz` 包含以下字段（与原 MediaPipe 方案完全兼容）：

| 字段 | 形状 | 说明 |
|---|---|---|
| `left_wrist_poses` | `(T, 6)` | 左手腕位姿 `[x, y, z, rx, ry, rz]`（米 + 弧度） |
| `right_wrist_poses` | `(T, 6)` | 右手腕位姿 |
| `left_gripper` | `(T,)` | 左手拇指-食指距离（米） |
| `right_gripper` | `(T,)` | 右手拇指-食指距离（米） |
| `left_valid` | `(T,)` | 左手直接检测帧标记（False=插值） |
| `right_valid` | `(T,)` | 右手直接检测帧标记 |
| `fps` | `float32` | 帧率 |

坐标系：ego 相机坐标系（使用 identity SLAM，world == camera frame）。

---

## 技术说明

### 为什么跳过 DROID-SLAM？

服务器 nvcc 11.5 与 PyTorch cu118 不兼容，无法编译 DROID-SLAM CUDA 扩展。
本方案使用 **identity SLAM**（world == camera frame），输出保持在相机坐标系下，
与原 MediaPipe 方案坐标系一致，不影响 cotrain 训练。

### Orbbec 深度 metric 校正

HaWoR 输出的 translation 是归一化相机坐标（无 metric scale）。
本脚本将手腕投影到深度图，用 Orbbec 深度值（mm → m）校正 Z 轴，
再等比缩放 X/Y，使输出具有真实 metric 尺度。

可用 `--no_depth_anchor` 跳过此步骤（输出为归一化坐标）。

### MANO_LEFT 说明

HaWoR 的 `run_mano_left` 在加载时自动应用 `shapedirs[:,0,:] *= -1` 修正
（参见 `hawor/utils/process.py` 中的 `fix_shapedirs=True`），
因此 MANO_LEFT.pkl 直接从 MANO_RIGHT.pkl 复制即可。

---

## 常见问题

**Q: `ModuleNotFoundError: No module named 'pytorch3d'`**

A: pytorch3d 仅用于可视化（mask 生成），位姿提取不需要。脚本已自动 mock，无需安装。

**Q: `No sequence directories found`**

A: 检查 `--data_dir` 下是否有 `camera_params.json` 和 `07/RGB/*.jpg`。

**Q: 检测率很低（< 50%）**

A: 正常，HaWoR infiller 会自动填充未检测帧。检查 `left_valid` / `right_valid` 比例。

**Q: 输出坐标单位不对**

A: 确认深度图为 Orbbec uint16 mm 格式。如使用其他相机，调整 `DEPTH_SCALE`（默认 0.001）。
