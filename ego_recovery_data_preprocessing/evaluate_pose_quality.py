#!/usr/bin/env python3
"""
评估 LeRobot 格式数据集中位姿估计质量。

指标：
  RPY:
    - 平均帧间跳变 (deg)
    - P99 帧间跳变 (deg)
    - 大跳变帧数 (>46 deg)
    - 自相关（平滑度）
  XYZ:
    - 平均帧间跳变 (mm)
    - P99 帧间跳变 (mm)
    - 大跳变帧数 (>20 mm)
    - >10cm 跳变的 episode 列表
    - 自相关（平滑度）

用法：
  python evaluate_pose_quality.py --dataset /path/to/lerobot_dataset
  python evaluate_pose_quality.py --dataset /path/to/lerobot_dataset --side left
  python evaluate_pose_quality.py --dataset /path/to/lerobot_dataset --side both --verbose
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# STATE_NAMES 索引
# left:  x=0, y=1, z=2, rx=3, ry=4, rz=5, gripper=6
# right: x=7, y=8, z=9, rx=10, ry=11, rz=12, gripper=13
SIDES = {
    "left":  {"xyz": slice(0, 3), "rpy": slice(3, 6)},
    "right": {"xyz": slice(7, 10), "rpy": slice(10, 13)},
}


def load_all_episodes(dataset_dir: Path):
    """读取所有 parquet 文件，按 episode_index 分组，返回 {ep_idx: np.ndarray (T, 14)}"""
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.rglob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到任何 parquet 文件")

    episodes = {}
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["observation.state", "episode_index"])
        states = np.array(table["observation.state"].to_pylist(), dtype=np.float32)  # (T, 14)
        ep_indices = table["episode_index"].to_pylist()
        ep_idx = ep_indices[0]
        episodes[ep_idx] = states

    return episodes


def rad_to_deg(arr):
    return arr * (180.0 / math.pi)


def autocorr_lag1(series: np.ndarray) -> float:
    """计算 lag-1 自相关，值越接近 1 表示越平滑。"""
    if len(series) < 2:
        return float("nan")
    x = series - series.mean()
    denom = np.dot(x, x)
    if denom < 1e-12:
        return 1.0
    return float(np.dot(x[:-1], x[1:]) / denom)


def compute_jump_stats(deltas_flat: np.ndarray, large_thresh: float):
    """给定所有帧间跳变量（绝对值），计算统计指标。"""
    if len(deltas_flat) == 0:
        return dict(mean=float("nan"), p99=float("nan"), large_count=0, large_ratio=0.0)
    mean_jump = float(np.mean(deltas_flat))
    p99_jump = float(np.percentile(deltas_flat, 99))
    large_count = int(np.sum(deltas_flat > large_thresh))
    large_ratio = large_count / len(deltas_flat)
    return dict(mean=mean_jump, p99=p99_jump, large_count=large_count, large_ratio=large_ratio)


def evaluate_side(episodes: dict, side: str, verbose: bool):
    xyz_slice = SIDES[side]["xyz"]
    rpy_slice = SIDES[side]["rpy"]

    all_xyz_deltas = []   # mm
    all_rpy_deltas = []   # deg
    all_xyz_autocorr = []
    all_rpy_autocorr = []
    episodes_with_large_xyz = []  # >10cm 跳变

    for ep_idx in sorted(episodes.keys()):
        states = episodes[ep_idx]  # (T, 14)
        if len(states) < 2:
            continue

        xyz = states[:, xyz_slice]  # (T, 3), meters
        rpy = states[:, rpy_slice]  # (T, 3), radians

        # 转换单位
        xyz_mm = xyz * 1000.0
        rpy_deg = rad_to_deg(rpy)

        # 帧间差分
        xyz_diff = np.diff(xyz_mm, axis=0)   # (T-1, 3)
        rpy_diff = np.diff(rpy_deg, axis=0)  # (T-1, 3)

        # 角度差归一化到 [-180, 180]
        rpy_diff = (rpy_diff + 180.0) % 360.0 - 180.0

        xyz_jump = np.linalg.norm(xyz_diff, axis=1)   # (T-1,) mm
        rpy_jump = np.linalg.norm(rpy_diff, axis=1)   # (T-1,) deg

        all_xyz_deltas.append(xyz_jump)
        all_rpy_deltas.append(rpy_jump)

        # 自相关（对每个轴分别计算，取均值）
        xyz_ac = np.mean([autocorr_lag1(xyz_mm[:, i]) for i in range(3)])
        rpy_ac = np.mean([autocorr_lag1(rpy_deg[:, i]) for i in range(3)])
        all_xyz_autocorr.append(xyz_ac)
        all_rpy_autocorr.append(rpy_ac)

        # >10cm 跳变 episode
        if np.any(xyz_jump > 100.0):
            count_large = int(np.sum(xyz_jump > 100.0))
            episodes_with_large_xyz.append((ep_idx, count_large, float(xyz_jump.max())))
            if verbose:
                bad_frames = np.where(xyz_jump > 100.0)[0]
                print(f"  [ep {ep_idx}] {side} XYZ >10cm 跳变帧: {bad_frames.tolist()}")

    all_xyz_flat = np.concatenate(all_xyz_deltas) if all_xyz_deltas else np.array([])
    all_rpy_flat = np.concatenate(all_rpy_deltas) if all_rpy_deltas else np.array([])

    xyz_stats = compute_jump_stats(all_xyz_flat, large_thresh=20.0)
    rpy_stats = compute_jump_stats(all_rpy_flat, large_thresh=46.0)

    mean_xyz_ac = float(np.mean(all_xyz_autocorr)) if all_xyz_autocorr else float("nan")
    mean_rpy_ac = float(np.mean(all_rpy_autocorr)) if all_rpy_autocorr else float("nan")

    return {
        "xyz": xyz_stats,
        "rpy": rpy_stats,
        "xyz_autocorr": mean_xyz_ac,
        "rpy_autocorr": mean_rpy_ac,
        "episodes_with_large_xyz_jump": episodes_with_large_xyz,
        "total_frames_evaluated": len(all_xyz_flat),
    }


def print_report(side: str, result: dict):
    xyz = result["xyz"]
    rpy = result["rpy"]
    eps = result["episodes_with_large_xyz_jump"]
    n = result["total_frames_evaluated"]

    print(f"\n{'='*55}")
    print(f"  [{side.upper()}] 位姿质量评估报告  (帧间跳变样本数: {n})")
    print(f"{'='*55}")

    print("\n--- RPY (旋转) ---")
    print(f"  平均帧间跳变:       {rpy['mean']:.4f} deg")
    print(f"  P99 帧间跳变:       {rpy['p99']:.4f} deg")
    print(f"  大跳变 (>46°) 帧数: {rpy['large_count']}  ({rpy['large_ratio']*100:.2f}%)")
    print(f"  自相关 (lag-1):     {result['rpy_autocorr']:.4f}  (越接近1越平滑)")

    print("\n--- XYZ (位置) ---")
    print(f"  平均帧间跳变:        {xyz['mean']:.4f} mm")
    print(f"  P99 帧间跳变:        {xyz['p99']:.4f} mm")
    print(f"  大跳变 (>20mm) 帧数: {xyz['large_count']}  ({xyz['large_ratio']*100:.2f}%)")
    print(f"  自相关 (lag-1):      {result['xyz_autocorr']:.4f}  (越接近1越平滑)")

    print(f"\n--- XYZ >10cm 跳变 Episodes ({len(eps)} 个) ---")
    if eps:
        for ep_idx, cnt, max_jump in eps:
            print(f"  episode {ep_idx:4d}: {cnt} 帧超过10cm, 最大跳变 {max_jump:.1f} mm")
    else:
        print("  无")


def main():
    ap = argparse.ArgumentParser(description="评估 LeRobot 格式数据集的位姿质量")
    ap.add_argument("--dataset", required=True, help="LeRobot 数据集根目录（含 data/ 子目录）")
    ap.add_argument(
        "--side",
        choices=["left", "right", "both"],
        default="both",
        help="评估哪侧手腕位姿 (默认: both)",
    )
    ap.add_argument("--verbose", action="store_true", help="打印每个 episode 的详细跳变帧信息")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    print(f"加载数据集: {dataset_dir}")
    episodes = load_all_episodes(dataset_dir)
    print(f"共加载 {len(episodes)} 个 episode")

    sides = ["left", "right"] if args.side == "both" else [args.side]
    for side in sides:
        result = evaluate_side(episodes, side, args.verbose)
        print_report(side, result)

    print()


if __name__ == "__main__":
    main()
