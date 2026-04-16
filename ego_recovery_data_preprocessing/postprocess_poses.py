#!/usr/bin/env python3
"""
对 HaWoR 输出的 wrist_poses.npz 做后处理，消除大跳变。

流程：
  1. 异常帧检测：XYZ 帧间跳变 > jump_thresh 的帧标记为异常
  2. 插值：用前后有效帧线性插值替换异常帧
  3. 平滑：Savitzky-Golay 或 median+savgol

用法（单文件）：
  python postprocess_poses.py --input wrist_poses.npz --output wrist_poses_clean.npz

用法（批量，覆盖原文件）：
  python postprocess_poses.py --input_dir /path/to/poses_dir --inplace

用法（批量，输出到新目录）：
  python postprocess_poses.py --input_dir /path/to/poses_dir --output_dir /path/to/clean_dir
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter


# ── 核心后处理 ──────────────────────────────────────────────────────────────

def detect_jump_outliers(xyz: np.ndarray, jump_thresh_m: float,
                         context: int = 1) -> np.ndarray:
    """
    检测 XYZ 跳变异常帧，返回 bool mask（True = 异常）。

    context: 异常帧前后各额外标记多少帧（避免插值锚点本身也是坏帧）
    """
    T = len(xyz)
    bad = np.zeros(T, dtype=bool)
    if T < 2:
        return bad

    jumps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)  # (T-1,)

    # 跳变发生在帧 i -> i+1，把帧 i+1 标为异常
    for i in np.where(jumps > jump_thresh_m)[0]:
        lo = max(0, i + 1 - context)
        hi = min(T - 1, i + 1 + context)
        bad[lo:hi + 1] = True

    return bad


def interpolate_bad(poses: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """用前后有效帧线性插值替换异常帧。"""
    result = poses.copy()
    good = ~bad
    vi = np.where(good)[0]
    if len(vi) < 2:
        return result
    ii = np.where(bad)[0]
    for d in range(poses.shape[1]):
        result[ii, d] = np.interp(ii, vi, poses[vi, d])
    return result


def smooth_xyz(xyz: np.ndarray, method: str, window: int) -> np.ndarray:
    """仅对 XYZ (T, 3) 做平滑，不触碰 RPY。"""
    T = len(xyz)
    win = min(window, T if T % 2 == 1 else T - 1)
    if win < 5 or method == "none":
        return xyz
    s = xyz.copy()
    for d in range(xyz.shape[1]):
        if method == "savgol":
            s[:, d] = savgol_filter(xyz[:, d], win, 3)
        elif method == "median_then_savgol":
            s[:, d] = median_filter(xyz[:, d], size=5)
            s[:, d] = savgol_filter(s[:, d], win, 3)
    return s.astype(np.float32)


def postprocess_side(poses: np.ndarray, jump_thresh_m: float,
                     smooth_method: str, smooth_window: int,
                     context: int) -> tuple[np.ndarray, int]:
    """
    处理单侧（left/right）的 poses (T, 6)，返回 (cleaned_poses, n_fixed_frames)。

    策略：
    - XYZ：异常帧检测 → 插值 → 平滑
    - RPY：仅在 XYZ 异常帧处插值（那些帧旋转也不可信），不做额外平滑
      （HaWoR 旋转质量本身较好，平滑反而引入误差）
    """
    result = poses.copy()
    xyz = poses[:, :3]
    rpy = poses[:, 3:6]

    bad = detect_jump_outliers(xyz, jump_thresh_m, context)
    n_fixed = int(bad.sum())

    if n_fixed > 0:
        # XYZ：插值异常帧
        result[:, :3] = interpolate_bad(xyz, bad)
        # RPY：仅插值 XYZ 异常帧（旋转在这些帧同样不可信）
        result[:, 3:6] = interpolate_bad(rpy, bad)

    # 平滑只作用于 XYZ
    result[:, :3] = smooth_xyz(result[:, :3], smooth_method, smooth_window)

    return result, n_fixed


def process_file(input_path: Path, output_path: Path, args) -> dict:
    data = dict(np.load(str(input_path)))

    jump_thresh_m = args.jump_thresh_mm / 1000.0

    results = {}
    for side in ("left", "right"):
        key = f"{side}_wrist_poses"
        if key not in data:
            continue
        original = data[key].astype(np.float32)
        cleaned, n_fixed = postprocess_side(
            original, jump_thresh_m, args.smooth_method,
            args.smooth_window, args.context
        )
        data[key] = cleaned
        results[side] = {"frames": len(original), "fixed": n_fixed}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **data)
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="HaWoR wrist_poses.npz 后处理：去除大跳变")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",     help="单个 .npz 文件路径")
    src.add_argument("--input_dir", help="包含多个 *_wrist_poses.npz 或 wrist_poses.npz 的目录")

    dst = ap.add_mutually_exclusive_group()
    dst.add_argument("--output",     help="单文件输出路径（与 --input 配合）")
    dst.add_argument("--output_dir", help="批量输出目录（与 --input_dir 配合）")
    dst.add_argument("--inplace",    action="store_true",
                     help="直接覆盖原文件（与 --input_dir 配合）")

    ap.add_argument("--jump_thresh_mm", type=float, default=50.0,
                    help="XYZ 帧间跳变阈值（mm），超过则视为异常帧（默认 50mm）")
    ap.add_argument("--context", type=int, default=1,
                    help="异常帧前后额外标记的帧数（默认 1）")
    ap.add_argument("--smooth_method", default="median_then_savgol",
                    choices=["savgol", "median_then_savgol", "none"],
                    help="平滑方法（默认 median_then_savgol）")
    ap.add_argument("--smooth_window", type=int, default=11,
                    help="Savitzky-Golay 窗口大小，必须为奇数（默认 11）")
    ap.add_argument("--dry_run", action="store_true",
                    help="只统计，不写文件")

    args = ap.parse_args()

    # 确保窗口为奇数
    if args.smooth_window % 2 == 0:
        args.smooth_window += 1

    # 收集输入文件
    if args.input:
        input_files = [Path(args.input)]
    else:
        d = Path(args.input_dir)
        input_files = sorted(d.rglob("*wrist_poses*.npz"))
        if not input_files:
            input_files = sorted(d.rglob("*.npz"))
        if not input_files:
            raise FileNotFoundError(f"在 {d} 下未找到任何 .npz 文件")

    print(f"找到 {len(input_files)} 个文件")
    print(f"参数: jump_thresh={args.jump_thresh_mm}mm  smooth={args.smooth_method}"
          f"  window={args.smooth_window}  context=±{args.context}帧")
    if args.dry_run:
        print("[dry_run] 不写文件\n")

    total_fixed = {"left": 0, "right": 0}

    for fp in input_files:
        # 确定输出路径
        if args.dry_run:
            out = fp  # 不会实际写入
        elif args.input:
            out = Path(args.output) if args.output else fp.with_name(fp.stem + "_clean.npz")
        elif args.inplace:
            out = fp
        else:
            out_dir = Path(args.output_dir)
            out = out_dir / fp.name

        if args.dry_run:
            # 只统计不写
            data = dict(np.load(str(fp)))
            jump_thresh_m = args.jump_thresh_mm / 1000.0
            for side in ("left", "right"):
                key = f"{side}_wrist_poses"
                if key not in data:
                    continue
                poses = data[key].astype(np.float32)
                bad = detect_jump_outliers(poses[:, :3], jump_thresh_m, args.context)
                n = int(bad.sum())
                total_fixed[side] += n
                if n > 0:
                    print(f"  {fp.name}  {side}: {n} 帧异常")
        else:
            results = process_file(fp, out, args)
            parts = []
            for side, r in results.items():
                total_fixed[side] += r["fixed"]
                parts.append(f"{side}: {r['fixed']}/{r['frames']} 帧修复")
            status = "  ".join(parts)
            arrow = f" -> {out}" if out != fp else " [inplace]"
            print(f"  {fp.name}{arrow}  {status}")

    print(f"\n汇总: left 修复 {total_fixed['left']} 帧, right 修复 {total_fixed['right']} 帧")


if __name__ == "__main__":
    main()
