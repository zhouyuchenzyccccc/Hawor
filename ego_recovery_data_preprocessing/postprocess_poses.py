#!/usr/bin/env python3
"""
对 HaWoR 输出的 wrist_poses.npz 做后处理，消除大跳变。

流程：
  XYZ：异常帧检测（>jump_thresh_mm）→ 插值 → 平滑
  RPY：异常帧检测（>rpy_jump_thresh_deg，可选）→ 插值 → 平滑（unwrap→smooth→wrap）

用法（先分析分布，再决定阈值）：
  python postprocess_poses.py --input_dir /path/to/poses --analyze

用法（单文件）：
  python postprocess_poses.py --input wrist_poses.npz --output wrist_poses_clean.npz

用法（批量，输出到新目录）：
  python postprocess_poses.py --input_dir /path/to/poses_dir --output_dir /path/to/clean_dir \\
      --jump_thresh_mm 30 --rpy_jump_thresh_deg 20
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter


# ── 核心后处理 ──────────────────────────────────────────────────────────────

def detect_jump_outliers(signal: np.ndarray, jump_thresh: float,
                         context: int = 1) -> np.ndarray:
    """
    检测帧间跳变异常帧，返回 bool mask（True = 异常）。
    signal: (T, D)，按行范数计算跳变；或 (T,) 标量序列。
    jump_thresh: 跳变阈值（与 signal 单位一致）。
    context: 异常帧前后各额外标记多少帧。
    """
    T = len(signal)
    bad = np.zeros(T, dtype=bool)
    if T < 2:
        return bad

    diff = np.diff(signal, axis=0)
    jumps = np.linalg.norm(diff, axis=1) if diff.ndim == 2 else np.abs(diff)

    for i in np.where(jumps > jump_thresh)[0]:
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
    """对 XYZ (T, 3) 做平滑。"""
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


def smooth_rpy(rpy: np.ndarray, method: str, window: int) -> np.ndarray:
    """
    对 RPY (T, 3) 做平滑，先 unwrap 再平滑再 wrap 回 [-π, π]。
    unwrap 避免 ±180° 边界导致平滑失真。
    """
    T = len(rpy)
    win = min(window, T if T % 2 == 1 else T - 1)
    if win < 5 or method == "none":
        return rpy
    unwrapped = np.unwrap(rpy, axis=0)  # (T, 3)
    s = unwrapped.copy()
    for d in range(3):
        if method == "savgol":
            s[:, d] = savgol_filter(unwrapped[:, d], win, 3)
        elif method == "median_then_savgol":
            s[:, d] = median_filter(unwrapped[:, d], size=5)
            s[:, d] = savgol_filter(s[:, d], win, 3)
    # wrap 回 [-π, π]
    s = ((s + np.pi) % (2 * np.pi) - np.pi)
    return s.astype(np.float32)


def postprocess_side(poses: np.ndarray, jump_thresh_m: float,
                     smooth_method: str, smooth_window: int, context: int,
                     rpy_jump_thresh_rad: float | None = None) -> tuple[np.ndarray, int, int]:
    """
    处理单侧 poses (T, 6)，返回 (cleaned_poses, n_xyz_fixed, n_rpy_fixed)。

    XYZ：异常帧检测 → 插值 → 平滑
    RPY：
      - 始终在 XYZ 异常帧处插值（那些帧旋转同样不可信）
      - 若设置 rpy_jump_thresh_rad，额外检测 RPY 自身的异常帧并插值
      - 最后做平滑（unwrap → smooth → wrap）
    """
    result = poses.copy()
    xyz = poses[:, :3]
    rpy = poses[:, 3:6]

    # XYZ 处理
    xyz_bad = detect_jump_outliers(xyz, jump_thresh_m, context)
    n_xyz_fixed = int(xyz_bad.sum())
    if n_xyz_fixed > 0:
        result[:, :3] = interpolate_bad(xyz, xyz_bad)
        result[:, 3:6] = interpolate_bad(rpy, xyz_bad)

    # RPY 额外异常帧检测
    n_rpy_fixed = 0
    if rpy_jump_thresh_rad is not None:
        # 对插值后的 RPY 做检测，避免重复计算 XYZ 已修复的帧
        rpy_current = result[:, 3:6]
        # 角度差归一化到 [-π, π]
        rpy_diff = np.diff(rpy_current, axis=0)
        rpy_diff = (rpy_diff + np.pi) % (2 * np.pi) - np.pi
        rpy_jump_norm = np.linalg.norm(rpy_diff, axis=1)
        rpy_bad = np.zeros(len(rpy_current), dtype=bool)
        for i in np.where(rpy_jump_norm > rpy_jump_thresh_rad)[0]:
            lo = max(0, i + 1 - context)
            hi = min(len(rpy_current) - 1, i + 1 + context)
            rpy_bad[lo:hi + 1] = True
        # 排除已被 XYZ 修复的帧，只统计新增的
        n_rpy_fixed = int((rpy_bad & ~xyz_bad).sum())
        if rpy_bad.any():
            result[:, 3:6] = interpolate_bad(result[:, 3:6], rpy_bad)

    # 平滑
    result[:, :3] = smooth_xyz(result[:, :3], smooth_method, smooth_window)
    result[:, 3:6] = smooth_rpy(result[:, 3:6], smooth_method, smooth_window)

    return result, n_xyz_fixed, n_rpy_fixed


def process_file(input_path: Path, output_path: Path, args) -> dict:
    data = dict(np.load(str(input_path)))

    jump_thresh_m = args.jump_thresh_mm / 1000.0
    rpy_thresh_rad = (args.rpy_jump_thresh_deg * np.pi / 180.0
                      if args.rpy_jump_thresh_deg is not None else None)

    results = {}
    for side in ("left", "right"):
        key = f"{side}_wrist_poses"
        if key not in data:
            continue
        original = data[key].astype(np.float32)
        cleaned, n_xyz, n_rpy = postprocess_side(
            original, jump_thresh_m, args.smooth_method,
            args.smooth_window, args.context, rpy_thresh_rad
        )
        data[key] = cleaned
        results[side] = {"frames": len(original), "xyz_fixed": n_xyz, "rpy_fixed": n_rpy}

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
    ap.add_argument("--rpy_jump_thresh_deg", type=float, default=None,
                    help="RPY 帧间跳变阈值（deg），不设则不单独检测 RPY 异常帧")
    ap.add_argument("--context", type=int, default=1,
                    help="异常帧前后额外标记的帧数（默认 1）")
    ap.add_argument("--smooth_method", default="median_then_savgol",
                    choices=["savgol", "median_then_savgol", "none"],
                    help="平滑方法，同时作用于 XYZ 和 RPY（默认 median_then_savgol）")
    ap.add_argument("--smooth_window", type=int, default=11,
                    help="Savitzky-Golay 窗口大小，必须为奇数（默认 11）")
    ap.add_argument("--dry_run", action="store_true",
                    help="只统计，不写文件")
    ap.add_argument("--analyze", action="store_true",
                    help="打印跳变分布直方图（P50/P90/P95/P99/max），帮助选择合适阈值")

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
    rpy_info = f"  rpy_thresh={args.rpy_jump_thresh_deg}deg" if args.rpy_jump_thresh_deg else ""
    print(f"参数: jump_thresh={args.jump_thresh_mm}mm{rpy_info}  smooth={args.smooth_method}"
          f"  window={args.smooth_window}  context=±{args.context}帧")
    if args.dry_run:
        print("[dry_run] 不写文件\n")

    # --analyze：打印跳变分布，帮助选阈值
    if args.analyze:
        all_xyz_jumps = {"left": [], "right": []}
        all_rpy_jumps = {"left": [], "right": []}
        for fp in input_files:
            data = dict(np.load(str(fp)))
            for side in ("left", "right"):
                key = f"{side}_wrist_poses"
                if key not in data:
                    continue
                poses = data[key].astype(np.float32)
                if len(poses) < 2:
                    continue
                xyz = poses[:, :3]
                rpy = poses[:, 3:6]
                all_xyz_jumps[side].append(
                    np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000.0)
                # RPY 差分归一化到 [-180, 180]
                rpy_diff = np.diff(rpy, axis=0) * (180.0 / np.pi)
                rpy_diff = (rpy_diff + 180.0) % 360.0 - 180.0
                all_rpy_jumps[side].append(
                    np.linalg.norm(rpy_diff, axis=1))

        xyz_thresholds = [10, 20, 30, 50, 100]
        rpy_thresholds = [5, 10, 20, 30, 46]

        print("\n=== XYZ 跳变分布（mm）===")
        for side in ("left", "right"):
            if not all_xyz_jumps[side]:
                continue
            j = np.concatenate(all_xyz_jumps[side])
            print(f"\n  [{side.upper()}]  样本数: {len(j)}")
            print(f"    P50={np.percentile(j,50):.1f}  P90={np.percentile(j,90):.1f}"
                  f"  P95={np.percentile(j,95):.1f}  P99={np.percentile(j,99):.1f}"
                  f"  max={j.max():.1f}")
            for t in xyz_thresholds:
                cnt = int(np.sum(j > t))
                print(f"      >{t:3d}mm: {cnt:6d} 帧  ({100*cnt/len(j):.2f}%)")

        print("\n=== RPY 跳变分布（deg）===")
        for side in ("left", "right"):
            if not all_rpy_jumps[side]:
                continue
            j = np.concatenate(all_rpy_jumps[side])
            print(f"\n  [{side.upper()}]  样本数: {len(j)}")
            print(f"    P50={np.percentile(j,50):.2f}  P90={np.percentile(j,90):.2f}"
                  f"  P95={np.percentile(j,95):.2f}  P99={np.percentile(j,99):.2f}"
                  f"  max={j.max():.2f}")
            for t in rpy_thresholds:
                cnt = int(np.sum(j > t))
                print(f"      >{t:2d}deg: {cnt:6d} 帧  ({100*cnt/len(j):.2f}%)")
        print()
        if not (args.dry_run or args.inplace or args.output or args.output_dir):
            return

    total_xyz_fixed = {"left": 0, "right": 0}
    total_rpy_fixed = {"left": 0, "right": 0}

    for fp in input_files:
        # 确定输出路径
        if args.dry_run:
            out = fp
        elif args.input:
            out = Path(args.output) if args.output else fp.with_name(fp.stem + "_clean.npz")
        elif args.inplace:
            out = fp
        else:
            out_dir = Path(args.output_dir)
            out = out_dir / fp.name

        if args.dry_run:
            data = dict(np.load(str(fp)))
            jump_thresh_m = args.jump_thresh_mm / 1000.0
            rpy_thresh_rad = (args.rpy_jump_thresh_deg * np.pi / 180.0
                              if args.rpy_jump_thresh_deg is not None else None)
            for side in ("left", "right"):
                key = f"{side}_wrist_poses"
                if key not in data:
                    continue
                poses = data[key].astype(np.float32)
                _, n_xyz, n_rpy = postprocess_side(
                    poses, jump_thresh_m, args.smooth_method,
                    args.smooth_window, args.context, rpy_thresh_rad)
                total_xyz_fixed[side] += n_xyz
                total_rpy_fixed[side] += n_rpy
                if n_xyz > 0 or n_rpy > 0:
                    print(f"  {fp.name}  {side}: xyz={n_xyz} rpy={n_rpy} 帧异常")
        else:
            results = process_file(fp, out, args)
            parts = []
            for side, r in results.items():
                total_xyz_fixed[side] += r["xyz_fixed"]
                total_rpy_fixed[side] += r["rpy_fixed"]
                parts.append(f"{side}: xyz={r['xyz_fixed']} rpy={r['rpy_fixed']}/{r['frames']}")
            arrow = f" -> {out}" if out != fp else " [inplace]"
            print(f"  {fp.name}{arrow}  {'  '.join(parts)}")

    print(f"\n汇总:")
    print(f"  left  XYZ修复={total_xyz_fixed['left']} RPY修复={total_rpy_fixed['left']}")
    print(f"  right XYZ修复={total_xyz_fixed['right']} RPY修复={total_rpy_fixed['right']}")


if __name__ == "__main__":
    main()
