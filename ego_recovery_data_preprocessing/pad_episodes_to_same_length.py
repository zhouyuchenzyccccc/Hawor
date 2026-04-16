#!/usr/bin/env python3
"""
将数据集中所有 episode 的图片帧数补齐到最长 episode 的帧数。

补齐策略：均匀间隔插入重复帧——对于每个目标位置 i，映射到原始帧
  floor(i * original_len / target_len)，即"当前位置对应的最近前一帧"。
  RGB 和深度图使用相同的逻辑。

目录结构（每个 episode 下含 06/07/08 三个相机，每相机含 RGB/ 和 Depth/）：
    data_dir/
    ├── episode_0/
    │   ├── 06/
    │   │   ├── RGB/    (*.jpg / *.png)
    │   │   └── Depth/  (*.png)
    │   ├── 07/
    │   │   ├── RGB/
    │   │   └── Depth/
    │   └── 08/
    │       ├── RGB/
    │       └── Depth/
    ├── episode_1/
    └── ...

用法：
  # 原地修改（直接覆盖原始 episode 目录）：
  python pad_episodes_to_same_length.py --data_dir /path/to/dataset

  # 写到新目录（保留原始数据）：
  python pad_episodes_to_same_length.py --data_dir /path/to/dataset --output_dir /path/to/output

  # 预览，不写入任何文件：
  python pad_episodes_to_same_length.py --data_dir /path/to/dataset --dry_run
"""
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


CAMERA_IDS = ["06", "07", "08"]
MODALITIES = ["RGB", "Depth"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ── 文件发现 ────────────────────────────────────────────────────────────────

def find_image_files(directory: Path) -> List[Path]:
    """返回目录中按名称排序的图片文件列表。"""
    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def is_episode_dir(path: Path) -> bool:
    """判断目录是否为 episode（至少含一个相机子目录）。"""
    return path.is_dir() and any((path / cam_id).is_dir() for cam_id in CAMERA_IDS)


def find_episode_dirs(data_dir: Path) -> List[Path]:
    """返回 data_dir 下按名称排序的所有 episode 目录。"""
    return sorted(d for d in data_dir.iterdir() if is_episode_dir(d))


def get_episode_frame_count(episode_dir: Path) -> int:
    """返回 episode 的帧数（取第一个找到的相机/模态）。"""
    for cam_id in CAMERA_IDS:
        for modality in MODALITIES:
            img_dir = episode_dir / cam_id / modality
            if img_dir.is_dir():
                files = find_image_files(img_dir)
                if files:
                    return len(files)
    raise ValueError(f"未找到图片文件：{episode_dir}")


# ── 插值索引计算 ─────────────────────────────────────────────────────────────

def compute_padded_indices(original_len: int, target_len: int) -> np.ndarray:
    """
    计算从 target_len 个位置到 original_len 帧的映射索引。

    使用 floor(i * original_len / target_len)，保证插入帧始终是前一帧的副本。
    示例：original_len=3, target_len=6 → [0, 0, 1, 1, 2, 2]
          original_len=4, target_len=7 → [0, 0, 1, 2, 2, 3, 3]
    """
    if original_len == target_len:
        return np.arange(original_len)
    return np.floor(np.arange(target_len) * original_len / target_len).astype(int)


# ── 文件复制 ─────────────────────────────────────────────────────────────────

def pad_camera_modality_dir(
    src_dir: Path, dst_dir: Path, indices: np.ndarray
) -> None:
    """按 indices 映射将 src_dir 的图片写入 dst_dir（保持六位数字命名）。"""
    src_files = find_image_files(src_dir)
    if not src_files:
        return
    ext = src_files[0].suffix
    dst_dir.mkdir(parents=True, exist_ok=True)
    for new_idx, orig_idx in enumerate(indices):
        shutil.copy2(src_files[orig_idx], dst_dir / f"{new_idx:06d}{ext}")


def copy_non_camera_files(src_dir: Path, dst_dir: Path) -> None:
    """将 episode 目录中非相机子目录的文件/目录复制到 dst_dir。"""
    for item in src_dir.iterdir():
        if item.name in CAMERA_IDS:
            continue
        dest = dst_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest)


# ── episode 处理 ─────────────────────────────────────────────────────────────

def process_episode(
    episode_dir: Path,
    target_len: int,
    output_dir: Optional[Path],
    dry_run: bool,
) -> None:
    """将单个 episode 补齐到 target_len 帧。"""
    current_len = get_episode_frame_count(episode_dir)

    if current_len == target_len:
        print(f"  [SKIP] {episode_dir.name}: {current_len} 帧（已是目标长度）")
        if output_dir is not None and not dry_run:
            dest = output_dir / episode_dir.name
            if not dest.exists():
                shutil.copytree(episode_dir, dest)
        return

    n_added = target_len - current_len
    indices = compute_padded_indices(current_len, target_len)
    print(
        f"  [PAD]  {episode_dir.name}: {current_len} → {target_len} 帧 "
        f"（插入 {n_added} 帧）"
    )

    if dry_run:
        return

    if output_dir is not None:
        # ── 写出模式 ────────────────────────────────────────────────────────
        dest_dir = output_dir / episode_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        copy_non_camera_files(episode_dir, dest_dir)
        for cam_id in CAMERA_IDS:
            for modality in MODALITIES:
                src_cam_dir = episode_dir / cam_id / modality
                if src_cam_dir.is_dir():
                    pad_camera_modality_dir(
                        src_cam_dir, dest_dir / cam_id / modality, indices
                    )
    else:
        # ── 原地模式：先写临时目录，再原子替换 ─────────────────────────────
        tmp_dir = episode_dir.parent / f"_tmp_{episode_dir.name}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        try:
            tmp_dir.mkdir(parents=True)
            copy_non_camera_files(episode_dir, tmp_dir)
            for cam_id in CAMERA_IDS:
                for modality in MODALITIES:
                    src_cam_dir = episode_dir / cam_id / modality
                    if src_cam_dir.is_dir():
                        pad_camera_modality_dir(
                            src_cam_dir, tmp_dir / cam_id / modality, indices
                        )
            # 原子替换
            backup_dir = episode_dir.parent / f"_backup_{episode_dir.name}"
            episode_dir.rename(backup_dir)
            tmp_dir.rename(episode_dir)
            shutil.rmtree(backup_dir)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="将数据集所有 episode 的图片帧数补齐到最长 episode 的帧数。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir", required=True, type=Path,
        help="包含各 episode 子目录的数据集根目录。",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="将补齐后的数据集写入此目录（不设则原地修改）。",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="仅打印操作计划，不写入任何文件。",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    # ── 发现 episode ─────────────────────────────────────────────────────────
    print(f"扫描目录：{data_dir}")
    episode_dirs = find_episode_dirs(data_dir)
    if not episode_dirs:
        print("未找到 episode 目录，退出。")
        return

    frame_counts: Dict[Path, int] = {}
    for ep in episode_dirs:
        try:
            frame_counts[ep] = get_episode_frame_count(ep)
        except ValueError as exc:
            print(f"  [WARN] {exc}")

    if not frame_counts:
        print("无有效 episode，退出。")
        return

    max_len = max(frame_counts.values())
    max_ep = max(frame_counts, key=frame_counts.get)

    # ── 打印概览 ─────────────────────────────────────────────────────────────
    print(f"\n共 {len(frame_counts)} 个有效 episode")
    print(f"基准（最长）：{max_ep.name} — {max_len} 帧\n")
    print("各 episode 帧数：")
    for ep in sorted(frame_counts):
        marker = " ← 最长" if ep == max_ep else ""
        print(f"  {ep.name:<30s} {frame_counts[ep]:5d} 帧{marker}")

    print(f"\n目标帧数：{max_len}")
    if args.dry_run:
        print("（DRY RUN — 不写入任何文件）\n")

    if args.output_dir is not None and not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 逐 episode 处理 ──────────────────────────────────────────────────────
    print()
    for ep in sorted(frame_counts):
        process_episode(ep, max_len, args.output_dir, dry_run=args.dry_run)

    print("\n完成。")


if __name__ == "__main__":
    main()
