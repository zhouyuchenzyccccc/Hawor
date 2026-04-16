#!/usr/bin/env python3
"""
将图片序列合成为视频。

用法示例：
    python images_to_video.py --input_dir ./frames --output_video ./output.mp4 --fps 30
"""

import argparse
import glob
import os
import re
import cv2
import numpy as np

def natural_sort_key(s):
    """
    用于自然排序的键函数，例如将 "frame10.jpg" 排在 "frame2.jpg" 之后。
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def find_image_files(input_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """
    递归或非递归地查找所有支持的图片文件，并返回按自然排序后的路径列表。
    """
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f'*{ext}')
        image_paths.extend(glob.glob(pattern))
        # 如果需要递归子目录，可以取消下面一行的注释
        # image_paths.extend(glob.glob(os.path.join(input_dir, '**', f'*{ext}'), recursive=True))
    # 去重并排序
    image_paths = sorted(set(image_paths), key=natural_sort_key)
    return image_paths

def main():
    parser = argparse.ArgumentParser(description='将文件夹中的图片合成为视频。')
    parser.add_argument('--input_dir', required=True, help='包含图片的输入文件夹路径')
    parser.add_argument('--output_video', required=True, help='输出视频文件路径（例如 output.mp4）')
    parser.add_argument('--fps', type=float, default=30.0, help='输出视频的帧率，默认 30')
    parser.add_argument('--codec', default='mp4v', help='编码器，默认 mp4v（对应 .mp4），可选 avc1, X264 等')
    parser.add_argument('--resize', type=str, default=None, help='调整图片尺寸，格式 "宽x高"，例如 640x480')
    parser.add_argument('--no_resize', action='store_true', help='遇到尺寸不一致的图片时跳过（不调整尺寸，可能导致写入失败）')
    args = parser.parse_args()

    # 查找所有图片文件
    image_paths = find_image_files(args.input_dir)
    if not image_paths:
        print(f"错误：在 {args.input_dir} 中没有找到支持的图片文件（{', '.join(find_image_files.__defaults__[0])}）")
        return

    print(f"找到 {len(image_paths)} 张图片，开始合成视频...")

    # 读取第一张图片以获取尺寸
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        print(f"错误：无法读取第一张图片 {image_paths[0]}")
        return
    height, width = first_frame.shape[:2]

    # 如果指定了 resize，则覆盖目标尺寸
    if args.resize:
        try:
            width, height = map(int, args.resize.lower().split('x'))
        except ValueError:
            print("错误：--resize 参数格式应为 '宽x高'，例如 640x480")
            return

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))

    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"警告：跳过无法读取的图片 {img_path}")
            continue

        # 调整尺寸（如果必要）
        if args.resize or (frame.shape[1] != width or frame.shape[0] != height):
            if args.no_resize:
                print(f"警告：图片 {img_path} 尺寸 {frame.shape[1]}x{frame.shape[0]} 与期望 {width}x{height} 不一致，已跳过")
                continue
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        out.write(frame)
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(image_paths)} 帧")

    out.release()
    print(f"视频已保存至: {args.output_video}")

if __name__ == '__main__':
    main()