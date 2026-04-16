#!/usr/bin/env python3
"""
Visualize HaWoR wrist poses overlaid on ego camera frames.
Draws MANO wrist position axes and pose text on each frame.

Usage:
    python visualize_hawor_poses.py \
        --data_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/test/0 \
        --poses wrist_poses.npz \
        --output hawor_viz.mp4 \
        --ego_cam_id 07
"""

import argparse, json
import numpy as np
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R

DEPTH_SCALE = 0.001


def get_by_alias(mapping, aliases):
    if not isinstance(mapping, dict):
        return None
    normalized = {str(k).lower(): v for k, v in mapping.items()}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def normalize_intrinsic(node):
    if node is None or not isinstance(node, dict):
        return None
    intrinsic = get_by_alias(node, ["intrinsic", "intrinsics"]) or node
    if not isinstance(intrinsic, dict):
        return None
    result = {}
    for key in ["fx", "fy", "cx", "cy", "width", "height"]:
        v = get_by_alias(intrinsic, [key])
        if v is not None:
            result[key] = v
    return result if all(k in result for k in ["fx", "fy", "cx", "cy"]) else None


def load_cam_params(data_dir, cam_id):
    with open(Path(data_dir) / "camera_params.json") as f:
        p = json.load(f)
    cam = get_by_alias(p, [cam_id, str(int(cam_id)) if str(cam_id).isdigit() else cam_id])
    if cam is None:
        raise KeyError(f"Camera {cam_id} not found")
    rgb_intr = (
        normalize_intrinsic(get_by_alias(cam, ["RGB", "rgb", "color", "Color"]))
        or normalize_intrinsic(get_by_alias(cam, ["rgb_intrinsic", "color_intrinsic"]))
        or normalize_intrinsic(cam)
    )
    depth_intr = (
        normalize_intrinsic(get_by_alias(cam, ["Depth", "depth"]))
        or normalize_intrinsic(get_by_alias(cam, ["depth_intrinsic"]))
        or rgb_intr
    )
    return {"rgb": rgb_intr, "depth": depth_intr or rgb_intr}


def proj(pt3d, intr):
    x, y, z = pt3d
    if z < 1e-4:
        return None
    u = int(x / z * intr["fx"] + intr["cx"])
    v = int(y / z * intr["fy"] + intr["cy"])
    return (u, v)


def draw_axes(img, pos, euler, intr, length=0.05):
    rot = R.from_euler("xyz", euler).as_matrix()
    o = proj(pos, intr)
    if o is None:
        return
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue
    for i, color in enumerate(colors):
        ep = proj(pos + rot[:, i] * length, intr)
        if ep is not None:
            cv2.arrowedLine(img, o, ep, color, 2, tipLength=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--poses",      required=True)
    ap.add_argument("--output",     default="hawor_viz.mp4")
    ap.add_argument("--ego_cam_id", default="07")
    ap.add_argument("--fps",        type=float, default=30.0)
    ap.add_argument("--max_frames", type=int,   default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    params   = load_cam_params(data_dir, args.ego_cam_id)
    intr     = params["depth"]

    rgb_files = sorted((data_dir / args.ego_cam_id / "RGB").glob("*.jpg"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB frames in {data_dir / args.ego_cam_id / 'RGB'}")

    sample = cv2.imread(str(rgb_files[0]))
    H, W   = sample.shape[:2]
    intr.setdefault("height", H)
    intr.setdefault("width",  W)

    data        = np.load(args.poses)
    left_poses  = data["left_wrist_poses"]
    right_poses = data["right_wrist_poses"]
    left_valid  = data["left_valid"]
    right_valid = data["right_valid"]
    left_gripper  = data.get("left_gripper",  np.zeros(len(left_poses)))
    right_gripper = data.get("right_gripper", np.zeros(len(right_poses)))

    if args.max_frames:
        rgb_files = rgb_files[:args.max_frames]
    T = len(rgb_files)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))

    for i, rgb_path in enumerate(rgb_files):
        if i % 50 == 0:
            print(f"  frame {i}/{T}")

        img = cv2.imread(str(rgb_path))

        for side, poses, valid, gripper in [
            ("left",  left_poses,  left_valid,  left_gripper),
            ("right", right_poses, right_valid, right_gripper),
        ]:
            if i >= len(poses):
                continue
            pos   = poses[i, :3]
            euler = poses[i, 3:6]
            color = (0, 255, 255) if side == "left" else (255, 165, 0)

            draw_axes(img, pos, euler, intr, length=0.05)

            o = proj(pos, intr)
            if o is not None:
                is_valid = bool(valid[i]) if i < len(valid) else False
                label  = f"{side[0].upper()} xyz=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})m"
                g_label = f"grip={gripper[i]:.3f}m"
                status = "DET" if is_valid else "INTERP"
                s_color = (0, 220, 0) if is_valid else (0, 140, 255)
                tx = min(o[0] + 12, W - 220)
                ty = o[1]
                cv2.putText(img, label,   (tx, ty),      cv2.FONT_HERSHEY_SIMPLEX, 0.42, color,   1, cv2.LINE_AA)
                cv2.putText(img, g_label, (tx, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color,   1, cv2.LINE_AA)
                cv2.putText(img, status,  (tx, ty + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, s_color, 1, cv2.LINE_AA)

        cv2.putText(img, f"Frame {i:04d}/{T}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "HaWoR | Axes: X=red Y=green Z=blue", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
