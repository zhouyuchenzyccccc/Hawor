#!/usr/bin/env python3
"""
Extract 3D wrist poses using HaWoR (CVPR 2025 Highlight).

HaWoR provides temporally consistent MANO hand reconstruction with built-in
temporal motion priors. Output format is identical to extract_wrist_pose.py
so it is plug-in compatible with convert_to_lerobot.py.

Pipeline:
  1. Encode ego camera frames to temp mp4 (HaWoR expects video input)
  2. detect_track_video  -> detection + tracking
  3. hawor_motion_estimation -> per-frame camera-space MANO params (saved as JSON)
  4. Create identity-camera dummy SLAM (world = camera frame, no DROID-SLAM needed)
  5. hawor_infiller -> temporally filled MANO params
  6. run_mano / run_mano_left -> wrist position + joints for gripper
  7. Optional Orbbec depth anchoring for metric scale
  8. Interpolate + optional extra smoothing -> wrist_poses.npz

MANO joints in OpenPose order (0=wrist, 1-4=thumb, 5-8=index, ...):
  thumb_tip = joint 4,  index_tip = joint 8
  Gripper = ||thumb_tip - index_tip||

Usage:
  python hawor_extract_poses.py \
      --data_dir /home/ubuntu/WorkSpace/ZYC/dataset/hand_insertion/test/0 \
      --output wrist_poses.npz \
      --ego_cam_id 07
"""

import argparse, json, sys, os, tempfile, subprocess, shutil, types
import numpy as np
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

DEPTH_SCALE = 0.001   # Orbbec: uint16 mm -> meters
THUMB_TIP   = 4       # OpenPose hand joint index
INDEX_TIP   = 8       # OpenPose hand joint index
HAND2IDX    = {"left": 0, "right": 1}  # HaWoR tid convention


# ── camera params helpers (identical to extract_wrist_pose.py) ─────────────

def sort_key(path):
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

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
        raise KeyError(f"Camera {cam_id} not found in camera_params.json")
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
    if rgb_intr is None:
        raise KeyError(f"Could not parse intrinsics for camera {cam_id}")
    return {"rgb": rgb_intr, "depth": depth_intr or rgb_intr}


# ── sequence discovery ──────────────────────────────────────────────────────

def is_sequence_dir(path, cam_id):
    return (path / "camera_params.json").is_file() and (path / cam_id / "RGB").is_dir()

def discover_sequence_dirs(root_dir, cam_id):
    root_dir = Path(root_dir)
    if is_sequence_dir(root_dir, cam_id):
        return [root_dir]
    seq_dirs = sorted(
        [p for p in root_dir.iterdir() if p.is_dir() and is_sequence_dir(p, cam_id)],
        key=sort_key,
    )
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence directories found under {root_dir}")
    return seq_dirs

def resolve_output_path(output_arg, seq_dir, batch_mode):
    if not output_arg:
        return seq_dir / "wrist_poses_hawor.npz"
    output_path = Path(output_arg)
    if not batch_mode and output_path.suffix == ".npz":
        return output_path
    if batch_mode and output_path.suffix == ".npz":
        batch_dir = output_path.parent if str(output_path.parent) not in ("", ".") else Path.cwd()
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir / f"{seq_dir.name}_{output_path.name}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{seq_dir.name}_wrist_poses.npz"


# ── depth helpers ───────────────────────────────────────────────────────────

def get_depth_at(depth_img, u, v, H, W, search_r=4):
    for r in range(search_r + 1):
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                if abs(du) != r and abs(dv) != r:
                    continue
                nu = int(np.clip(u + du, 0, W - 1))
                nv = int(np.clip(v + dv, 0, H - 1))
                d = depth_img[nv, nu]
                if 100 < d < 3000:
                    return d
    return 0

def project_3d_to_px(pt3d, intr):
    x, y, z = pt3d
    if z < 1e-4:
        return None
    return int(x / z * intr["fx"] + intr["cx"]), int(y / z * intr["fy"] + intr["cy"])


# ── smoothing / interpolation ───────────────────────────────────────────────

def interpolate_invalid(poses, valid):
    result = poses.copy()
    vi = np.where(valid)[0]
    if len(vi) < 2:
        return result
    ii = np.where(~valid)[0]
    for d in range(poses.shape[1]):
        result[ii, d] = np.interp(ii, vi, poses[vi, d])
    return result

def smooth_poses(poses, method):
    T = len(poses)
    win = min(11, T if T % 2 == 1 else T - 1)
    if win < 5 or method == "none":
        return poses
    s = poses.copy()
    for d in range(poses.shape[1]):
        if method == "savgol":
            s[:, d] = savgol_filter(poses[:, d], win, 3)
        elif method == "ema":
            for t in range(1, T):
                s[t, d] = 0.3 * poses[t, d] + 0.7 * s[t - 1, d]
        elif method == "median_then_savgol":
            s[:, d] = median_filter(poses[:, d], size=5)
            s[:, d] = savgol_filter(s[:, d], win, 3)
    return s


# ── HaWoR environment setup ─────────────────────────────────────────────────

def setup_hawor_env(hawor_dir):
    """
    Add HaWoR to sys.path and mock pytorch3d + lib.vis.renderer.
    The Renderer is only needed for mask visualisation, not for pose extraction.
    pytorch3d is optional; we replace it with lightweight dummies so imports work.
    """
    hd = str(hawor_dir)
    if hd not in sys.path:
        sys.path.insert(0, hd)

    # Mock pytorch3d sub-modules
    for mod in [
        "pytorch3d", "pytorch3d.renderer", "pytorch3d.structures",
        "pytorch3d.ops", "pytorch3d.loss", "pytorch3d.io",
        "pytorch3d.renderer.mesh", "pytorch3d.renderer.cameras",
        "pytorch3d.renderer.lighting", "pytorch3d.renderer.materials",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # Mock Renderer: render_multiple must return arrays so model_masks works
    class _MockRenderer:
        def __init__(self, W, H, *a, **kw):
            self.W, self.H = int(W), int(H)
        def create_camera_from_cv(self, *a, **kw):
            return None, None
        def render_multiple(self, *a, **kw):
            return (np.zeros((self.H, self.W, 3), dtype=np.uint8),
                    np.zeros((self.H, self.W), dtype=bool))

    _rmod = types.ModuleType("lib.vis.renderer")
    _rmod.Renderer = _MockRenderer
    sys.modules["lib.vis.renderer"] = _rmod


def encode_frames_to_video(rgb_dir, fps, tmp_video_path):
    """Encode sorted *.jpg frames to mp4 via ffmpeg concat demuxer."""
    frames = sorted(Path(rgb_dir).glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No jpg frames in {rgb_dir}")
    list_file = Path(tmp_video_path).parent / "_frame_list.txt"
    with open(list_file, "w") as f:
        for fr in frames:
            f.write(f"file '{fr.resolve()}'\n")
            f.write(f"duration {1.0 / fps:.6f}\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_file),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
           str(tmp_video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-600:]}")


def create_dummy_slam(slam_path, num_frames):
    """
    Identity-camera SLAM file for hawor_infiller without DROID-SLAM.
    With identity poses, world == camera frame, so output stays in camera space.

    load_slam_cam expects: traj[t] = [tx, ty, tz, qx, qy, qz, qw], scale=scalar
    """
    slam_path = Path(slam_path)
    slam_path.parent.mkdir(parents=True, exist_ok=True)
    traj = np.zeros((num_frames, 7), dtype=np.float32)
    traj[:, 6] = 1.0   # qw = 1 => identity rotation
    np.savez(str(slam_path), traj=traj, scale=np.float32(1.0))
    print(f"  Created dummy identity SLAM ({num_frames} frames): {slam_path}")


# ── main inference pipeline ─────────────────────────────────────────────────

def run_hawor_on_sequence(seq_dir, cam_params, rgb_files, args):
    """
    Full HaWoR pipeline for one sequence.
    Returns (pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid).
    """
    import argparse as _ap

    rgb_dir = seq_dir / args.ego_cam_id / "RGB"
    T = len(rgb_files)

    tmp_dir   = Path(tempfile.mkdtemp(prefix="hawor_"))
    tmp_video = tmp_dir / f"{seq_dir.name}.mp4"
    print(f"  Encoding {T} frames -> {tmp_video} ...")
    encode_frames_to_video(rgb_dir, args.fps, tmp_video)

    fx = float(cam_params["rgb"].get("fx", 600))
    hawk_args = _ap.Namespace(
        video_path=str(tmp_video),
        input_type="file",
        checkpoint=args.checkpoint,
        infiller_weight=args.infiller_weight,
        img_focal=fx,
        vis_mode="cam",
    )

    setup_hawor_env(args.hawor_dir)
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller

    # All HaWoR internals use relative paths (weights/, _DATA/) so we must
    # run from the hawor root directory throughout the entire pipeline.
    orig_cwd = os.getcwd()
    os.chdir(args.hawor_dir)
    try:
        print("  [HaWoR] Detecting and tracking hands ...")
        start_idx, end_idx, seq_folder, imgfiles = detect_track_video(hawk_args)

        print("  [HaWoR] Motion estimation (camera-space MANO params) ...")
        frame_chunks_all, img_focal = hawor_motion_estimation(
            hawk_args, start_idx, end_idx, seq_folder)

        # Identity SLAM: world == camera frame, so infiller output is in camera space
        slam_path = os.path.join(seq_folder,
                                 f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        create_dummy_slam(slam_path, len(imgfiles))

        print("  [HaWoR] Temporal infilling ...")
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
            hawk_args, start_idx, end_idx, frame_chunks_all)
    finally:
        os.chdir(orig_cwd)

    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid


def extract_poses(pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid,
                  rgb_files, depth_dir, cam_params, args):
    """
    Convert HaWoR tensors -> wrist_poses.npz numpy arrays.
    pred_trans shape: (2, T_hawor, 3) in camera space.
    """
    import torch
    from hawor.utils.process import run_mano, run_mano_left

    T   = len(rgb_files)
    T_h = pred_trans.shape[1]

    def to_np(t):
        return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    # MANO forward for joints (for gripper distance)
    # hawor_dir is already in sys.path; _DATA paths are relative so chdir first
    ri, li = HAND2IDX["right"], HAND2IDX["left"]
    orig_cwd = os.getcwd()
    os.chdir(args.hawor_dir)
    try:
        out_r = run_mano(pred_trans[ri:ri+1, :T_h], pred_rot[ri:ri+1, :T_h],
                         pred_hand_pose[ri:ri+1, :T_h],
                         betas=pred_betas[ri:ri+1, :T_h])
        out_l = run_mano_left(pred_trans[li:li+1, :T_h], pred_rot[li:li+1, :T_h],
                              pred_hand_pose[li:li+1, :T_h],
                              betas=pred_betas[li:li+1, :T_h])
    finally:
        os.chdir(orig_cwd)

    right_joints = to_np(out_r["joints"][0])  # (T_h, 21, 3) OpenPose order
    left_joints  = to_np(out_l["joints"][0])  # (T_h, 21, 3)
    # joint[0] = wrist joint (more accurate than pred_trans root translation)
    right_wrist_pos = right_joints[:, 0, :]  # (T_h, 3)
    left_wrist_pos  = left_joints[:,  0, :]  # (T_h, 3)

    pv = (to_np(pred_valid) > 0)   # (2, T_h) bool
    depth_intr = cam_params["depth"]
    H = int(depth_intr.get("height", 480))
    W = int(depth_intr.get("width",  640))

    left_poses    = np.zeros((T, 6), dtype=np.float32)
    right_poses   = np.zeros((T, 6), dtype=np.float32)
    left_gripper  = np.zeros(T,      dtype=np.float32)
    right_gripper = np.zeros(T,      dtype=np.float32)
    left_valid    = np.zeros(T,      dtype=bool)
    right_valid   = np.zeros(T,      dtype=bool)

    for i, rgb_path in enumerate(rgb_files):
        if i >= T_h:
            break

        depth_img = None
        if not args.no_depth_anchor:
            dp = depth_dir / (rgb_path.stem + ".png")
            if dp.exists():
                depth_img = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)

        for side, hand_idx in HAND2IDX.items():
            if not pv[hand_idx, i]:
                continue

            # Use MANO joint[0] as wrist position (more accurate than pred_trans)
            wrist_pos_arr = right_wrist_pos if side == "right" else left_wrist_pos
            trans  = wrist_pos_arr[i].copy().astype(np.float32)
            rot_aa = to_np(pred_rot[hand_idx,   i]).copy().astype(np.float32)

            # Depth anchoring: scale translation so wrist Z matches Orbbec depth
            if (depth_img is not None and depth_img.ndim >= 2
                    and abs(trans[2]) > 1e-4):
                px = project_3d_to_px(trans, depth_intr)
                if px is not None:
                    u = int(np.clip(px[0], 0, W - 1))
                    v = int(np.clip(px[1], 0, H - 1))
                    d_raw = get_depth_at(depth_img, u, v, H, W)
                    if d_raw > 0:
                        scale = (d_raw * DEPTH_SCALE) / trans[2]
                        if 0.2 < scale < 5.0:   # reject outliers
                            trans = trans * scale

            euler = R.from_rotvec(rot_aa).as_euler("xyz").astype(np.float32)
            pose  = np.concatenate([trans, euler])

            if side == "left":
                jts = left_joints[i]
                left_poses[i]   = pose
                left_gripper[i] = float(np.linalg.norm(jts[THUMB_TIP] - jts[INDEX_TIP]))
                left_valid[i]   = True
            else:
                jts = right_joints[i]
                right_poses[i]   = pose
                right_gripper[i] = float(np.linalg.norm(jts[THUMB_TIP] - jts[INDEX_TIP]))
                right_valid[i]   = True

    return left_poses, right_poses, left_gripper, right_gripper, left_valid, right_valid


def process_sequence(seq_dir, output_path, args):
    cam_params = load_cam_params(seq_dir, args.ego_cam_id)
    rgb_dir    = seq_dir / args.ego_cam_id / "RGB"
    depth_dir  = seq_dir / args.ego_cam_id / "Depth"
    rgb_files  = sorted(rgb_dir.glob("*.jpg"))
    T = len(rgb_files)
    if T == 0:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")

    sample = cv2.imread(str(rgb_files[0]))
    if sample is not None:
        H, W = sample.shape[:2]
        for k in ("rgb", "depth"):
            cam_params[k].setdefault("height", H)
            cam_params[k].setdefault("width",  W)

    print(f"\n[Sequence] {seq_dir.name}: {T} frames, ego_cam={args.ego_cam_id}")

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        run_hawor_on_sequence(seq_dir, cam_params, rgb_files, args)

    left_poses, right_poses, left_gripper, right_gripper, left_valid, right_valid = \
        extract_poses(pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid,
                      rgb_files, depth_dir, cam_params, args)

    print(f"  Detected: left={left_valid.sum()}/{T} ({100*left_valid.mean():.1f}%)  "
          f"right={right_valid.sum()}/{T} ({100*right_valid.mean():.1f}%)")

    left_poses  = interpolate_invalid(left_poses,  left_valid)
    right_poses = interpolate_invalid(right_poses, right_valid)
    left_poses  = smooth_poses(left_poses,  args.smooth_method)
    right_poses = smooth_poses(right_poses, args.smooth_method)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path),
             left_wrist_poses  = left_poses,
             right_wrist_poses = right_poses,
             left_gripper      = left_gripper,
             right_gripper     = right_gripper,
             left_valid        = left_valid,
             right_valid       = right_valid,
             fps               = np.float32(args.fps))
    print(f"  Saved -> {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract wrist poses via HaWoR (temporally consistent MANO)")
    ap.add_argument("--data_dir",   required=True,
                    help="Single sequence dir or parent dir of multiple sequences")
    ap.add_argument("--output",     default=None,
                    help="Output .npz (single) or dir/stem (batch)")
    ap.add_argument("--ego_cam_id", default="07",
                    help="Camera folder id for ego view (default: 07)")
    ap.add_argument("--hawor_dir",  default="/home/ubuntu/WorkSpace/ZYC/hawor",
                    help="HaWoR root directory")
    ap.add_argument("--checkpoint", default=None,
                    help="hawor.ckpt path (auto-detected from hawor_dir)")
    ap.add_argument("--infiller_weight", default=None,
                    help="infiller.pt path (auto-detected from hawor_dir)")
    ap.add_argument("--fps",        type=float, default=30.0)
    ap.add_argument("--no_depth_anchor", action="store_true",
                    help="Skip Orbbec depth metric anchoring")
    ap.add_argument("--smooth_method", default="none",
                    choices=["savgol", "ema", "median_then_savgol", "none"],
                    help="Extra smoothing (default: none; HaWoR infiller provides temporal consistency)")
    args = ap.parse_args()

    hawor_dir = Path(args.hawor_dir)
    args.hawor_dir = str(hawor_dir)
    if args.checkpoint is None:
        args.checkpoint = str(hawor_dir / "weights/hawor/checkpoints/hawor.ckpt")
    if args.infiller_weight is None:
        args.infiller_weight = str(hawor_dir / "weights/hawor/checkpoints/infiller.pt")

    seq_dirs   = discover_sequence_dirs(Path(args.data_dir), args.ego_cam_id)
    batch_mode = len(seq_dirs) > 1

    print(f"Sequences : {len(seq_dirs)}")
    print(f"HaWoR dir : {hawor_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Infiller  : {args.infiller_weight}")

    for seq_dir in seq_dirs:
        out = resolve_output_path(args.output, seq_dir, batch_mode)
        process_sequence(seq_dir, out, args)


if __name__ == "__main__":
    main()
