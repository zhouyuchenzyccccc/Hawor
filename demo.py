import argparse
import sys
import os

import torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import joblib
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
# ── pytorch3d mock (added by patch_demo_world.py) ────────────────────────────
import sys as _sys, types as _types

# pyrender video renderer (replaces aitviewer)
def _render_to_video(left_dict, right_dict, output_pth, img_focal, image_names, fps=30, alpha=0.85, **kw):
    import os, cv2, numpy as np, pyrender, trimesh
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.makedirs(output_pth, exist_ok=True)
    out_path = os.path.join(output_pth, "render_overlay.mp4")
    img0 = cv2.imread(image_names[0])
    H, W = img0.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    faces = None
    for d in (right_dict, left_dict):
        if d and "faces" in d and d["faces"] is not None:
            f = d["faces"]
            faces = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
            faces = faces.astype(np.int32)
            break
    fx = fy = float(img_focal)
    cx, cy = W / 2.0, H / 2.0
    # Camera at origin looking down -z (OpenGL convention).
    # demo.py already applies R_x to convert verts from OpenCV to OpenGL space,
    # so verts have negative z -> in front of camera. Use identity cam_pose.
    cam_pose = np.eye(4)
    def _v(arr, t):
        if arr is None: return None
        if hasattr(arr, "detach"): arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 4:
            if t >= arr.shape[1]: return None
            v = arr[0, t]
        elif arr.ndim == 3:
            if t >= arr.shape[0]: return None
            v = arr[t]
        else:
            return None
        return v if v.shape[0] > 0 else None
    hand_cfgs = [(right_dict, [0.90, 0.70, 0.50]), (left_dict, [0.50, 0.70, 0.90])]
    print(f"Rendering {len(image_names)} frames -> {out_path}")
    mesh_count = 0
    for t, img_path in enumerate(image_names):
        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None: continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.4,0.4,0.4])
        has_mesh = False
        if faces is not None:
            for d, col in hand_cfgs:
                if not d: continue
                v = _v(d.get("vertices"), t)
                if v is None: continue
                if t == 0:
                    print(f"  t=0 verts: shape={v.shape}, z range [{v[:,2].min():.3f}, {v[:,2].max():.3f}]")
                tri = trimesh.Trimesh(vertices=v, faces=faces, process=False)
                mat = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[col[0],col[1],col[2],1.0],
                    metallicFactor=0.0, roughnessFactor=0.7)
                scene.add(pyrender.Mesh.from_trimesh(tri, material=mat, smooth=True))
                has_mesh = True
        if has_mesh:
            cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.001, zfar=100.0)
            scene.add(cam, pose=cam_pose)
            scene.add(pyrender.DirectionalLight([1.0,1.0,1.0], intensity=3.0), pose=cam_pose)
            r = pyrender.OffscreenRenderer(W, H)
            rgba, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES)
            r.delete()
            if t == 0:
                print(f"  t=0 rgba: nonzero alpha pixels = {(rgba[...,3]>0).sum()}")
            mask = rgba[..., 3] > 0
            overlay = frame_rgb.copy()
            overlay[mask] = (alpha * rgba[...,:3][mask].astype(np.float32) + (1-alpha)*frame_rgb[mask])
            out_rgb = np.clip(overlay, 0, 255).astype(np.uint8)
            mesh_count += 1
        else:
            out_rgb = frame_rgb.astype(np.uint8)
        writer.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        if t % 50 == 0: print(f"  frame {t}/{len(image_names)} has_mesh={has_mesh}", flush=True)
    writer.release()
    print(f"Saved ({mesh_count}/{len(image_names)} frames with mesh): {out_path}")
    return out_path
# end _render_to_video
for _mod in [
    "pytorch3d", "pytorch3d.renderer", "pytorch3d.renderer.mesh",
    "pytorch3d.renderer.cameras", "pytorch3d.renderer.lighting",
    "pytorch3d.structures", "pytorch3d.transforms",
    "pytorch3d.loss", "pytorch3d.ops",
]:
    if _mod not in _sys.modules:
        _sys.modules[_mod] = _types.ModuleType(_mod)
# ─────────────────────────────────────────────────────────────────────────────
# hawor_slam import is deferred to avoid DROID-SLAM dependency when not needed
def hawor_slam(*a, **kw):
    from scripts.scripts_test_video.hawor_slam import hawor_slam as _hs
    return _hs(*a, **kw)
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='example/video_0.mp4')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight",  type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--vis_mode",  type=str, default='world', help='cam | world')
    args = parser.parse_args()

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        # DROID-SLAM unavailable (nvcc incompatibility) — create identity SLAM (camera-space)
        import os as _os2
        import numpy as _np2
        _os2.makedirs(_os2.path.dirname(slam_path), exist_ok=True)
        _n = end_idx - start_idx
        _traj = _np2.zeros((_n, 7), dtype=_np2.float32)
        _traj[:, 6] = 1.0  # identity quaternion (qw=1)
        _np2.savez(slam_path, traj=_traj, scale=_np2.float32(1.0))
        print(f"[demo] Created identity SLAM (camera-space): {slam_path}")
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

    # vis sequence for this video
    hand2idx = {
        "right": 1,
        "left": 0
    }
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1
            
    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)

    # get right hand vertices
    hand = 'right'
    hand_idx = hand2idx[hand]
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0]
    right_dict = {
            'vertices': right_verts.unsqueeze(0),
            'faces': faces_right,
        }

    # get left hand vertices
    faces_left = faces_right[:,[0,2,1]]
    hand = 'left'
    hand_idx = hand2idx[hand]
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    left_verts = pred_glob_l['vertices'][0]
    left_dict = {
            'vertices': left_verts.unsqueeze(0),
            'faces': faces_left,
        }

    R_x = torch.tensor([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
    
    # Here we use aitviewer(https://github.com/eth-ait/aitviewer) for simple visualization.
    if args.vis_mode == 'world': 
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        _render_to_video(left_dict, right_dict, output_pth, img_focal, image_names)
    elif args.vis_mode == 'cam':
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names, R_w2c=R_w2c_sla_all[vis_start:vis_end], t_w2c=t_w2c_sla_all[vis_start:vis_end])

    print("finish")



