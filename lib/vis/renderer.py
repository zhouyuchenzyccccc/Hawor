# lib/vis/renderer.py — pytorch3d-free replacement using pyrender + trimesh
# Drop-in replacement: same Renderer class interface as the original.

import cv2
import torch
import numpy as np
import trimesh
import pyrender
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"   # headless rendering on server

from .tools import get_colors, checkerboard_geometry


# ── helpers kept from original (no pytorch3d) ─────────────────────────────

def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy() if isinstance(bbox, torch.Tensor) else np.array(bbox[0], dtype=int)
    roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    roi_image[mask] = image[mask]
    out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image
    return out_image


def perspective_projection(x3d, K, R=None, T=None):
    if R is not None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T is not None:
        x3d = x3d + T.transpose(1, 2)
    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left   = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right  = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top    = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width  = right - left
    height = bottom - top
    new_left   = torch.clamp(cx - width  / 2 * scaleFactor, min=0, max=img_w - 1)
    new_right  = torch.clamp(cx + width  / 2 * scaleFactor, min=1, max=img_w)
    new_top    = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h - 1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)
    bbox = torch.stack((new_left.detach(), new_top.detach(),
                        new_right.detach(), new_bottom.detach())).int().float().T
    return bbox


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype
    K = torch.zeros((K_org.shape[0], 4, 4)).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1
    image_sizes = []
    for idx, bb in enumerate(bbox):
        left, upper, right, lower = bb
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]
        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width  = max(right - left, 1)
        new_cx = new_width  - new_cx
        new_cy = new_height - new_cy
        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))
    return K, image_sizes


# ── pyrender-based Renderer ────────────────────────────────────────────────

def _to_np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


class Renderer:
    """
    Drop-in replacement for the pytorch3d Renderer.
    Uses pyrender (EGL headless) for mesh rasterisation.
    """

    def __init__(self, width, height, focal_length, device,
                 bin_size=None, max_faces_per_bin=None):
        self.width  = int(width)
        self.height = int(height)
        self.focal_length = float(focal_length)
        self.device = device

        # Keep these for callers that read them
        self.K = torch.tensor(
            [[self.focal_length, 0, self.width  / 2],
             [0, self.focal_length, self.height / 2],
             [0, 0, 1]]
        ).unsqueeze(0).float()
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)

        self._scene  = None
        self._camera = None
        self._cam_node = None

    # ── camera helpers (called by hawor_video.py) ──────────────────────────

    def create_camera_from_cv(self, R, T, K=None, image_size=None):
        """
        Returns (cameras, lights) — we return plain numpy/torch objects that
        are passed straight back into render_multiple().
        """
        if K is None:
            K = self.K
        cam_info = {
            "R": _to_np(R) if R is not None else np.eye(3)[None],
            "T": _to_np(T) if T is not None else np.zeros((1, 3)),
            "K": _to_np(K),
        }
        lights = {"location": _to_np(T) if T is not None else np.zeros((1, 3))}
        return cam_info, lights

    # ── main render entry point ────────────────────────────────────────────

    def render_multiple(self, verts_list, faces, colors_list, cameras, lights):
        """
        verts_list : list of (1, V, 3) tensors  OR  (B, V, 3) tensor
        faces      : (F, 3) numpy array or tensor
        colors_list: (N, 4) tensor  [r,g,b,a] in [0,1]
        cameras    : dict from create_camera_from_cv  (or None)
        lights     : dict  (or None)

        Returns:
            image (H, W, 3) uint8
            mask  (H, W)    bool
        """
        faces_np = _to_np(faces).astype(np.int32)

        # Build pyrender scene
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

        # Add meshes
        if isinstance(verts_list, torch.Tensor):
            # (B, V, 3) — iterate over batch dim
            items = [verts_list[i] for i in range(verts_list.shape[0])]
        else:
            items = list(verts_list)

        for i, verts in enumerate(items):
            v = _to_np(verts)
            if v.ndim == 3:
                v = v[0]   # (1, V, 3) -> (V, 3)

            if colors_list is not None and i < len(colors_list):
                c = _to_np(colors_list[i])
                if c.ndim == 1:
                    rgba = c[:4] if len(c) >= 4 else np.array([*c[:3], 1.0])
                else:
                    rgba = c.flatten()[:4]
                color = (rgba[:3] * 255).astype(np.uint8).tolist()
            else:
                color = [180, 180, 180]

            tri = trimesh.Trimesh(vertices=v, faces=faces_np, process=False)
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[color[0]/255, color[1]/255, color[2]/255, 1.0],
                metallicFactor=0.0, roughnessFactor=0.8)
            mesh = pyrender.Mesh.from_trimesh(tri, material=mat, smooth=True)
            scene.add(mesh)

        # Camera
        fx = self.focal_length
        fy = self.focal_length
        cx = self.width  / 2.0
        cy = self.height / 2.0

        if cameras is not None and isinstance(cameras, dict):
            R_cv = cameras["R"][0] if cameras["R"].ndim == 3 else cameras["R"]
            T_cv = cameras["T"][0] if cameras["T"].ndim == 2 else cameras["T"]
            K_arr = cameras["K"]
            if K_arr.ndim == 3:
                K_arr = K_arr[0]
            if K_arr.shape == (4, 4):
                fx, fy = float(K_arr[0, 0]), float(K_arr[1, 1])
                cx, cy = float(K_arr[0, 2]), float(K_arr[1, 2])
            elif K_arr.shape == (3, 3):
                fx, fy = float(K_arr[0, 0]), float(K_arr[1, 1])
                cx, cy = float(K_arr[0, 2]), float(K_arr[1, 2])
            # Convert OpenCV -> OpenGL camera pose
            R_gl = R_cv.copy()
            R_gl[1:] *= -1
            T_gl = T_cv.copy()
            T_gl[1:] *= -1
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = R_gl.T
            cam_pose[:3,  3] = -R_gl.T @ T_gl
        else:
            cam_pose = np.eye(4)

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                           znear=0.01, zfar=100.0)
        scene.add(camera, pose=cam_pose)

        # Light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=cam_pose)

        # Render
        r = pyrender.OffscreenRenderer(self.width, self.height)
        color_img, depth_img = r.render(scene,
                                        flags=pyrender.RenderFlags.RGBA |
                                              pyrender.RenderFlags.SKIP_CULL_FACES)
        r.delete()

        image = color_img[..., :3]          # (H, W, 3) uint8
        mask  = color_img[..., 3] > 0       # (H, W) bool
        return image, mask

    # ── stubs for methods called elsewhere ────────────────────────────────

    def set_ground(self, *a, **kw):
        pass

    def update_bbox(self, *a, **kw):
        pass

    def reset_bbox(self, *a, **kw):
        pass

    def render_mesh(self, vertices, background, colors=[0.8, 0.8, 0.8]):
        """Single-mesh convenience wrapper."""
        v = _to_np(vertices)
        if not hasattr(self, 'faces'):
            return background
        verts_t = torch.from_numpy(v).unsqueeze(0)
        c = torch.tensor([[*colors[:3], 1.0]])
        image, mask = self.render_multiple([verts_t], self.faces, c, None, None)
        image_t = torch.from_numpy(image)
        mask_t  = torch.from_numpy(mask)
        return overlay_image_onto_background(image_t, mask_t, self.bboxes, background)

    def render_with_ground(self, verts, faces, colors, cameras, lights):
        image, _ = self.render_multiple(
            [verts[i:i+1] for i in range(len(verts))], faces, colors, cameras, lights)
        return image


# ── stubs for pytorch3d symbols imported elsewhere ─────────────────────────

def get_global_cameras(verts, device, distance=5, position=(-5.0, 5.0, 0.0)):
    """Stub — not used in demo.py cam-space mode."""
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets   = verts.mean(1)
    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1, keepdim=True) * distance
    positions  = targets - directions
    R = torch.eye(3).unsqueeze(0).repeat(len(verts), 1, 1)
    T = positions
    lights = {"location": position}
    return R, T, lights
