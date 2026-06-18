from pathlib import Path
import json
import math
import os
import subprocess

import cv2
import numpy as np
from PIL import Image
import trimesh


ROOT = Path("/root/HW3_task1/final_fusion_abc_garden_world_static_v2")
BASE_FRAMES = ROOT / "base_garden_a_table_static_v2/train/ours_30000/renders"
OUT_FRAMES = ROOT / "final_frames_v2"
OUT_VIDEO = ROOT / "videos/final_abc_garden_world_static_v2.mp4"
CAM_JSON = Path("/root/HW3_task1/data/background/gs/garden/cameras.json")
LAYOUT = ROOT / "world_static_layout_v2.npz"
B_OBJ = Path("/root/HW3_task1/third_party/threestudio/outputs/objectB_dreamfusion_sd15_gpu3_5k_v3/Phase1/save/it5000-export/model.obj")
C_OBJ = Path("/root/HW3_task1/third_party/threestudio/outputs/objectC_sunscreen_zero123_0611_gpu0/Phase1/save/it600-export/model.obj")


def load_scene_mesh(path):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node in loaded.graph.nodes_geometry:
            mat, geom_name = loaded.graph[node]
            geom = loaded.geometry[geom_name].copy()
            geom.apply_transform(mat)
            meshes.append(geom)
        return trimesh.util.concatenate(meshes)
    return loaded


def rot_cam_y(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def mesh_zup_to_camera(v):
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1).astype(np.float32)


def prepare_mesh(path, target_height, yaw):
    mesh = load_scene_mesh(path)
    verts_raw = mesh.vertices.astype(np.float32)
    verts_raw -= (verts_raw.min(0) + verts_raw.max(0)) * 0.5
    verts = mesh_zup_to_camera(verts_raw)
    height = float(verts[:, 1].max() - verts[:, 1].min())
    if height > 1e-8:
        verts *= target_height / height
    verts = verts @ rot_cam_y(yaw).T
    faces = mesh.faces.astype(np.int32)
    colors = np.full((len(faces), 3), 220, dtype=np.uint8)
    if getattr(mesh.visual, "kind", None) == "texture" and mesh.visual.uv is not None:
        img = getattr(mesh.visual.material, "image", None)
        if img is not None:
            uv = mesh.visual.uv
            tex = np.array(img.convert("RGB"))
            h, w = tex.shape[:2]
            face_uv = uv[faces].mean(axis=1)
            u = np.mod(face_uv[:, 0], 1.0)
            v = np.mod(face_uv[:, 1], 1.0)
            x = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
            y = np.clip(((1.0 - v) * (h - 1)).astype(np.int32), 0, h - 1)
            colors = tex[y, x].astype(np.uint8)
    elif getattr(mesh.visual, "vertex_colors", None) is not None and len(mesh.visual.vertex_colors):
        vc = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
        colors = vc[faces].mean(axis=1).astype(np.uint8)
    return verts, faces, colors


def project_world(points, cam, w, h):
    pts = np.asarray(points, dtype=np.float32)
    c = np.asarray(cam["position"], dtype=np.float32)
    r = np.asarray(cam["rotation"], dtype=np.float32)
    pc = (pts - c) @ r
    z = pc[:, 2]
    sx = w / float(cam["width"])
    sy = h / float(cam["height"])
    fx = cam["fx"] * sx
    fy = cam["fy"] * sy
    x = pc[:, 0] / (z + 1e-8) * fx + w * 0.5
    y = pc[:, 1] / (z + 1e-8) * fy + h * 0.5
    return np.stack([x, y, z], axis=1)


def local_to_world(local, center, r0):
    return center.astype(np.float32) + local @ r0.T


def draw_mesh(img, verts_world, faces, face_colors, cam):
    h, w = img.shape[:2]
    pp = project_world(verts_world, cam, w, h)
    out = img.copy()
    tri_z = pp[faces, 2].mean(axis=1)
    order = np.argsort(tri_z)[::-1]
    light_dir = np.array([0.25, -0.35, 0.9], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    for fi in order:
        f = faces[fi]
        tri = pp[f]
        if np.any(tri[:, 2] <= 0.08):
            continue
        pts2 = tri[:, :2].astype(np.float32)
        if pts2[:, 0].max() < 0 or pts2[:, 0].min() >= w or pts2[:, 1].max() < 0 or pts2[:, 1].min() >= h:
            continue
        area = (pts2[1, 0] - pts2[0, 0]) * (pts2[2, 1] - pts2[0, 1]) - (pts2[2, 0] - pts2[0, 0]) * (pts2[1, 1] - pts2[0, 1])
        if abs(area) < 1e-6:
            continue
        v0, v1, v2 = verts_world[f]
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n)
        shade = 0.88
        if n_norm > 1e-8:
            n = n / n_norm
            shade = 0.68 + 0.32 * max(0.0, float(np.dot(n, light_dir)))
        color = np.clip(face_colors[fi].astype(np.float32) * shade, 0, 255).astype(np.uint8)
        cv2.fillConvexPoly(out, np.round(pts2).astype(np.int32), color.tolist(), lineType=cv2.LINE_AA)
    return out


def main():
    OUT_FRAMES.mkdir(parents=True, exist_ok=True)
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    cams = json.loads(CAM_JSON.read_text())
    r0 = np.asarray(cams[0]["rotation"], dtype=np.float32).T
    layout = np.load(LAYOUT)
    b_center = layout["b_center"].astype(np.float32) + np.array([0, -0.06, 0], dtype=np.float32)
    c_center = layout["c_center"].astype(np.float32) + np.array([0, -0.06, 0], dtype=np.float32)
    b_local = prepare_mesh(B_OBJ, 0.34, -0.35)
    c_local = prepare_mesh(C_OBJ, 0.34, 0.45)
    b_world = local_to_world(b_local[0], b_center, r0)
    c_world = local_to_world(c_local[0], c_center, r0)
    frames = sorted(BASE_FRAMES.glob("*.png"))
    n = min(len(frames), len(cams))
    indices = [0, 5, 10, 36, 72, 108, 144] if os.environ.get("KEYFRAMES") else list(range(n))
    for i in indices:
        img = np.array(Image.open(frames[i]).convert("RGB"))
        img = draw_mesh(img, b_world, b_local[1], b_local[2], cams[i])
        img = draw_mesh(img, c_world, c_local[1], c_local[2], cams[i])
        Image.fromarray(img).save(OUT_FRAMES / f"{i:05d}.png")
        print("frame", i)
    if not os.environ.get("KEYFRAMES"):
        subprocess.check_call([
            "ffmpeg", "-y", "-framerate", "24", "-i", str(OUT_FRAMES / "%05d.png"),
            "-vf", "format=yuv420p", "-c:v", "mpeg4", "-q:v", "3", str(OUT_VIDEO)
        ])
        print("saved", OUT_VIDEO)


if __name__ == "__main__":
    main()
