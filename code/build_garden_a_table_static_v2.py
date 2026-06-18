from pathlib import Path
import json
import math
import shutil

import numpy as np
from plyfile import PlyData, PlyElement


ROOT = Path("/root/HW3_task1/final_fusion_abc_garden_world_static_v2")
OUT = ROOT / "base_garden_a_table_static_v2"
GARDEN_MODEL = Path("/root/HW3_task1/data/background/gs/garden")
GARDEN_PLY = GARDEN_MODEL / "point_cloud/iteration_30000/point_cloud.ply"
CAM_JSON = GARDEN_MODEL / "cameras.json"
A_PLY = Path("/root/HW3_task1/data/objectA_rot90_recon_0612_relaxed_fullframe_pinhole/gs_eval_7000/point_cloud/iteration_7000/point_cloud.ply")
TABLE_NPZ = Path("/root/HW3_task1/final_fusion_abc_garden_world_static/debug_table_points/table_candidates.npz")


def read_ply(path):
    return PlyData.read(path)["vertex"].data


def xyz_of(v):
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def rotate_y(points, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    r = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return points @ r.T


def yaw_quat(yaw, n):
    half = yaw * 0.5
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = math.cos(half)
    q[:, 2] = math.sin(half)
    return q


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1).astype(np.float32)


def filter_a(a):
    xyz = xyz_of(a)
    p_low = np.percentile(xyz, 3, axis=0)
    p_high = np.percentile(xyz, 97, axis=0)
    keep = np.all((xyz >= p_low) & (xyz <= p_high), axis=1)
    keep &= a["opacity"] > -3.0
    max_s = np.maximum.reduce([a["scale_0"], a["scale_1"], a["scale_2"]])
    keep &= max_s < np.percentile(max_s, 96)
    print("A filter", len(a), "->", int(keep.sum()))
    return a[keep]


def normalize_to_unit(xyz):
    p_low = np.percentile(xyz, 2, axis=0)
    p_high = np.percentile(xyz, 98, axis=0)
    center = (p_low + p_high) / 2.0
    scale = float((p_high - p_low).max())
    return (xyz - center) / scale, scale


def transform_a_records(a, template_dtype, target_center, target_scale, yaw):
    a_xyz = xyz_of(a)
    a_norm, a_scale = normalize_to_unit(a_xyz)
    out = np.empty(len(a), dtype=template_dtype)
    for name in template_dtype.names:
        out[name] = a[name] if name in a.dtype.names else 0.0
    xyz = rotate_y(a_norm, yaw) * target_scale + target_center
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    scale_delta = math.log(target_scale / a_scale)
    for name in ["scale_0", "scale_1", "scale_2"]:
        s = a[name] + scale_delta
        out[name] = np.minimum(s, np.percentile(s, 94))
    q_old = np.stack([a["rot_0"], a["rot_1"], a["rot_2"], a["rot_3"]], axis=1).astype(np.float32)
    q_new = quat_mul(yaw_quat(yaw, len(a)), q_old)
    q_new /= np.linalg.norm(q_new, axis=1, keepdims=True) + 1e-8
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = q_new[:, 0], q_new[:, 1], q_new[:, 2], q_new[:, 3]
    out["opacity"] = np.clip(out["opacity"], -1.8, 1.6)
    return out


def table_point(name):
    d = np.load(TABLE_NPZ)
    names = [str(x) for x in d["names"]]
    return d["world"][names.index(name)].astype(np.float32)


def camera_up_world():
    cams = json.loads(CAM_JSON.read_text())
    r_w2c = np.asarray(cams[0]["rotation"], dtype=np.float32)
    # Camera Y is down, so negative camera Y is up in the image.
    up = r_w2c.T @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
    return up / (np.linalg.norm(up) + 1e-8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    garden = read_ply(GARDEN_PLY)
    # v1 used q12L + [0, -0.08, 0]. v2 keeps that anchor and adds a small
    # camera-up lift so the lower part of A is no longer hidden by the table.
    a_center = table_point("q12L") + np.array([0.0, -0.08, 0.0], dtype=np.float32)
    a_center = a_center + 0.18 * camera_up_world()
    a = filter_a(read_ply(A_PLY))
    a_rec = transform_a_records(a, garden.dtype, a_center, 0.38, 0.25)
    fused = np.concatenate([garden, a_rec])

    out_ply = OUT / "point_cloud/iteration_30000/point_cloud.ply"
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(fused, "vertex")], text=False).write(out_ply)

    cfg = (GARDEN_MODEL / "cfg_args").read_text()
    cfg = cfg.replace("model_path='/root/HW3_task1/data/background/gs/garden'", f"model_path='{OUT}'")
    (OUT / "cfg_args").write_text(cfg)
    for name in ["cameras.json", "exposure.json", "input.ply"]:
        src = GARDEN_MODEL / name
        if src.exists():
            shutil.copy2(src, OUT / name)
    np.savez(ROOT / "world_static_layout_v2.npz",
             a_center=a_center,
             b_center=table_point("q12M"),
             c_center=table_point("q12R"))
    print("saved", out_ply, "A center", a_center)


if __name__ == "__main__":
    main()
