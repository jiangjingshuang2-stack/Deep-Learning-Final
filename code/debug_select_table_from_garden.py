from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData


ROOT = Path("/root/HW3_task1/final_fusion_abc_garden_world_static/debug_table_points")
FRAMES = Path("/root/HW3_task1/data/background/gs/garden/train/ours_30000/renders")
CAM_JSON = Path("/root/HW3_task1/data/background/gs/garden/cameras.json")
GARDEN_PLY = Path("/root/HW3_task1/data/background/gs/garden/point_cloud/iteration_30000/point_cloud.ply")


def xyz_of(v):
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def project_world(points, cam, w, h):
    pts = np.asarray(points, dtype=np.float32)
    c = np.asarray(cam["position"], dtype=np.float32)
    r = np.asarray(cam["rotation"], dtype=np.float32)
    # Match the convention that made frame-0 camera offsets project correctly:
    # x_cam = R * (x_world - C)
    pc = (pts - c) @ r
    z = pc[:, 2]
    sx = w / float(cam["width"])
    sy = h / float(cam["height"])
    fx = cam["fx"] * sx
    fy = cam["fy"] * sy
    x = pc[:, 0] / (z + 1e-8) * fx + w * 0.5
    y = pc[:, 1] / (z + 1e-8) * fy + h * 0.5
    return np.stack([x, y, z], axis=1)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cams = json.loads(CAM_JSON.read_text())
    img0 = Image.open(FRAMES / "00000.png").convert("RGB")
    w, h = img0.size
    v = PlyData.read(GARDEN_PLY)["vertex"].data
    xyz = xyz_of(v)

    # Use a deterministic subset for speed, dense enough for table localization.
    rng = np.random.default_rng(2026)
    if len(xyz) > 900000:
        idx = rng.choice(len(xyz), size=900000, replace=False)
        xyz_s = xyz[idx]
    else:
        xyz_s = xyz

    p0 = project_world(xyz_s, cams[0], w, h)
    # Hand-picked frame-0 table-top ellipse. This avoids background wall/grass
    # and focuses on the visible wooden table surface.
    cx, cy = 800.0, 560.0
    rx, ry = 390.0, 115.0
    in_ellipse = ((p0[:, 0] - cx) / rx) ** 2 + ((p0[:, 1] - cy) / ry) ** 2 < 1.0
    valid = in_ellipse & (p0[:, 2] > 0)
    z = p0[valid, 2]
    pts = xyz_s[valid]
    pix = p0[valid]
    print("region points", len(pts), "z range", float(z.min()), float(z.max()))

    # Use near-depth bands; far bands are background that projects through the
    # same pixels. Save multiple candidates for visual choice.
    qs = [2, 5, 8, 12, 16, 22, 30, 40]
    centers = []
    names = []
    for q in qs:
        zq = np.percentile(z, q)
        band = np.abs(z - zq) < 0.035
        if band.sum() < 20:
            band = np.abs(z - zq) < 0.08
        if band.sum() == 0:
            continue
        # Split left/center/right on the table surface so A/B/C can be placed.
        for label, mask_x in [
            ("L", pix[:, 0] < cx - 120),
            ("M", np.abs(pix[:, 0] - cx) <= 120),
            ("R", pix[:, 0] > cx + 120),
        ]:
            m = band & mask_x
            if m.sum() < 10:
                continue
            centers.append(np.median(pts[m], axis=0))
            names.append(f"q{q}{label}")

    centers = np.asarray(centers, dtype=np.float32)
    np.savez(ROOT / "table_candidates.npz", names=np.asarray(names), world=centers)

    keyframes = [0, 5, 10, 36, 72, 108, 144]
    for k in keyframes:
        img = Image.open(FRAMES / f"{k:05d}.png").convert("RGB")
        draw = ImageDraw.Draw(img)
        pp = project_world(centers, cams[k], img.size[0], img.size[1])
        for name, (u, vv, zz) in zip(names, pp):
            if zz <= 0 or not (0 <= u < img.size[0] and 0 <= vv < img.size[1]):
                continue
            color = (255, 30, 30) if name.endswith("L") else ((30, 200, 30) if name.endswith("M") else (0, 160, 255))
            r = 7
            draw.ellipse([u - r, vv - r, u + r, vv + r], fill=color)
            draw.text((u + 8, vv - 8), name, fill=color)
        img.save(ROOT / f"{k:05d}.jpg", quality=92)
    print("saved", ROOT, names)


if __name__ == "__main__":
    main()
