from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/root/HW3_task1/final_fusion_abc_garden_world_static/debug_candidates")
FRAMES = Path("/root/HW3_task1/data/background/gs/garden/train/ours_30000/renders")
CAM_JSON = Path("/root/HW3_task1/data/background/gs/garden/cameras.json")


def project_world(points, cam, w, h):
    pts = np.asarray(points, dtype=np.float32)
    c = np.asarray(cam["position"], dtype=np.float32)
    r_w2c = np.asarray(cam["rotation"], dtype=np.float32)
    pc = (pts - c) @ r_w2c.T
    z = pc[:, 2]
    sx = w / float(cam["width"])
    sy = h / float(cam["height"])
    fx = cam["fx"] * sx
    fy = cam["fy"] * sy
    x = pc[:, 0] / (z + 1e-8) * fx + w * 0.5
    y = pc[:, 1] / (z + 1e-8) * fy + h * 0.5
    return np.stack([x, y, z], axis=1)


def cam_offset_to_world(cam, offset):
    pos = np.asarray(cam["position"], dtype=np.float32)
    r_w2c = np.asarray(cam["rotation"], dtype=np.float32)
    return pos + r_w2c.T @ np.asarray(offset, dtype=np.float32)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cams = json.loads(CAM_JSON.read_text())
    # Candidate table points from frame-0 camera coordinates. These are not used
    # as AR placement; they are converted once into fixed world points and then
    # projected into later frames to test whether they remain on the table.
    specs = []
    idx = 0
    for x in [-0.65, -0.35, 0.0, 0.35, 0.65]:
        for y in [-0.25, -0.05, 0.15, 0.35, 0.55]:
            for z in [2.5, 3.0, 3.5, 4.0]:
                specs.append((f"P{idx}", np.array([x, y, z], dtype=np.float32)))
                idx += 1
    world = [(name, cam_offset_to_world(cams[0], off), off) for name, off in specs]
    keyframes = [0, 5, 10, 36, 72, 108, 144]
    for k in keyframes:
        img = Image.open(FRAMES / f"{k:05d}.png").convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        draw = ImageDraw.Draw(img)
        pts = project_world([p for _, p, _ in world], cams[k], w, h)
        for (name, _, off), (u, v, z) in zip(world, pts):
            if z <= 0 or u < 0 or u >= w or v < 0 or v >= h:
                continue
            color = (255, 0, 0) if off[1] < 0 else ((0, 180, 255) if off[1] < 0.25 else (0, 220, 0))
            r = 6
            draw.ellipse([u - r, v - r, u + r, v + r], fill=color)
            draw.text((u + 7, v - 7), name, fill=color)
        img.save(ROOT / f"{k:05d}.jpg", quality=92)
    np.savez(ROOT / "candidates_world.npz",
             names=np.array([n for n, _, _ in world]),
             world=np.stack([p for _, p, _ in world]),
             offsets=np.stack([o for _, _, o in world]))
    print("saved", ROOT)


if __name__ == "__main__":
    main()
