"""
Faithful renderer: place every part exactly where Onshape's solver put it.

Reads robot_intake/onshape_frames.json (per-part solved transforms across the deploy range)
and robot_intake/onshape_assets/<hash>.stl (each part's real mesh), and renders the assembly
by setting each part's pose directly from Onshape's transform -- so it matches Onshape exactly,
for any mate type (the cam included). Renders verification PNGs + a GIF headlessly.

Run: python render_faithful.py
"""

import json
import numpy as np
import pybullet as p
from PIL import Image
from stl import mesh as stlmesh
from transforms3d.quaternions import mat2quat

OUT = "robot_intake"


def load_visual(stl_path, rgba=(0.6, 0.6, 0.62, 1.0)):
    """Create a PyBullet visual shape from an STL (via vertices, format-agnostic)."""
    m = stlmesh.Mesh.from_file(stl_path)
    verts = m.vectors.reshape(-1, 3).tolist()
    idx = list(range(len(verts)))
    return p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=idx, rgbaColor=list(rgba))


def pose_from_T(T):
    """Onshape 16-float row-major 4x4 -> (position, quaternion xyzw)."""
    R = np.array([[T[0], T[1], T[2]], [T[4], T[5], T[6]], [T[8], T[9], T[10]]])
    pos = [T[3], T[7], T[11]]
    w, x, y, z = mat2quat(R)
    return pos, [x, y, z, w]


def main():
    import os
    data = json.load(open(f"{OUT}/onshape_frames.json"))
    part_refs = data["part_refs"]
    frames = data["frames"]
    colors = json.load(open(f"{OUT}/onshape_colors.json")) if os.path.exists(f"{OUT}/onshape_colors.json") else {}

    p.connect(p.DIRECT)

    # One visual shape per unique mesh, reused across occurrences.
    shape_cache = {}
    bodies = {}  # occurrence key -> body id
    for key, ref in part_refs.items():
        mesh = ref["mesh"]
        if mesh not in shape_cache:
            try:
                shape_cache[mesh] = load_visual(f"{OUT}/onshape_assets/{mesh}.stl",
                                                colors.get(mesh, (0.6, 0.6, 0.62, 1.0)))
            except Exception as e:
                shape_cache[mesh] = -1
        vs = shape_cache[mesh]
        if vs < 0:
            continue
        bodies[key] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs)
    print(f"created {len(bodies)} part bodies (from {len(shape_cache)} meshes)")

    frame_by_deg = {f["deg"]: f for f in frames}

    def apply(deg):
        f = frame_by_deg[deg]
        for key, T in f["poses"].items():
            if key in bodies:
                pos, quat = pose_from_T(T)
                p.resetBasePositionAndOrientation(bodies[key], pos, quat)

    # Camera framing from the rest pose.
    apply(0)
    lo = np.array([1e9] * 3); hi = -lo
    for b in bodies.values():
        a, c = p.getAABB(b)
        lo = np.minimum(lo, a); hi = np.maximum(hi, c)
    center = ((lo + hi) / 2).tolist(); size = float(np.linalg.norm(hi - lo))
    proj = p.computeProjectionMatrixFOV(50, 1.0, 0.01, 100)

    def shot(deg, yaw):
        apply(deg)
        vm = p.computeViewMatrixFromYawPitchRoll(center, size * 1.2, yaw, -25, 0, 2)
        _, _, rgb, _, _ = p.getCameraImage(700, 700, vm, proj, renderer=p.ER_TINY_RENDERER)
        return Image.fromarray(np.reshape(rgb, (700, 700, 4))[:, :, :3].astype("uint8"))

    for deg in (0, -60, -120):
        shot(deg, 45).save(f"{OUT}/faithful_{abs(deg)}.png")
        print(f"  rendered deg {deg}")

    degs = [f["deg"] for f in frames]
    seq = degs + degs[::-1]
    gif = [shot(d, 40) for d in seq]
    gif[0].save(f"{OUT}/intake_faithful.gif", save_all=True, append_images=gif[1:],
                duration=60, loop=0)
    print(f"saved {OUT}/intake_faithful.gif")


if __name__ == "__main__":
    main()
