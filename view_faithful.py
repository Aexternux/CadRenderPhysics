"""
Interactive faithful viewer -- the real thing.

Loads each part's real mesh + color and places it exactly where Onshape's solver put it.
A slider deploys the slapdown; every part (arm, cam-driven hopper wall, everything) follows
Onshape's exact solution. No robot-joint approximation, no physics jamming.

Controls:
    "deploy (deg)" slider = move the mechanism (snaps to Onshape's sampled solution)
    RIGHT-drag = orbit   MIDDLE-drag = pan   SCROLL = zoom

Usage:
    python view_faithful.py [robot_dir]      (default: robot_intake)
"""

import json
import os
import sys
import time

import numpy as np
import pybullet as p
from stl import mesh as stlmesh
from transforms3d.quaternions import mat2quat

MOUSE_MOVE_EVENT, MOUSE_BUTTON_EVENT = 1, 2
KEY_IS_DOWN, KEY_WAS_TRIGGERED, KEY_WAS_RELEASED = 1, 2, 4


def pose_from_T(T):
    R = np.array([[T[0], T[1], T[2]], [T[4], T[5], T[6]], [T[8], T[9], T[10]]])
    w, x, y, z = mat2quat(R)
    return [T[3], T[7], T[11]], [x, y, z, w]


def main():
    rd = sys.argv[1] if len(sys.argv) > 1 else "robot_intake"
    data = json.load(open(f"{rd}/onshape_frames.json"))
    part_refs = data["part_refs"]
    frames = data["frames"]
    colors = {}
    if os.path.exists(f"{rd}/onshape_colors.json"):
        colors = json.load(open(f"{rd}/onshape_colors.json"))

    if p.connect(p.GUI) < 0:
        raise RuntimeError("Could not open a PyBullet GUI window.")
    p.setGravity(0, 0, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    shape_cache, bodies = {}, {}
    for key, ref in part_refs.items():
        mesh = ref["mesh"]
        if mesh not in shape_cache:
            try:
                m = stlmesh.Mesh.from_file(f"{rd}/onshape_assets/{mesh}.stl")
                verts = m.vectors.reshape(-1, 3).tolist()
                rgba = colors.get(mesh, [0.6, 0.6, 0.62, 1.0])
                shape_cache[mesh] = p.createVisualShape(
                    p.GEOM_MESH, vertices=verts, indices=list(range(len(verts))),
                    rgbaColor=rgba)
            except Exception:
                shape_cache[mesh] = -1
        if shape_cache[mesh] >= 0:
            bodies[key] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape_cache[mesh])
    print(f"loaded {len(bodies)} parts, {len(colors)} colors")

    frame_degs = np.array([f["deg"] for f in frames])
    frame_by_deg = {f["deg"]: f for f in frames}

    def apply(deg):
        nearest = float(frame_degs[np.argmin(np.abs(frame_degs - deg))])
        for key, T in frame_by_deg[nearest]["poses"].items():
            if key in bodies:
                pos, quat = pose_from_T(T)
                p.resetBasePositionAndOrientation(bodies[key], pos, quat)

    apply(0)
    lo = np.array([1e9] * 3); hi = -lo
    for b in bodies.values():
        a, c = p.getAABB(b); lo = np.minimum(lo, a); hi = np.maximum(hi, c)
    center = ((lo + hi) / 2).tolist(); size = float(np.linalg.norm(hi - lo))
    p.resetDebugVisualizerCamera(size * 1.2, 40, -25, center)

    deg_min = float(frame_degs.min())
    s = p.addUserDebugParameter("deploy (deg)", deg_min, 0.0, 0.0)
    print("\nDrag 'deploy (deg)' to move the intake. RIGHT-drag orbit, MIDDLE-drag pan, scroll zoom.")

    drag = {"right": False, "middle": False, "x": 0.0, "y": 0.0}

    def cam():
        c = p.getDebugVisualizerCamera()
        return c[8], c[9], c[10], c[11], c[4], c[5]

    while p.isConnected():
        apply(p.readUserDebugParameter(s))
        for ev in p.getMouseEvents():
            et, mx, my, btn, st = ev
            if et == MOUSE_BUTTON_EVENT and btn in (1, 2):
                k = "right" if btn == 2 else "middle"
                if st & (KEY_WAS_TRIGGERED | KEY_IS_DOWN):
                    drag[k] = True; drag["x"], drag["y"] = mx, my
                if st & KEY_WAS_RELEASED:
                    drag[k] = False
            elif et == MOUSE_MOVE_EVENT:
                dx, dy = mx - drag["x"], my - drag["y"]
                if drag["right"]:
                    yaw, pit, dist, tgt, _u, _f = cam()
                    p.resetDebugVisualizerCamera(dist, yaw + dx * 0.4,
                                                 max(-89, min(89, pit - dy * 0.4)), tgt)
                elif drag["middle"]:
                    yaw, pit, dist, tgt, up, fwd = cam()
                    r = np.cross(fwd, up); r /= (np.linalg.norm(r) or 1.0)
                    sc = 0.0015 * dist
                    tgt = [tgt[i] - r[i] * dx * sc + up[i] * dy * sc for i in range(3)]
                    p.resetDebugVisualizerCamera(dist, yaw, pit, tgt)
                drag["x"], drag["y"] = mx, my
        p.stepSimulation()
        time.sleep(1.0 / 120.0)


if __name__ == "__main__":
    main()
