"""
Faithful viewer driven by Onshape's own solver.

The cam coupling (slapdown -> hopper wall) was sampled directly from Onshape's constraint
solver via the API and saved to <robot_dir>/coupling.json. Here we drive the slapdown with a
slider and move the wall along that exact, real curve -- so it matches Onshape, including the
nonlinear cam, with no physics and no guessing.

Controls:
    "slapdown (deg)" slider = deploy the arm; the wall follows Onshape's curve
    RIGHT-drag = orbit   MIDDLE-drag = pan   SCROLL = zoom

Usage:
    python view_onshape.py [robot_dir]      (default: robot_intake)
"""

import json
import math
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

MOUSE_MOVE_EVENT = 1
MOUSE_BUTTON_EVENT = 2
KEY_IS_DOWN = 1
KEY_WAS_TRIGGERED = 2
KEY_WAS_RELEASED = 4

WALL_SIGN = 1.0  # flip to -1.0 if the wall slides the wrong way


def main():
    robot_dir = sys.argv[1] if len(sys.argv) > 1 else "robot_intake"
    urdf_path = f"{robot_dir}/robot.urdf"

    with open(f"{robot_dir}/coupling.json") as f:
        curve = json.load(f)["curve"]
    degs = np.array([c[0] for c in curve])
    disps = np.array([c[1] for c in curve])
    order = np.argsort(degs)              # np.interp needs ascending x
    xp, fp = degs[order], disps[order]
    deg_min = float(degs.min())

    if p.connect(p.GUI) < 0:
        raise RuntimeError("Could not open a PyBullet GUI window.")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    driver = None
    walls = []
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        if info[2] == p.JOINT_REVOLUTE and driver is None:
            driver = j
        elif info[2] == p.JOINT_PRISMATIC:
            walls.append(j)
            # widen the wall travel limit so the full cam curve (to ~0.27 m) isn't clipped
            try:
                p.changeDynamics(robot, j, jointLowerLimit=-1.0, jointUpperLimit=1.0)
            except TypeError:
                pass

    s_drive = p.addUserDebugParameter("slapdown (deg)", deg_min, 0.0, 0.0)

    print(f"Loaded {urdf_path}: driver joint {driver}, wall joints {walls}, "
          f"coupling curve {len(curve)} pts ({deg_min:.0f}..0 deg)")
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45,
                                 cameraPitch=-30, cameraTargetPosition=[0, 0, 0.3])
    print("\nDrive 'slapdown (deg)' -- the wall follows Onshape's exact cam curve.")
    print("RIGHT-drag = orbit, MIDDLE-drag = pan, SCROLL = zoom.")

    drag = {"right": False, "middle": False, "x": 0.0, "y": 0.0}

    def cam():
        c = p.getDebugVisualizerCamera()
        return c[8], c[9], c[10], c[11], c[4], c[5]

    while p.isConnected():
        deg = p.readUserDebugParameter(s_drive)
        # Drive the slapdown to the exact angle, and the wall to Onshape's solved displacement.
        p.setJointMotorControl2(robot, driver, p.POSITION_CONTROL,
                                targetPosition=math.radians(deg), force=500)
        disp = WALL_SIGN * float(np.interp(deg, xp, fp))
        for wj in walls:
            p.setJointMotorControl2(robot, wj, p.POSITION_CONTROL,
                                    targetPosition=disp, force=500)

        for ev in p.getMouseEvents():
            etype, mx, my, btn, state = ev
            if etype == MOUSE_BUTTON_EVENT and btn in (1, 2):
                key = "right" if btn == 2 else "middle"
                if state & (KEY_WAS_TRIGGERED | KEY_IS_DOWN):
                    drag[key] = True; drag["x"], drag["y"] = mx, my
                if state & KEY_WAS_RELEASED:
                    drag[key] = False
            elif etype == MOUSE_MOVE_EVENT:
                dx, dy = mx - drag["x"], my - drag["y"]
                if drag["right"]:
                    yaw, pitch, dist, tgt, _u, _f = cam()
                    p.resetDebugVisualizerCamera(dist, yaw + dx * 0.4,
                                                 max(-89, min(89, pitch - dy * 0.4)), tgt)
                elif drag["middle"]:
                    yaw, pitch, dist, tgt, up, fwd = cam()
                    right = np.cross(fwd, up); right /= (np.linalg.norm(right) or 1.0)
                    s = 0.0015 * dist
                    tgt = [tgt[i] - right[i] * dx * s + up[i] * dy * s for i in range(3)]
                    p.resetDebugVisualizerCamera(dist, yaw, pitch, tgt)
                drag["x"], drag["y"] = mx, my

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
