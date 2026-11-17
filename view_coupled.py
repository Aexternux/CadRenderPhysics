"""
Kinematic coupled viewer -- drive the slapdown, everything follows. Smooth, no tuning.

One slider drives the slapdown; the hopper wall(s) track it automatically (coupling ratio is
auto-computed so a full deploy = full wall travel). Solved as a driven sim that actually
steps, so it moves visibly and smoothly.

Controls:
    "slapdown" slider = move the whole mechanism
    RIGHT-drag = orbit   MIDDLE-drag = pan   SCROLL = zoom

Usage:
    python view_coupled.py [robot_dir]      (default: robot_intake)
"""

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


def main():
    robot_dir = sys.argv[1] if len(sys.argv) > 1 else "robot_intake"
    urdf_path = f"{robot_dir}/robot.urdf"

    if p.connect(p.GUI) < 0:
        raise RuntimeError("Could not open a PyBullet GUI window.")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    driver = None
    coupled = []
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        if info[2] == p.JOINT_REVOLUTE and driver is None:
            driver = (j, info)
        elif info[2] == p.JOINT_PRISMATIC:
            coupled.append((j, info))
    if driver is None:
        print("No revolute driver joint found.")
        return

    dj, dinfo = driver
    lo, hi = dinfo[8], dinfo[9]
    if lo >= hi:
        lo, hi = -3.14159, 3.14159
    theta_stowed = hi
    span = abs(hi - lo) or 1.0

    # Auto coupling: a full slapdown sweep = full wall travel.
    ratios = []
    for cj, cinfo in coupled:
        clo, chi = cinfo[8], cinfo[9]
        travel = (chi - clo) if chi > clo else 0.2
        ratios.append(travel / span)

    s_drive = p.addUserDebugParameter(f"slapdown ({dinfo[1].decode()})", lo, hi, hi)

    print(f"Loaded {urdf_path}: driver='{dinfo[1].decode()}', "
          f"{len(coupled)} coupled wall joint(s), auto ratios={[round(r,3) for r in ratios]}")
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45,
                                 cameraPitch=-30, cameraTargetPosition=[0, 0, 0.3])
    print("\nDrag the 'slapdown' slider -- the whole mechanism moves together.")
    print("RIGHT-drag = orbit, MIDDLE-drag = pan, SCROLL = zoom.")

    drag = {"right": False, "middle": False, "x": 0.0, "y": 0.0}

    def cam():
        c = p.getDebugVisualizerCamera()
        return c[8], c[9], c[10], c[11], c[4], c[5]

    while p.isConnected():
        theta = p.readUserDebugParameter(s_drive)
        # Drive the slapdown to the slider value.
        p.setJointMotorControl2(robot, dj, p.POSITION_CONTROL, targetPosition=theta, force=300)
        # Couple each wall to the slapdown's deployment.
        deploy = theta_stowed - theta
        for (cj, cinfo), r in zip(coupled, ratios):
            clo, chi = cinfo[8], cinfo[9]
            target = r * deploy
            if clo < chi:
                target = max(clo, min(chi, target))
            p.setJointMotorControl2(robot, cj, p.POSITION_CONTROL,
                                    targetPosition=target, force=300)

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
