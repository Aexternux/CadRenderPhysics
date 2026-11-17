"""
Direct-manipulation viewer for an imported robot.

Loads <robot_dir>/robot.urdf in PyBullet and lets you GRAB AND DRAG the mechanism with the
mouse -- just like dragging mates in Onshape. Each movable joint is left free (with a touch
of friction so parts stay where you put them), so pulling a link articulates the chain
along its real degrees of freedom, respecting the joint limits.

Controls:
    LEFT-drag a part   = move it (drag the mechanism, like Onshape)
    RIGHT-drag         = orbit camera
    MIDDLE-drag        = pan camera
    SCROLL             = zoom

Usage:
    python view_robot.py [robot_dir]        (default: robot_intake)

Run this in your OWN terminal so the 3D window appears.
"""

import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

# getMouseEvents() codes (not exposed as named constants in all pybullet builds).
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

    robot = p.loadURDF(urdf_path, useFixedBase=True,
                       flags=p.URDF_USE_INERTIA_FROM_FILE)

    # Free every movable joint so the mechanism can be dragged by hand. A small velocity
    # friction keeps parts where you release them; joint limits are still enforced.
    movable = 0
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            movable += 1
            p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL,
                                    targetVelocity=0, force=0.3)
            p.changeDynamics(robot, j, jointDamping=0.1)
    print(f"Loaded {urdf_path} with {movable} draggable joints")

    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45,
                                 cameraPitch=-30, cameraTargetPosition=[0, 0, 0.3])
    print("\nLEFT-drag a part to move it (like Onshape).  "
          "RIGHT-drag = orbit, MIDDLE-drag = pan, SCROLL = zoom.")

    drag = {"right": False, "middle": False, "x": 0.0, "y": 0.0}
    ORBIT_SPEED = 0.4
    PAN_SPEED = 0.0015

    def cam_state():
        c = p.getDebugVisualizerCamera()
        return c[8], c[9], c[10], c[11], c[4], c[5]  # yaw, pitch, dist, target, up, fwd

    while p.isConnected():
        # Camera: right-drag orbit, middle-drag pan. (Left mouse is left to PyBullet so it
        # can pick and drag parts, and orbit on empty space.)
        for ev in p.getMouseEvents():
            etype, mx, my, btn, state = ev
            if etype == MOUSE_BUTTON_EVENT and btn in (1, 2):
                key = "right" if btn == 2 else "middle"
                if state & (KEY_WAS_TRIGGERED | KEY_IS_DOWN):
                    drag[key] = True
                    drag["x"], drag["y"] = mx, my
                if state & KEY_WAS_RELEASED:
                    drag[key] = False
            elif etype == MOUSE_MOVE_EVENT:
                dx, dy = mx - drag["x"], my - drag["y"]
                if drag["right"]:
                    yaw, pitch, dist, tgt, _u, _f = cam_state()
                    yaw += dx * ORBIT_SPEED
                    pitch = max(-89.0, min(89.0, pitch - dy * ORBIT_SPEED))
                    p.resetDebugVisualizerCamera(dist, yaw, pitch, tgt)
                elif drag["middle"]:
                    yaw, pitch, dist, tgt, up, fwd = cam_state()
                    right = np.cross(fwd, up)
                    right /= (np.linalg.norm(right) or 1.0)
                    s = PAN_SPEED * dist
                    tgt = [tgt[i] - right[i] * dx * s + up[i] * dy * s for i in range(3)]
                    p.resetDebugVisualizerCamera(dist, yaw, pitch, tgt)
                drag["x"], drag["y"] = mx, my

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
