"""
Stage 3: physics + contact simulator.

Loads <robot_dir>/robot.urdf with gravity and SELF-COLLISION enabled, so links of the same
robot can push on each other. The slapdown (revolute) is motor-driven by a "DEPLOY" slider;
the hopper wall(s) are left free, so when the slapdown rotates into the wall it physically
cams/pushes it along its slide -- reproducing the Onshape Tangent mate through real contact
instead of a kinematic joint.

Controls:
    DEPLOY slider (top-right) = drive the slapdown
    LEFT-drag a part          = grab/push it
    RIGHT-drag = orbit   MIDDLE-drag = pan   SCROLL = zoom

Usage:
    python sim_robot.py [robot_dir] [--headless]
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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    robot_dir = args[0] if args else "robot_intake"
    headless = "--headless" in sys.argv
    urdf_path = f"{robot_dir}/robot.urdf"

    p.connect(p.DIRECT if headless else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1)
    p.loadURDF("plane.urdf", [0, 0, -1.0])

    flags = (p.URDF_USE_INERTIA_FROM_FILE
             | p.URDF_USE_SELF_COLLISION
             | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=flags)

    # Classify joints: the slapdown (revolute) is the driver; everything else is left free.
    drivers, free = [], []
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        if info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            continue
        (drivers if info[2] == p.JOINT_REVOLUTE else free).append((j, info))

    # Free joints: small friction so they hold but can still be pushed by contact.
    for j, _info in free:
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0.5)
        p.changeDynamics(robot, j, jointDamping=0.2)

    # Decent contact friction on every link (and the base).
    for link in range(-1, p.getNumJoints(robot)):
        p.changeDynamics(robot, link, lateralFriction=0.9, restitution=0.0,
                         contactStiffness=1e5, contactDamping=1e3)

    # A DEPLOY slider per driver joint.
    sliders = []
    for j, info in drivers:
        lo, hi = info[8], info[9]
        if lo >= hi:
            lo, hi = -3.14159, 3.14159
        name = "DEPLOY " + info[1].decode()
        sliders.append((j, p.addUserDebugParameter(name, lo, hi, 0.0) if not headless else None))

    print(f"Loaded {urdf_path}: {len(drivers)} driven, {len(free)} free joints, self-collision ON")

    if headless:
        # Stability smoke-test: drive the slapdown through its range and check states stay finite.
        for step in range(600):
            for (j, info) in drivers:
                lo, hi = info[8], info[9]
                if lo >= hi:
                    lo, hi = -1.0, 1.0
                target = lo + (hi - lo) * (0.5 - 0.5 * np.cos(step / 600.0 * np.pi))
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, target, force=200)
            p.stepSimulation()
        states = [p.getJointState(robot, j)[0] for j in range(p.getNumJoints(robot))]
        ok = all(np.isfinite(s) for s in states)
        base = p.getBasePositionAndOrientation(robot)[0]
        print("HEADLESS RESULT:", "STABLE" if ok and np.isfinite(base).all() else "UNSTABLE/NaN")
        print("  final joint positions:", [round(float(s), 4) for s in states])
        p.disconnect()
        return

    # ---- interactive GUI loop ----
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45,
                                 cameraPitch=-30, cameraTargetPosition=[0, 0, 0.3])
    print("\nDrag the DEPLOY slider to swing the slapdown; watch it push the wall.")
    print("RIGHT-drag = orbit, MIDDLE-drag = pan, SCROLL = zoom.")

    drag = {"right": False, "middle": False, "x": 0.0, "y": 0.0}

    def cam():
        c = p.getDebugVisualizerCamera()
        return c[8], c[9], c[10], c[11], c[4], c[5]

    while p.isConnected():
        for j, sid in sliders:
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    targetPosition=p.readUserDebugParameter(sid), force=200)
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
