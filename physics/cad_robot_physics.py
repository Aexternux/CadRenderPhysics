"""
CAD -> working physics robot.

Takes the already-uploaded intake CAD (robot_intake) and builds a MuJoCo model where:
  - the static structure is fixed collision geometry,
  - the slapdown arm is a MOTORIZED revolute joint (axis extracted from the real sampled motion),
  - a calibrated FUEL ball can be scooped/pushed by the deploying arm.

No API needed -- uses meshes + motion we already have.

Run: python physics/cad_robot_physics.py            (headless: reports interaction)
     python physics/cad_robot_physics.py render      (renders a GIF of the intake + ball)
"""

import json
import sys

import mujoco
import numpy as np
from PIL import Image
from transforms3d.quaternions import mat2quat

DIR = "robot_intake"
R_BALL, M_BALL = 0.0751, 0.215


def _m(T):
    return np.array(T, float).reshape(4, 4)


def posquat(T):
    R = np.array([[T[0], T[1], T[2]], [T[4], T[5], T[6]], [T[8], T[9], T[10]]])
    w, x, y, z = mat2quat(R)
    return [T[3], T[7], T[11]], [w, x, y, z]


def load():
    data = json.load(open(f"{DIR}/onshape_frames.json"))
    part_refs = data["part_refs"]
    frames = {f["deg"]: f["poses"] for f in data["frames"]}
    rest = frames[0]
    deployed_deg = min(frames)                 # most-negative angle = fully deployed
    deployed = frames[deployed_deg]
    return part_refs, rest, deployed, deployed_deg


def arm_group_and_axis(rest, deployed):
    """Find the largest rigidly-rotating group (the arm) and its world hinge (origin, axis, angle)."""
    deltas = {}
    for k in deployed:
        if k in rest:
            D = _m(deployed[k]) @ np.linalg.inv(_m(rest[k]))
            if not np.allclose(D, np.eye(4), atol=1e-4):
                deltas[k] = D
    # cluster moved parts by their delta -> rigid groups; take the largest (the arm)
    groups = {}
    for k, D in deltas.items():
        key = tuple(np.round(D.reshape(16), 3))
        groups.setdefault(key, []).append(k)
    arm = max(groups.values(), key=len)
    D = deltas[arm[0]]
    R = D[:3, :3]
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (np.linalg.norm(axis) or 1)
    origin = np.linalg.lstsq(np.eye(3) - R, D[:3, 3], rcond=None)[0]
    return set(arm), origin, axis, ang


def build():
    part_refs, rest, deployed, dep_deg = load()
    arm_keys, origin, axis, dep_ang = arm_group_and_axis(rest, deployed)

    meshes = {}
    for r in part_refs.values():
        meshes.setdefault(r["mesh"], f"m{len(meshes)}")
    assets = "".join(f'<mesh name="{n}" file="onshape_assets/{h}.stl"/>' for h, n in meshes.items())

    static_geoms, arm_geoms, pts = [], [], []
    for key, ref in part_refs.items():
        if key not in rest:
            continue
        pos, quat = posquat(rest[key])
        pts.append(pos)
        g = (f'<geom type="mesh" mesh="{meshes[ref["mesh"]]}" pos="{pos[0]} {pos[1]} {pos[2]}" '
             f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" ')
        if key in arm_keys:
            arm_geoms.append(g + 'rgba="0.75 0.45 0.8 1" contype="4" conaffinity="1" density="600"/>')
        else:
            static_geoms.append(g + 'rgba="0.55 0.57 0.62 1" contype="2" conaffinity="1"/>')

    pts = np.array(pts)
    ctr = pts.mean(0)
    zmin = pts[:, 2].min()
    # place the ball on the floor in front of the intake, in the arm's reach
    ball_xy = pts[[k in arm_keys for k in [k for k in part_refs if k in rest]]].mean(0) \
        if len(arm_keys) else ctr
    ball = [ctr[0], ctr[1] - 0.15, zmin + R_BALL + 0.02]

    xml = f"""
<mujoco>
  <compiler meshdir="{DIR}" angle="degree" balanceinertia="true" discardvisual="false"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <asset>{assets}</asset>
  <worldbody>
    <light pos="{ctr[0]} {ctr[1]-1} {pts[:,2].max()+2}" dir="0 0.3 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="5 5 0.1" pos="{ctr[0]} {ctr[1]} {zmin-0.02}"
          rgba="0.2 0.22 0.26 1" contype="2" conaffinity="1"/>
    {''.join(static_geoms)}
    <body name="arm">
      <joint name="deploy" type="hinge" pos="{origin[0]} {origin[1]} {origin[2]}"
             axis="{axis[0]} {axis[1]} {axis[2]}" damping="0.5"/>
      {''.join(arm_geoms)}
    </body>
    <body name="ball" pos="{ball[0]} {ball[1]} {ball[2]}">
      <freejoint/>
      <geom type="sphere" size="{R_BALL}" mass="{M_BALL}" friction="1.6 0.02 0.001"
            solref="0.004 0.11" solimp="0.95 0.99 0.001" rgba="0.95 0.8 0.1 1"
            contype="1" conaffinity="6"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="deploy" kp="80" name="deploy_act"/>
  </actuator>
</mujoco>
"""
    return xml, ctr, dep_ang, len(arm_keys), len(static_geoms)


def main():
    render = len(sys.argv) > 1 and sys.argv[1] == "render"
    xml, ctr, dep_ang, n_arm, n_static = build()
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    print(f"arm = {n_arm} moving parts (motorized hinge), {n_static} static parts + floor + FUEL ball")
    print(f"deploy angle: {np.degrees(dep_ang):.0f} deg")

    act = m.actuator("deploy_act").id
    bqadr = m.jnt_qposadr[m.body("ball").jntadr[0]]
    ball0 = d.qpos[bqadr:bqadr + 3].copy()

    renderer = cam = None
    frames = []
    if render:
        renderer = mujoco.Renderer(m, 720, 960)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = ctr
        cam.distance, cam.azimuth, cam.elevation = 1.9, 135, -15

    target = -dep_ang       # deploy in the negative direction (stowed 0 -> deployed)
    for step in range(3000):
        # ramp the deploy command, then hold
        d.ctrl[act] = target * min(1.0, step / 1200.0)
        mujoco.mj_step(m, d)
        if render and step % 15 == 0:
            renderer.update_scene(d, cam)
            frames.append(Image.fromarray(renderer.render()))

    ball1 = d.qpos[bqadr:bqadr + 3]
    moved = float(np.linalg.norm(ball1 - ball0))
    if render:
        frames[0].save("physics/intake_ball.gif", save_all=True, append_images=frames[1:],
                       duration=45, loop=0)
        print(f"saved physics/intake_ball.gif ({len(frames)} frames)")
    print(f"ball moved {moved*100:.1f} cm during deploy -> "
          f"{'INTAKE INTERACTS WITH THE BALL' if moved > 0.03 else 'little interaction (tune ball position)'}")


if __name__ == "__main__":
    main()
