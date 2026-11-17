"""
Shoot the FUEL ball into a goal -- the payoff clip.

Combines everything: the calibrated ball, the drag-corrected exit speed needed to reach a goal at
a set distance/height (solved with air drag), launched at the team's 65 deg, arcing through real
aerodynamics into the target. Renders a well-lit, well-framed GIF.

Run: python physics/shoot_into_goal.py            (measures the shot)
     python physics/shoot_into_goal.py render      (renders the GIF)
"""

import sys

import mujoco
import numpy as np
from PIL import Image

from targeting import solve   # drag-aware exit-speed solver

R_BALL, M_BALL = 0.0751, 0.215
LAUNCH_DEG, LAUNCH_H = 65.0, 0.6
GX, GZ = 5.0, 2.0          # goal: 5 m away, 2 m high
GOAL_R = 0.30             # goal opening radius (generous target)


def model():
    return f"""
<mujoco>
  <option timestep="0.001" density="1.225" viscosity="1.8e-5" integrator="RK4"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <global offwidth="1200" offheight="620"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.28 0.30 0.34" rgb2="0.20 0.22 0.26"
             width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="2 -3 5" dir="0 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="5 3 5" dir="0 -0.5 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="20 8 0.1" pos="6 0 0" material="grid"/>
    <!-- shooter marker -->
    <geom type="box" size="0.12 0.15 0.3" pos="0 0 0.3" rgba="0.55 0.3 0.75 1"/>
    <!-- goal: a ring of posts around the target opening -->
    <geom type="cylinder" fromto="{GX} -{GOAL_R} {GZ} {GX} {GOAL_R} {GZ}" size="0.02"
          rgba="0.2 0.85 0.4 1"/>
    <geom type="cylinder" fromto="{GX} -{GOAL_R} {GZ-GOAL_R} {GX} -{GOAL_R} {GZ+GOAL_R}" size="0.02"
          rgba="0.2 0.85 0.4 1"/>
    <geom type="cylinder" fromto="{GX} {GOAL_R} {GZ-GOAL_R} {GX} {GOAL_R} {GZ+GOAL_R}" size="0.02"
          rgba="0.2 0.85 0.4 1"/>
    <geom type="cylinder" fromto="{GX} -{GOAL_R} {GZ+GOAL_R} {GX} {GOAL_R} {GZ+GOAL_R}" size="0.02"
          rgba="0.2 0.85 0.4 1"/>
    <geom type="box" size="0.02 {GOAL_R} 0.35" pos="{GX+0.35} 0 {GZ}" rgba="0.85 0.85 0.9 0.25"/>
    <body name="ball" pos="0 0 {LAUNCH_H}">
      <freejoint/>
      <geom type="sphere" size="{R_BALL}" mass="{M_BALL}"
            fluidshape="ellipsoid" fluidcoef="0.47 0.25 1.5 1.0 1.0" rgba="0.95 0.8 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def run(render=False):
    exit_speed = solve(GX, GZ)
    print(f"goal: {GX} m away, {GZ} m high. drag-solved exit speed: {exit_speed:.1f} m/s "
          f"(~{exit_speed/0.85/(2*np.pi*0.0381)*60:.0f} RPM on 3\" wheels)")

    m = mujoco.MjModel.from_xml_string(model())
    d = mujoco.MjData(m)
    va = np.radians(LAUNCH_DEG)
    vadr = m.jnt_dofadr[m.body("ball").jntadr[0]]
    qadr = m.jnt_qposadr[m.body("ball").jntadr[0]]
    d.qvel[vadr:vadr + 3] = [exit_speed * np.cos(va), 0, exit_speed * np.sin(va)]
    d.qvel[vadr + 3:vadr + 6] = [0, exit_speed * 1.2, 0]

    renderer = cam = None
    frames = []
    if render:
        renderer = mujoco.Renderer(m, 620, 1200)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [GX / 2, 0, 1.4]
        cam.distance, cam.azimuth, cam.elevation = 8.5, 90, -8

    closest = 1e9
    for step in range(2400):
        mujoco.mj_step(m, d)
        x, y, z = d.qpos[qadr:qadr + 3]
        closest = min(closest, np.hypot(x - GX, z - GZ))
        if render and step % 10 == 0:
            renderer.update_scene(d, cam)
            frames.append(Image.fromarray(renderer.render()))
        if x > GX + 0.5 or z < 0:
            break

    if render:
        frames[0].save("physics/goal_shot.gif", save_all=True, append_images=frames[1:],
                       duration=40, loop=0)
        print(f"saved physics/goal_shot.gif ({len(frames)} frames)")
    print(f"closest approach to goal center: {closest*100:.0f} cm "
          f"-> {'SCORED (within opening)' if closest < GOAL_R else 'missed'}")


if __name__ == "__main__":
    run(render=len(sys.argv) > 1 and sys.argv[1] == "render")
