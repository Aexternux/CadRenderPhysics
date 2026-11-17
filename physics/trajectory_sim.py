"""
FUEL ball flight with real aerodynamics (drag + Magnus lift from backspin).

Uses MuJoCo's built-in fluid model (option density = air, geom fluidshape="ellipsoid"). Launches
the calibrated ball at the team's 65 deg angle at the calibrated exit speeds, and measures real
range vs the vacuum (no-drag) range the team's projectile math assumes. Foam is light + big, so
drag matters a lot.

Run: python physics/trajectory_sim.py            (range table: drag vs vacuum)
     python physics/trajectory_sim.py render      (renders the 4000 RPM arc)
"""

import sys

import mujoco
import numpy as np
from PIL import Image

R_BALL, M_BALL = 0.0751, 0.215
LAUNCH_DEG = 65.0
LAUNCH_H = 0.6

# calibrated (RPM -> exit speed m/s, backspin rad/s) from shooter_sim
SHOTS = {2000: (9.33, 12.8), 3000: (13.61, 17.3), 4000: (17.63, 20.7),
         5000: (21.48, 21.8), 6000: (24.99, 25.2)}


def model(drag):
    dens = 1.225 if drag else 0.0
    fluid = 'fluidshape="ellipsoid" fluidcoef="0.47 0.25 1.5 1.0 1.0"' if drag else ''
    return f"""
<mujoco>
  <option timestep="0.001" density="{dens}" viscosity="1.8e-5" integrator="RK4"/>
  <visual><global offwidth="1200" offheight="500"/></visual>
  <worldbody>
    <light pos="5 -3 6" dir="0 0.4 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="40 6 0.1" pos="10 0 0" rgba="0.2 0.22 0.26 1"/>
    <body name="ball" pos="0 0 {LAUNCH_H}">
      <freejoint/>
      <geom type="sphere" size="{R_BALL}" mass="{M_BALL}" {fluid} rgba="0.95 0.8 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def fly(speed, spin, drag, capture=False):
    m = mujoco.MjModel.from_xml_string(model(drag))
    d = mujoco.MjData(m)
    va = np.radians(LAUNCH_DEG)
    vadr = m.jnt_dofadr[m.body("ball").jntadr[0]]
    qadr = m.jnt_qposadr[m.body("ball").jntadr[0]]
    d.qvel[vadr:vadr + 3] = [speed * np.cos(va), 0, speed * np.sin(va)]
    d.qvel[vadr + 3:vadr + 6] = [0, spin, 0]     # backspin about +Y (Magnus lift)
    traj = []
    apex = LAUNCH_H
    prev_z = LAUNCH_H
    for step in range(8000):
        mujoco.mj_step(m, d)
        x, y, z = d.qpos[qadr:qadr + 3]
        apex = max(apex, z)
        if capture and step % 12 == 0:
            traj.append((x, z))
        # landed (came back down to launch height, past the top)
        if z < LAUNCH_H and prev_z >= LAUNCH_H and step > 50:
            return x, apex, d.time, traj
        prev_z = z
    return d.qpos[qadr], apex, d.time, traj


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        render()
        return
    print(f"FUEL ball @ {LAUNCH_DEG:.0f} deg launch. range measured back to launch height.\n")
    print(f"{'RPM':>6} | {'exit v':>7} | {'vacuum range':>13} | {'REAL range (air)':>16} | {'drag loss':>10}")
    for rpm, (v, spin) in SHOTS.items():
        r_vac, _, _, _ = fly(v, spin, drag=False)
        r_air, apex, t, _ = fly(v, spin, drag=True)
        loss = (r_vac - r_air) / r_vac if r_vac else 0
        print(f"{rpm:>6} | {v:>6.1f}m | {r_vac:>11.1f}m | {r_air:>14.1f}m | {loss:>9.0%}")
    print("\n(air drag on light foam is large -> the team's vacuum math over-predicts range; "
          "this is why real shooters need calibration/lookup tables.)")


def render():
    v, spin = SHOTS[4000]
    _, _, _, traj = fly(v, spin, drag=True, capture=True)
    m = mujoco.MjModel.from_xml_string(model(True))
    d = mujoco.MjData(m)
    va = np.radians(LAUNCH_DEG)
    vadr = m.jnt_dofadr[m.body("ball").jntadr[0]]
    d.qvel[vadr:vadr + 3] = [v * np.cos(va), 0, v * np.sin(va)]
    d.qvel[vadr + 3:vadr + 6] = [0, spin, 0]
    renderer = mujoco.Renderer(m, 500, 1200)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [max(4, traj[-1][0] / 2 if traj else 5), 0, 1.2]
    cam.distance, cam.azimuth, cam.elevation = 14, 90, -6
    frames = []
    for step in range(2600):
        mujoco.mj_step(m, d)
        if step % 12 == 0:
            renderer.update_scene(d, cam)
            frames.append(Image.fromarray(renderer.render()))
    frames[0].save("physics/shot_arc.gif", save_all=True, append_images=frames[1:],
                   duration=40, loop=0)
    print(f"saved physics/shot_arc.gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()
