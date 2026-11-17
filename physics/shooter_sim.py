"""
Parametric flywheel-shooter physics prototype (MuJoCo).

Dual counter-rotating flywheels grip a ball, compress it, and launch it. This is a physics
sandbox to (a) prove the shooting dynamics come out realistic and (b) get calibrated against a
real ball + a real data point. Later we swap the parametric wheels/hood for the actual CAD
shooter geometry and the ball for the user's real ball parameters.

Contact compliance (solref/solimp) models BALL COMPRESSION: the ball penetrates the wheels by
roughly its compression depth under the shooter's normal force. Friction transfers the wheel
surface speed to the ball (with slip) -> launch.

Run: python physics/shooter_sim.py
"""

import numpy as np
import mujoco

# ---- 2026 REBUILT "FUEL" ball (AndyMark) ----
# Diameter 5.91 in = 0.150 m ; mass 0.448-0.5 lb ~ 0.215 kg ; high-density polyurethane foam.
R_BALL = 0.0751          # radius (m)  = 5.91 in / 2
M_BALL = 0.215           # mass (kg)   ~ 0.474 lb midpoint
# NOTE ON CONTACT: the ball's free-bounce CoR is calibrated to 0.71 (see calibrate_ball.py,
# solref "0.004 0.11"). But during a SHOT the ball is captured/compressed against the wheels
# by the hood, where sustained grip (friction) dominates, so here we use a grippier normal
# contact. Both are the same real ball; the regime differs (impact-bounce vs held-compression).
SOLREF = "0.006 1"       # grippy held-compression contact for launching
SOLIMP = "0.90 0.95 0.001"
COMPRESSION = 0.030      # squish under shooter grip (m)
FRICTION = 2.2           # tuned toward the team's ~90% surface->exit efficiency
# ---- shooter geometry (from robot code: dual 3" flywheels, 10% top/bottom differential) ----
R_WHEEL = 0.051          # PLACEHOLDER wheel radius (real is 3"=0.0381; swap for CAD hood+wheels)
HW = 0.025               # flywheel half-width (m)
SPIN_DIFF = 0.10         # top wheel 10% faster than bottom -> backspin (SPIN_RATIO from code)
ZC = 0.6                 # launch height (m)

GAP = 2 * R_BALL - COMPRESSION          # wheel-surface gap (< ball diameter -> compression)
ZTOP = ZC + R_WHEEL + GAP / 2
ZBOT = ZC - R_WHEEL - GAP / 2


def build_model(rpm):
    return f"""
<mujoco>
  <compiler angle="degree"/>
  <option timestep="0.0002" integrator="implicitfast" cone="elliptic"/>
  <default>
    <geom solref="{SOLREF}" solimp="{SOLIMP}"/>
  </default>
  <visual><global offwidth="960" offheight="640"/></visual>
  <worldbody>
    <light pos="0.4 -1.2 2.5" dir="-0.1 0.4 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="6 3 0.1" pos="1.5 0 0" rgba="0.22 0.24 0.28 1"/>
    <body name="top" pos="0 0 {ZTOP}">
      <joint name="jt" type="hinge" axis="0 1 0" damping="0.01" armature="0.01"/>
      <geom type="cylinder" size="{R_WHEEL} {HW}" euler="90 0 0"
            friction="{FRICTION} 0.02 0.001" mass="0.8" rgba="0.7 0.3 0.3 1"/>
    </body>
    <body name="bot" pos="0 0 {ZBOT}">
      <joint name="jb" type="hinge" axis="0 1 0" damping="0.01" armature="0.01"/>
      <geom type="cylinder" size="{R_WHEEL} {HW}" euler="90 0 0"
            friction="{FRICTION} 0.02 0.001" mass="0.8" rgba="0.3 0.3 0.7 1"/>
    </body>
    <body name="ball" pos="-0.13 0 {ZC}">
      <freejoint/>
      <geom name="ball" type="sphere" size="{R_BALL}" mass="{M_BALL}"
            friction="{FRICTION} 0.02 0.001" rgba="0.95 0.8 0.1 1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="jt" kv="20" name="at"/>
    <velocity joint="jb" kv="20" name="ab"/>
  </actuator>
</mujoco>
"""


def shoot(rpm):
    m = mujoco.MjModel.from_xml_string(build_model(rpm))
    d = mujoco.MjData(m)
    w = rpm * 2 * np.pi / 60.0
    # counter-rotate (both gap surfaces move +X); top 10% faster -> backspin
    ctrl_top, ctrl_bot = -w * (1 + SPIN_DIFF), +w * (1 - SPIN_DIFF)

    ball_qadr = m.jnt_qposadr[m.body("ball").jntadr[0]]
    ball_vadr = m.jnt_dofadr[m.body("ball").jntadr[0]]
    top_dof = m.jnt_dofadr[m.joint("jt").id]
    bot_dof = m.jnt_dofadr[m.joint("jb").id]

    # start the wheels already at speed (avoids the ball falling during a slow spin-up)
    d.qvel[top_dof] = ctrl_top
    d.qvel[bot_dof] = ctrl_bot
    d.ctrl[:] = [ctrl_top, ctrl_bot]
    mujoco.mj_forward(m, d)
    # feed the ball into the wheels
    d.qvel[ball_vadr:ball_vadr + 3] = [3.0, 0, 0]

    exit_speed = 0.0
    exit_vx = 0.0
    spin = 0.0
    launched = False
    for _ in range(20000):
        d.ctrl[:] = [ctrl_top, ctrl_bot]
        mujoco.mj_step(m, d)
        pos = d.qpos[ball_qadr:ball_qadr + 3]
        vel = d.qvel[ball_vadr:ball_vadr + 3]
        # once the ball is past the wheels and airborne, record its speed
        if pos[0] > R_WHEEL + R_BALL:
            s = float(np.linalg.norm(vel))
            if s > exit_speed:
                exit_speed = s
                exit_vx = float(vel[0])
                spin = float(np.linalg.norm(d.qvel[ball_vadr + 3:ball_vadr + 6]))
            launched = True
        if launched and pos[0] > 0.6:
            break

    surface = w * R_WHEEL
    return exit_speed, exit_vx, spin, surface


def render_shot(rpm, fname="physics/shot.gif"):
    from PIL import Image
    m = mujoco.MjModel.from_xml_string(build_model(rpm))
    d = mujoco.MjData(m)
    w = rpm * 2 * np.pi / 60.0
    ball_vadr = m.jnt_dofadr[m.body("ball").jntadr[0]]
    top_dof = m.jnt_dofadr[m.joint("jt").id]
    bot_dof = m.jnt_dofadr[m.joint("jb").id]
    d.qvel[top_dof], d.qvel[bot_dof] = -w, w
    d.ctrl[:] = [-w, w]
    mujoco.mj_forward(m, d)
    d.qvel[ball_vadr:ball_vadr + 3] = [3.0, 0, 0]

    renderer = mujoco.Renderer(m, 640, 960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [1.2, 0, 0.4]
    cam.distance, cam.azimuth, cam.elevation = 3.4, 90, -8
    frames = []
    for step in range(1400):
        d.ctrl[:] = [-w, w]
        mujoco.mj_step(m, d)
        if step % 8 == 0:
            renderer.update_scene(d, cam)
            frames.append(Image.fromarray(renderer.render()))
    frames[0].save(fname, save_all=True, append_images=frames[1:], duration=40, loop=0)
    print(f"saved {fname} ({len(frames)} frames) — {rpm} RPM shot of the FUEL ball")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        render_shot(int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
        raise SystemExit
    print(f"ball: r={R_BALL*100:.1f}cm m={M_BALL*1000:.0f}g compression={COMPRESSION*1000:.0f}mm "
          f"friction={FRICTION}")
    print(f"{'RPM':>6} | {'surface v (m/s)':>15} | {'exit v (m/s)':>12} | {'exit vx':>8} | "
          f"{'spin (rad/s)':>12} | efficiency")
    for rpm in (2000, 3000, 4000, 5000, 6000):
        es, evx, spin, surf = shoot(rpm)
        eff = es / surf if surf else 0
        print(f"{rpm:>6} | {surf:>15.2f} | {es:>12.2f} | {evx:>8.2f} | {spin:>12.1f} | {eff:>.0%}")
