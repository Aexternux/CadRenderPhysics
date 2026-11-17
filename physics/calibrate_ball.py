"""
Calibrate the FUEL ball's contact restitution to the real bounce test:
  1 m drop -> 20 in (0.508 m) rebound  => coefficient of restitution ~0.71.

MuJoCo bounce is governed by the contact solref damping ratio (2nd value): lower = bouncier.
Sweep it, drop the ball, measure rebound, and report which value reproduces 20 in.

Run: python physics/calibrate_ball.py
"""

import mujoco
import numpy as np

R_BALL, M_BALL = 0.0751, 0.215
DROP_H = 1.0
TARGET_BOUNCE = 0.508     # 20 inches


def bounce_height(dampratio, timeconst=0.004):
    xml = f"""
<mujoco>
  <option timestep="0.0005" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"
          solref="{timeconst} {dampratio}" solimp="0.95 0.99 0.001"/>
    <body name="ball" pos="0 0 {R_BALL + DROP_H}">
      <freejoint/>
      <geom type="sphere" size="{R_BALL}" mass="{M_BALL}"
            solref="{timeconst} {dampratio}" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    qadr = m.jnt_qposadr[m.body("ball").jntadr[0]]
    hit = False
    peak = 0.0
    for _ in range(6000):
        mujoco.mj_step(m, d)
        z = d.qpos[qadr + 2] - R_BALL     # height of ball bottom above floor
        if z < 0.002:
            hit = True
        if hit and z > peak:
            peak = z
    return peak


if __name__ == "__main__":
    print(f"target: 1.0 m drop -> {TARGET_BOUNCE:.3f} m bounce (CoR {np.sqrt(TARGET_BOUNCE):.2f})\n")
    print(f"{'dampratio':>10} | {'bounce (m)':>10} | {'bounce (in)':>11} | {'CoR':>5}")
    best = None
    for dr in (1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1):
        h = bounce_height(dr)
        cor = np.sqrt(max(h, 0) / DROP_H)
        print(f"{dr:>10.2f} | {h:>10.3f} | {h*39.37:>11.1f} | {cor:>5.2f}")
        if best is None or abs(h - TARGET_BOUNCE) < abs(best[1] - TARGET_BOUNCE):
            best = (dr, h)
    print(f"\nclosest: dampratio={best[0]} -> {best[1]*39.37:.1f} in "
          f"(target 20 in). Use this solref for the ball.")
