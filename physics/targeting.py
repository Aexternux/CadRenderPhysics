"""
Drag-corrected shooter targeting.

Your StationaryShooterCalculator solves the flywheel RPM from vacuum projectile math. With the
FUEL ball's real air drag, that under-shoots badly. This finds the RPM that ACTUALLY lands the
ball in a goal at a given distance+height (via the calibrated drag sim) and compares it to the
vacuum answer -> a corrected lookup table your robot could use.

Run: python physics/targeting.py
"""

import numpy as np
import mujoco

R_BALL, M_BALL = 0.0751, 0.215
LAUNCH_DEG, LAUNCH_H = 65.0, 0.6
R_WHEEL = 0.0381          # 3" flywheel (from robot code)
EFFICIENCY = 0.85         # calibrated surface->exit efficiency
G = 9.81


def model(drag):
    dens = 1.225 if drag else 0.0
    fluid = 'fluidshape="ellipsoid" fluidcoef="0.47 0.25 1.5 1.0 1.0"' if drag else ''
    return f"""
<mujoco><option timestep="0.001" density="{dens}" viscosity="1.8e-5" integrator="RK4"/>
  <worldbody><body name="ball" pos="0 0 {LAUNCH_H}"><freejoint/>
    <geom type="sphere" size="{R_BALL}" mass="{M_BALL}" {fluid}/></body></worldbody></mujoco>"""


def z_at_distance(speed, target_x, drag=True):
    """Height of the ball when it crosses target_x (descending branch), or None if it never reaches."""
    m = mujoco.MjModel.from_xml_string(model(drag))
    d = mujoco.MjData(m)
    va = np.radians(LAUNCH_DEG)
    vadr = m.jnt_dofadr[m.body("ball").jntadr[0]]
    qadr = m.jnt_qposadr[m.body("ball").jntadr[0]]
    spin = speed * 1.2                                  # backspin ~ proportional to exit speed
    d.qvel[vadr:vadr + 3] = [speed * np.cos(va), 0, speed * np.sin(va)]
    d.qvel[vadr + 3:vadr + 6] = [0, spin, 0]
    px, pz = 0.0, LAUNCH_H
    rising = True
    for _ in range(8000):
        mujoco.mj_step(m, d)
        x, _, z = d.qpos[qadr:qadr + 3]
        if z < pz:
            rising = False
        if x >= target_x:                               # crossed the goal plane
            f = (target_x - px) / (x - px) if x != px else 0
            return pz + f * (z - pz), (not rising)
        px, pz = x, z
        if z < -1:
            break
    return None, False


def rpm_for(speed):
    surface = speed / EFFICIENCY
    return surface / (2 * np.pi * R_WHEEL) * 60.0


def solve(distance, goal_h, drag=True):
    """Binary-search exit speed so the ball is at goal_h when it crosses `distance` (descending)."""
    lo, hi = 3.0, 45.0
    best = None
    for _ in range(40):
        mid = (lo + hi) / 2
        z, descending = z_at_distance(mid, distance, drag)
        if z is None:
            lo = mid                                    # didn't reach -> need more speed
            continue
        if abs(z - goal_h) < 0.02 and descending:
            return mid
        # if too low at the goal plane, need more speed; if too high, less
        if z < goal_h:
            lo = mid
        else:
            hi = mid
        best = mid
    return best


def vacuum_rpm(distance, goal_h):
    va = np.radians(LAUNCH_DEG)
    dh = goal_h - LAUNCH_H
    denom = 2 * np.cos(va) ** 2 * (distance * np.tan(va) - dh)
    if denom <= 0:
        return None
    v = np.sqrt(G * distance ** 2 / denom)
    return rpm_for(v)


if __name__ == "__main__":
    GOAL_H = 2.0     # goal height above floor (placeholder -- set to real REBUILT goal height)
    print(f"Goal height {GOAL_H} m, launch {LAUNCH_DEG:.0f} deg, 3\" flywheels, eff {EFFICIENCY:.0%}\n")
    print(f"{'distance':>9} | {'vacuum RPM (your code)':>22} | {'REAL RPM (with drag)':>21} | {'under by':>9}")
    for dist in (3, 4, 5, 6, 7):
        vac = vacuum_rpm(dist, GOAL_H)
        real_v = solve(dist, GOAL_H)
        real_rpm = rpm_for(real_v) if real_v else None
        if vac and real_rpm:
            under = (real_rpm - vac) / real_rpm
            print(f"{dist:>7} m | {vac:>19.0f}    | {real_rpm:>18.0f}    | {under:>8.0%}")
        else:
            print(f"{dist:>7} m | {'--':>19}    | {'out of range':>18}    |")
    print("\n=> With drag you need MUCH higher RPM than the vacuum math predicts. "
          "This corrected mapping is what a real shooter lookup table needs.")
