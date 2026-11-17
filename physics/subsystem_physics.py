"""
User-tagged mechanisms -> working physics.

Consumes what the web viewer produces: a job's model.json (parts + rest poses + meshes) and
subsystems.json (the mechanisms YOU selected: rollers, pivots, sliders, each with an axis).
Builds a MuJoCo robot where:
  - every 'roller' subsystem is a hinge with a velocity actuator -> it SPINS continuously,
  - every 'revolute'/'slider' subsystem is a motorized joint you can command,
  - all other parts are fixed collision geometry (optional for speed on huge bots),
and drops a calibrated FUEL ball onto each roller to show it gets carried/thrown.

No Onshape API needed -- runs entirely off cached meshes + your selections.

Run: python physics/subsystem_physics.py <job>            (report interaction)
     python physics/subsystem_physics.py <job> render      (render a GIF)
     python physics/subsystem_physics.py <job> render --all-static   (include every static part)
where <job> is a folder under web/jobs/ (e.g. b221a95616), or a path to a model.json.
"""

import json
import os
import sys

import mujoco
import numpy as np
from PIL import Image
from transforms3d.quaternions import mat2quat

R_BALL, M_BALL = 0.0751, 0.215          # calibrated 2026 REBUILT FUEL ball
SPIN_RPS = 12.0                         # roller surface spin (rev/s) for the demo
MAX_STATIC_DEFAULT = 300                # cap static meshes unless --all-static (keeps compile fast)


def posquat(T):
    R = np.array([[T[0], T[1], T[2]], [T[4], T[5], T[6]], [T[8], T[9], T[10]]])
    w, x, y, z = mat2quat(R)
    return (T[3], T[7], T[11]), (w, x, y, z)


def resolve_job(job):
    """Return (model.json path, mesh dir) for a job id under web/jobs, or a direct model.json path."""
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    if job.endswith(".json"):
        d = os.path.dirname(os.path.abspath(job))
        return job, d
    d = os.path.join(proj, "web", "jobs", job)
    return os.path.join(d, "model.json"), d


def load(job):
    mpath, mdir = resolve_job(job)
    model = json.load(open(mpath))
    subs_path = os.path.join(mdir, "subsystems.json")
    subs = json.load(open(subs_path)) if os.path.exists(subs_path) else []
    rest = model.get("rest")
    if rest is None:                                  # fall back to a DOF's zero frame
        for dof in model.get("dofs", []):
            fr = next((f for f in dof["frames"] if f["val"] == 0), dof["frames"][0])
            rest = fr["poses"]
            break
    return model, subs, rest or {}, mdir


def build(job, all_static=False):
    model, subs, rest, mdir = load(job)
    if not subs:
        raise SystemExit(f"no subsystems.json for '{job}' -- tag some mechanisms in the web viewer first.")

    parts = model["parts"]
    member_of = {}                                    # part key -> subsystem index
    for i, s in enumerate(subs):
        for k in s["members"]:
            member_of[k] = i

    def has_mesh(h):
        return os.path.exists(os.path.join(mdir, "onshape_assets", f"{h}.stl"))

    meshes = {}                                       # hash -> mujoco mesh name (only meshes on disk)
    for r in parts.values():
        h = r["mesh"]
        if h not in meshes and has_mesh(h):
            meshes[h] = f"m{len(meshes)}"
    assets = "".join(f'<mesh name="{n}" file="onshape_assets/{h}.stl"/>' for h, n in meshes.items())

    sub_geoms = [[] for _ in subs]
    static_geoms, pts = [], []
    n_static_total = 0
    cap = 10 ** 9 if all_static else MAX_STATIC_DEFAULT
    for key, ref in parts.items():
        if key not in rest or ref["mesh"] not in meshes:
            continue
        pos, quat = posquat(rest[key])
        pts.append(pos)
        g = (f'<geom type="mesh" mesh="{meshes[ref["mesh"]]}" pos="{pos[0]} {pos[1]} {pos[2]}" '
             f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" ')
        if key in member_of:
            si = member_of[key]
            sub_geoms[si].append(g + f'rgba="0.75 0.45 0.8 1" contype="4" conaffinity="1" density="700"/>')
        else:
            n_static_total += 1
            if len(static_geoms) < cap:
                static_geoms.append(g + 'rgba="0.5 0.52 0.57 1" contype="2" conaffinity="1"/>')

    pts = np.array(pts)
    ctr = pts.mean(0)
    zmax = pts[:, 2].max()

    bodies, actuators, balls = [], [], []
    for i, s in enumerate(subs):
        o = s["axis"]["origin"]
        d = np.array(s["axis"]["dir"], float)
        d = d / (np.linalg.norm(d) or 1)
        jtype = "slide" if s["type"] == "slider" else "hinge"
        name = f"sub{i}"
        # armature + damping keep velocity actuators stable (avoids RK4/velocity NaN)
        joint = (f'<joint name="{name}" type="{jtype}" pos="{o[0]} {o[1]} {o[2]}" '
                 f'axis="{d[0]} {d[1]} {d[2]}" damping="0.4" armature="0.01"/>')
        bodies.append(f'<body name="{name}">{joint}{"".join(sub_geoms[i])}</body>')
        if s["type"] == "roller":
            actuators.append(f'<velocity joint="{name}" kv="3" name="{name}_act"/>')
        else:
            actuators.append(f'<position joint="{name}" kp="60" name="{name}_act"/>')
        # drop a ball just above this mechanism's centroid
        mc = np.array(o, float)
        if sub_geoms[i]:
            balls.append((mc[0], mc[1], zmax + 0.10 + i * 0.02, i))

    ball_xml = ""
    for bx, by, bz, i in balls:
        ball_xml += (
            f'<body name="ball{i}" pos="{bx} {by} {bz}"><freejoint/>'
            f'<geom type="sphere" size="{R_BALL}" mass="{M_BALL}" friction="1.8 0.02 0.001" '
            f'solref="0.004 0.11" solimp="0.95 0.99 0.001" rgba="0.95 0.8 0.1 1" '
            f'contype="1" conaffinity="6"/></body>')

    xml = f"""
<mujoco>
  <compiler meshdir="{mdir}" angle="degree" balanceinertia="true" discardvisual="false"/>
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="1000" offheight="760"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/></visual>
  <asset>{assets}</asset>
  <worldbody>
    <light pos="{ctr[0]} {ctr[1]-1} {zmax+2}" dir="0 0.3 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="6 6 0.1" pos="{ctr[0]} {ctr[1]} {pts[:,2].min()-0.02}"
          rgba="0.18 0.2 0.24 1" contype="2" conaffinity="1"/>
    {''.join(static_geoms)}
    {''.join(bodies)}
    {ball_xml}
  </worldbody>
  <actuator>{''.join(actuators)}</actuator>
</mujoco>
"""
    meta = {"ctr": ctr, "zmax": zmax, "subs": subs, "n_static_total": n_static_total,
            "n_static_shown": len(static_geoms), "balls": balls}
    return xml, meta


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    job = sys.argv[1]
    render = "render" in sys.argv[2:]
    all_static = "--all-static" in sys.argv

    xml, meta = build(job, all_static=all_static)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    subs = meta["subs"]
    print(f"{len(subs)} tagged mechanism(s); {meta['n_static_shown']}/{meta['n_static_total']} static "
          f"parts shown{' (all)' if all_static else ''}, {len(meta['balls'])} ball(s)")
    for i, s in enumerate(subs):
        print(f"  - {s['name']:<24} {s['type']:<9} {len(s['members'])} parts")

    # command: rollers spin, positioned joints hold 0
    spin_radps = SPIN_RPS * 2 * np.pi
    for i, s in enumerate(subs):
        aid = m.actuator(f"sub{i}_act").id
        d.ctrl[aid] = spin_radps if s["type"] == "roller" else 0.0

    ball_qadr = {i: m.jnt_qposadr[m.body(f"ball{i}").jntadr[0]] for _, _, _, i in meta["balls"]}
    ball0 = {i: d.qpos[a:a + 3].copy() for i, a in ball_qadr.items()}

    renderer = cam = None
    frames = []
    if render:
        renderer = mujoco.Renderer(m, 760, 1000)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = meta["ctr"]
        cam.distance = float(np.linalg.norm([1, 1, 1])) * 0.9 + 0.6
        cam.azimuth, cam.elevation = 130, -18

    for step in range(4000):
        for i, s in enumerate(subs):
            aid = m.actuator(f"sub{i}_act").id
            d.ctrl[aid] = spin_radps if s["type"] == "roller" else 0.0
        mujoco.mj_step(m, d)
        if render and step % 15 == 0:
            renderer.update_scene(d, cam)
            frames.append(Image.fromarray(renderer.render()))

    moved = {i: float(np.linalg.norm(d.qpos[a:a + 3] - ball0[i])) for i, a in ball_qadr.items()}
    if render:
        out = f"physics/subsystem_{os.path.basename(job).replace('.json','')}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=45, loop=0)
        print(f"saved {out} ({len(frames)} frames)")
    for i, s in enumerate(subs):
        if i in moved:
            verdict = "MOVED by mechanism" if moved[i] > 0.02 else "little motion"
            print(f"  ball on {s['name']!r}: {moved[i]*100:.1f} cm -> {verdict}")


if __name__ == "__main__":
    main()
