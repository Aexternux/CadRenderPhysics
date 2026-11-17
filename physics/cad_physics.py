"""
CAD -> physics: load the ACTUAL robot CAD meshes into MuJoCo as collision geometry and drop a
FUEL ball onto them, so the ball interacts with the real uploaded geometry (not a placeholder).

This proves the core of Stage 4: the ball passes through / collides with the real robot CAD.
(MuJoCo collides meshes as their convex hull — exact for the shooter wheels & ball; concave parts
like hoods/funnels will need convex decomposition for full fidelity, a later step.)

Run: python physics/cad_physics.py            (headless: measures the drop)
     python physics/cad_physics.py render      (renders a GIF)
"""

import json
import sys

import mujoco
import numpy as np
from PIL import Image
from transforms3d.quaternions import mat2quat

DIR = "robot_intake"
R_BALL, M_BALL = 0.0751, 0.215     # real FUEL ball


def load_rest():
    data = json.load(open(f"{DIR}/onshape_frames.json"))
    part_refs = data["part_refs"]
    rest = next(f for f in data["frames"] if f["deg"] == 0)["poses"]
    return part_refs, rest


def T_to_posquat(T):
    R = np.array([[T[0], T[1], T[2]], [T[4], T[5], T[6]], [T[8], T[9], T[10]]])
    pos = [T[3], T[7], T[11]]
    w, x, y, z = mat2quat(R)
    return pos, [w, x, y, z]


def build():
    part_refs, rest = load_rest()
    # one mesh asset per unique STL, reused by all its occurrences
    meshes = {}
    for r in part_refs.values():
        meshes.setdefault(r["mesh"], f"m{len(meshes)}")
    assets = "".join(
        f'<mesh name="{name}" file="onshape_assets/{h}.stl"/>' for h, name in meshes.items())

    geoms = []
    pts = []
    for key, ref in part_refs.items():
        if key not in rest:
            continue
        pos, quat = T_to_posquat(rest[key])
        pts.append(pos)
        geoms.append(f'<geom type="mesh" mesh="{meshes[ref["mesh"]]}" '
                     f'pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" '
                     f'rgba="0.6 0.6 0.65 1" contype="1" conaffinity="1"/>')
    pts = np.array(pts)
    ctr = pts.mean(0)
    top = pts[:, 2].max()
    hi = pts[pts[:, 2].argmax()]                 # highest part -> drop right above it
    drop = [hi[0], hi[1], top + 0.25]
    zmin = pts[:, 2].min()

    xml = f"""
<mujoco>
  <compiler meshdir="{DIR}" angle="degree"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <asset>{assets}</asset>
  <worldbody>
    <light pos="{ctr[0]} {ctr[1]-1} {top+2}" dir="0 0.3 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="{ctr[0]} {ctr[1]} {zmin-0.15}"
          rgba="0.2 0.22 0.26 1" contype="1" conaffinity="1"/>
    {''.join(geoms)}
    <body name="ball" pos="{drop[0]} {drop[1]} {drop[2]}">
      <freejoint/>
      <geom type="sphere" size="{R_BALL}" mass="{M_BALL}" rgba="0.95 0.8 0.1 1"
            friction="1.0 0.02 0.001" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""
    return xml, ctr, top, drop


def main():
    xml, ctr, top, drop = build()
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    print(f"loaded {m.ngeom - 1} CAD collision geoms + 1 ball; "
          f"mechanism spans z up to {top:.2f} m, ball dropped from {drop[2]:.2f} m")

    if len(sys.argv) > 1 and sys.argv[1] == "render":
        renderer = mujoco.Renderer(m, 720, 960)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = ctr
        cam.distance, cam.azimuth, cam.elevation = 1.9, 120, -12
        frames = []
        for step in range(1600):
            mujoco.mj_step(m, d)
            if step % 12 == 0:
                renderer.update_scene(d, cam)
                frames.append(Image.fromarray(renderer.render()))
        frames[0].save("physics/cad_drop.gif", save_all=True, append_images=frames[1:],
                       duration=45, loop=0)
        print(f"saved physics/cad_drop.gif ({len(frames)} frames)")
        return

    ballq = m.jnt_qposadr[m.body("ball").jntadr[0]]
    ball_gid = m.geom("ball").id if False else None  # ball geom is last
    z0 = d.qpos[ballq + 2]
    max_con = 0
    hit_cad = False
    floor_gid = m.geom("floor").id
    for _ in range(2500):
        mujoco.mj_step(m, d)
        max_con = max(max_con, d.ncon)
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if floor_gid not in (g1, g2):     # a contact that isn't with the floor = CAD contact
                hit_cad = True
    zf = d.qpos[ballq + 2]
    print(f"ball z: {z0:.3f} -> {zf:.3f} m | peak contacts: {max_con} | "
          f"touched a CAD part: {hit_cad}")
    print("=> " + ("BALL COLLIDES WITH THE REAL CAD [OK]" if hit_cad
                   else "no CAD contact (fell through an opening or collision off)"))


if __name__ == "__main__":
    main()
