"""Offline test for compose.py composition math (no API): placement @ local, prefixing, merge."""
import numpy as np

import compose


def T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return [float(x) for x in M.reshape(16)]


# placement: rotate the whole sub 90 deg about Z, then shift it to (1, 2, 3) in the robot
Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
PLACE = T(Rz, [1.0, 2.0, 3.0])

# a synthetic imported subassembly: 2 parts, one revolute DOF with 2 frames
sub_model = {
    "parts": {"a": {"mesh": "h1", "color": [1, 0, 0, 1]},
              "b": {"mesh": "h2", "color": [0, 1, 0, 1]}},
    "rest": {"a": T(np.eye(3), [1, 0, 0]),      # part a at (1,0,0) in the sub
             "b": T(np.eye(3), [0, 1, 0])},     # part b at (0,1,0) in the sub
    "dofs": [{"name": "left roller", "type": "Revolute", "unit": "deg", "min": 0, "max": 90,
              "frames": [{"val": 0,  "poses": {"a": T(np.eye(3), [1, 0, 0])}},
                         {"val": 90, "poses": {"a": T(Rz,        [0, 1, 0])}}]}]}

# ---- world_pose: world = placement @ local ----
w = compose.mat(compose.world_pose(PLACE, sub_model["rest"]["a"]))
expected = compose.mat(PLACE) @ compose.mat(sub_model["rest"]["a"])
assert np.allclose(w, expected), "world_pose must equal placement @ local"
# part a at sub-(1,0,0) -> Rz maps (1,0,0)->(0,1,0), +offset(1,2,3) => robot (1,3,3)
assert np.allclose(w[:3, 3], [1, 3, 3]), w[:3, 3]

# ---- place_model: keys prefixed, poses composed, dof name namespaced ----
pm = compose.place_model(sub_model, PLACE, "Intake")
assert set(pm["parts"]) == {"Intake/a", "Intake/b"}, pm["parts"].keys()
assert pm["parts"]["Intake/a"]["mesh"] == "h1"
# rest of b: sub-(0,1,0) -> Rz maps to (-1,0,0), +offset => robot (0,2,3)
assert np.allclose(compose.mat(pm["rest"]["Intake/b"])[:3, 3], [0, 2, 3])
dof = pm["dofs"][0]
assert dof["name"] == "Intake · left roller", dof["name"]
assert set(dof["frames"][0]["poses"]) == {"Intake/a"}
# frame val=90: pose a is Rz@..., composed by placement -> R = Rz@Rz (180 about Z)
f90 = compose.mat(dof["frames"][1]["poses"]["Intake/a"])
assert np.allclose(f90[:3, :3], Rz @ Rz), "DOF frame rotation must be composed by placement"

# ---- merge_models: two subs combine, keys stay unique ----
pm2 = compose.place_model(sub_model, T(np.eye(3), [0, 0, 0]), "Shooter")
merged = compose.merge_models([pm, pm2], "robotX")
assert len(merged["parts"]) == 4, merged["parts"].keys()          # 2 + 2, no collision
assert len(merged["dofs"]) == 2                                    # one per sub
assert merged["_composed"] is True
assert {d["name"] for d in merged["dofs"]} == {"Intake · left roller", "Shooter · left roller"}

print("ALL ASSERTIONS PASSED — compose math is correct:")
print("  world = placement @ local (rest + every DOF frame), keys namespaced, subs merge cleanly.")
