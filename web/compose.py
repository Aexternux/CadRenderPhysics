"""
Import-and-compose: build a whole robot from its linked subassemblies.

FRC robots keep each subsystem in its OWN Onshape document and LINK it into the robot assembly.
Their mates are therefore nested + in a different doc, so they can't be driven on a copy of the
robot (pipeline.owner_of returns None for linked subs). This module sidesteps that:

  1. read the robot assembly (READ-only) -> list its top-level subassembly instances + each
     instance's PLACEMENT transform (where that sub sits in the robot),
  2. import each subassembly on its OWN url via pipeline.import_assembly -- there its mates ARE
     top-level and fully driveable (the hybrid engine handles rollers/arms/elevators),
  3. COMPOSE: map every sub part + every sub mechanism frame into robot space by
     world = placement @ sub_local, then merge into one model.json.

The composition MATH is pure and unit-tested offline (test_compose.py). The Onshape calls in
compose() need live API budget and are NOT exercisable until the annual quota resets.

Usage:  python web/compose.py "<robot assembly url>"
"""

import hashlib
import json
import os
import shutil

import numpy as np

import pipeline
from pipeline import get, parse_url, JOBS_DIR


# ----------------------------- pure composition math (unit-tested) -----------------------------
def mat(T16):
    return np.array(T16, float).reshape(4, 4)


def t16(M):
    return [round(float(x), 7) for x in np.asarray(M).reshape(16)]


def world_pose(placement16, local16):
    """Compose a sub-local pose into robot space: world = placement @ local."""
    return t16(mat(placement16) @ mat(local16))


def place_model(sub_model, placement16, prefix):
    """
    Map a subassembly's imported model into robot space.

    Returns {parts, rest, dofs} with:
      - part keys namespaced by `prefix` (so two subs never collide),
      - every rest pose and every DOF frame pose composed by placement @ local,
      - DOF names namespaced by prefix so the UI shows "Intake · left roller".
    Meshes are referenced by their original hash (identical parts still dedupe across subs).
    """
    P = mat(placement16)

    def key(k):
        return f"{prefix}/{k}"

    parts = {key(k): dict(v) for k, v in sub_model["parts"].items()}
    rest = {key(k): t16(P @ mat(T)) for k, T in sub_model.get("rest", {}).items()}

    dofs = []
    for dof in sub_model.get("dofs", []):
        frames = [{"val": f["val"],
                   "poses": {key(k): t16(P @ mat(T)) for k, T in f["poses"].items()}}
                  for f in dof["frames"]]
        d = dict(dof)
        d["name"] = f"{prefix} · {dof['name']}"
        d["frames"] = frames
        dofs.append(d)
    return {"parts": parts, "rest": rest, "dofs": dofs}


def merge_models(placed, job_id):
    """Merge several placed sub-models (output of place_model) into one robot model.json dict."""
    parts, rest, dofs = {}, {}, []
    for pm in placed:
        parts.update(pm["parts"])
        rest.update(pm["rest"])
        dofs.extend(pm["dofs"])
    return {"job": job_id, "parts": parts, "rest": rest, "dofs": dofs, "_composed": True}


# ----------------------------- Onshape orchestration (needs live API) -----------------------------
def default_workspace(did):
    """Linked subs are referenced by version/microversion, but import needs a workspace to copy."""
    doc = get(f"/api/documents/{did}").json()
    return (doc.get("defaultWorkspace") or {}).get("id")


def sub_instances(adef):
    """Top-level subassembly instances of the robot, each with its placement transform."""
    ra = adef["rootAssembly"]
    placement = {tuple(o["path"])[0]: o["transform"]
                 for o in ra["occurrences"] if len(o["path"]) == 1}
    out = []
    for inst in ra["instances"]:
        if inst.get("type") != "Assembly":
            continue
        p = placement.get(inst["id"])
        if p is None:
            continue                     # no placement occurrence -> skip
        out.append({"id": inst["id"], "name": inst.get("name") or inst["id"],
                    "did": inst["documentId"], "eid": inst["elementId"],
                    "version": inst.get("documentVersion"),
                    "microversion": inst.get("documentMicroversion"),
                    "config": inst.get("configuration"), "placement": p})
    return out


def sub_url(sub):
    """Build an assembly URL for a linked subassembly's own document (default workspace)."""
    wid = default_workspace(sub["did"])
    if not wid:
        return None
    return f"{pipeline.API}/documents/{sub['did']}/w/{wid}/e/{sub['eid']}"


def compose(url, job_id=None, progress=None):
    ref = parse_url(url)
    job_id = job_id or ("compose_" + hashlib.md5(url.encode()).hexdigest()[:10])
    jobdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(os.path.join(jobdir, "onshape_assets"), exist_ok=True)

    pipeline.log(progress, "reading robot assembly…", 0.02)
    base = f"/api/assemblies/d/{ref['did']}/{ref['wvm']}/{ref['wvmid']}/e/{ref['eid']}"
    adef = get(base).json()
    subs = sub_instances(adef)
    pipeline.log(progress, f"{len(subs)} subassemblies to import", 0.05)

    placed = []
    for i, sub in enumerate(subs, 1):
        su = sub_url(sub)
        if not su:
            pipeline.log(progress, f"skip {sub['name']} (no workspace)", None)
            continue
        pipeline.log(progress, f"importing {sub['name']} ({i}/{len(subs)})…", 0.05 + 0.8 * i / len(subs))
        try:
            sub_job = pipeline.import_assembly(su, progress=None)
        except Exception as e:
            pipeline.log(progress, f"  {sub['name']} failed: {e}", None)
            continue
        sub_model = json.load(open(os.path.join(JOBS_DIR, sub_job, "model.json")))
        placed.append(place_model(sub_model, sub["placement"], sub["name"]))
        # bring the sub's meshes into the composed job (same hash dedupes identical parts)
        src = os.path.join(JOBS_DIR, sub_job, "onshape_assets")
        for fn in os.listdir(src):
            dst = os.path.join(jobdir, "onshape_assets", fn)
            if not os.path.exists(dst):
                shutil.copy(os.path.join(src, fn), dst)

    model = merge_models(placed, job_id)
    json.dump(model, open(os.path.join(jobdir, "model.json"), "w"))
    pipeline.log(progress, f"composed {len(model['parts'])} parts, {len(model['dofs'])} mechanism(s)", 1.0)
    return job_id


if __name__ == "__main__":
    import sys
    print("JOB:", compose(sys.argv[1]))
