"""
Sample Onshape's SOLVED part positions across a driven DOF, and download each part's mesh.

This is the faithful engine: instead of reconstructing motion from approximate robot joints,
we ask Onshape's solver for the exact transform of EVERY part at each driver angle, then place
each part's real mesh exactly there. Result is pixel-faithful to Onshape, for any mate type.

Outputs into <out_dir>/:
  onshape_frames.json   { part_refs: {key: {did,mid,eid,pid,cfg,mesh}}, frames: [{deg, poses:{key: T16}}] }
  onshape_assets/<hash>.stl   one mesh per unique part

Run: python sample_onshape.py
"""

import hashlib
import json
import math
import os
import urllib.parse

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv(override=True)
API = os.environ["ONSHAPE_API"]
AUTH = HTTPBasicAuth(os.environ["ONSHAPE_ACCESS_KEY"], os.environ["ONSHAPE_SECRET_KEY"])
HJ = {"Accept": "application/json", "Content-Type": "application/json"}

# Scratch COPY of the intake (original never touched).
DID = "bb81c3341160f55352b7c2cf"
WID = "8afd11b0c8479725477214c4"
EID = "42acbc7faa8769574358429d"
DRIVER_MATE = "Revolute 1"
DEG_RANGE = range(0, -131, -5)
OUT = "robot_intake"

base = f"/api/assemblies/d/{DID}/w/{WID}/e/{EID}"


def get(path, **kw):
    return requests.get(API + path, auth=AUTH, headers=HJ, **kw)


def main():
    os.makedirs(f"{OUT}/onshape_assets", exist_ok=True)

    feats = get(base + "/features").json()["features"]
    feat = next(f for f in feats if f["message"].get("name") == DRIVER_MATE)
    fidq = urllib.parse.quote(feat["message"]["featureId"], safe="")

    def pin(deg):
        for pm in feat["message"]["parameters"]:
            m = pm["message"]
            if m.get("parameterId") in ("limitAxialZMin", "limitAxialZMax"):
                m["isNull"] = False
                m["expression"] = f"{deg} deg"
                m["value"] = math.radians(deg)
        requests.post(API + base + f"/features/featureid/{fidq}",
                      auth=AUTH, headers=HJ, json={"feature": feat})

    # Resolve every leaf part occurrence -> part reference (with microversion for cross-doc fetch).
    ad = get(base).json()
    ra = ad["rootAssembly"]
    top = {i["id"]: i for i in ra["instances"]}
    subs = {(s.get("documentId"), s.get("elementId"), s.get("configuration")): s
            for s in ad.get("subAssemblies", [])}

    def leaf(path):
        inst = top.get(path[0])
        for nxt in path[1:]:
            if inst is None:
                return None
            s = subs.get((inst.get("documentId"), inst.get("elementId"), inst.get("configuration")))
            if not s:
                return None
            inst = {x["id"]: x for x in s["instances"]}.get(nxt)
        return inst

    part_refs = {}
    for o in ra["occurrences"]:
        inst = leaf(o["path"])
        if inst and inst.get("type") == "Part":
            key = "/".join(o["path"])
            uk = (inst["documentId"], inst.get("documentMicroversion"),
                  inst["elementId"], inst["partId"], inst.get("configuration", ""))
            part_refs[key] = {"did": uk[0], "mid": uk[1], "eid": uk[2], "pid": uk[3], "cfg": uk[4],
                              "mesh": hashlib.md5(repr(uk).encode()).hexdigest()[:16]}
    print(f"leaf part occurrences: {len(part_refs)}")

    # Sample solved transforms across the deploy range.
    frames = []
    for deg in DEG_RANGE:
        pin(deg)
        ad = get(base).json()
        poses = {}
        for o in ad["rootAssembly"]["occurrences"]:
            key = "/".join(o["path"])
            if key in part_refs:
                poses[key] = o["transform"]
        frames.append({"deg": deg, "poses": poses})
        print(f"  sampled {deg} deg ({len(poses)} parts)")

    json.dump({"part_refs": part_refs, "frames": frames},
              open(f"{OUT}/onshape_frames.json", "w"))
    print("saved onshape_frames.json")

    # Download one STL per unique part.
    uniq = {}
    for r in part_refs.values():
        uniq[r["mesh"]] = r
    print(f"downloading {len(uniq)} unique part meshes...")
    for i, (mesh, r) in enumerate(uniq.items(), 1):
        fn = f"{OUT}/onshape_assets/{mesh}.stl"
        if os.path.exists(fn):
            continue
        pid = urllib.parse.quote(r["pid"], safe="")
        url = (API + f"/api/parts/d/{r['did']}/m/{r['mid']}/e/{r['eid']}/partid/{pid}/stl"
               f"?mode=binary&units=meter")
        if r["cfg"]:
            url += "&configuration=" + urllib.parse.quote(r["cfg"], safe="")
        resp = requests.get(url, auth=AUTH, headers={"Accept": "*/*"}, allow_redirects=True)
        if resp.status_code == 200 and resp.content[:4] not in (b"<htm", b"{\"me"):
            open(fn, "wb").write(resp.content)
        else:
            print(f"  WARN mesh {i} ({r['pid']}) HTTP {resp.status_code} len {len(resp.content)}")
        if i % 15 == 0:
            print(f"  {i}/{len(uniq)} meshes")
    got = len(os.listdir(f"{OUT}/onshape_assets"))
    print(f"done: {got} mesh files in {OUT}/onshape_assets")


if __name__ == "__main__":
    main()
