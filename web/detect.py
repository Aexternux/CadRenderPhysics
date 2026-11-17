"""
Lean mechanism detector: list EVERY moving subsystem/mechanism in an Onshape assembly.

Unlike pipeline.py (which copies the doc and DRIVES each mate to generate faithful motion --
API-heavy), this only ANSWERS "what can move?".  That needs just two read-only calls and
never copies the document, so it is safe to run against a nearly-exhausted API quota:

  1. GET the assembly definition   -> instances / subassemblies / occurrence names
  2. GET /matevalues               -> every movable mate at EVERY nesting level

It resolves each mate to the top-level subsystem it lives in and prints a grouped inventory
(rollers, arms, elevators, turrets ...).  Coupling (which mates are followers of another, e.g.
belt-linked rollers or a cam) can only be known by DRIVING, so every movable mate is listed
here; pipeline.py collapses couplings later when it generates motion.

Usage:  python web/detect.py "<assembly url>"
Writes: web/jobs/<id>/mechanisms.json
"""

import hashlib
import json
import os
import re
import sys

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv(override=True)
API = os.environ.get("ONSHAPE_API", "https://cad.onshape.com")
AUTH = HTTPBasicAuth(os.environ.get("ONSHAPE_ACCESS_KEY", ""),
                     os.environ.get("ONSHAPE_SECRET_KEY", ""))
HJ = {"Accept": "application/json", "Content-Type": "application/json"}
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")


def parse_url(url):
    m = re.search(r"/documents/([0-9a-f]+)/(w|v|m)/([0-9a-f]+)(?:/e/([0-9a-f]+))?", url)
    if not m:
        raise ValueError("Not a recognizable Onshape document URL")
    return {"did": m.group(1), "wvm": m.group(2), "wvmid": m.group(3), "eid": m.group(4)}


def get(path):
    r = requests.get(API + path, auth=AUTH, headers=HJ)
    if r.status_code == 402:
        raise RuntimeError("Onshape API quota exhausted (HTTP 402). Wait for it to reset.")
    r.raise_for_status()
    return r.json()


# Classify a mate into a human role from its type + name. Best-effort; the user can rename.
ROLE_WORDS = [
    ("roller/wheel", ("roll", "wheel", "flywheel", "spin", "feed", "belt", "conveyor")),
    ("turret", ("turret", "yaw", "azimuth")),
    ("wrist", ("wrist",)),
    ("arm/pivot", ("arm", "pivot", "deploy", "slap", "shoulder", "elbow", "hinge", "climb", "hook")),
]


def role(mate):
    jt = mate["jsonType"]
    nm = (mate.get("mateName") or "").lower()
    if jt == "Slider":
        return "linear (elevator/telescope)"
    for label, words in ROLE_WORDS:
        if any(w in nm for w in words):
            return label
    return "rotary"


# ----------------------------- kinematics (pure, offline) -----------------------------
# Turns a detected mate into something the viewer can DRIVE: a world-space axis, the set of
# occurrences that move with it, its limits and its current (rest) value. Everything comes from
# the assembly definition (fetched with includeMateFeatures=true) + matevalues -- no extra calls.
KINEMATICS_VERSION = 1
RIGID_MATES = ("FASTENED",)


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _mat16(T):
    return [T[0:4], T[4:8], T[8:12], T[12:16]]


def _apply(M, p, w):
    return [M[i][0] * p[0] + M[i][1] * p[1] + M[i][2] * p[2] + M[i][3] * w for i in range(3)]


def _norm(v):
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def resolve_kinematics(adef, mvals):
    """Map (ownerPath tuple, featureId) -> drive info for every movable mate it can resolve.

    Drive info: {axis: {origin, dir} (robot/world frame, meters), movingPrefixes: [occurrence
    paths joined by '/'; every part at or under one of them moves], min/max/rest (rad or m, the
    mate's own value), limited: bool}. Mates that can't be resolved are simply absent.
    """
    ra = adef.get("rootAssembly") or {}
    subs = {(s.get("documentId"), s.get("elementId"), s.get("configuration")): s
            for s in adef.get("subAssemblies", [])}
    occ = {tuple(o["path"]): _mat16(o["transform"]) for o in ra.get("occurrences", [])}
    top = {i["id"]: i for i in ra.get("instances", [])}

    def assembly_at(path):
        """(instances, features) of the assembly that lives at occurrence `path` ([] = root)."""
        if not path:
            return ra.get("instances", []), ra.get("features", [])
        insts = top
        inst = None
        for pid in path:
            inst = insts.get(pid)
            if not inst or inst.get("type") != "Assembly":
                return None
            s = subs.get((inst.get("documentId"), inst.get("elementId"), inst.get("configuration")))
            if not s:
                return None
            insts = {x["id"]: x for x in s.get("instances", [])}
        return list(insts.values()), s.get("features", [])

    def ground_hints(owner):
        """Children of `owner` that an ANCESTOR assembly mates to -> they're the fixed base."""
        hints = set()
        for depth in range(len(owner)):
            anc = owner[:depth]
            got = assembly_at(anc)
            if not got:
                continue
            for f in got[1]:
                if f.get("featureType") != "mate" or f.get("suppressed"):
                    continue
                for e in f["featureData"].get("matedEntities", []):
                    full = anc + list(e.get("matedOccurrence") or [])
                    if len(full) > len(owner) and full[:len(owner)] == owner:
                        hints.add(full[len(owner)])
        return hints

    out = {}
    cache = {}
    for m in mvals:
        if m.get("jsonType") not in ("Revolute", "Slider", "Cylindrical"):
            continue
        owner = list(m.get("ownerOccurrencePath") or [])
        key = tuple(owner)
        if key not in cache:
            got = assembly_at(owner)
            if not got:
                cache[key] = None
            else:
                insts, feats = got
                uf = _UF()
                for i in insts:
                    uf.find(i["id"])
                mates = []
                for f in feats:
                    if f.get("suppressed"):
                        continue
                    fd = f.get("featureData", {})
                    if f.get("featureType") == "mateGroup":
                        ids = [o["occurrence"][0] for o in fd.get("occurrences", []) if o.get("occurrence")]
                        for a, b in zip(ids, ids[1:]):
                            uf.union(a, b)
                    elif f.get("featureType") == "mate":
                        ents = fd.get("matedEntities", [])
                        nodes = [(e.get("matedOccurrence") or [None])[0] for e in ents]
                        if len(nodes) == 2 and all(nodes):
                            if fd.get("mateType") in RIGID_MATES:
                                uf.union(*nodes)
                            else:
                                mates.append((nodes[0], nodes[1]))
                cache[key] = {"insts": insts, "feats": {f.get("id"): f for f in feats},
                              "uf": uf, "mates": mates, "ground": ground_hints(owner)}
        ctx = cache[key]
        if not ctx:
            continue
        f = ctx["feats"].get(m.get("featureId"))
        if not f or f.get("featureType") != "mate":
            continue
        fd = f["featureData"]
        ents = fd.get("matedEntities", [])
        if len(ents) != 2 or not all(e.get("matedOccurrence") for e in ents):
            continue
        uf = ctx["uf"]
        comp = [uf.find(e["matedOccurrence"][0]) for e in ents]
        if comp[0] == comp[1]:
            continue                                   # both sides rigidly joined: can't move

        # Bodies reachable from each side through the OTHER movable mates (children ride along),
        # ignoring every mate between these same two bodies (redundant revolute+cylindrical pairs).
        adj = {}
        for a, b in ctx["mates"]:
            ca, cb = uf.find(a), uf.find(b)
            if ca == cb or {ca, cb} == set(comp):
                continue
            adj.setdefault(ca, set()).add(cb)
            adj.setdefault(cb, set()).add(ca)

        def reach(start):
            seen, stack = {start}, [start]
            while stack:
                for n in adj.get(stack.pop(), ()):
                    if n not in seen:
                        seen.add(n)
                        stack.append(n)
            return seen

        sides = [reach(comp[0]), reach(comp[1])]
        if sides[0] & sides[1]:                        # closed loop: just move the rigid body
            sides = [{comp[0]}, {comp[1]}]
        ids = [i["id"] for i in ctx["insts"]]
        members = [[i for i in ids if uf.find(i) in s] for s in sides]
        names = {i["id"]: (i.get("name") or "").lower() for i in ctx["insts"]}
        grounded = [bool(set(mem) & ctx["ground"]) for mem in members]
        if grounded[0] != grounded[1]:
            moving = 1 if grounded[0] else 0
        else:
            cube = [any("origin cube" in names[i] for i in mem) for mem in members]
            if cube[0] != cube[1]:
                moving = 1 if cube[0] else 0
            else:                                      # fall back: the smaller side moves
                moving = 0 if len(members[0]) <= len(members[1]) else 1

        # Axis: entity 0's mate frame, in world. Onshape's mate value moves entity 0 relative
        # to entity 1 along/about +Z of that frame, so flip the sign if entity 1 is the mover.
        e0 = ents[0]
        T = occ.get(tuple(owner + list(e0["matedOccurrence"])))
        cs = e0.get("matedCS") or {}
        if T is None or "origin" not in cs or "zAxis" not in cs:
            continue
        origin = _apply(T, cs["origin"], 1.0)
        direction = _norm(_apply(T, cs["zAxis"], 0.0))
        if moving == 1:
            direction = [-x for x in direction]

        rot = m["jsonType"] != "Slider"
        lim = fd.get("mateLimits") or {}
        lo = lim.get("limitAxialZMin" if rot else "limitZMin")
        hi = lim.get("limitAxialZMax" if rot else "limitZMax")
        rest = (m.get("rotationZ") if rot else m.get("translationZ")) or 0.0
        out[(key, m.get("featureId"))] = {
            "axis": {"origin": [round(x, 6) for x in origin], "dir": [round(x, 6) for x in direction]},
            "movingPrefixes": sorted("/".join(owner + [i]) for i in members[moving]),
            "min": lo, "max": hi, "rest": rest,
            "limited": lo is not None and hi is not None,
        }
    return out


def detect(url, save=True):
    ref = parse_url(url)
    base = f"/api/assemblies/d/{ref['did']}/{ref['wvm']}/{ref['wvmid']}/e/{ref['eid']}"

    print(f"[detect] reading assembly {ref['eid']} ...")
    adef = get(base + "?includeMateFeatures=true")   # call 1: structure + mate axes/limits
    ra = adef["rootAssembly"]
    top = {i["id"]: i for i in ra["instances"]}

    print("[detect] reading mate values ...")
    mvals = get(base + "/matevalues")["mateValues"]   # call 2: every movable mate, any depth
    movable = [m for m in mvals if m["jsonType"] in ("Revolute", "Slider", "Cylindrical")]
    try:
        kin = resolve_kinematics(adef, movable)
    except Exception as e:                # never let drive-info break the inventory itself
        print(f"[detect] could not resolve mate kinematics: {e}")
        kin = {}

    # Group each movable mate under the TOP-LEVEL subsystem instance it belongs to.
    # ownerOccurrencePath = path to the subassembly that owns the mate; [0] = the top-level
    # subsystem instance. Empty path => a mate defined at the robot's own top level.
    groups = {}   # subsystem label -> list of mechanism dicts
    for m in movable:
        owner = m.get("ownerOccurrencePath") or []
        if owner:
            sub_inst = top.get(owner[0])
            label = (sub_inst.get("name") if sub_inst else None) or owner[0]
            linked = bool(sub_inst and sub_inst.get("documentId") != ref["did"])
        else:
            label, linked = "Robot (top level)", False
        groups.setdefault(label, {"linked": linked, "mechs": []})
        groups[label]["mechs"].append({
            "mateName": m.get("mateName"),
            "type": m["jsonType"],
            "role": role(m),
            "featureId": m.get("featureId"),
            "ownerPath": owner,
            "depth": len(owner),
            "drive": kin.get((tuple(owner), m.get("featureId"))),
        })

    # ---- report ----
    total = len(movable)
    print()
    print("=" * 66)
    print(f"  MOVING MECHANISMS  --  {total} across {len(groups)} subsystem(s)")
    print("=" * 66)
    for label in sorted(groups, key=lambda k: (-len(groups[k]["mechs"]), k)):
        g = groups[label]
        tag = "  [linked doc]" if g["linked"] else ""
        print(f"\n{label}  ({len(g['mechs'])}){tag}")
        for mech in g["mechs"]:
            depth = f"  @depth {mech['depth']}" if mech["depth"] else ""
            print(f"    - {mech['type']:<11} {str(mech['mateName']):<24} {mech['role']}{depth}")
    print()

    result = {"url": url, "eid": ref["eid"], "total": total,
              "kinematicsVersion": KINEMATICS_VERSION,
              "subsystems": [{"name": k, "linked": v["linked"], "mechanisms": v["mechs"]}
                             for k, v in groups.items()]}
    if save:
        job_id = hashlib.md5(url.encode()).hexdigest()[:10]
        jobdir = os.path.join(JOBS_DIR, job_id)
        os.makedirs(jobdir, exist_ok=True)
        out = os.path.join(jobdir, "mechanisms.json")
        json.dump(result, open(out, "w"), indent=2)
        print(f"[detect] saved -> {out}  (job {job_id})")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python web/detect.py \"<onshape assembly url>\"")
        sys.exit(1)
    try:
        detect(sys.argv[1])
    except RuntimeError as e:
        print(f"[detect] {e}")
        sys.exit(2)
