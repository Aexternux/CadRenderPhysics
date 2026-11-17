"""
General Onshape importer: paste any assembly URL -> faithful, drivable model.

Pipeline (all via Onshape's own solver, so any mate type is reproduced faithfully):
  1. parse URL -> document/workspace/element
  2. copy the document to a scratch doc (user's original is never touched)
  3. find the assembly; read top-level movable mates (matevalues)
  4. detect INDEPENDENT degrees of freedom: drive each candidate; any other mate it moves is
     a coupled follower (e.g. a cam-driven slider) and is not offered as its own slider
  5. for each independent DOF, sweep its range and record every part's solved transform
  6. download each unique part's mesh + color
  7. write web/jobs/<id>/model.json (+ onshape_assets/) for the web viewer

Usage (standalone test):  python web/pipeline.py "<assembly url>"
"""

import hashlib
import json
import math
import os
import re
import urllib.parse

import numpy as np
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv(override=True)
API = os.environ.get("ONSHAPE_API", "https://cad.onshape.com")
AUTH = HTTPBasicAuth(os.environ.get("ONSHAPE_ACCESS_KEY", ""),
                     os.environ.get("ONSHAPE_SECRET_KEY", ""))
HJ = {"Accept": "application/json", "Content-Type": "application/json"}
# Part colors are read in batches: ONE metadata request per Part Studio (+configuration) returns
# the appearance of every part in it, and the result is cached with the job. Set
# ONSHAPE_DOWNLOAD_COLORS=false in .env to skip colors entirely and save that quota.
DOWNLOAD_COLORS = os.environ.get("ONSHAPE_DOWNLOAD_COLORS", "true").lower() != "false"
DEFAULT_COLOR = [0.6, 0.6, 0.62, 1.0]
# Bumped when imports change in a way old cached models should be repaired for
# (2 = suppressed parts removed, missing meshes retried, part colors).
IMPORT_VERSION = 2


# ---- local kinematics helpers (row-major 16-float <-> 4x4) ----
def _m(T):
    return np.array(T, float).reshape(4, 4)


def _t16(M):
    return [round(float(x), 7) for x in M.reshape(16)]


def _unit(v):
    v = np.array(v, float)
    n = np.linalg.norm(v)
    return v / n if n else v


def rot_about(origin, axis, ang):
    """4x4 world transform: rotate by `ang` (rad) about the line (origin, axis)."""
    x, y, z = _unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                  [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                  [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]])
    M = np.eye(4)
    M[:3, :3] = R
    o = np.array(origin, float)
    M[:3, 3] = o - R @ o
    return M


def trans_along(axis, dist):
    """4x4 world transform: translate by `dist` along `axis`."""
    M = np.eye(4)
    M[:3, 3] = _unit(axis) * dist
    return M

JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")


def log(progress, msg, frac=None):
    print(f"[pipeline] {msg}" + (f" ({int(frac*100)}%)" if frac is not None else ""))
    if progress is not None:
        progress(msg, frac)


def parse_url(url):
    m = re.search(r"/documents/([0-9a-f]+)/(w|v|m)/([0-9a-f]+)(?:/e/([0-9a-f]+))?", url)
    if not m:
        raise ValueError("Not a recognizable Onshape document URL")
    return {"did": m.group(1), "wvm": m.group(2), "wvmid": m.group(3), "eid": m.group(4)}


def get(path):
    return requests.get(API + path, auth=AUTH, headers=HJ)


def fetch_stl(inst, link_did, tries=3):
    wmv, wmvid = (("v", inst["documentVersion"]) if inst.get("documentVersion")
                  else ("m", inst["documentMicroversion"]))
    pid = urllib.parse.quote(inst["partId"], safe="")
    u = (API + f"/api/parts/d/{inst['documentId']}/{wmv}/{wmvid}/e/{inst['elementId']}"
         f"/partid/{pid}/stl?mode=binary&units=meter&linkDocumentId={link_did}")
    if inst.get("configuration"):
        u += "&configuration=" + urllib.parse.quote(inst["configuration"], safe="")
    for _ in range(tries):   # large meshes (belts, chains) occasionally time out: retry
        try:
            r = requests.get(u, auth=AUTH, headers={"Accept": "*/*"}, allow_redirects=False,
                             timeout=120)
            if r.status_code in (301, 302, 303, 307, 308):
                r = requests.get(r.headers["Location"], auth=AUTH, headers={"Accept": "*/*"},
                                 timeout=120)
        except requests.RequestException:
            continue
        if r.status_code == 402:
            raise RuntimeError("Onshape API quota exhausted (HTTP 402).")
        if r.status_code == 200 and len(r.content) > 84:
            return r.content
    return None


def studio_key(inst):
    """Identity of the Part Studio (at a version + configuration) a part instance comes from."""
    return (inst["documentId"], inst.get("documentVersion") or "",
            inst.get("documentMicroversion") or "", inst["elementId"], inst.get("configuration") or "")


def _rgba(value):
    if isinstance(value, dict) and "color" in value:
        c = value["color"]
        return [round(c["red"] / 255, 4), round(c["green"] / 255, 4), round(c["blue"] / 255, 4),
                round(value.get("opacity", 255) / 255, 4)]
    return None


def fetch_studio_colors(inst, link_did):
    """{partId: rgba} for EVERY part in this instance's Part Studio, in one metadata request."""
    wmv, wmvid = (("v", inst["documentVersion"]) if inst.get("documentVersion")
                  else ("m", inst["documentMicroversion"]))
    u = (API + f"/api/metadata/d/{inst['documentId']}/{wmv}/{wmvid}/e/{inst['elementId']}"
         f"/p?linkDocumentId={link_did}")
    if inst.get("configuration"):
        u += "&configuration=" + urllib.parse.quote(inst["configuration"], safe="")
    out = {}
    while u:
        try:
            r = requests.get(u, auth=AUTH, headers=HJ, timeout=60)
        except requests.RequestException:
            break
        if r.status_code == 402:
            raise RuntimeError("Onshape API quota exhausted (HTTP 402).")
        if r.status_code != 200:
            break
        body = r.json()
        for item in body.get("items", []):
            for prop in item.get("properties", []):
                if prop.get("name") == "Appearance":
                    col = _rgba(prop.get("value"))
                    if col and item.get("partId"):
                        out[item["partId"]] = col
        u = body.get("next")
    return out


def fetch_colors(rep_inst, link_did, progress=None, start=0.0, span=0.0):
    """{mesh hash: rgba} for every mesh, batching one request per Part Studio."""
    groups = {}
    for mesh, inst in rep_inst.items():
        groups.setdefault(studio_key(inst), []).append((mesh, inst))
    colors = {}
    for n, members in enumerate(groups.values(), 1):
        got = fetch_studio_colors(members[0][1], link_did)
        for mesh, inst in members:
            colors[mesh] = got.get(inst["partId"], DEFAULT_COLOR)
        log(progress, f"reading part colors {n}/{len(groups)}…", start + span * n / len(groups))
    return colors


def instance_chain(adef, path):
    """Instances along an occurrence path (root -> leaf), or None if it can't be resolved."""
    ra = adef["rootAssembly"]
    subs = {(s.get("documentId"), s.get("elementId"), s.get("configuration")): s
            for s in adef.get("subAssemblies", [])}
    insts = {i["id"]: i for i in ra["instances"]}
    chain = []
    for pid in path:
        inst = insts.get(pid)
        if inst is None:
            return None
        chain.append(inst)
        if inst.get("type") == "Assembly":
            s = subs.get((inst.get("documentId"), inst.get("elementId"), inst.get("configuration")))
            insts = {x["id"]: x for x in s["instances"]} if s else {}
    return chain


def assembly_paths(adef, part_keys):
    """Occurrence path + sub-assembly names for each part, so the viewer can select whole
    sub-assemblies (instance ids can contain '/', so keys alone can't be split reliably).

    Returns (partPaths {key: [ids]}, occNames {"id/id/...": name} for every assembly prefix)."""
    occ_paths = {"/".join(o["path"]): o["path"] for o in adef["rootAssembly"]["occurrences"]}
    part_paths, occ_names = {}, {}
    for k in part_keys:
        path = occ_paths.get(k)
        if not path:
            continue
        part_paths[k] = path
        chain = instance_chain(adef, path) or []
        for i, inst in enumerate(chain[:-1]):
            occ_names.setdefault("/".join(path[:i + 1]), inst.get("name", path[i]))
    return part_paths, occ_names


def part_names(adef, part_paths):
    """Instance name of each part ("52t Custom HTD 5mm Pulley <1>" -> without the "<1>")."""
    out = {}
    for k, path in part_paths.items():
        chain = instance_chain(adef, path) or []
        if chain:
            out[k] = re.sub(r"\s*<\d+>$", "", chain[-1].get("name", ""))
    return out


def is_suppressed(adef, path):
    """True if the occurrence or ANY assembly above it is suppressed (e.g. by a configuration).

    Onshape still lists suppressed occurrences -- parked at leftover positions -- so importing
    them scatters stray parts and overlapping alternate-configuration geometry across the robot.
    """
    chain = instance_chain(adef, path)
    return chain is None or any(i.get("suppressed") for i in chain)


def fetch_color(inst, link_did):
    wmv, wmvid = (("v", inst["documentVersion"]) if inst.get("documentVersion")
                  else ("m", inst["documentMicroversion"]))
    pid = urllib.parse.quote(inst["partId"], safe="")
    u = (API + f"/api/metadata/d/{inst['documentId']}/{wmv}/{wmvid}/e/{inst['elementId']}"
         f"/p/{pid}?linkDocumentId={link_did}")
    if inst.get("configuration"):
        u += "&configuration=" + urllib.parse.quote(inst["configuration"], safe="")
    r = requests.get(u, auth=AUTH, headers=HJ)
    col = [0.6, 0.6, 0.62, 1.0]
    if r.status_code == 200:
        for e in r.json().get("properties", []):
            v = e.get("value")
            if isinstance(v, dict) and "color" in v:
                c = v["color"]
                col = [c["red"] / 255, c["green"] / 255, c["blue"] / 255, v.get("opacity", 255) / 255]
    return col


def download_meshes(rep_inst, jobdir, link_did, progress=None, start=0.0, span=0.0):
    """Fetch every mesh not already on disk. Returns the hashes that still failed."""
    failed = []
    total = max(1, len(rep_inst))
    for i, (mesh, inst) in enumerate(rep_inst.items(), 1):
        fn = os.path.join(jobdir, "onshape_assets", f"{mesh}.stl")
        if not (os.path.exists(fn) and os.path.getsize(fn) > 84):
            data = fetch_stl(inst, link_did)
            if data:
                open(fn, "wb").write(data)
            else:
                failed.append(mesh)
        log(progress, f"downloading parts {i}/{total}…", start + span * (i / total))
    return failed


def mesh_hash(inst):
    uk = (inst["documentId"], inst.get("documentMicroversion"), inst["elementId"],
          inst["partId"], inst.get("configuration", ""))
    return hashlib.md5(repr(uk).encode()).hexdigest()[:16]


def repair_model(job_id, url, progress=None):
    """Upgrade an already-imported model in place, without re-running the motion import.

    One assembly read (instance IDs are identical between the original doc and our scratch
    copy) lets us drop suppressed parts, retry meshes that failed to download, and add part
    colors (one request per Part Studio). Costs ~1 + #part-studios API calls, once.
    """
    jobdir = os.path.join(JOBS_DIR, job_id)
    path = os.path.join(jobdir, "model.json")
    model = json.load(open(path, encoding="utf-8"))
    ref = parse_url(url)
    log(progress, "reading assembly…", 0.03)
    r = get(f"/api/assemblies/d/{ref['did']}/{ref['wvm']}/{ref['wvmid']}/e/{ref['eid']}")
    if r.status_code == 402:
        raise RuntimeError("Onshape API quota exhausted (HTTP 402).")
    r.raise_for_status()
    adef = r.json()
    occ_paths = {"/".join(o["path"]): o["path"] for o in adef["rootAssembly"]["occurrences"]}

    # 1. drop suppressed parts everywhere they're referenced
    drop = {k for k in model["parts"] if k in occ_paths and is_suppressed(adef, occ_paths[k])}
    for k in drop:
        model["parts"].pop(k, None)
        model.get("rest", {}).pop(k, None)
    for dof in model.get("dofs", []):
        if "members" in dof:
            dof["members"] = [k for k in dof["members"] if k not in drop]
        for fr in dof.get("frames", []):
            for k in drop:
                fr["poses"].pop(k, None)
    log(progress, f"removed {len(drop)} suppressed part(s)", 0.06)

    # 2. representative instance per mesh (meshes are cached per job, so this is free)
    rep_inst = {}
    for k, part in model["parts"].items():
        chain = instance_chain(adef, occ_paths.get(k, []))
        if chain and chain[-1].get("type") == "Part":
            rep_inst.setdefault(part["mesh"], chain[-1])
    link = (model.get("_copy") or {}).get("did") or ref["did"]
    failed = download_meshes(rep_inst, jobdir, link, progress, 0.06, 0.04)

    # 3. colors
    if DOWNLOAD_COLORS:
        colors = fetch_colors(rep_inst, link, progress, 0.1, 0.88)
        for part in model["parts"].values():
            part["color"] = colors.get(part["mesh"], part.get("color", DEFAULT_COLOR))
    model.update(url=url, importVersion=IMPORT_VERSION, colors=DOWNLOAD_COLORS)
    model["partPaths"], model["occNames"] = assembly_paths(adef, model["parts"])
    model["partNames"] = part_names(adef, model["partPaths"])
    model.setdefault("topLevelNames", {i["id"]: i.get("name", i["id"])
                                       for i in adef["rootAssembly"].get("instances", [])})
    json.dump(model, open(path, "w"))
    log(progress, f"repaired: -{len(drop)} suppressed, {len(failed)} mesh(es) still missing", 1.0)
    return {"removed": len(drop), "missingMeshes": len(failed)}


def import_assembly(url, job_id=None, progress=None):
    ref = parse_url(url)
    job_id = job_id or hashlib.md5(url.encode()).hexdigest()[:10]
    jobdir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(os.path.join(jobdir, "onshape_assets"), exist_ok=True)

    # 1. Copy the document (original untouched).
    log(progress, "copying document…", 0.02)
    cp = requests.post(API + f"/api/documents/{ref['did']}/workspaces/{ref['wvmid']}/copy",
                       auth=AUTH, headers=HJ, json={"newName": "FRC Sim scratch", "isPublic": True})
    cp.raise_for_status()
    did = cp.json()["newDocumentId"]
    wid = cp.json()["newWorkspaceId"]

    # 2. Find the target assembly in the copy.
    els = get(f"/api/documents/d/{did}/w/{wid}/elements").json()
    asms = [e for e in els if e.get("elementType") == "ASSEMBLY"]
    if not asms:
        raise ValueError("No assembly found in document")
    want_name = None
    if ref["eid"]:
        try:
            oels = get(f"/api/documents/d/{ref['did']}/{ref['wvm']}/{ref['wvmid']}/elements").json()
            want_name = next((e["name"] for e in oels if e["id"] == ref["eid"]), None)
        except Exception:
            pass
    eid = next((a["id"] for a in asms if a["name"] == want_name), asms[0]["id"])
    base = f"/api/assemblies/d/{did}/w/{wid}/e/{eid}"
    log(progress, f"assembly: {next((a['name'] for a in asms if a['id']==eid), eid)}", 0.05)

    # 3. Resolve leaf part references + an instance lookup.
    adef = get(base).json()
    ra = adef["rootAssembly"]
    top = {i["id"]: i for i in ra["instances"]}
    subs = {(s.get("documentId"), s.get("elementId"), s.get("configuration")): s
            for s in adef.get("subAssemblies", [])}

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

    # Skip parts that aren't real geometry: hidden occurrences and mating reference parts.
    SKIP_NAMES = ("origin cube", "mate connector", "reference")
    part_refs, rep_inst = {}, {}
    for o in ra["occurrences"]:
        if o.get("hidden") or is_suppressed(adef, o["path"]):
            continue
        inst = leaf(o["path"])
        if inst and inst.get("type") == "Part":
            nm = (inst.get("name") or "").lower()
            if any(s in nm for s in SKIP_NAMES):
                continue
            mesh = mesh_hash(inst)
            part_refs["/".join(o["path"])] = {"mesh": mesh}
            rep_inst.setdefault(mesh, inst)

    def poses_now():
        a = get(base).json()
        return {"/".join(o["path"]): o["transform"] for o in a["rootAssembly"]["occurrences"]
                if "/".join(o["path"]) in part_refs}

    # 4. EVERY movable mate at ANY nesting level (top-level OR inside subassemblies).
    #    Revolute = arm/roller/turret, Slider = elevator/telescope, Cylindrical = both.
    #    HYBRID engine: a simple joint is confirmed with ONE deflection sample and then its whole
    #    motion is generated LOCALLY (0 further API calls); only coupled/cam mates fall back to
    #    full Onshape-solver sampling. This keeps API usage low enough to scale.
    mvals = get(base + "/matevalues").json()["mateValues"]
    candidates = [m for m in mvals if m["jsonType"] in ("Revolute", "Slider", "Cylindrical")]
    candidates.sort(key=lambda m: (len(m.get("ownerOccurrencePath") or []),
                                   0 if m["jsonType"] != "Slider" else 1))

    def mate_key(m):
        return (tuple(m.get("ownerOccurrencePath") or []), m["featureId"])

    rest_mv = {mate_key(m): (m.get("rotationZ", 0) or m.get("translationZ", 0)) for m in mvals}

    # World transform of every occurrence + rest poses of the tracked parts (from the rest def).
    occ_world = {tuple(o["path"]): _m(o["transform"]) for o in ra["occurrences"]}
    rest = {"/".join(o["path"]): o["transform"] for o in ra["occurrences"]
            if "/".join(o["path"]) in part_refs}

    def mate_featuredata(m):
        path = m.get("ownerOccurrencePath") or []
        if not path:
            feats = ra.get("features", [])
        else:
            inst = leaf(list(path))
            sub = (subs.get((inst.get("documentId"), inst.get("elementId"),
                             inst.get("configuration"))) if inst else None)
            feats = sub.get("features", []) if sub else []
        for f in feats:
            if f.get("featureType") == "mate" and (
                    f.get("id") == m["featureId"]
                    or f.get("featureData", {}).get("name") == m["mateName"]):
                return f.get("featureData")
        return None

    def mate_axis(m, fdata):
        ents = (fdata or {}).get("matedEntities") or []
        if not ents:
            return None
        cs = ents[0].get("matedCS")
        if not cs or "zAxis" not in cs or "origin" not in cs:
            return None
        full = tuple(list(m.get("ownerOccurrencePath") or [])
                     + list(ents[0].get("matedOccurrence") or []))
        Tw = occ_world.get(full)
        if Tw is None:
            return None
        o = np.array(cs["origin"], float)
        return (Tw @ np.append(o, 1.0))[:3], Tw[:3, :3] @ np.array(cs["zAxis"], float)

    feat_cache = {}

    def owner_of(m):
        path = m.get("ownerOccurrencePath") or []
        if not path:
            return (did, wid, eid)               # the top-level assembly in our copy
        inst = leaf(path)
        if not inst or inst.get("type") != "Assembly" or inst.get("documentId") != did:
            return None                          # externally-linked sub: not editable on our copy
        return (did, wid, inst["elementId"])

    def feature_of(owner, fid):
        if owner not in feat_cache:
            try:
                fs = get(f"/api/assemblies/d/{owner[0]}/w/{owner[1]}/e/{owner[2]}"
                         "/features").json()["features"]
                feat_cache[owner] = {f["message"]["featureId"]: f for f in fs}
            except Exception:
                feat_cache[owner] = {}
        return feat_cache[owner].get(fid)

    def pin(owner, feat, jtype, val):
        ids = (("limitAxialZMin", "limitAxialZMax", "rad") if jtype != "Slider"
               else ("limitZMin", "limitZMax", "m"))
        for pm in feat["message"]["parameters"]:
            mm = pm["message"]
            if mm.get("parameterId") in ids[:2]:
                mm["isNull"] = False
                mm["expression"] = f"{val} {ids[2]}"
                mm["value"] = val
            if mm.get("parameterId") == "limitsEnabled":
                mm["value"] = True
        fidq = urllib.parse.quote(feat["message"]["featureId"], safe="")
        requests.post(API + f"/api/assemblies/d/{owner[0]}/w/{owner[1]}/e/{owner[2]}"
                      f"/features/featureid/{fidq}", auth=AUTH, headers=HJ, json={"feature": feat})

    def gen_local(origin, axis, is_rot, members):
        vals = ([math.radians(d) for d in range(-135, 136, 5)] if is_rot
                else [d / 1000.0 for d in range(-250, 251, 10)])
        restM = {k: _m(rest[k]) for k in members}
        out = []
        for v in vals:
            delta = rot_about(origin, axis, v) if is_rot else trans_along(axis, v)
            out.append({"val": v, "poses": {k: _t16(delta @ restM[k]) for k in members}})
        return out

    def sample_solver(owner, feat, jtype):
        step = math.radians(10) if jtype != "Slider" else 0.015
        out = [{"val": 0.0, "poses": rest}]
        for sgn in (-1, 1):
            prev = rest
            for k in range(1, nmax + 1):
                pin(owner, feat, jtype, sgn * step * k)
                pz = poses_now()
                disp = max((abs(pz[key][3] - prev[key][3]) + abs(pz[key][7] - prev[key][7])
                            + abs(pz[key][11] - prev[key][11])) for key in pz) if pz else 0
                if disp < 1e-5:
                    break
                out.append({"val": sgn * step * k, "poses": pz})
                prev = pz
            pin(owner, feat, jtype, 0.0)
        out.sort(key=lambda f: f["val"])
        return out

    def differs(A, B):
        return any(abs(A[j] - B[j]) > 1e-5 for j in (3, 7, 11, 0, 5, 10))

    assigned = set()
    dofs = []
    MAX_DOFS, nmax = 16, 10
    est = max(1, len(candidates))
    for ci, cand in enumerate(candidates):
        if len(dofs) >= MAX_DOFS:
            break
        if mate_key(cand) in assigned:
            continue
        owner = owner_of(cand)
        if owner is None:
            continue
        feat = feature_of(owner, cand["featureId"])
        if not feat:
            continue
        jtype = cand["jsonType"]
        is_rot = jtype != "Slider"
        log(progress, f"analyzing '{cand['mateName']}'…", 0.05 + 0.45 * (ci / est))

        # ONE small deflection: find which parts move + detect coupled mates.
        test = math.radians(15) if is_rot else 0.03
        pin(owner, feat, jtype, test)
        pz = poses_now()
        members = [k for k in pz if k in rest and differs(pz[k], rest[k])]
        moved = set()
        for mv in get(base + "/matevalues").json()["mateValues"]:
            cur = mv.get("rotationZ", 0) or mv.get("translationZ", 0)
            if abs(cur - rest_mv.get(mate_key(mv), 0)) > 1e-3 and mate_key(mv) != mate_key(cand):
                moved.add(mate_key(mv))
        assigned.add(mate_key(cand))
        assigned |= moved
        if not members:
            pin(owner, feat, jtype, 0.0)
            continue

        # Simple rigid joint? (members move exactly like a rotation/translation about the axis)
        frames = None
        ax = mate_axis(cand, mate_featuredata(cand))
        if ax:
            origin, axis = ax
            for sgn in (1, -1):
                delta = (rot_about(origin, axis, sgn * test) if is_rot
                         else trans_along(axis, sgn * test))
                if all(np.allclose(_m(pz[k]), delta @ _m(rest[k]), atol=3e-3) for k in members):
                    frames = gen_local(origin, np.array(axis, float) * sgn, is_rot, members)
                    break
        pin(owner, feat, jtype, 0.0)  # undo the test deflection

        if frames is None:            # coupled / cam mate -> use Onshape's solver (rare, expensive)
            frames = sample_solver(owner, feat, jtype)
        if len(frames) < 2:
            continue
        frames.sort(key=lambda f: f["val"])
        members_set = {key for f in frames for key, T in f["poses"].items()
                       if key in rest and differs(T, rest[key])}
        unit = "deg" if is_rot else "m"
        conv = (lambda v: round(math.degrees(v), 1)) if is_rot else (lambda v: round(v, 4))
        dofs.append({"name": cand["mateName"], "type": jtype, "unit": unit,
                     "ownerPath": cand.get("ownerOccurrencePath") or [],
                     "members": sorted(members_set),
                     "min": conv(frames[0]["val"]), "max": conv(frames[-1]["val"]),
                     "frames": [{"val": conv(f["val"]),
                                 "poses": {k: f["poses"][k] for k in members_set if k in f["poses"]}}
                                for f in frames]})

    log(progress, f"{len(dofs)} mechanism(s) found; downloading {len(rep_inst)} parts…", 0.5)

    # 6. Meshes + colors (colors: one batched request per Part Studio).
    download_meshes(rep_inst, jobdir, did, progress, 0.5, 0.35 if DOWNLOAD_COLORS else 0.5)
    colors = fetch_colors(rep_inst, did, progress, 0.85, 0.15) if DOWNLOAD_COLORS else {}

    parts = {k: {"mesh": r["mesh"], "color": colors.get(r["mesh"], DEFAULT_COLOR)}
             for k, r in part_refs.items()}
    # Always store the assembled rest positions so the model renders correctly even with 0 DOFs.
    top_level_names = {item["id"]: item.get("name", item["id"])
                       for item in ra.get("instances", []) if item.get("id")}
    part_paths, occ_names = assembly_paths(adef, parts)
    model = {"job": job_id, "parts": parts, "rest": rest, "dofs": dofs,
             "partPaths": part_paths, "occNames": occ_names,
             "partNames": part_names(adef, part_paths),
             "topLevelNames": top_level_names, "url": url, "importVersion": IMPORT_VERSION,
             "colors": DOWNLOAD_COLORS,
             "_copy": {"did": did, "wid": wid, "eid": eid}}
    json.dump(model, open(os.path.join(jobdir, "model.json"), "w"))
    log(progress, f"done: {len(parts)} parts, {len(dofs)} mechanism(s)", 1.0)
    return job_id


if __name__ == "__main__":
    import sys
    jid = import_assembly(sys.argv[1])
    print("JOB:", jid)
