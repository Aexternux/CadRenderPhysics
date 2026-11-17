"""
Web app backend for the FRC robot simulator.

Serves the browser viewer and imported models. The Import button POSTs an Onshape assembly
URL; pipeline.import_assembly runs in a background thread (copy doc -> detect DOFs -> sample
Onshape's solved poses -> fetch meshes/colors) and the page polls /api/job/<id> for progress,
then loads the faithful, drivable model.

Run:  python web/app.py   then open http://localhost:5000
"""

import json
import math
import os
import threading
import time
import traceback
from collections import Counter

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_from_directory

load_dotenv(override=True)

import pipeline  # noqa: E402  (after load_dotenv so creds are set)
import detect  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
JOBS_DIR = os.path.join(ROOT, "jobs")

# Preloaded job that ships with the project.
BUILTIN = {"intake": "robot_intake"}

app = Flask(__name__, static_folder=None)

# job_id -> {status: 'running'|'done'|'error', message, done(bool)}
JOB_STATUS = {}


def job_id_for_url(url):
    """Return a stable cache key for an Onshape assembly.

    People routinely paste the same assembly URL with different query strings or
    fragments. Those must reuse the same locally cached model and meshes instead
    of starting another API-heavy import.
    """
    import hashlib
    try:
        ref = pipeline.parse_url(url)
        identity = "/".join(str(ref.get(key) or "") for key in ("did", "wvm", "wvmid", "eid"))
    except ValueError:
        identity = url.strip()
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def builtin_model(job):
    d = os.path.join(PROJ, BUILTIN[job])
    frames = json.load(open(os.path.join(d, "onshape_frames.json")))
    cpath = os.path.join(d, "onshape_colors.json")
    colors = json.load(open(cpath)) if os.path.exists(cpath) else {}
    refs = frames["part_refs"]
    parts = {k: {"mesh": r["mesh"], "color": colors.get(r["mesh"], [0.6, 0.6, 0.62, 1.0])}
             for k, r in refs.items()}
    degs = [f["deg"] for f in frames["frames"]]
    fr = [{"val": f["deg"], "poses": f["poses"]} for f in frames["frames"]]
    dofs = [{"name": "deploy", "type": "Revolute", "unit": "deg",
             "min": min(degs), "max": max(degs), "frames": fr}]
    return {"job": job, "parts": parts, "dofs": dofs}


def job_model_path(job):
    return os.path.join(JOBS_DIR, job, "model.json")


def subsystems_path(job):
    return os.path.join(JOBS_DIR, job, "subsystems.json")


def detected_subsystems_path(job):
    return os.path.join(JOBS_DIR, job, "detected_subsystems.json")


def load_subsystems(job):
    p = subsystems_path(job)
    return json.load(open(p)) if os.path.exists(p) else []


def load_detected_subsystems(job):
    path = detected_subsystems_path(job)
    return json.load(open(path)) if os.path.exists(path) else None


def ensure_detected_subsystems(url, job):
    """Read nested/linkable mates once when the motion importer found no DOFs.

    Re-reads (two calls) only if the cached file predates drive info (axis/members/limits).
    """
    path = detected_subsystems_path(job)
    cached = load_detected_subsystems(job) if os.path.exists(path) else None
    if cached and cached.get("kinematicsVersion") == detect.KINEMATICS_VERSION:
        return cached
    url = url or (cached or {}).get("url")
    if not url:
        return cached
    result = detect.detect(url, save=False)  # exactly two read-only API calls
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    return result


def drivable_mechanisms(mechanisms, part_keys):
    """Turn detected mates (with `drive` info) into viewer-ready joints.

    Expands moving occurrence prefixes into the model's part keys, converts rotary values to
    degrees, and merges redundant mates that move the exact same parts (e.g. a roller held by
    both a Revolute and a Cylindrical mate) into one control.
    """
    merged = {}
    for mech in mechanisms:
        drive = mech.get("drive")
        if not drive:
            continue
        prefixes = drive["movingPrefixes"]
        members = sorted(k for k in part_keys
                         if any(k == p or k.startswith(p + "/") for p in prefixes))
        if not members:
            continue
        rotary = mech["type"] != "Slider"
        sig = (rotary, tuple(prefixes))
        conv = (lambda v: None if v is None else round(math.degrees(v), 3)) if rotary \
            else (lambda v: None if v is None else round(v, 5))
        joint = {"name": mech.get("mateName") or "Joint",
                 "type": "pivot" if rotary else "slider",
                 "unit": "deg" if rotary else "m",
                 "axis": drive["axis"], "members": members,
                 "rest": conv(drive.get("rest") or 0.0),
                 "min": conv(drive.get("min")), "max": conv(drive.get("max")),
                 "limited": bool(drive.get("limited")),
                 "role": mech.get("role")}
        if sig in merged:
            prev = merged[sig]
            prev["name"] = f"{prev['name']} + {joint['name']}"
            if not prev["limited"] and joint["limited"]:
                prev.update(rest=joint["rest"], min=joint["min"], max=joint["max"], limited=True,
                            axis=joint["axis"])
            continue
        merged[sig] = joint
    joints = list(merged.values())
    for j in joints:   # unlimited rotary -> free spin (roller); unlimited slider -> +/-0.3 m
        if not j["limited"]:
            span = 180.0 if j["type"] == "pivot" else 0.3
            j["min"], j["max"] = j["rest"] - span, j["rest"] + span
        j["continuous"] = j["type"] == "pivot" and not j["limited"]
    return joints


def auto_subsystems(model, detected=None):
    """Group detected CAD degrees of freedom into human-scale subsystems.

    Each imported DOF includes the parts it moves and the nested assembly that
    owns its mate. Prefer that owner (an intake assembly, shooter assembly,
    etc.); for a mate on the top-level robot, use the dominant moved part's
    top-level instance. This is read-only inference from the already-cached
    model, so it costs no additional Onshape calls.
    """
    if not model.get("dofs") and detected:
        part_keys = list(model.get("parts", {}))
        out = []
        for item in detected.get("subsystems", []):
            drivable = drivable_mechanisms(item.get("mechanisms", []), part_keys)
            moving = {k for d in drivable for k in d["members"]}
            out.append({"name": item["name"],
                        "joints": [mechanism.get("mateName", "Unnamed mate")
                                   for mechanism in item.get("mechanisms", [])],
                        "partCount": len(moving) if drivable else None,
                        "linked": item.get("linked", False),
                        "mechanisms": drivable})
        return out

    names = model.get("topLevelNames", {})
    grouped = {}
    for index, dof in enumerate(model.get("dofs", [])):
        owner = dof.get("ownerPath") or []
        members = dof.get("members") or [key for frame in dof.get("frames", [])
                                           for key in frame.get("poses", {})]
        tops = [key.split("/", 1)[0] for key in members if key]
        key = owner[0] if owner else (Counter(tops).most_common(1)[0][0] if tops else f"dof-{index}")
        grouped.setdefault(key, {"name": names.get(key, key), "dofs": [], "members": set()})
        grouped[key]["dofs"].append(dof.get("name", f"Joint {index + 1}"))
        grouped[key]["members"].update(members)
    return sorted(
        [{"name": value["name"], "joints": value["dofs"], "partCount": len(value["members"])}
         for value in grouped.values()],
        key=lambda value: value["name"].lower(),
    )


@app.route("/")
def index():
    # never let the browser serve a stale viewer after the code changes
    resp = send_from_directory(ROOT, "index.html", max_age=0)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/physics.js")
def physics_js():
    resp = send_from_directory(ROOT, "physics.js", mimetype="text/javascript", max_age=0)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/hinge.js")
def hinge_js():
    resp = send_from_directory(ROOT, "hinge.js", mimetype="text/javascript", max_age=0)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/model/<job>")
def model(job):
    if job in BUILTIN:
        m = builtin_model(job)
    elif os.path.exists(job_model_path(job)):
        m = json.load(open(job_model_path(job)))
    else:
        abort(404)
    detected = load_detected_subsystems(job)
    if (detected and not m.get("dofs")
            and detected.get("kinematicsVersion") != detect.KINEMATICS_VERSION):
        try:   # one-time upgrade of an older inventory so its subsystems become drivable
            detected = ensure_detected_subsystems(None, job)
        except Exception:
            traceback.print_exc()
    m["subsystems"] = load_subsystems(job)   # user-defined mechanisms
    m["rollerSets"] = load_rollersets(job)   # user-defined roller sets
    m["autoSubsystems"] = auto_subsystems(m, detected)
    resp = jsonify(m)
    resp.headers["Cache-Control"] = "no-store"   # repairs/colors must show up on the next load
    return resp


@app.route("/api/subsystems/<job>", methods=["GET"])
def get_subsystems(job):
    return jsonify(load_subsystems(job))


@app.route("/api/subsystems/<job>", methods=["POST"])
def save_subsystems(job):
    subs = request.json
    if not isinstance(subs, list):
        return jsonify({"error": "expected a list of subsystems"}), 400
    os.makedirs(os.path.join(JOBS_DIR, job), exist_ok=True)
    json.dump(subs, open(subsystems_path(job), "w"), indent=1)
    return jsonify({"ok": True, "count": len(subs)})


def rollersets_path(job):
    return os.path.join(JOBS_DIR, job, "rollersets.json")


def load_rollersets(job):
    p = rollersets_path(job)
    return json.load(open(p)) if os.path.exists(p) else []


@app.route("/api/rollersets/<job>", methods=["GET"])
def get_rollersets(job):
    return jsonify(load_rollersets(job))


@app.route("/api/rollersets/<job>", methods=["POST"])
def save_rollersets(job):
    """User-defined roller sets: [{id, name, rps, rollers: [{id, members, axis, dir}]}]."""
    sets = request.json
    if not isinstance(sets, list):
        return jsonify({"error": "expected a list of roller sets"}), 400
    os.makedirs(os.path.join(JOBS_DIR, job), exist_ok=True)
    json.dump(sets, open(rollersets_path(job), "w"), indent=1)
    return jsonify({"ok": True, "count": len(sets)})


@app.route("/assets/<job>/<path:fn>")
def assets(job, fn):
    if job in BUILTIN:
        return send_from_directory(os.path.join(PROJ, BUILTIN[job], "onshape_assets"), fn)
    return send_from_directory(os.path.join(JOBS_DIR, job, "onshape_assets"), fn)


def run_import(url, job_id):
    def progress(msg, frac=None):
        st = JOB_STATUS[job_id]
        st["message"] = msg
        if frac is not None:
            st["progress"] = max(st.get("progress", 0.0), float(frac))
    try:
        pipeline.import_assembly(url, job_id=job_id, progress=progress)
        model_data = json.load(open(job_model_path(job_id), encoding="utf-8"))
        if not model_data.get("dofs"):
            progress("reading nested movable mates…", 0.95)
            ensure_detected_subsystems(url, job_id)
        JOB_STATUS[job_id].update(status="done", done=True, message="done", progress=1.0)
    except Exception as e:
        traceback.print_exc()
        JOB_STATUS[job_id].update(status="error", done=True, message=str(e))


def run_repair(url, job_id):
    def progress(msg, frac=None):
        st = JOB_STATUS[job_id]
        st["message"] = msg
        if frac is not None:
            st["progress"] = max(st.get("progress", 0.0), float(frac))
    try:
        pipeline.repair_model(job_id, url, progress=progress)
        JOB_STATUS[job_id].update(status="done", done=True, message="done", progress=1.0)
    except Exception as e:
        traceback.print_exc()
        JOB_STATUS[job_id].update(status="error", done=True, message=str(e))


@app.route("/api/import", methods=["POST"])
def do_import():
    url = (request.json or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "Paste an Onshape assembly URL."}), 400
    job_id = job_id_for_url(url)
    if os.path.exists(job_model_path(job_id)):
        # Older cached linked-assembly imports may have no movable DOFs. Add
        # their cheap read-only mate inventory the first time they are reopened.
        cached = json.load(open(job_model_path(job_id), encoding="utf-8"))
        if not cached.get("dofs"):
            try:
                ensure_detected_subsystems(url, job_id)
            except Exception:
                pass  # The cached CAD remains viewable even when quota is unavailable.
        if cached.get("importVersion", 1) < pipeline.IMPORT_VERSION:
            # Older import: drop suppressed parts, retry missing meshes, add colors -- in place,
            # with progress, without redoing the expensive motion import.
            running = JOB_STATUS.get(job_id, {})
            if not (running.get("status") == "running"):
                JOB_STATUS[job_id] = {"status": "running", "message": "upgrading import…",
                                      "done": False, "progress": 0.0, "started": time.time()}
                threading.Thread(target=run_repair, args=(url, job_id), daemon=True).start()
            return jsonify({"job": job_id, "cached": False})
        return jsonify({"job": job_id, "cached": True})
    JOB_STATUS[job_id] = {"status": "running", "message": "starting…", "done": False,
                          "progress": 0.0, "started": time.time()}
    threading.Thread(target=run_import, args=(url, job_id), daemon=True).start()
    return jsonify({"job": job_id, "cached": False})


@app.route("/api/job/<job_id>")
def job_status(job_id):
    st = JOB_STATUS.get(job_id, {"status": "unknown", "message": "", "done": True, "progress": 1.0})
    out = dict(st)
    if "started" in st:
        out["elapsed"] = time.time() - st["started"]
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
