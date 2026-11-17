"""Local browser interface for inspecting cached CadParser assemblies.

Run ``python -m cadparser.web`` and open http://127.0.0.1:8080.
The server deliberately works only with files in ``cache/``; it never makes an
Onshape API request.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import threading
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cadparser.flatten import flatten
from cadparser.joints import build_tree, choose_root
from cadparser.kinematics import forward_kinematics, part_poses
from cadparser.onshape import OnshapeClient, fetch_assembly, parse_url
from cadparser.rigid_groups import build_rigid_groups
from cadparser.subsystems import find_subsystems


def load_model(cache_dir: Path, filename: str):
    """Build the browser-friendly model for one cached assembly file."""
    path = (cache_dir / filename).resolve()
    if path.parent != cache_dir.resolve() or path.suffix != ".json" or not path.is_file():
        raise FileNotFoundError(filename)
    with path.open(encoding="utf-8") as source:
        assembly = json.load(source)
    flat = flatten(assembly)
    groups = build_rigid_groups(flat)
    root = choose_root(flat, groups)
    tree = build_tree(flat, groups, root)
    subsystems = find_subsystems(flat, groups, tree)
    return flat, groups, tree, subsystems


def matrix(matrix):
    return matrix.tolist()


def occurrence_refs(raw, flat):
    """Return the Onshape part reference for every flattened occurrence."""
    root = raw.get("rootAssembly", {})
    definitions = {(item.get("documentId"), item.get("elementId"), item.get("configuration", "")): item
                   for item in raw.get("subAssemblies", [])}
    refs = {}
    for path in flat.parts:
        instance = next((item for item in root.get("instances", []) if item.get("id") == path[0]), None)
        owner = root
        for depth, instance_id in enumerate(path):
            if depth:
                if instance is None:
                    break
                definition = definitions.get((instance.get("documentId"), instance.get("elementId"), instance.get("configuration", "")))
                if definition is None:
                    break
                owner = definition
            instance = next((item for item in owner.get("instances", []) if item.get("id") == instance_id), None)
            if instance is None:
                break
        if instance and instance.get("type") == "Part" and instance.get("partId"):
            refs[path] = {
                "documentId": instance.get("documentId"), "elementId": instance.get("elementId"),
                "partId": instance.get("partId"), "documentVersion": instance.get("documentVersion"),
                "documentMicroversion": instance.get("documentMicroversion"),
                "configuration": instance.get("fullConfiguration", instance.get("configuration", "")),
            }
    return refs


def mesh_refs(raw, flat):
    """Collapse repeated occurrences to the distinct STL files they need."""
    return {mesh_id(ref): ref for ref in occurrence_refs(raw, flat).values()}


def mesh_inventory(cache_dir: Path, raw, flat):
    refs = mesh_refs(raw, flat)
    mesh_dir = cache_dir / "meshes"
    available = sorted(key for key in refs if (mesh_dir / f"{key}.stl").is_file())
    return {"total": len(refs), "cached": available}


def summary(cache_dir: Path, filename: str, values=None):
    flat, groups, tree, subsystems = load_model(cache_dir, filename)
    with (cache_dir / filename).open(encoding="utf-8") as source:
        raw = json.load(source)
    refs = occurrence_refs(raw, flat)
    poses = part_poses(flat, groups, forward_kinematics(tree, values or {}))
    return {
        "filename": filename,
        "parts": [
            {
                "path": list(path), "name": part.name,
                "topLevelName": part.top_level_name, "pose": matrix(poses[path]),
                "mesh": mesh_id(refs[path]) if path in refs else None,
            }
            for path, part in flat.parts.items()
        ],
        "groups": [{"id": group.id, "parts": [list(path) for path in group.parts]} for group in groups],
        "root": tree.root,
        "joints": [
            {"name": joint.name, "kind": joint.kind, "parent": joint.parent, "child": joint.child,
             "frame": matrix(joint.frame_world), "reversed": joint.reversed}
            for joint in tree.joints
        ],
        "subsystems": [
            {"name": subsystem.name, "groups": subsystem.groups, "joints": subsystem.joints}
            for subsystem in subsystems
        ],
        "loopClosures": [mate.name for mate in tree.loop_closures],
        "floatingGroups": tree.floating_groups,
        "warnings": flat.warnings + tree.warnings,
        "meshes": mesh_inventory(cache_dir, raw, flat),
    }


def mesh_id(ref):
    return hashlib.sha1(json.dumps(ref, sort_keys=True).encode()).hexdigest()[:20]


def cache_meshes(cache_dir: Path, filename: str, progress=None):
    """Download missing unique meshes, reporting after each completed request."""
    with (cache_dir / filename).open(encoding="utf-8") as source:
        raw = json.load(source)
    flat, _, _, _ = load_model(cache_dir, filename)
    root_document = raw.get("rootAssembly", {}).get("documentId")
    target_dir = cache_dir / "meshes"
    target_dir.mkdir(exist_ok=True)
    refs = mesh_refs(raw, flat)
    missing = [(key, ref) for key, ref in refs.items() if not (target_dir / f"{key}.stl").is_file()]
    if not missing:
        return 0
    client = OnshapeClient()
    for index, (key, ref) in enumerate(missing, 1):
        mesh = client.download_part_mesh(ref, link_document_id=root_document)
        temporary = target_dir / f".{key}.{os.getpid()}.tmp"
        temporary.write_bytes(mesh)
        temporary.replace(target_dir / f"{key}.stl")
        if progress:
            progress(index, len(missing), key)
    return len(missing)


@dataclass
class MeshJob:
    filename: str
    total: int = 0
    completed: int = 0
    available: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    done: bool = False


def start_mesh_job(cache_dir: Path, filename: str):
    """Start a non-blocking mesh download so the viewer remains responsive."""
    job = MeshJob(filename)
    with (cache_dir / filename).open(encoding="utf-8") as source:
        raw = json.load(source)
    flat, _, _, _ = load_model(cache_dir, filename)
    inventory = mesh_inventory(cache_dir, raw, flat)
    job.available = inventory["cached"]
    job.total = inventory["total"] - len(job.available)

    def run():
        def update(completed, total, key):
            job.completed, job.total = completed, total
            job.available.append(key)
        try:
            cache_meshes(cache_dir, filename, update)
        except Exception as error:  # Preserve any successfully cached parts for the viewer.
            job.errors.append(str(error))
        finally:
            job.done = True

    threading.Thread(target=run, name=f"cadparser-mesh-{filename}", daemon=True).start()
    return job


class CadHandler(SimpleHTTPRequestHandler):
    cache_dir: Path
    static_dir: Path
    mesh_jobs: dict[str, MeshJob] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.static_dir), **kwargs)

    def send_json(self, status, value):
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        request = urlparse(self.path)
        if request.path == "/api/assemblies":
            files = sorted(path.name for path in self.cache_dir.glob("*.json"))
            return self.send_json(HTTPStatus.OK, {"assemblies": files})
        if request.path == "/api/model":
            filename = parse_qs(request.query).get("file", [""])[0]
            try:
                return self.send_json(HTTPStatus.OK, summary(self.cache_dir, filename))
            except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as error:
                return self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        if request.path == "/api/meshes":
            filename = parse_qs(request.query).get("file", [""])[0]
            try:
                job = self.mesh_jobs.get(filename)
                if job is None:
                    with (self.cache_dir / filename).open(encoding="utf-8") as source:
                        raw = json.load(source)
                    flat, _, _, _ = load_model(self.cache_dir, filename)
                    inventory = mesh_inventory(self.cache_dir, raw, flat)
                    return self.send_json(HTTPStatus.OK, {"running": False, "done": True,
                                                         "completed": len(inventory["cached"]),
                                                         "total": inventory["total"],
                                                         "available": inventory["cached"], "errors": []})
                return self.send_json(HTTPStatus.OK, {"running": not job.done, "done": job.done,
                                                     "completed": job.completed, "total": job.total,
                                                     "available": job.available, "errors": job.errors})
            except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as error:
                return self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        if request.path.startswith("/assets/meshes/"):
            name = Path(request.path.rsplit("/", 1)[-1]).name
            path = self.cache_dir / "meshes" / name
            if path.is_file() and name.endswith(".stl"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "model/stl")
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
                with path.open("rb") as mesh_file:
                    self.wfile.write(mesh_file.read())
                return
            return self.send_error(HTTPStatus.NOT_FOUND)
        if request.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        endpoint = urlparse(self.path).path
        if endpoint not in {"/api/pose", "/api/import", "/api/meshes"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if endpoint == "/api/meshes":
                filename = payload["file"]
                job = self.mesh_jobs.get(filename)
                if job is None or job.done:
                    job = start_mesh_job(self.cache_dir, filename)
                    self.mesh_jobs[filename] = job
                return self.send_json(HTTPStatus.ACCEPTED, {"running": True, "done": False,
                                                            "completed": job.completed, "total": job.total,
                                                            "available": job.available, "errors": job.errors})
            if endpoint == "/api/import":
                ref = parse_url(payload["url"])
                fetch_assembly(OnshapeClient(), ref, self.cache_dir, refresh=payload.get("refresh", False))
                filename = "{did}_{wvm}_{wvmid}_{eid}.json".format(**ref)
                return self.send_json(HTTPStatus.OK, summary(self.cache_dir, filename))
            result = summary(self.cache_dir, payload["file"], payload.get("values", {}))
            self.send_json(HTTPStatus.OK, {"parts": result["parts"]})
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    CadHandler.cache_dir = cache_dir
    CadHandler.static_dir = Path(__file__).with_name("static")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CadHandler)
    print(f"CadParser viewer: http://127.0.0.1:{args.port}")
    print(f"Reading cached assemblies from {cache_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
