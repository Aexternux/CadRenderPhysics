"""
Auto-importer for onshape-to-robot.

onshape-to-robot only turns a mate into a robot joint if the mate is NAMED with a
`dof_` prefix in Onshape ("other mates are not considered"). Our FRC CAD uses normal
Revolute/Slider/Cylindrical mates that are NOT tagged, so a plain export would yield a
rigid robot with zero joints.

This wrapper monkeypatches onshape_to_robot so that EVERY movable mate
(REVOLUTE / CYLINDRICAL / SLIDER) is automatically renamed with a `dof_` prefix right
before the library processes mates -- giving the "upload any robot, get a joint for
everything that moves" behaviour, while reusing all of onshape-to-robot's (well tested)
geometry / transform / tree-building / URDF machinery.

We rename the mate in BOTH places the library reads names from:
  - assembly_data["rootAssembly"]["features"]  (drives joint detection)
  - self.features["features"]                  (drives joint LIMIT lookup)
keyed by the original (unique) Onshape mate name, so slider/revolute travel limits set
in CAD are preserved.

Usage:  python auto_import.py <robot_dir>      (same arg as `onshape-to-robot <robot_dir>`)
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

from onshape_to_robot import assembly as _assembly_mod
from onshape_to_robot import export as _export_mod

# Default travel given to a movable joint that came out of CAD with no usable limits
# (e.g. a slider whose mate had no "limits" enabled). These are guesses so the joint is
# draggable; set real limits on the mate in Onshape and re-export for accurate travel.
DEFAULT_PRISMATIC_RANGE = (0.0, 0.20)        # meters
DEFAULT_REVOLUTE_RANGE = (-3.14159, 3.14159)  # radians

# Mate types we auto-promote to a degree of freedom.
# NOTE: onshape-to-robot maps CYLINDRICAL -> revolute (it drops the sliding DOF).
# PLANAR is intentionally excluded (the library has no joint type for it and would crash);
# BALL is excluded for now to keep the first pass simple.
MOVABLE_MATE_TYPES = {"REVOLUTE", "CYLINDRICAL", "SLIDER"}

# Prefixes that already have special meaning to onshape-to-robot -- never touch these.
RESERVED_PREFIXES = ("dof_", "fix_", "frame_", "link_", "closing_")


def _sanitize(name: str, used: set) -> str:
    """Turn an arbitrary Onshape mate name into a safe, unique dof_ joint name."""
    base = name.lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    if not base:
        base = "joint"
    # A trailing _inv / _inverted would be misread by the library as an axis flip.
    if base.endswith("_inv") or base.endswith("_inverted"):
        base += "_j"
    candidate = base
    i = 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


BASE_NAME_HINTS = ("static", "base", "chassis", "frame", "drivetrain",
                   "drive base", "drivebase", "root", "ground")


def _choose_base(self):
    """Reorder top-level instances so a sensible fixed base is first.

    onshape-to-robot roots the kinematic tree at the FIRST top-level instance, which is
    arbitrary (assembly order). We instead pick the instance that is (a) grounded/fixed in
    CAD if any, else (b) involved in the most mates (the most likely fixed frame), with a
    name-keyword tie-break. That instance becomes the pinned base of the robot.
    """
    from collections import Counter

    ra = self.assembly_data["rootAssembly"]
    instances = ra["instances"]
    if len(instances) < 2:
        return
    ids = [i["id"] for i in instances]
    idset = set(ids)
    namemap = {i["id"]: i.get("name", "") for i in instances}

    fixed_ids = {
        o["path"][0]
        for o in ra.get("occurrences", [])
        if len(o.get("path", [])) == 1 and o.get("fixed") and o["path"][0] in idset
    }

    deg = Counter()
    for f in ra.get("features", []):
        if f.get("featureType") != "mate":
            continue
        for me in f.get("featureData", {}).get("matedEntities", []):
            occ = me.get("matedOccurrence", [])
            if occ and occ[0] in idset:
                deg[occ[0]] += 1

    def score(iid):
        nm = namemap[iid].lower()
        keyword = any(k in nm for k in BASE_NAME_HINTS)
        return (iid in fixed_ids, deg.get(iid, 0), keyword)

    base_id = max(ids, key=score)
    instances.sort(key=lambda i: 0 if i["id"] == base_id else 1)  # stable: base first
    print(
        f"[auto_import] Base link: '{namemap[base_id]}' "
        f"(fixed={base_id in fixed_ids}, mate-degree={deg.get(base_id, 0)})"
    )


def _auto_tag(self):
    """Rename movable mates to dof_* in both the assembly data and the feature list."""
    _choose_base(self)

    used = set()
    rename = {}  # original Onshape name -> new dof_ name

    features = self.assembly_data.get("rootAssembly", {}).get("features", [])
    for feature in features:
        if feature.get("featureType") != "mate":
            continue
        data = feature.get("featureData", {})
        original = data.get("name", "")
        mate_type = data.get("mateType")
        if mate_type not in MOVABLE_MATE_TYPES:
            continue
        if original.startswith(RESERVED_PREFIXES):
            continue
        new_name = "dof_" + _sanitize(original, used)
        data["name"] = new_name
        rename[original] = new_name

    # Mirror the rename into the FeatureScript feature list so get_limits() still matches
    # (it looks joints up by name), preserving any travel limits set in CAD.
    for feature in self.features.get("features", []):
        msg = feature.get("message", {})
        if msg.get("name") in rename:
            msg["name"] = rename[msg["name"]]

    if rename:
        print(f"[auto_import] Auto-tagged {len(rename)} movable mate(s) as joints:")
        for original, new_name in rename.items():
            print(f"[auto_import]   '{original}'  ->  {new_name}")
    else:
        print("[auto_import] WARNING: no REVOLUTE/CYLINDRICAL/SLIDER mates found to tag.")

    return self._orig_process_mates()


# Install the monkeypatch: wrap Assembly.process_mates so auto-tagging runs first.
_assembly_mod.Assembly._orig_process_mates = _assembly_mod.Assembly.process_mates
_assembly_mod.Assembly.process_mates = _auto_tag


def postprocess_urdf(urdf_path: str):
    """Clean up the exported URDF so every movable joint is unique and draggable."""
    if not os.path.exists(urdf_path):
        print(f"[auto_import] postprocess: {urdf_path} not found, skipping.")
        return

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    seen = set()
    renamed = 0
    relimited = 0
    for joint in root.findall("joint"):
        jtype = joint.get("type")

        # 1) Make joint names unique (onshape-to-robot can emit duplicates).
        name = joint.get("name", "joint")
        if name in seen:
            i = 2
            while f"{name}_{i}" in seen:
                i += 1
            name = f"{name}_{i}"
            joint.set("name", name)
            renamed += 1
        seen.add(name)

        # 2) Give degenerate (lower==upper) movable joints a usable default range.
        if jtype in ("revolute", "prismatic"):
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            try:
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
            except ValueError:
                lower, upper = 0.0, 0.0
            if lower == upper:
                lo, hi = (
                    DEFAULT_PRISMATIC_RANGE if jtype == "prismatic"
                    else DEFAULT_REVOLUTE_RANGE
                )
                limit.set("lower", str(lo))
                limit.set("upper", str(hi))
                if limit.get("effort") in (None, "", "0"):
                    limit.set("effort", "100")
                if limit.get("velocity") in (None, "", "0"):
                    limit.set("velocity", "10")
                relimited += 1

    tree.write(urdf_path)
    print(
        f"[auto_import] postprocess: {renamed} joint name(s) de-duplicated, "
        f"{relimited} joint(s) given default limits -> {urdf_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_import.py <robot_dir>")
        sys.exit(1)
    robot_dir = sys.argv[1]
    # export.main() reads the robot dir from argv via argparse; hand it through unchanged.
    sys.argv = ["onshape-to-robot"] + sys.argv[1:]
    _export_mod.main()
    # onshape-to-robot writes <robot_dir>/robot.urdf by default.
    postprocess_urdf(os.path.join(robot_dir, "robot.urdf"))
