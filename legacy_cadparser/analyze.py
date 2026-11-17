"""Run the offline pipeline on a cached assembly. Usage: python analyze.py cache/<file>.json"""

import json
import sys

from cadparser.flatten import flatten
from cadparser.joints import build_tree, choose_root
from cadparser.rigid_groups import build_rigid_groups
from cadparser.subsystems import find_subsystems


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as f:
        assembly = json.load(f)

    flat = flatten(assembly)
    groups = build_rigid_groups(flat)
    tree = build_tree(flat, groups, choose_root(flat, groups))
    subsystems = find_subsystems(flat, groups, tree)

    root_group = next(g for g in groups if g.id == tree.root)
    print(f"Parts: {len(flat.parts)}  Mates: {len(flat.mates)}  Rigid groups: {len(groups)}")
    print(f"Root: group {tree.root} with {len(root_group.parts)} parts")
    print(f"Joints: {len(tree.joints)}  Loops: {len(tree.loop_closures)}  "
          f"Floating groups: {len(tree.floating_groups)}")

    print("\nSubsystems:")
    for subsystem in subsystems:
        print(f"  {subsystem.name}: {len(subsystem.groups)} groups")
        for joint in subsystem.joints:
            print(f"    - {joint}")

    warnings = flat.warnings + tree.warnings
    print(f"\nWarnings: {len(warnings)}")
    for warning in warnings:
        print(f"  ! {warning}")


if __name__ == "__main__":
    main()
