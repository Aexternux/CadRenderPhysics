from cadparser.flatten import flatten
from cadparser.joints import build_tree, choose_root
from cadparser.model import Joint, KinematicTree, RigidGroup
from cadparser.rigid_groups import build_rigid_groups
from cadparser.subsystems import find_subsystems
from tests.fixture_assembly import assembly
import numpy as np


def test_fake_robot_has_three_named_subsystems():
    flat = flatten(assembly())
    groups = build_rigid_groups(flat)
    tree = build_tree(flat, groups, choose_root(flat, groups))
    subsystems = find_subsystems(flat, groups, tree)
    assert [s.name for s in subsystems] == ["Arm <1>", "Module <1>", "Module <2>"]
    assert subsystems[0].joints == ["Arm pivot"]
    assert subsystems[1].joints == ["M1/Axle"]


def test_chained_joints_stay_in_one_subsystem():
    flat = flatten(assembly())
    groups = [
        RigidGroup(0, [("CH",)]),
        RigidGroup(1, [("ARM", "TUBE")]),
        RigidGroup(2, [("ARM", "BRACKET")]),
    ]
    tree = KinematicTree(root=0, joints=[
        Joint("shoulder", "hinge", parent=0, child=1, frame_world=np.eye(4)),
        Joint("wrist", "hinge", parent=1, child=2, frame_world=np.eye(4)),
    ])
    subsystems = find_subsystems(flat, groups, tree)
    assert len(subsystems) == 1
    assert subsystems[0].joints == ["shoulder", "wrist"]
    assert sorted(subsystems[0].groups) == [1, 2]
