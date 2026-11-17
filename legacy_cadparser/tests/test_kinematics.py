import math

import numpy as np

from cadparser.flatten import flatten
from cadparser.joints import build_tree, choose_root
from cadparser.kinematics import forward_kinematics, joint_motion, part_poses
from cadparser.model import Joint, KinematicTree
from cadparser.rigid_groups import build_rigid_groups
from tests.fixture_assembly import assembly


def analyze():
    flat = flatten(assembly())
    groups = build_rigid_groups(flat)
    return flat, groups, build_tree(flat, groups, choose_root(flat, groups))


def at(x, y, z):
    frame = np.eye(4)
    frame[:3, 3] = [x, y, z]
    return frame


def test_hinge_rotates_about_z():
    motion = joint_motion("hinge", math.pi / 2)
    assert np.allclose(motion @ [1, 0, 0, 1], [0, 1, 0, 1])


def test_slide_moves_along_z():
    assert np.allclose(joint_motion("slide", 0.5) @ [0, 0, 0, 1], [0, 0, 0.5, 1])


def test_all_zero_values_reproduce_the_cad():
    flat, groups, tree = analyze()
    poses = part_poses(flat, groups, forward_kinematics(tree, {}))
    for path, part in flat.parts.items():
        assert np.allclose(poses[path], part.pose), path


def test_moving_the_arm_moves_only_the_arm():
    flat, groups, tree = analyze()
    poses = part_poses(flat, groups, forward_kinematics(tree, {"Arm pivot": math.pi / 2}))
    # The bracket sits 0.5 m along +X from the pivot. A +90 degree turn about +Y
    # (right-hand rule) carries +X to -Z.
    assert np.allclose(poses[("ARM", "BRACKET")][:3, 3], [0.1, 0, -0.3])
    assert np.allclose(poses[("BOLT",)], flat.parts[("BOLT",)].pose)
    assert np.allclose(poses[("M1", "WHEEL")], flat.parts[("M1", "WHEEL")].pose)


def test_child_rides_on_parent_and_both_motions_combine():
    # Elevator along Z, with an arm on top pivoting about Z at x = 1.
    tree = KinematicTree(root=0, joints=[
        Joint("elevator", "slide", parent=0, child=1, frame_world=at(0, 0, 0)),
        Joint("arm", "hinge", parent=1, child=2, frame_world=at(1, 0, 0)),
    ])
    deltas = forward_kinematics(tree, {"elevator": 2.0, "arm": math.pi / 2})
    # A point on the arm 1 m past the pivot starts at (2, 0, 0).
    assert np.allclose(deltas[2] @ [2, 0, 0, 1], [1, 1, 2, 1])
    assert np.allclose(deltas[0], np.eye(4))


def test_reversed_joint_moves_the_opposite_way():
    tree = KinematicTree(root=0, joints=[
        Joint("slider", "slide", parent=0, child=1, frame_world=at(0, 0, 0), reversed=True),
    ])
    deltas = forward_kinematics(tree, {"slider": 1.0})
    assert np.allclose(deltas[1] @ [0, 0, 0, 1], [0, 0, -1, 1])


def test_floating_groups_stay_where_they_are():
    tree = KinematicTree(root=0, joints=[], floating_groups=[7])
    assert np.allclose(forward_kinematics(tree, {})[7], np.eye(4))
