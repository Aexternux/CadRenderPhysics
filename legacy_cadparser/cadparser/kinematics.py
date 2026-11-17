"""Step 5: calculate where everything is for a given set of joint values."""

import numpy as np


def joint_motion(kind, q):
    """Return the 4x4 motion of a joint in its own frame, where Z is the axis.

    hinge: rotate by q radians about Z.
    slide: translate by q meters along Z.
    """
    motion = np.eye(4)
    if kind == "hinge":
        cosine, sine = np.cos(q), np.sin(q)
        motion[:3, :3] = ((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1))
    elif kind == "slide":
        motion[2, 3] = q
    else:
        raise ValueError(f"Unknown joint kind: {kind!r}")
    return motion


def forward_kinematics(tree, q):
    """Return {group id: 4x4 delta}, the movement of each group from its CAD pose.

    `q` maps joint names to values. Missing joints default to 0.

    A group's new pose is delta @ cad_pose. The root's delta is the identity.
    For each joint in tree.joints, in order:

        F = joint.frame_world
        delta[child] = delta[parent] @ F @ joint_motion(kind, value) @ inverse(F)

    Read it right to left: move into the joint's frame, apply the motion, move
    back, then carry along whatever the parent did. That last step is what lets
    an arm ride on an elevator, which the old pose snapshots couldn't do.

    If joint.reversed is True, negate the value.

    Floating groups have no joint, so give them the identity too.
    """
    deltas = {tree.root: np.eye(4)}
    for joint in tree.joints:
        value = q.get(joint.name, 0.0)
        if joint.reversed:
            value = -value
        frame = joint.frame_world
        deltas[joint.child] = (
            deltas[joint.parent] @ frame @ joint_motion(joint.kind, value) @ np.linalg.inv(frame)
        )
    for group_id in tree.floating_groups:
        deltas[group_id] = np.eye(4)
    return deltas


def part_poses(flat, groups, deltas):
    """Return {part path: 4x4 pose}: delta of the part's group @ part's CAD pose."""
    path_to_group = {path: group.id for group in groups for path in group.parts}
    return {path: deltas[path_to_group[path]] @ part.pose for path, part in flat.parts.items()}
