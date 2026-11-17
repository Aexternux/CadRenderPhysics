"""Step 4: connect rigid groups with joints and arrange them as a tree."""

from collections import deque

from cadparser.model import Joint, KinematicTree, MOVING_MATES, RIGID_MATES


def choose_root(flat, groups):
    """Return the id of the group that acts as the chassis.

    Pick the group with the most parts. Ties go to the lowest id.

    This is only a first guess. At the real-CAD checkpoint, see whether occurrences
    marked "fixed" do better. An override can replace the rule later.
    """
    return min(groups, key=lambda group: (-len(group.parts), group.id)).id


def build_tree(flat, groups, root):
    """Return a KinematicTree, using breadth-first search from `root`.

    1. Map each part path to its group id.
    2. Go through flat.mates in order and sort each one:
         - In MOVING_MATES: an edge between two groups.
         - In RIGID_MATES: already handled in step 3, ignore it.
         - Anything else: add a warning naming the mate and its type.
    3. Run BFS from root. When an edge reaches a new group, create a Joint:
         - parent = the group you came from; child = the new group.
         - reversed = True if occurrence_a is on the child side.
         - name = "/".join(owner_path + (mate name,)).
    4. An edge between two groups already in the tree is a loop. Put its Mate
       in loop_closures instead of creating a Joint.
    5. Groups BFS never reached go in floating_groups, with a warning.

    A moving mate with both ends in the same group can't move at all. Add a
    warning and ignore it.

    Why a tree? Each group gets exactly one parent, so a pose can be calculated
    by walking from the root. Loops need separate handling later.
    """
    path_to_group = {path: group.id for group in groups for path in group.parts}
    edges = []
    warnings = []
    for mate in flat.mates:
        if mate.mate_type in RIGID_MATES:
            continue
        if mate.mate_type not in MOVING_MATES:
            warnings.append(f"Unsupported mate {mate.name!r} ({mate.mate_type}).")
            continue
        group_a = path_to_group.get(mate.occurrence_a)
        group_b = path_to_group.get(mate.occurrence_b)
        if group_a is None or group_b is None:
            warnings.append(f"Moving mate {mate.name!r} refers to a non-part occurrence.")
        elif group_a == group_b:
            warnings.append(f"Moving mate {mate.name!r} connects one rigid group and was ignored.")
        else:
            edges.append((mate, group_a, group_b))

    adjacent = {group.id: [] for group in groups}
    for index, (_, group_a, group_b) in enumerate(edges):
        adjacent[group_a].append((index, group_b))
        adjacent[group_b].append((index, group_a))

    visited = {root}
    queue = deque([root])
    joints = []
    tree_edges = set()
    loop_edges = set()
    while queue:
        parent = queue.popleft()
        for index, other in adjacent[parent]:
            if index in tree_edges or index in loop_edges:
                continue
            mate, group_a, group_b = edges[index]
            if other in visited:
                loop_edges.add(index)
                continue
            visited.add(other)
            queue.append(other)
            tree_edges.add(index)
            reversed_joint = group_a == other
            joints.append(Joint(
                name="/".join(mate.owner_path + (mate.name,)),
                kind=MOVING_MATES[mate.mate_type],
                parent=parent,
                child=other,
                frame_world=mate.frame_world,
                reversed=reversed_joint,
            ))

    floating = sorted(group.id for group in groups if group.id not in visited)
    warnings.extend(f"Rigid group {group_id} is floating (not connected to the root)." for group_id in floating)
    return KinematicTree(root, joints, [edges[index][0] for index in sorted(loop_edges)], floating, warnings)
