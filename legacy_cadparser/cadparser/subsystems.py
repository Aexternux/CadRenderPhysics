"""Step 6: group joints into the mechanisms a person would recognize."""

from collections import Counter, defaultdict

from cadparser.model import Subsystem


def find_subsystems(flat, groups, tree):
    """Return a list of Subsystem, one per connected piece beyond the chassis.

    Idea: imagine deleting the root group from the tree. Whatever stays connected
    moves as one mechanism. An arm with a wrist is one subsystem; each wheel is
    its own.

    1. Each joint whose parent is tree.root starts a new subsystem.
    2. Follow joints downward to collect its groups and joint names.
    3. Name it with the most common top_level_name among its parts. Ties go to
       the alphabetically first name.

    Sort the result by name so the output is stable. Floating groups aren't
    included because they aren't connected to anything.

    This is a first guess. Later, overrides.json lets a person correct it, for
    example by merging four separate wheels into one "Drivetrain".
    """
    children = defaultdict(list)
    for joint in tree.joints:
        children[joint.parent].append(joint)
    names_by_path = {path: part.top_level_name for path, part in flat.parts.items()}
    group_parts = {group.id: group.parts for group in groups}
    subsystems = []

    for root_joint in children[tree.root]:
        stack = [root_joint]
        subsystem_groups = []
        joint_names = []
        while stack:
            joint = stack.pop()
            subsystem_groups.append(joint.child)
            joint_names.append(joint.name)
            stack.extend(reversed(children[joint.child]))
        top_names = [names_by_path[path] for group_id in subsystem_groups
                     for path in group_parts[group_id]]
        counts = Counter(top_names)
        name = min((candidate for candidate, count in counts.items() if count == max(counts.values())))
        subsystems.append(Subsystem(name, subsystem_groups, joint_names))
    return sorted(subsystems, key=lambda subsystem: subsystem.name)
