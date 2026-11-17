"""Step 3: merge parts that can't move relative to each other."""

from cadparser.model import RigidGroup, RIGID_MATES


class UnionFind:
    """Finished for you. Tracks which items have been merged into one set."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def build_rigid_groups(flat):
    """Return a list of RigidGroup with ids 0, 1, 2, ...

    Union two parts when any of these is true:
      1. A FASTENED mate connects them.
      2. They're in the same mate group.
      3. They're inside a subassembly occurrence with no moving mate anywhere
         inside it. Moving means anything other than a FASTENED mate.

    Rule 3 test: a mate is inside occurrence P if its owner_path starts with P.
    Check every prefix of each part's path, because subassemblies nest.

    Open question: Onshape subassemblies can be rigid or flexible. Rule 3 is a
    starting guess. Check it against your cached JSON at the step 1 checkpoint.

    If a mate points at a path that isn't a part, add a warning to flat.warnings.
    """
    union_find = UnionFind(flat.parts)

    for mate in flat.mates:
        if mate.mate_type not in RIGID_MATES:
            continue
        if mate.occurrence_a not in flat.parts or mate.occurrence_b not in flat.parts:
            flat.warnings.append(f"Rigid mate {mate.name!r} refers to a non-part occurrence.")
            continue
        union_find.union(mate.occurrence_a, mate.occurrence_b)

    for member_paths in flat.mate_groups:
        valid = [path for path in member_paths if path in flat.parts]
        for path in member_paths:
            if path not in flat.parts:
                flat.warnings.append(f"Mate group refers to non-part occurrence {'/'.join(path)}.")
        for path in valid[1:]:
            union_find.union(valid[0], path)

    prefixes = {path[:length] for path in flat.parts for length in range(1, len(path) + 1)}
    for prefix in prefixes:
        has_motion = any(
            mate.owner_path[:len(prefix)] == prefix and mate.mate_type not in RIGID_MATES
            for mate in flat.mates
        )
        if has_motion:
            continue
        contained = [path for path in flat.parts if path[:len(prefix)] == prefix]
        for path in contained[1:]:
            union_find.union(contained[0], path)

    merged = {}
    for path in flat.parts:
        merged.setdefault(union_find.find(path), []).append(path)
    return [RigidGroup(id, sorted(parts)) for id, parts in enumerate(merged.values())]
