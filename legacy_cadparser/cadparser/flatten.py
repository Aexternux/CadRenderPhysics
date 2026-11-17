"""Step 2: turn nested Onshape JSON into flat parts and mates."""

import numpy as np

from cadparser.model import FlatAssembly, Mate, Part


def flatten(assembly):
    """Walk the assembly from the root and return a FlatAssembly.

    Shape of the JSON (see tests/fixture_assembly.py):
      rootAssembly.instances    What the root contains
      rootAssembly.occurrences  World transform for every path (16 numbers, row-major)
      rootAssembly.features     Mates defined at the root
      subAssemblies[]           Definitions, each with its own instances and features

    Algorithm: write a recursive walk(instances, features, prefix):
      - For each instance, path = prefix + (instance id,).
      - If it's a Part, add a Part, using the transform for that path.
      - If it's an Assembly, find its definition and recurse with prefix=path.
      - For each mate in features, prepend prefix to its occurrence paths.

    Gotchas the tests check:
      - One definition can be used many times. Four swerve modules means one
        definition but four sets of parts and mates. That's why you walk from
        the root instead of looping over subAssemblies.
      - Look up definitions by (documentId, elementId, configuration).
      - occurrences includes subassemblies too, not only parts.
      - Skip features with "suppressed": true.
      - matedOccurrence is relative to the subassembly that owns the mate.
      - featureType "mateGroup" lists members as
        featureData.occurrences[i].occurrence. Save them in mate_groups.
      - If a definition is missing, add a warning. Don't crash or skip silently.

    Mate frame: matedCS is relative to the part it's attached to. Build a 4x4
    matrix whose columns are xAxis, yAxis, zAxis, and origin, then:
        frame_world = part_world_pose @ matedCS_matrix
    Use matedEntities[0] for the frame and occurrence_a, [1] for occurrence_b.
    """
    flat = FlatAssembly()
    root = assembly.get("rootAssembly", {})
    occurrences = {
        tuple(item.get("path", [])): item
        for item in root.get("occurrences", [])
    }
    definitions = {
        (item.get("documentId"), item.get("elementId"), item.get("configuration", "")): item
        for item in assembly.get("subAssemblies", [])
    }

    def pose_for(path):
        occurrence = occurrences.get(path)
        if occurrence is None or len(occurrence.get("transform", [])) != 16:
            flat.warnings.append(f"No usable world transform for {'/'.join(path)}.")
            return np.eye(4)
        return np.asarray(occurrence["transform"], dtype=float).reshape(4, 4)

    def path_from(entity):
        path = entity.get("matedOccurrence", [])
        if isinstance(path, dict):
            path = path.get("path", [])
        return tuple(path)

    def mate_frame(part_path, cs):
        try:
            frame = np.eye(4)
            frame[:3, 0] = cs["xAxis"]
            frame[:3, 1] = cs["yAxis"]
            frame[:3, 2] = cs["zAxis"]
            frame[:3, 3] = cs["origin"]
            return pose_for(part_path) @ frame
        except (KeyError, TypeError, ValueError):
            flat.warnings.append(f"Mate frame on {'/'.join(part_path)} is malformed.")
            return pose_for(part_path)

    def add_features(features, prefix):
        for feature in features:
            if feature.get("suppressed"):
                continue
            data = feature.get("featureData", {})
            if feature.get("featureType") == "mateGroup":
                members = [prefix + tuple(item.get("occurrence", []))
                           for item in data.get("occurrences", [])]
                if members:
                    flat.mate_groups.append(members)
                continue
            if feature.get("featureType") != "mate":
                continue
            entities = data.get("matedEntities", [])
            if len(entities) < 2:
                flat.warnings.append(f"Mate {data.get('name', feature.get('id', '<unnamed>'))!r} has fewer than two entities.")
                continue
            occurrence_a = prefix + path_from(entities[0])
            occurrence_b = prefix + path_from(entities[1])
            name = data.get("name", feature.get("id", "<unnamed mate>"))
            flat.mates.append(Mate(
                name=name,
                mate_type=data.get("mateType", ""),
                owner_path=prefix,
                occurrence_a=occurrence_a,
                occurrence_b=occurrence_b,
                frame_world=mate_frame(occurrence_a, entities[0].get("matedCS", {})),
            ))

    def walk(instances, features, prefix, top_level_name=None):
        add_features(features, prefix)
        for instance in instances:
            instance_id = instance.get("id")
            if not instance_id:
                flat.warnings.append("Encountered an instance with no id.")
                continue
            path = prefix + (instance_id,)
            name = instance.get("name", instance_id)
            top_name = top_level_name or name
            if instance.get("type") == "Part":
                flat.parts[path] = Part(path, name, top_name, pose_for(path))
            elif instance.get("type") == "Assembly":
                key = (instance.get("documentId"), instance.get("elementId"),
                       instance.get("configuration", ""))
                definition = definitions.get(key)
                if definition is None:
                    flat.warnings.append(f"No definition found for subassembly {'/'.join(path)}.")
                    continue
                walk(definition.get("instances", []), definition.get("features", []), path, top_name)
            else:
                flat.warnings.append(f"Unsupported instance type {instance.get('type')!r} at {'/'.join(path)}.")

    walk(root.get("instances", []), root.get("features", []), ())
    return flat
