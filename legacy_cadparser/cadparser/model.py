"""Shared data structures. Every pipeline step reads and writes these.

Conventions:
  - A path is a tuple of instance IDs from the root, e.g. ("M1", "WHEEL").
  - Poses are 4x4 numpy arrays in world coordinates, in meters.
  - A joint frame's Z axis is the joint axis.
  - Joint value q = 0 means "exactly as modeled in the CAD".
"""

from dataclasses import dataclass, field

import numpy as np

Path = tuple[str, ...]

# Onshape mate type -> joint kind. Mates that aren't listed here or in
# RIGID_MATES can't be simulated yet and must produce a warning.
MOVING_MATES = {"REVOLUTE": "hinge", "SLIDER": "slide", "CYLINDRICAL": "hinge"}
RIGID_MATES = {"FASTENED"}


@dataclass
class Part:
    path: Path
    name: str
    top_level_name: str  # Name of the root-level instance this part sits under
    pose: np.ndarray


@dataclass
class Mate:
    name: str
    mate_type: str
    owner_path: Path  # Subassembly occurrence that defines this mate; () for root
    occurrence_a: Path  # Full path from the root, not relative to the owner
    occurrence_b: Path
    frame_world: np.ndarray  # World pose of the mate frame on occurrence_a


@dataclass
class FlatAssembly:
    parts: dict[Path, Part] = field(default_factory=dict)
    mates: list[Mate] = field(default_factory=list)  # Excludes suppressed mates
    mate_groups: list[list[Path]] = field(default_factory=list)  # Full paths
    warnings: list[str] = field(default_factory=list)


@dataclass
class RigidGroup:
    id: int
    parts: list[Path]


@dataclass
class Joint:
    name: str  # Unique: "/".join(owner_path + (mate name,))
    kind: str  # "hinge" or "slide"
    parent: int  # RigidGroup id
    child: int  # RigidGroup id
    frame_world: np.ndarray
    reversed: bool = False  # True if the mate's a->b order is child->parent here


@dataclass
class KinematicTree:
    root: int
    joints: list[Joint]  # Ordered so a parent always appears before its children
    loop_closures: list[Mate] = field(default_factory=list)
    floating_groups: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Subsystem:
    name: str
    groups: list[int]
    joints: list[str]
