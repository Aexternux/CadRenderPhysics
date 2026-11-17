import numpy as np

from cadparser.flatten import flatten
from tests.fixture_assembly import assembly


def test_finds_every_part_and_skips_subassembly_occurrences():
    assert set(flatten(assembly()).parts) == {
        ("CH",), ("BOLT",), ("BATTERY",),
        ("ARM", "TUBE"), ("ARM", "BRACKET"),
        ("M1", "HOUSING"), ("M1", "WHEEL"),
        ("M2", "HOUSING"), ("M2", "WHEEL"),
    }


def test_part_pose_is_a_4x4_from_row_major_data():
    pose = flatten(assembly()).parts[("M2", "WHEEL")].pose
    assert pose.shape == (4, 4)
    assert np.allclose(pose[:3, 3], [-0.3, 0, 0.05])


def test_part_records_its_top_level_name():
    parts = flatten(assembly()).parts
    assert parts[("M2", "WHEEL")].top_level_name == "Module <2>"
    assert parts[("CH",)].top_level_name == "Chassis"


def test_one_definition_used_twice_gives_two_mates():
    axles = [m for m in flatten(assembly()).mates if m.name == "Axle"]
    assert {m.owner_path for m in axles} == {("M1",), ("M2",)}


def test_nested_mate_paths_are_full_paths():
    axle = next(m for m in flatten(assembly()).mates if m.owner_path == ("M2",))
    assert axle.occurrence_a == ("M2", "HOUSING")
    assert axle.occurrence_b == ("M2", "WHEEL")


def test_mate_frame_is_in_world_coordinates():
    axle = next(m for m in flatten(assembly()).mates if m.owner_path == ("M2",))
    assert np.allclose(axle.frame_world[:3, 3], [-0.3, 0, 0.05])


def test_mate_axis_is_column_two_of_the_frame():
    pivot = next(m for m in flatten(assembly()).mates if m.name == "Arm pivot")
    assert np.allclose(pivot.frame_world[:3, 2], [0, 1, 0])
    assert np.allclose(pivot.frame_world[:3, 3], [0.1, 0, 0.2])


def test_suppressed_mates_are_ignored():
    assert "Old fasten" not in {m.name for m in flatten(assembly()).mates}


def test_mate_groups_are_read():
    assert flatten(assembly()).mate_groups == [[("CH",), ("BATTERY",)]]


def test_missing_definition_warns_instead_of_crashing():
    data = assembly()
    data["subAssemblies"] = [s for s in data["subAssemblies"] if s["elementId"] != "E_ARM"]
    flat = flatten(data)
    assert ("ARM", "TUBE") not in flat.parts
    assert flat.warnings
