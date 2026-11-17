from cadparser.flatten import flatten
from cadparser.rigid_groups import build_rigid_groups
from tests.fixture_assembly import assembly


def group_sets(data):
    return {frozenset(g.parts) for g in build_rigid_groups(flatten(data))}


def test_expected_groups():
    assert group_sets(assembly()) == {
        frozenset({("CH",), ("BOLT",), ("BATTERY",), ("M1", "HOUSING"), ("M2", "HOUSING")}),
        frozenset({("ARM", "TUBE"), ("ARM", "BRACKET")}),
        frozenset({("M1", "WHEEL")}),
        frozenset({("M2", "WHEEL")}),
    }


def test_group_ids_count_up_from_zero():
    groups = build_rigid_groups(flatten(assembly()))
    assert sorted(g.id for g in groups) == list(range(len(groups)))


def test_every_part_is_in_exactly_one_group():
    flat = flatten(assembly())
    grouped = [path for g in build_rigid_groups(flat) for path in g.parts]
    assert len(grouped) == len(flat.parts)
    assert set(grouped) == set(flat.parts)


def test_a_moving_mate_inside_keeps_a_subassembly_flexible():
    wheel_group = next(
        g for g in build_rigid_groups(flatten(assembly())) if ("M1", "WHEEL") in g.parts
    )
    assert ("M1", "HOUSING") not in wheel_group.parts
