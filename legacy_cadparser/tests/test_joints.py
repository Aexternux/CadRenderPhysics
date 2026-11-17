from cadparser.flatten import flatten
from cadparser.joints import build_tree, choose_root
from cadparser.rigid_groups import build_rigid_groups
from tests.fixture_assembly import assembly, cs, mate


def analyze(data):
    flat = flatten(data)
    groups = build_rigid_groups(flat)
    root = choose_root(flat, groups)
    return flat, groups, root, build_tree(flat, groups, root)


def group_of(groups, path):
    return next(g.id for g in groups if path in g.parts)


def test_root_is_the_chassis_group():
    _, groups, root, _ = analyze(assembly())
    assert root == group_of(groups, ("CH",))


def test_finds_each_joint_with_a_unique_name():
    _, _, _, tree = analyze(assembly())
    assert {j.name for j in tree.joints} == {"Arm pivot", "M1/Axle", "M2/Axle"}
    assert all(j.kind == "hinge" for j in tree.joints)


def test_every_joint_hangs_off_the_chassis_here():
    _, _, root, tree = analyze(assembly())
    assert all(j.parent == root for j in tree.joints)
    assert tree.loop_closures == [] and tree.floating_groups == []


def test_reversed_is_false_when_the_mate_already_points_outward():
    _, _, _, tree = analyze(assembly())
    assert not any(j.reversed for j in tree.joints)


def test_reversed_is_true_when_the_mate_points_toward_the_root():
    data = assembly()
    pivot = next(f for f in data["rootAssembly"]["features"] if f["id"] == "Arm pivot")
    entities = pivot["featureData"]["matedEntities"]
    entities[0]["matedOccurrence"], entities[1]["matedOccurrence"] = (
        entities[1]["matedOccurrence"], entities[0]["matedOccurrence"],
    )
    _, _, _, tree = analyze(data)
    assert next(j for j in tree.joints if j.name == "Arm pivot").reversed


def test_second_path_between_groups_is_a_loop_not_a_joint():
    data = assembly()
    data["rootAssembly"]["features"].append(
        mate("Link", "REVOLUTE", ["CH"], ["ARM", "BRACKET"], cs())
    )
    _, _, _, tree = analyze(data)
    assert len(tree.joints) == 3
    assert [m.name for m in tree.loop_closures] == ["Link"]


def test_disconnected_group_is_reported_as_floating():
    data = assembly()
    features = data["rootAssembly"]["features"]
    data["rootAssembly"]["features"] = [f for f in features if f["id"] != "Arm pivot"]
    _, groups, _, tree = analyze(data)
    assert tree.floating_groups == [group_of(groups, ("ARM", "TUBE"))]
    assert tree.warnings


def test_unsupported_mate_produces_a_warning_naming_it():
    data = assembly()
    data["rootAssembly"]["features"].append(
        mate("Odd joint", "BALL", ["CH"], ["ARM", "BRACKET"], cs())
    )
    _, _, _, tree = analyze(data)
    assert any("Odd joint" in warning for warning in tree.warnings)
