"""A tiny fake robot in Onshape's JSON shape. Draw it on paper before step 2.

Root
  CH       Chassis (part)
  BOLT     Bolt, fastened to the chassis
  BATTERY  Battery, in a mate group with the chassis
  ARM      Arm subassembly: TUBE + BRACKET, with no moving mates inside
  M1, M2   Two uses of ONE module definition: HOUSING + WHEEL, joined by a revolute

Root mates
  Bolt fasten        CH <-> BOLT
  Arm pivot          CH <-> ARM/TUBE, revolute about world Y at (0.1, 0, 0.2)
  Old fasten         CH <-> ARM/TUBE, SUPPRESSED, so it must be ignored
  M1 mount, M2 mount CH <-> each HOUSING

Expected: 9 parts, 4 rigid groups, 3 joints, 3 subsystems.
"""

import copy


def translate(x, y, z):
    return [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, 0.0, 0.0, 0.0, 1.0]


def cs(origin=(0.0, 0.0, 0.0), x=(1.0, 0.0, 0.0), y=(0.0, 1.0, 0.0), z=(0.0, 0.0, 1.0)):
    return {"origin": list(origin), "xAxis": list(x), "yAxis": list(y), "zAxis": list(z)}


def part(instance_id, name):
    return {"id": instance_id, "type": "Part", "name": name, "documentId": "D", "elementId": "PS",
            "partId": instance_id, "configuration": "default"}


def subassembly(instance_id, name, element_id):
    return {"id": instance_id, "type": "Assembly", "name": name, "documentId": "D",
            "elementId": element_id, "configuration": "default"}


def mate(name, mate_type, path_a, path_b, frame_a=None, suppressed=False):
    return {
        "featureType": "mate",
        "id": name,
        "suppressed": suppressed,
        "featureData": {
            "name": name,
            "mateType": mate_type,
            "matedEntities": [
                {"matedOccurrence": list(path_a), "matedCS": frame_a or cs()},
                {"matedOccurrence": list(path_b), "matedCS": cs()},
            ],
        },
    }


# Revolute about world Y: the frame's Z axis points along +Y.
PIVOT_FRAME = cs(origin=(0.1, 0.0, 0.2), x=(1.0, 0.0, 0.0), y=(0.0, 0.0, -1.0), z=(0.0, 1.0, 0.0))

_ASSEMBLY = {
    "rootAssembly": {
        "instances": [
            part("CH", "Chassis"),
            part("BOLT", "Bolt"),
            part("BATTERY", "Battery"),
            subassembly("ARM", "Arm <1>", "E_ARM"),
            subassembly("M1", "Module <1>", "E_MOD"),
            subassembly("M2", "Module <2>", "E_MOD"),
        ],
        "occurrences": [
            {"path": ["CH"], "transform": translate(0, 0, 0)},
            {"path": ["BOLT"], "transform": translate(0.2, 0, 0)},
            {"path": ["BATTERY"], "transform": translate(-0.1, 0, 0)},
            {"path": ["ARM"], "transform": translate(0.1, 0, 0.2)},
            {"path": ["ARM", "TUBE"], "transform": translate(0.1, 0, 0.2)},
            {"path": ["ARM", "BRACKET"], "transform": translate(0.6, 0, 0.2)},
            {"path": ["M1"], "transform": translate(0.3, 0, 0)},
            {"path": ["M1", "HOUSING"], "transform": translate(0.3, 0, 0)},
            {"path": ["M1", "WHEEL"], "transform": translate(0.3, 0, 0.05)},
            {"path": ["M2"], "transform": translate(-0.3, 0, 0)},
            {"path": ["M2", "HOUSING"], "transform": translate(-0.3, 0, 0)},
            {"path": ["M2", "WHEEL"], "transform": translate(-0.3, 0, 0.05)},
        ],
        "features": [
            mate("Bolt fasten", "FASTENED", ["CH"], ["BOLT"]),
            mate("Arm pivot", "REVOLUTE", ["CH"], ["ARM", "TUBE"], PIVOT_FRAME),
            mate("Old fasten", "FASTENED", ["CH"], ["ARM", "TUBE"], suppressed=True),
            mate("M1 mount", "FASTENED", ["CH"], ["M1", "HOUSING"]),
            mate("M2 mount", "FASTENED", ["CH"], ["M2", "HOUSING"]),
            {
                "featureType": "mateGroup",
                "id": "Group 1",
                "suppressed": False,
                "featureData": {
                    "name": "Group 1",
                    "occurrences": [{"occurrence": ["CH"]}, {"occurrence": ["BATTERY"]}],
                },
            },
        ],
    },
    "subAssemblies": [
        {
            "documentId": "D",
            "elementId": "E_ARM",
            "configuration": "default",
            "instances": [part("TUBE", "Arm tube"), part("BRACKET", "Arm bracket")],
            "features": [],
        },
        {
            "documentId": "D",
            "elementId": "E_MOD",
            "configuration": "default",
            "instances": [part("HOUSING", "Housing"), part("WHEEL", "Wheel")],
            "features": [
                mate("Axle", "REVOLUTE", ["HOUSING"], ["WHEEL"], cs(origin=(0.0, 0.0, 0.05))),
            ],
        },
    ],
}


def assembly():
    """Return a fresh copy, so a test can edit it without affecting other tests."""
    return copy.deepcopy(_ASSEMBLY)
