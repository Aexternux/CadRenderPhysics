"""Offline test for detect.py: feed a synthetic assembly (no API) and check the grouping."""
import detect

DID = "aaaaaaaaaaaaaaaaaaaaaaaa"          # the robot doc (ref did)
LINKED = "bbbbbbbbbbbbbbbbbbbbbbbb"       # a different doc => "linked" subsystem

# top-level instances: two subassembly instances + one loose part
ADEF = {"rootAssembly": {"instances": [
    {"id": "INTAKE", "type": "Assembly", "name": "Intake", "documentId": LINKED},
    {"id": "SHOOTER", "type": "Assembly", "name": "Shooter", "documentId": DID},
    {"id": "PLATE", "type": "Part", "name": "Base Plate", "documentId": DID},
]}}

# every movable mate lives inside a subassembly (owner path non-empty), plus a top-level one
MATEVALUES = {"mateValues": [
    {"jsonType": "Revolute", "mateName": "left roller",  "featureId": "f1", "ownerOccurrencePath": ["INTAKE"]},
    {"jsonType": "Revolute", "mateName": "right roller", "featureId": "f2", "ownerOccurrencePath": ["INTAKE", "SUBARM"]},
    {"jsonType": "Revolute", "mateName": "arm pivot",    "featureId": "f3", "ownerOccurrencePath": ["INTAKE"]},
    {"jsonType": "Revolute", "mateName": "flywheel",     "featureId": "f4", "ownerOccurrencePath": ["SHOOTER"]},
    {"jsonType": "Revolute", "mateName": "turret yaw",   "featureId": "f5", "ownerOccurrencePath": ["SHOOTER"]},
    {"jsonType": "Slider",   "mateName": "elevator",     "featureId": "f6", "ownerOccurrencePath": []},
    {"jsonType": "Fastened", "mateName": "bolt",         "featureId": "f7", "ownerOccurrencePath": ["SHOOTER"]},  # ignore
]}

# Monkeypatch the two API reads with our fixtures.
_calls = []
def fake_get(path):
    _calls.append(path)
    return MATEVALUES if path.endswith("/matevalues") else ADEF
detect.get = fake_get

res = detect.detect(f"https://cad.onshape.com/documents/{DID}/w/cccccccccccccccccccccccc/e/dddddddddddddddddddddddd", save=False)

# ---- assertions ----
assert res["total"] == 6, f"expected 6 movable (fastened ignored), got {res['total']}"
byname = {s["name"]: s for s in res["subsystems"]}
assert set(byname) == {"Intake", "Shooter", "Robot (top level)"}, byname.keys()
assert byname["Intake"]["linked"] is True,  "Intake is a separate doc -> linked"
assert byname["Shooter"]["linked"] is False, "Shooter shares the robot doc -> not linked"
assert byname["Robot (top level)"]["linked"] is False
assert len(byname["Intake"]["mechanisms"]) == 3
assert len(byname["Shooter"]["mechanisms"]) == 2   # flywheel + turret, bolt excluded
assert len(byname["Robot (top level)"]["mechanisms"]) == 1

roles = {m["mateName"]: m["role"] for s in res["subsystems"] for m in s["mechanisms"]}
assert roles["left roller"] == "roller/wheel", roles["left roller"]
assert roles["flywheel"] == "roller/wheel", roles["flywheel"]
assert roles["arm pivot"] == "arm/pivot", roles["arm pivot"]
assert roles["turret yaw"] == "turret", roles["turret yaw"]
assert roles["elevator"] == "linear (elevator/telescope)", roles["elevator"]

# nested roller keeps its depth
nested = next(m for s in res["subsystems"] for m in s["mechanisms"] if m["mateName"] == "right roller")
assert nested["depth"] == 2, nested["depth"]

assert len(_calls) == 2, f"detection must be exactly 2 API calls, got {len(_calls)}"

print("\nALL ASSERTIONS PASSED — detect.py logic is correct (2 calls, fastened ignored,")
print("linked flag right, roles/rollers classified, nesting depth preserved).")
