# CadParser

Turn an Onshape robot assembly into a model of its moving subsystems, then move them.

```
Onshape URL -> cached JSON -> flat parts + mates -> rigid groups
            -> joint tree -> subsystems -> pose any joint values
```

## What "any CAD" means

The goal is to accept any Onshape assembly and never fail silently.

| The CAD has | What the parser does |
|---|---|
| Revolute, slider, or cylindrical mates | Creates joints automatically |
| Linked or nested subassemblies | Handles them the same as local ones |
| Loops (four-bars) | Builds the tree and reports the loop closure |
| Unsupported mates (ball, planar, pin slot) | Adds a warning |
| Parts with no path to the chassis | Reports them as floating |
| A moving part that was fastened in place | Can't tell from the data; needs a manual override |

## Rules

1. Only `cadparser/onshape.py` talks to Onshape.
2. Fetch once and cache. Everything after that runs offline.
3. Never drop data silently. If something is skipped, add a warning.
4. Motion is calculated locally. Never drive joints through the Onshape API.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
git init
```

Put your API keys in `.env`. Check that `git status` doesn't list `.env` before your first commit.

## Fetch an assembly

Fetch once to preserve API quota, then use the cached JSON for all local analysis
and viewing:

```powershell
python fetch_assembly.py "<intake assembly URL>"
```

The raw response is saved in `cache/`. Check the first response from each new
robot document for the following details:

- Do mates inside a **linked** subassembly appear under `subAssemblies[].features`?
- What exact `featureType` and `mateType` values appear?
- Is there any field that says a subassembly is flexible or rigid?
- Do mate relations appear, and what do they look like?

The fixture is my best understanding of Onshape's format. If the real JSON differs, update the fixture so the tests describe reality.

**After step 6**, run `python analyze.py cache/<file>.json` and compare the result with what you know about the robot.

## Current limitations and next work

7. Cross-check every joint against the `/matevalues` endpoint.
8. Add `overrides.json` to mark fastened parts as movable and to rename or merge subsystems.
9. Handle coupled motion: read mate relations first, then loops such as the hopper wall. `coupling.json` from the old project is real test data.
10. Test `forward_kinematics` against `robot_intake/onshape_frames.json` from the old project.
11. Improve CAD mesh appearance and add MuJoCo export.
12. Export nested bodies to MuJoCo for `frc_robot_sim`.

## Local viewer

The browser viewer works entirely from cached assembly JSON and does not make API
calls. Fetch an assembly first, then start the server:

```powershell
python -m cadparser.web
```

Open `http://127.0.0.1:8080`. It shows cached CAD meshes when available,
detected subsystems and diagnostics, and sliders for each detected joint. On a
new assembly, click **Download and render CAD** once: meshes download in the
background, are cached under `cache/meshes/`, and appear progressively. Paste
an Onshape assembly URL in the **Import Onshape assembly** field to fetch and
cache another assembly. The small blue markers remain only for parts whose CAD
mesh has not been downloaded yet.
