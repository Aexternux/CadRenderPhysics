"""Fetch an assembly once and print a summary. Usage: python fetch_assembly.py "<url>" [--refresh]"""

import sys
from collections import Counter

from cadparser.onshape import OnshapeClient, fetch_assembly, parse_url


def summarize(assembly):
    root = assembly["rootAssembly"]
    subs = assembly.get("subAssemblies", [])
    print(f"Root instances:  {len(root['instances'])}")
    print(f"Occurrences:     {len(root['occurrences'])}")
    print(f"Subassemblies:   {len(subs)}")
    print(f"Root features:   {len(root.get('features', []))}")
    print(f"Nested features: {sum(len(s.get('features', [])) for s in subs)}")

    features = root.get("features", []) + [f for s in subs for f in s.get("features", [])]
    print("Feature types:", dict(Counter(f.get("featureType") for f in features)))
    print("Mate types:   ", dict(Counter(
        f.get("featureData", {}).get("mateType")
        for f in features if f.get("featureType") == "mate"
    )))


def main():
    args = [a for a in sys.argv[1:] if a != "--refresh"]
    if len(args) != 1:
        raise SystemExit(__doc__)
    client = OnshapeClient()
    assembly = fetch_assembly(client, parse_url(args[0]), "cache", "--refresh" in sys.argv)
    summarize(assembly)
    print(f"API calls this run: {client.calls}")


if __name__ == "__main__":
    main()
