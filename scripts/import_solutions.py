"""Write the organisers' annotations.json as a labeller file, same format as the label tool output.

  python scripts/import_solutions.py <annotations.json> [--name sensei]   -> labels/<name>.json

Then `python -m pipeline.eval.merge_labels` folds it into labels/dev.json like any other labeller.
Do NOT merge it while the same frames are the hidden validation set (CLAUDE.md hard rule 1).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOCK = {"empty": 0.0, "half": 0.5, "full": 1.0}

ap = argparse.ArgumentParser()
ap.add_argument("annotations", type=Path)
ap.add_argument("--name", default="sensei")
args = ap.parse_args()

src = json.loads(args.annotations.read_text())
out = {
    cam: {
        fn: {
            area: {
                "planogram_compliant": bool(a["planogram_compliant"]),
                "stock_level": a["stock_level"],
                "stock_value": STOCK[a["stock_level"]],
            }
            for area, a in areas.items()
        }
        for fn, areas in frames.items()
    }
    for cam, frames in src.items()
}
dst = ROOT / "labels" / f"{args.name}.json"
dst.write_text(json.dumps(out, indent=1))
print(f"{sum(len(a) for f in out.values() for a in f.values())} areas -> {dst}")
