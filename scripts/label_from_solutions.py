"""The normal label tool (pipeline/eval/label_tool.py), with every area prefilled from the solutions json.

  python -m scripts.label_from_solutions <annotations.json> <images dir> [--name sensei]

Opens the same window over the frames in out/run_frames.json (the last run). Each area shows the solution as
"already labelled"; n/p move, 1/2/3 and X override it, q saves. Labels go to labels/<name>.json.
Never merge this file into labels/dev.json while these frames are the hidden validation set (CLAUDE.md rule 1).
"""
import argparse
import json
from pathlib import Path

from pipeline.common.config import ROOT
from pipeline.eval import label_tool

STOCK = {"empty": 0.0, "half": 0.5, "full": 1.0}

ap = argparse.ArgumentParser()
ap.add_argument("annotations", type=Path)
ap.add_argument("images", type=Path, help="folder with <camera>/<file>.jpg of the annotated frames")
ap.add_argument("--name", default="sensei")
args = ap.parse_args()

out = ROOT / "labels" / f"{args.name}.json"
labels = json.loads(out.read_text()) if out.exists() else {}
for cam, frames in json.loads(args.annotations.read_text()).items():
    for fn, areas in frames.items():
        for area, a in areas.items():
            labels.setdefault(cam, {}).setdefault(fn, {}).setdefault(area, {
                "planogram_compliant": bool(a["planogram_compliant"]),
                "stock_level": a["stock_level"],
                "stock_value": STOCK[a["stock_level"]],
            })
out.write_text(json.dumps(labels, indent=2))

label_tool.IMAGES_DIR = args.images
label_tool.run(args.name, None, out)
