"""
Flatten the per-case image folders into one folder that syn_data.py can read.

  Cases Bleaching/Case 4/Face 4.jpg  ->  flat/Case 4__Face 4.jpg

The case name is prefixed to each file so names stay unique and the case can be
recovered later. A manifest.json is also written next to the images, mapping
each flat file stem back to its original case and path.

Usage:
  python flatten_cases.py --input "Cases Bleaching" --output flat
  python syn_data.py --input flat --output synthetic --n 5
  python split_cases.py --input synthetic --manifest flat/manifest.json --output "Synthetic Bleaching"
"""
import argparse
import json
import shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEP = "__"


def main():
    p = argparse.ArgumentParser(description="Flatten case folders into one folder for syn_data.py.")
    p.add_argument("--input", default="Cases Bleaching", help="folder containing 'Case N' subfolders")
    p.add_argument("--output", default="flat", help="flat folder to write images into")
    args = p.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        raise SystemExit(f"input folder not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for case_dir in sorted(d for d in in_dir.iterdir() if d.is_dir()):
        for img in sorted(case_dir.iterdir()):
            if img.suffix.lower() not in EXTS or img.name.startswith("."):
                continue
            flat_name = f"{case_dir.name}{SEP}{img.name}"
            shutil.copy2(img, out_dir / flat_name)
            manifest[Path(flat_name).stem] = {
                "case": case_dir.name,
                "original": str(img.relative_to(in_dir)),
            }

    if not manifest:
        raise SystemExit(f"no images found under {in_dir}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_cases = len({v["case"] for v in manifest.values()})
    print(f"copied {len(manifest)} images from {n_cases} cases into {out_dir}")
    print(f"manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
