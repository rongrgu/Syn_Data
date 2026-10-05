"""
Split syn_data.py output back into per-case folders.

  synthetic/Case 4__Face 4_syn0.jpg  ->  <output>/Case 4/Face 4_syn0.jpg

The case is looked up in the manifest written by flatten_cases.py. If no
manifest is given (or a stem is missing from it), the case is taken from the
"Case N__" prefix in the file name.

Usage:
  python split_cases.py --input synthetic --manifest flat/manifest.json --output "Synthetic Bleaching"
"""
import argparse
import json
import shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEP = "__"
SYN_TAG = "_syn"


def main():
    p = argparse.ArgumentParser(description="Split synthetic images back into their original case folders.")
    p.add_argument("--input", default="synthetic", help="folder of syn_data.py output")
    p.add_argument("--output", default="Synthetic Bleaching", help="folder to create 'Case N' subfolders in")
    p.add_argument("--manifest", default="flat/manifest.json", help="manifest.json from flatten_cases.py")
    p.add_argument("--move", action="store_true", help="move files instead of copying")
    args = p.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        raise SystemExit(f"input folder not found: {in_dir}")

    manifest = {}
    if Path(args.manifest).is_file():
        with open(args.manifest) as f:
            manifest = json.load(f)
    else:
        print(f"no manifest at {args.manifest}; using file-name prefixes only")

    transfer = shutil.move if args.move else shutil.copy2
    counts, skipped = {}, []
    for path in sorted(in_dir.iterdir()):
        if path.suffix.lower() not in EXTS or path.name.startswith("."):
            continue
        # "Case 4__Face 4_syn0" -> source stem "Case 4__Face 4"
        src_stem = path.stem.rsplit(SYN_TAG, 1)[0] if SYN_TAG in path.stem else path.stem
        if src_stem in manifest:
            case = manifest[src_stem]["case"]
        elif SEP in src_stem:
            case = src_stem.split(SEP, 1)[0]
        else:
            skipped.append(path.name)
            continue

        new_name = path.name.split(SEP, 1)[1] if path.name.startswith(case + SEP) else path.name
        case_dir = out_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        transfer(str(path), str(case_dir / new_name))
        counts[case] = counts.get(case, 0) + 1

    for case in sorted(counts, key=lambda c: (len(c), c)):
        print(f"{case}: {counts[case]} images")
    if skipped:
        print(f"skipped {len(skipped)} files with no case info: {', '.join(skipped)}")
    print(f"finished: {sum(counts.values())} images written to {out_dir}")


if __name__ == "__main__":
    main()
