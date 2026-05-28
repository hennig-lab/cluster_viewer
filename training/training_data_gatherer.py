#!/usr/bin/env python3
"""
Searches subdirectories of a given input directory for 'neuron_data.json' and
'clusters_excluded.csv', then copies any found files into a matching folder
structure in the output directory.
"""

import argparse
import shutil
from pathlib import Path


TARGET_FILES = {"neuron_data.json", "clusters_excluded.csv"}


def find_targets(folder: Path) -> dict[str, Path]:
    """Recursively search a folder for target files. Returns a dict of {filename: path}."""
    found = {}
    for target in TARGET_FILES:
        matches = list(folder.rglob(target))
        if matches:
            # If multiple matches exist, take the first one and warn
            if len(matches) > 1:
                print(f"  [!] Multiple '{target}' found in '{folder.name}', using: {matches[0]}")
            found[target] = matches[0]
    return found


def collect_files(input_dir: Path, output_dir: Path, dry_run: bool = False) -> None:
    """Main logic: iterate top-level subdirectories and copy found target files."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    subdirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if not subdirs:
        print("No subdirectories found in input directory.")
        return

    copied_count = 0
    skipped_count = 0

    for subdir in subdirs:
        found = find_targets(subdir)

        if not found:
            print(f"[ skip ] '{subdir.name}' — no target files found")
            skipped_count += 1
            continue

        out_folder = output_dir / subdir.name
        print(f"[ copy ] '{subdir.name}' — found: {', '.join(found.keys())}")

        if not dry_run:
            out_folder.mkdir(parents=True, exist_ok=True)
            for filename, src_path in found.items():
                dest_path = out_folder / filename
                shutil.copy2(src_path, dest_path)
                print(f"         {src_path.relative_to(input_dir)}  →  {dest_path.relative_to(output_dir.parent)}")

        copied_count += 1

    print(f"\nDone. {copied_count} folder(s) processed, {skipped_count} skipped.")
    if dry_run:
        print("(Dry run — no files were written.)")


def main():
    parser = argparse.ArgumentParser(
        description="Copy neuron_data.json and clusters_excluded.csv from subdirectories to an output directory."
    )
    parser.add_argument("input_dir", help="Directory whose subdirectories will be searched")
    parser.add_argument("output_dir", help="Directory where matching output folders will be created")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be copied without writing any files"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    if args.dry_run:
        print("Mode   : dry run\n")
    else:
        print()

    collect_files(input_dir, output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
