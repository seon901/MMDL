"""Restore archived project sources to their original relative paths; dry-run by default."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json


def plan_restoration(repo, root):
    manifest = json.loads(
        (repo / "results/server_source_manifest.json").read_text(encoding="utf-8")
    )
    plans = []
    for record in manifest["files"]:
        relative = PurePosixPath(record["source"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Invalid source path")
        stored = (repo / record["destination"]).resolve()
        if repo not in stored.parents:
            raise ValueError("Stored source is outside the repository")
        data = stored.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Stored source checksum mismatch: " + record["destination"])
        target = root.joinpath(*relative.parts).resolve()
        if root not in target.parents:
            raise ValueError("Target is outside the selected root")
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise ValueError("Existing target differs; nothing overwritten: " + str(target))
            continue
        parent = target.parent
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir():
            raise ValueError("Target parent is not a directory: " + str(parent))
        plans.append((target, data))
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.root.expanduser().resolve()
    plans = plan_restoration(repo, root)
    for target, data in plans:
        print(target)
    print("Files to restore:", len(plans))
    if not args.apply:
        print("Dry-run only. Add --apply after reviewing the target paths.")
        return
    for target, data in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    print("Source restoration complete. No training or evaluation was started.")


if __name__ == "__main__":
    main()
