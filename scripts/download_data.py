"""Acquire only the BrainVision Oddball recordings and their metadata."""
import argparse
import csv
from collections import Counter
import hashlib
from http.client import HTTPException
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

DATASET = "ds004621"
BUCKET = "https://s3.amazonaws.com/openneuro.org"
SUBJECTS = tuple(f"sub-{number:02d}" for number in range(1, 43))
REPO = Path(__file__).resolve().parents[1]
SHARED_FILES = (
    "dataset_description.json",
    "task-oddball_eeg.json",
    "task-oddball_events.json",
)
RECORDING_SUFFIXES = ("eeg.eeg", "eeg.vhdr", "eeg.vmrk", "channels.tsv", "events.tsv")
CHUNK_SIZE = 1024 * 1024


def selected_paths(subjects):
    paths = list(SHARED_FILES)
    for subject in subjects:
        stem = f"{subject}/eeg/{subject}_task-oddball"
        paths.extend(f"{stem}_{suffix}" for suffix in RECORDING_SUFFIXES)
    return paths


def object_url(path):
    return f"{BUCKET}/{quote(f'{DATASET}/{path}')}"


def list_selected(paths):
    expected = set(paths)
    selected = {}
    query = {"list-type": 2, "prefix": f"{DATASET}/"}
    while True:
        with urlopen(f"{BUCKET}?{urlencode(query)}", timeout=30) as response:
            page = ElementTree.parse(response).getroot()
        for obj in page.findall("{*}Contents"):
            key = obj.findtext("{*}Key")
            path = key.removeprefix(f"{DATASET}/")
            if path in expected:
                selected[path] = {
                    "path": path,
                    "url": object_url(path),
                    "bytes": int(obj.findtext("{*}Size")),
                    "etag": obj.findtext("{*}ETag"),
                }
        if page.findtext("{*}IsTruncated") != "true":
            break
        token = page.findtext("{*}NextContinuationToken")
        if not token:
            raise ValueError("The S3 listing did not supply its next-page token.")
        query["continuation-token"] = token
    missing = expected - selected.keys()
    if missing:
        raise ValueError("Files absent from the S3 listing:\n  " + "\n  ".join(sorted(missing)))
    return [selected[path] for path in paths]


@contextmanager
def atomic_output(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            yield output
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_current(local, obj, cached):
    return (
        cached.get("etag") == obj["etag"]
        and local.is_file()
        and local.stat().st_size == obj["bytes"]
        and sha256_file(local) == cached.get("sha256")
    )


def download_object(obj, destination):
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            size = 0
            with urlopen(obj["url"], timeout=60) as response, atomic_output(destination) as output:
                if response.headers.get("ETag") != obj["etag"]:
                    raise ValueError(f"S3 object changed since listing: {obj['path']}. Run again.")
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                if size != obj["bytes"]:
                    raise OSError(f"Expected {obj['bytes']} bytes, received {size}: {obj['path']}")
            return digest.hexdigest()
        except (OSError, HTTPException) as error:
            if isinstance(error, HTTPError) and error.code < 500 and error.code != 429:
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def read_tsv(path, field):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source, delimiter="\t")
        if field not in (rows.fieldnames or ()):
            raise ValueError(f"Missing {field} column: {path}")
        return [row[field] for row in rows]


def summarize_recordings(destination, subjects):
    summary = {}
    for subject in subjects:
        folder = destination / subject / "eeg"
        stem = f"{subject}_task-oddball"
        events = read_tsv(folder / f"{stem}_events.tsv", "event_type")
        channels = read_tsv(folder / f"{stem}_channels.tsv", "type")
        counts = Counter(" ".join(event.split()) for event in events)
        summary[subject] = {
            "event_counts": dict(sorted(counts.items())),
            "eeg_channels": sum(channel.strip().upper() == "EEG" for channel in channels),
        }
    return summary


def run(args):
    subjects = sorted(set(args.subjects or SUBJECTS))
    paths = selected_paths(subjects)
    objects = list_selected(paths)
    total = sum(obj["bytes"] for obj in objects)
    print(f"{len(subjects)} subjects | {len(objects)} files | {total / 1024**3:.2f} GiB")
    if args.dry_run:
        for obj in objects:
            print(f"{obj['bytes']:>12}  {obj['path']}")
        return

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "download_manifest.json"
    if subjects != list(SUBJECTS):
        suffix = subjects[0] if len(subjects) == 1 else "subset"
        manifest = destination / f"download_manifest_{suffix}.json"
    previous = {}
    for cached_manifest in sorted(destination.glob("download_manifest*.json")):
        cached = json.loads(cached_manifest.read_text("utf-8"))
        if cached.get("dataset") == DATASET:
            previous.update((entry["path"], entry) for entry in cached["files"])

    pending = [
        obj for obj in objects
        if not is_current(destination / obj["path"], obj, previous.get(obj["path"], {}))
    ]
    pending_paths = {obj["path"] for obj in pending}
    if sum(obj["bytes"] for obj in pending) > shutil.disk_usage(destination).free:
        raise OSError(f"Insufficient free space at {destination}")

    record = {
        "dataset": DATASET,
        "source": f"{BUCKET}/{DATASET}/",
        "subjects": subjects,
        "files": [previous[path] for path in paths if path not in pending_paths],
        "complete": False,
    }
    try:
        for index, obj in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {obj['path']}", flush=True)
            checksum = download_object(obj, destination / obj["path"])
            record["files"].append({**obj, "sha256": checksum})
        record["recordings"] = summarize_recordings(destination, subjects)
        record["complete"] = True
    finally:
        with atomic_output(manifest) as output:
            output.write((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    print(f"Download manifest: {manifest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, metavar="sub-NN",
                        help="Download only these participants. Defaults to all 42.")
    parser.add_argument("--output-dir", type=Path, default=REPO / "data" / "raw" / DATASET)
    parser.add_argument("--dry-run", action="store_true", help="List files and size without downloading.")
    try:
        run(parser.parse_args())
    except (OSError, HTTPException, ValueError, ElementTree.ParseError) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
