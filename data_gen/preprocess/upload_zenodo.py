#!/usr/bin/env python3
"""Create a draft Zenodo deposition and upload a packaged tarball + sidecar.

Reads the metadata from a JSON file, creates a NEW draft deposition (which
reserves a DOI), uploads the tarball via the bucket URL, then (optionally)
uploads the .sha256 sidecar. Does NOT publish — you review and publish in
the Zenodo web UI.

Env:
  ZENODO_TOKEN    Personal access token (scopes: deposit:write, deposit:actions)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_API = "https://zenodo.org/api"
CHUNK = 4 * 1024 * 1024


def make_session(token):
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s


def create_deposition(s, metadata):
    r = s.post(f"{ZENODO_API}/deposit/depositions", json={"metadata": metadata}, timeout=60)
    if not r.ok:
        sys.exit(f"Failed to create deposition: {r.status_code}\n{r.text}")
    return r.json()


class _ProgressStream:
    # File-like wrapper with read() + __len__ so requests sets Content-Length
    # and avoids chunked transfer encoding (which Zenodo's bucket API rejects).
    def __init__(self, path, bar):
        self._f = open(path, "rb")
        self._bar = bar
        self._size = path.stat().st_size

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._f.close()

    def __len__(self):
        return self._size

    def read(self, size=-1):
        chunk = self._f.read(size)
        if chunk:
            self._bar.update(len(chunk))
        return chunk


def upload_one_file(s, bucket_url, path, retries=5):
    size = path.stat().st_size
    url = f"{bucket_url}/{path.name}"
    headers = {"Content-Type": "application/octet-stream"}
    for attempt in range(1, retries + 1):
        try:
            with tqdm(
                total=size, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"upload {path.name} (try {attempt}/{retries})",
            ) as bar, _ProgressStream(path, bar) as stream:
                r = s.put(url, data=stream, headers=headers, timeout=(30, 3600))
            if r.ok:
                body = r.json()
                if body.get("size") != size:
                    print(f"[warn] server reports size={body.get('size')} vs local={size}; retrying...")
                else:
                    return body
            else:
                print(f"[warn] upload returned {r.status_code}: {r.text[:200]}")
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"[warn] upload interrupted ({e}); retrying...")
        if attempt < retries:
            time.sleep(5 * attempt)
    sys.exit(f"Upload of {path.name} failed after {retries} attempts.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", dest="files", type=Path, action="append", default=[],
                    help="File to upload. Pass multiple times for multi-part tarballs + sidecars.")
    ap.add_argument("--tarball", type=Path, default=None,
                    help="(Legacy) single-tarball shortcut; appended to --file list.")
    ap.add_argument("--sidecar", type=Path, default=None,
                    help="(Legacy) sidecar shortcut; appended to --file list.")
    ap.add_argument("--metadata", type=Path, required=True,
                    help="JSON file with a top-level 'metadata' key")
    ap.add_argument("--deposition-id", type=int, default=None,
                    help="Upload into an existing draft deposition instead of creating a new one.")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN env var is required.")

    files = list(args.files)
    if args.tarball is not None:
        files.append(args.tarball)
    if args.sidecar is not None:
        files.append(args.sidecar)
    if not files:
        sys.exit("must pass at least one --file (or --tarball)")
    for p in files:
        if not p.is_file():
            sys.exit(f"file not found: {p}")

    with open(args.metadata) as f:
        meta_blob = json.load(f)
    metadata = meta_blob["metadata"]

    s = make_session(token)
    if args.deposition_id is None:
        print(f"Creating draft deposition: {metadata['title']!r}")
        dep = create_deposition(s, metadata)
    else:
        print(f"Reusing existing draft deposition {args.deposition_id}")
        r = s.get(f"{ZENODO_API}/deposit/depositions/{args.deposition_id}", timeout=60)
        if not r.ok:
            sys.exit(f"Failed to load deposition {args.deposition_id}: {r.status_code} {r.text}")
        dep = r.json()
    dep_id = dep["id"]
    bucket_url = dep["links"]["bucket"]
    html_url = dep["links"]["html"]
    reserved_doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    print(f"  deposition id : {dep_id}")
    print(f"  reserved DOI  : {reserved_doi}")
    print(f"  draft URL     : {html_url}")
    print(f"  uploading     : {len(files)} file(s)")

    for path in files:
        upload_one_file(s, bucket_url, path)

    print()
    print("DONE — draft is staged, NOT yet published.")
    print(f"Review: {html_url}")
    print(f"When ready, paste this DOI into the repo README and paper, then publish in the web UI:")
    print(f"  {reserved_doi}")


if __name__ == "__main__":
    main()
