#!/usr/bin/env python3
"""Package the reified mol-dir lists + minimal per-mol files into Zenodo tarballs.

Streams directly into a .tar.gz per dataset (no staging dir). Each archive
contains the 3 split pkls, a README / MANIFEST, and exactly the 4 files per
mol dir that the loader actually reads: mol.pkl, smiles.txt, system.pdb,
traj.xtc. scalars.csv is deliberately omitted (never read by the pipeline).
"""

import argparse
import hashlib
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPLITS_DIR = REPO_ROOT / "data_gen" / "splits"
OFFICIAL_CODE_URL = "https://github.com/ani11452/Align_Your_Structures"

KEEP_FILES = ("mol.pkl", "smiles.txt", "system.pdb", "traj.xtc")

DATASETS = {
    "qm9":   {"geom_dir": "GEOM-QM9",   "pretty": "QM9"},
    "drugs": {"geom_dir": "GEOM-DRUGS", "pretty": "DRUGS"},
}

README_TEMPLATE = """# GEOM-{DS_UPPER} Pre-simulated MD Trajectories

Companion release for:

> *Align Your Structures: Generating Trajectories with Structure Pretraining
> for Molecular Dynamics* (ICLR 2026).
> https://arxiv.org/abs/2604.03911

## Counts (matches paper §5.2)

| split | unique SMILES | mol dirs | reps / SMILES |
|---|---|---|---|
| train | {n_train_smiles} | {n_train_mols} | ~5 |
| val   | {n_val_smiles}   | {n_val_mols}   | ~5 |
| test  | {n_test_smiles}  | {n_test_mols}  | ~5 |

Each SMILES was simulated with five all-atom, explicit-solvent replicas at
4 fs timestep (HMR 1.5 amu), 5 ns per trajectory, 100-step frame interval.

## Layout

```
{archive_root}/
|-- README.md                (this file)
|-- MANIFEST.json            (counts, provenance, source repo + commit)
|-- splits/
|   |-- {ds}_train_mol_dirs.pkl    # sorted list of mol-dir basenames
|   |-- {ds}_val_mol_dirs.pkl
|   `-- {ds}_test_mol_dirs.pkl
`-- output_trajectories/{GEOM_DIR}/4fs_HMR15_5ns_actual/
    |-- train/{{gen}}_results/{{mol_basename}}/
    |   |-- mol.pkl          # RDKit Mol with hydrogens
    |   |-- smiles.txt       # canonical SMILES (first line) + OpenMM timings
    |   |-- system.pdb       # OpenMM topology + initial positions
    |   `-- traj.xtc         # 5 ns trajectory, 4 fs dt, frame_interval=100
    |-- val/...
    `-- test/...
```

`scalars.csv` (OpenMM energy / temperature / density log) is NOT included.
It was written during simulation but the training and evaluation pipelines
never read it, so omitting it cuts archive size by ~30-40% without
affecting reproducibility.

## Using this data

Reference code: {code_url}

```bash
git clone {code_url}
cd $(basename {code_url})

# Option 1: extract alongside sibling dataset, strip the archive root
tar xf {tarball_name} --strip-components=1 -C $MD_DATA_ROOT

# Option 2: keep the archive root, set MD_DATA_ROOT to the extracted folder
tar xf {tarball_name} -C /path/to/place
export MD_DATA_ROOT=/path/to/place/{archive_root}
```

Then verify against the original split counts:

```bash
python data_gen/preprocess/reify_splits.py --dry-run
```

All six `[OK]` lines should print with the paper's numbers.
"""


def build_basename_index(split_dir):
    index = {}
    for gen in os.listdir(split_dir):
        gen_dir = split_dir / gen
        if not gen_dir.is_dir():
            continue
        for mol in os.listdir(gen_dir):
            mol_dir = gen_dir / mol
            if mol_dir.is_dir():
                index[mol] = mol_dir
    return index


def load_split_pkl(splits_dir, ds, split):
    with open(splits_dir / f"{ds}_{split}_mol_dirs.pkl", "rb") as f:
        return pickle.load(f)


def git_rev(arg):
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *arg], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def add_bytes(tar, arcname, data, mtime):
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mtime = mtime
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def sha256_file(path, block=4 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_outer_smiles_counts(splits_dir, ds):
    mf = splits_dir / "MANIFEST.json"
    out = {"train": "-", "val": "-", "test": "-"}
    if not mf.exists():
        return out
    with open(mf) as f:
        data = json.load(f)
    for split in out:
        key = f"{ds}_{split}"
        if key in data.get("splits", {}):
            out[split] = data["splits"][key].get("n_smiles", "-")
    return out


def package_dataset(ds, args):
    ds_info = DATASETS[ds]
    geom_dir = ds_info["geom_dir"]
    traj_root = args.md_data_root / "output_trajectories" / geom_dir / "4fs_HMR15_5ns_actual"
    if not traj_root.is_dir():
        sys.exit(f"[{ds}] missing trajectory root: {traj_root}")

    splits = {s: load_split_pkl(args.splits_dir, ds, s) for s in ("train", "val", "test")}

    print(f"[{ds}] scanning mol dirs on disk...")
    indices = {}
    for split in tqdm(("train", "val", "test"), desc=f"{ds}: index splits", unit="split"):
        indices[split] = build_basename_index(traj_root / split)

    missing = {s: [b for b in splits[s] if b not in indices[s]] for s in splits}
    missing = {s: m for s, m in missing.items() if m}
    if missing:
        for s, m in missing.items():
            print(f"[ERROR] {ds} {s}: {len(m)} basenames missing on disk (e.g. {m[:3]})")
        sys.exit(1)

    total_mol_dirs = sum(len(splits[s]) for s in splits)
    print(f"[{ds}] mol-dir totals: "
          + ", ".join(f"{s}={len(splits[s])}" for s in ("train", "val", "test"))
          + f"  (total={total_mol_dirs})")

    if args.dry_run:
        est_gb = total_mol_dirs * (2.0 if ds == "qm9" else 3.25) / 1024
        est_gz_gb = est_gb * (0.61 if ds == "qm9" else 0.78)
        print(f"[{ds}] --dry-run; estimated raw={est_gb:.1f} GB, tar.gz~{est_gz_gb:.1f} GB (not written).")
        return None

    archive_root = f"geom-{ds}-align-your-structures-{args.version}"
    tarball_name = f"{archive_root}.tar.gz"
    out_path = args.out_dir / tarball_name
    if out_path.exists() and not args.force:
        sys.exit(f"[{ds}] {out_path} already exists; pass --force to overwrite.")

    smiles_counts = load_outer_smiles_counts(args.splits_dir, ds)

    manifest = {
        "dataset": ds,
        "version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper": "https://arxiv.org/abs/2604.03911",
        "source_repo": args.code_url,
        "source_commit": git_rev(["rev-parse", "HEAD"]),
        "files_per_mol": list(KEEP_FILES),
        "excluded_from_mol": ["scalars.csv"],
        "splits": {
            s: {"n_mol_dirs": len(splits[s]), "n_smiles": smiles_counts[s]}
            for s in ("train", "val", "test")
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    readme = README_TEMPLATE.format(
        DS_UPPER=ds_info["pretty"],
        GEOM_DIR=geom_dir,
        ds=ds,
        archive_root=archive_root,
        tarball_name=tarball_name,
        n_train_smiles=smiles_counts["train"],
        n_val_smiles=smiles_counts["val"],
        n_test_smiles=smiles_counts["test"],
        n_train_mols=len(splits["train"]),
        n_val_mols=len(splits["val"]),
        n_test_mols=len(splits["test"]),
        code_url=args.code_url,
    )
    readme_bytes = readme.encode("utf-8")
    build_mtime = int(datetime.now(timezone.utc).timestamp())

    # Prepare (split, basename) tasks preserving train -> val -> test order.
    tasks = []
    for split in ("train", "val", "test"):
        for basename in splits[split]:
            tasks.append((split, basename))

    def read_mol_dir(task):
        split, basename = task
        src_mol_dir = indices[split][basename]
        gen_parent = src_mol_dir.parent.name
        arc_mol_dir = (
            f"{archive_root}/output_trajectories/{geom_dir}/4fs_HMR15_5ns_actual/"
            f"{split}/{gen_parent}/{basename}"
        )
        out = []
        for fn in KEEP_FILES:
            src = src_mol_dir / fn
            if not src.is_file():
                continue
            with open(src, "rb") as f:
                data = f.read()
            st = src.stat()
            out.append({
                "arcname": f"{arc_mol_dir}/{fn}",
                "data": data,
                "mtime": int(st.st_mtime),
                "mode": (st.st_mode & 0o777) or 0o644,
            })
        return split, out

    # Open the output tarball, optionally piping through pigz for parallel gzip.
    pigz_bin = shutil.which("pigz") if args.use_pigz else None
    out_f = None
    pigz_proc = None
    if pigz_bin:
        out_f = open(out_path, "wb")
        pigz_proc = subprocess.Popen(
            [pigz_bin, f"-{args.compresslevel}", "-p", str(args.jobs)],
            stdin=subprocess.PIPE,
            stdout=out_f,
        )
        tar_kwargs = dict(fileobj=pigz_proc.stdin, mode="w|")
        print(f"[{ds}] writing {out_path}  (pigz -{args.compresslevel} -p {args.jobs}, {args.read_workers} readers)")
    else:
        tar_kwargs = dict(name=str(out_path), mode="w:gz", compresslevel=args.compresslevel)
        print(f"[{ds}] writing {out_path}  (stdlib gzip -{args.compresslevel}, {args.read_workers} readers)")

    try:
        with tarfile.open(**tar_kwargs) as tar:
            add_bytes(tar, f"{archive_root}/README.md", readme_bytes, build_mtime)
            add_bytes(tar, f"{archive_root}/MANIFEST.json", manifest_bytes, build_mtime)
            for split in ("train", "val", "test"):
                src_pkl = args.splits_dir / f"{ds}_{split}_mol_dirs.pkl"
                tar.add(src_pkl, arcname=f"{archive_root}/splits/{ds}_{split}_mol_dirs.pkl")

            pbar = tqdm(total=total_mol_dirs, desc=f"{ds}: tar", unit="mol", smoothing=0.02)
            with ThreadPoolExecutor(max_workers=args.read_workers) as ex:
                for split, entries in ex.map(read_mol_dir, tasks, chunksize=8):
                    for e in entries:
                        info = tarfile.TarInfo(name=e["arcname"])
                        info.size = len(e["data"])
                        info.mtime = e["mtime"]
                        info.mode = e["mode"]
                        tar.addfile(info, io.BytesIO(e["data"]))
                    pbar.update(1)
                    pbar.set_postfix_str(split, refresh=False)
            pbar.close()
    finally:
        if pigz_proc is not None:
            pigz_proc.stdin.close()
            rc = pigz_proc.wait()
            out_f.close()
            if rc != 0:
                sys.exit(f"[{ds}] pigz exited with code {rc}")

    print(f"[{ds}] hashing {out_path.name} ...")
    sha = sha256_file(out_path)
    sidecar = out_path.with_name(out_path.name + ".sha256")
    with open(sidecar, "w") as f:
        f.write(f"{sha}  {out_path.name}\n")
    size_gb = out_path.stat().st_size / (1024 ** 3)
    print(f"[{ds}] done. size={size_gb:.2f} GB  sha256={sha}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--md-data-root", type=Path,
                    default=Path(os.environ.get("MD_DATA_ROOT", "")),
                    help="Path to md_data")
    ap.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR,
                    help="Where the *_mol_dirs.pkl + MANIFEST.json live")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write the tarballs (e.g. /atlas2/u/aniketh/zenodo_stage)")
    ap.add_argument("--datasets", type=str, default="qm9,drugs",
                    help="Comma-separated subset of qm9,drugs")
    ap.add_argument("--version", type=str, default="v1",
                    help="Version tag embedded in the tarball + archive root name")
    ap.add_argument("--code-url", type=str, default=OFFICIAL_CODE_URL,
                    help=f"Repo URL embedded in the archive README + MANIFEST (default: {OFFICIAL_CODE_URL})")
    ap.add_argument("--compresslevel", type=int, default=6,
                    help="gzip level (1=fast, 9=small). Level 6 is ~5%% larger than 9 but ~3x faster.")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1),
                    help="Parallel gzip workers for pigz (default: $(nproc))")
    ap.add_argument("--read-workers", type=int, default=16,
                    help="Parallel threads reading mol dirs from NFS (default: 16)")
    ap.add_argument("--use-pigz", action=argparse.BooleanOptionalAction, default=True,
                    help="Pipe the tar stream through pigz for parallel gzip (default: on if pigz is on PATH)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Verify indices and print size estimates without writing")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing tarballs")
    args = ap.parse_args()

    if not str(args.md_data_root) or not args.md_data_root.exists():
        sys.exit(f"MD_DATA_ROOT not set or missing: {args.md_data_root!r}")
    if not args.splits_dir.exists():
        sys.exit(f"splits dir missing: {args.splits_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            sys.exit(f"unknown dataset: {d}")

    for ds in datasets:
        package_dataset(ds, args)


if __name__ == "__main__":
    main()
