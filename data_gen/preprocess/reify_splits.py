#!/usr/bin/env python3
"""Reify the six (dataset x split) splits used in the paper to mol-dir pkls.

Mirrors the dir-enumeration / filtering / subsampling logic of
experiments/data_load/data_loader.py:TrajectoryDataset and the canonical-SMILES
matching done by the Test* variants, then writes a deterministic list of
mol-dir basenames per split under data_gen/splits/.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_LOADER_PATH = REPO_ROOT / "experiments" / "data_load" / "data_loader.py"

EXPECTED_COUNTS = {
    "qm9_train":   {"n_smiles": 1109, "n_mol_dirs": 5534},
    "qm9_val":     {"n_smiles": 1018, "n_mol_dirs": 5080},
    "qm9_test":    {"n_smiles": 239,  "n_mol_dirs": 1193},
    "drugs_train": {"n_smiles": 1137, "n_mol_dirs": 5682},
    "drugs_val":   {"n_smiles": 1044, "n_mol_dirs": 5209},
    "drugs_test":  {"n_smiles": 100,  "n_mol_dirs": 496},
}

TEST_PKLS = {
    "qm9":   REPO_ROOT / "model_outputs_official/qm9_trajectory_official/qm9_noH_1000_kabsch_traj_interpolator_pretrain_final_539_25/qm9_test_set.pkl",
    "drugs": REPO_ROOT / "model_outputs_official/drugs_trajectory_official/drugs_noH_1000_kabsch_traj_interpolator_pretrain_fs_25/drugs_test_subset.pkl",
}


def parse_ignore_from_data_loader():
    # Mirror of experiments/data_load/data_loader.py:42 (IGNORE list for DRUGS oversize mols).
    src = DATA_LOADER_PATH.read_text()
    m = re.search(r"^IGNORE\s*=\s*\[(.*?)\]", src, re.DOTALL | re.MULTILINE)
    if m is None:
        raise RuntimeError(f"Could not locate `IGNORE = [...]` in {DATA_LOADER_PATH}")
    return set(re.findall(r"'([^']+)'", m.group(1)))


def remove_suffix(s):
    return s.rsplit("_", 1)[0]


def build_data_dict(split_dir, apply_ignore, ignore_set, desc=None):
    # Mirror of TrajectoryDataset.__init__ at data_loader.py:507-519.
    data_dict = {}
    gens = os.listdir(split_dir)
    for gen in tqdm(gens, desc=desc or f"scan {split_dir.name}", unit="gen", leave=False):
        gen_dir = split_dir / gen
        if not gen_dir.is_dir():
            continue
        for mol in os.listdir(gen_dir):
            mol_dir = gen_dir / mol
            if not mol_dir.is_dir():
                continue
            if not (mol_dir / "system.pdb").exists():
                continue
            if apply_ignore and mol in ignore_set:
                continue
            key = remove_suffix(mol)
            data_dict.setdefault(key, []).append(mol)
    return data_dict


def reify_train(ds, split_dir, md_data_root, ignore_set):
    apply_ignore = (ds == "drugs")
    dd = build_data_dict(split_dir, apply_ignore, ignore_set, desc=f"{ds} train: scan")

    pkl_path = md_data_root / "output_trajectories" / f"GEOM-{ds.upper()}" / "train_subset_keys.pkl"
    with open(pkl_path, "rb") as f:
        pinned_keys = pickle.load(f)
    pinned_set = set(pinned_keys)

    missing = pinned_set - set(dd.keys())
    if missing:
        raise RuntimeError(
            f"{ds} train: {len(missing)} keys from {pkl_path.name} missing in filesystem dd "
            f"(e.g. {sorted(missing)[:3]})"
        )

    # Independent reproducibility check: Random(0).sample(k=25%) on the filesystem-order
    # key list must equal the pinned set exactly. Guards against a silently-changed pkl.
    keys_order = list(dd.keys())
    rng = random.Random(0)
    chosen_idxs = rng.sample(range(len(keys_order)), k=int(len(keys_order) * 0.25))
    random_set = {keys_order[i] for i in chosen_idxs}
    if random_set != pinned_set:
        only_in_pkl = pinned_set - random_set
        only_in_random = random_set - pinned_set
        raise RuntimeError(
            f"{ds} train: Random(0) 25% disagrees with {pkl_path.name}. "
            f"only_in_pkl={len(only_in_pkl)} only_in_random={len(only_in_random)}"
        )

    mol_dirs = [m for k in pinned_keys for m in dd[k]]
    return sorted(mol_dirs), len(pinned_set)


def reify_val(ds, split_dir, ignore_set):
    apply_ignore = (ds == "drugs")
    dd = build_data_dict(split_dir, apply_ignore, ignore_set, desc=f"{ds} val: scan")
    mol_dirs = [m for v in dd.values() for m in v]
    return sorted(mol_dirs), len(dd)


def reify_test(ds, split_dir, test_pkl_path):
    with open(test_pkl_path, "rb") as f:
        canonical_smiles = pickle.load(f)
    smi_set = set(canonical_smiles)

    mol_dirs = []
    matched_smiles = set()
    gens = os.listdir(split_dir)
    for gen in tqdm(gens, desc=f"{ds} test: scan", unit="gen", leave=False):
        gen_dir = split_dir / gen
        if not gen_dir.is_dir():
            continue
        for mol in os.listdir(gen_dir):
            mol_dir = gen_dir / mol
            if not mol_dir.is_dir():
                continue
            if not (mol_dir / "system.pdb").exists():
                continue
            smi_file = mol_dir / "smiles.txt"
            try:
                with open(smi_file) as f:
                    # First line only; later lines contain OpenMM load/simulation timers.
                    smi = f.readline().strip()
            except FileNotFoundError:
                continue
            if smi in smi_set:
                mol_dirs.append(mol)
                matched_smiles.add(smi)
    return sorted(mol_dirs), len(matched_smiles)


def sha256_of_list(lst):
    h = hashlib.sha256()
    for s in lst:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def git_hash_of_data_loader():
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "log", "-n1", "--format=%H", "--",
             str(DATA_LOADER_PATH.relative_to(REPO_ROOT))],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--md-data-root", type=Path,
                    default=Path(os.environ.get("MD_DATA_ROOT", "")),
                    help="Path to md_data (expects output_trajectories/GEOM-{QM9,DRUGS}/...)")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "data_gen" / "splits",
                    help="Where to write the 6 *_mol_dirs.pkl files + MANIFEST.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute counts and verify assertions but do not write any files")
    args = ap.parse_args()

    if not str(args.md_data_root) or not args.md_data_root.exists():
        sys.exit(f"MD_DATA_ROOT not set or missing: {args.md_data_root!r}")

    ignore_set = parse_ignore_from_data_loader()
    print(f"[info] IGNORE list size: {len(ignore_set)}  (from {DATA_LOADER_PATH.relative_to(REPO_ROOT)})")

    splits_todo = [(ds, split) for ds in ("qm9", "drugs") for split in ("train", "val", "test")]
    results = {}
    outer = tqdm(splits_todo, desc="reifying splits", unit="split")
    for ds, split in outer:
        key = f"{ds}_{split}"
        outer.set_postfix_str(key)
        traj_root = args.md_data_root / "output_trajectories" / f"GEOM-{ds.upper()}" / "4fs_HMR15_5ns_actual"
        split_dir = traj_root / split
        if not split_dir.is_dir():
            sys.exit(f"Missing split dir: {split_dir}")
        if split == "train":
            mol_dirs, n_smiles = reify_train(ds, split_dir, args.md_data_root, ignore_set)
        elif split == "val":
            mol_dirs, n_smiles = reify_val(ds, split_dir, ignore_set)
        else:
            mol_dirs, n_smiles = reify_test(ds, split_dir, TEST_PKLS[ds])

        exp = EXPECTED_COUNTS[key]
        ok = (n_smiles == exp["n_smiles"] and len(mol_dirs) == exp["n_mol_dirs"])
        tag = "OK" if ok else "MISMATCH"
        outer.write(f"[{tag:>8}] {key:>12}: n_smiles={n_smiles:>5}  n_mol_dirs={len(mol_dirs):>5}  "
                    f"(expected {exp['n_smiles']} / {exp['n_mol_dirs']})")
        if not ok:
            sys.exit(f"Count mismatch on {key}; refusing to proceed.")
        results[key] = {"mol_dirs": mol_dirs, "n_smiles": n_smiles}

    if args.dry_run:
        print("[info] --dry-run set; no files written.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "md_data_root": str(args.md_data_root),
        "data_loader_git_hash": git_hash_of_data_loader(),
        "ignore_list_size": len(ignore_set),
        "splits": {},
    }
    for key, info in tqdm(results.items(), desc="writing pkls", unit="pkl", total=len(results)):
        out_path = args.out_dir / f"{key}_mol_dirs.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(info["mol_dirs"], f, protocol=4)
        manifest["splits"][key] = {
            "n_smiles": info["n_smiles"],
            "n_mol_dirs": len(info["mol_dirs"]),
            "sha256_of_sorted_basenames": sha256_of_list(info["mol_dirs"]),
            "pkl_path": str(out_path.relative_to(REPO_ROOT)),
        }
        tqdm.write(f"[wrote] {out_path.relative_to(REPO_ROOT)}  ({len(info['mol_dirs'])} entries)")

    manifest_path = args.out_dir / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"[wrote] {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
