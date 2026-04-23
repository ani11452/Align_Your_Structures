import os, sys
_PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PIPELINE_DIR)

import mdtraj as md
import types

md.version = types.SimpleNamespace(version=md.__version__)

import argparse
import os
import pickle
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm
import pyemma
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import acovf
from rdkit import Chem

from utils import (
    get_bond_angles_in_gen,
    get_bond_angles_in_traj,
    get_bond_lengths_in_conformer_ref,
    get_bond_lengths_in_gen,
    get_bond_lengths_in_traj,
    get_metastate_prob,
    get_torsion_index_noH,
    get_torsions_in_gen,
    get_torsions_idx_mol,
)


##############################
# Helper functions
##############################
def histogram_metrics(ref: np.ndarray, gen: np.ndarray, *, bins: int, value_range: Tuple[float, float]):
    ref_hist, _ = np.histogram(ref, bins=bins, range=value_range, density=True)
    gen_hist, _ = np.histogram(gen, bins=bins, range=value_range, density=True)
    if np.sum(ref_hist) == 0 or np.sum(gen_hist) == 0:
        return float("nan"), float("nan")
    jsd = float(jensenshannon(ref_hist, gen_hist))
    w1 = float(wasserstein_distance(ref, gen))
    return jsd, w1


def pairwise_hist(ref: np.ndarray, gen: np.ndarray, *, bins: int, value_range: Tuple[float, float]):
    n_feat = min(ref.shape[1], gen.shape[1])
    jsds, w1s = [], []
    for i in range(n_feat):
        jsd, w1 = histogram_metrics(ref[:, i], gen[:, i], bins=bins, value_range=value_range)
        if not np.isnan(jsd):
            jsds.append(jsd)
        if not np.isnan(w1):
            w1s.append(w1)
    return {
        "mean_jsd": float(np.mean(jsds)) if jsds else float("nan"),
        "mean_w1": float(np.mean(w1s)) if w1s else float("nan"),
    }


def compute_decorrelation(trajs: List[np.ndarray], time_step: float, max_lag: int = 498):
    # trajs: list of (T, N) torsion arrays
    per_feat = {}
    if not trajs:
        return per_feat
    n_feat = trajs[0].shape[1]
    for i in range(n_feat):
        raw, shuffled = [], []
        for traj in trajs:
            torsion = traj[:, i]
            autocorr = acovf(np.sin(torsion), demean=False, adjusted=True, nlag=max_lag) + acovf(
                np.cos(torsion), demean=False, adjusted=True, nlag=max_lag
            )
            baseline = np.sin(torsion).mean() ** 2 + np.cos(torsion).mean() ** 2
            if 1 - baseline < 1e-8:
                raw.append(-1)
            else:
                autocorr = (autocorr - baseline) / (1 - baseline)
                idx = np.argmax(autocorr <= 1 / np.e)
                raw.append(float(idx * time_step) if autocorr[idx] <= 1 / np.e else -1)

            shuffled_torsion = np.random.permutation(torsion)
            autocorr = acovf(np.sin(shuffled_torsion), demean=False, adjusted=True, nlag=max_lag) + acovf(
                np.cos(shuffled_torsion), demean=False, adjusted=True, nlag=max_lag
            )
            baseline = np.sin(shuffled_torsion).mean() ** 2 + np.cos(shuffled_torsion).mean() ** 2
            if 1 - baseline < 1e-8:
                shuffled.append(-1)
            else:
                autocorr = (autocorr - baseline) / (1 - baseline)
                idx = np.argmax(autocorr <= 1 / np.e)
                shuffled.append(float(idx * time_step) if autocorr[idx] <= 1 / np.e else -1)
        per_feat[f"torsion_{i}"] = {"raw": raw, "shuffled": shuffled}
    return per_feat


def load_npz_traj(npz_path: Path, pdb_path: Path):
    data = np.load(npz_path)
    positions = data["positions"]
    traj = md.Trajectory(positions, md.load_pdb(str(pdb_path)).topology)
    time = data["time"]
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.4
    return traj, dt


def compute_backbone_sidechain(traj: md.Trajectory):
    backbone = {}
    sidechain = {}
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)
    if phi.size:
        backbone["phi"] = phi
    if psi.size:
        backbone["psi"] = psi
    for name, fn in [("chi1", md.compute_chi1), ("chi2", md.compute_chi2), ("chi3", md.compute_chi3), ("chi4", md.compute_chi4)]:
        try:
            _, arr = fn(traj)
        except ValueError:
            arr = np.empty((traj.n_frames, 0))
        if arr.size:
            sidechain[name] = arr
    return backbone, sidechain


def dihedral_summary(ref: Dict[str, np.ndarray], gen: Dict[str, np.ndarray], bins: int = 100):
    out = {}
    for key in sorted(set(ref) & set(gen)):
        ref_flat = ref[key].ravel()
        gen_flat = gen[key].ravel()
        jsd, w1 = histogram_metrics(ref_flat, gen_flat, bins=bins, value_range=(-np.pi, np.pi))
        out[key] = {"jsd": jsd, "w1": w1}
    return out


##############################
# Main evaluation per peptide
##############################
def main(peptide_key: str):
    out = {}
    split, name = peptide_key.split("/", 1) if "/" in peptide_key else ("", peptide_key)
    npz_path = Path(args.traj_ref_root) / split / f"{name}-traj-arrays.npz"
    pdb_path = Path(args.traj_ref_root) / split / f"{name}-traj-state0.pdb"

    ref_traj, time_step = load_npz_traj(npz_path, pdb_path)
    topology = ref_traj.topology

    ref_conf = ref_conformers.get(peptide_key)
    gen_conf = gen_conformers.get(peptide_key)
    if ref_conf is None or gen_conf is None:
        return peptide_key, out

    # Molecule with Hs if available in conformer payload; otherwise from PDB
    mol = ref_conf.get("rdkit_mol") or gen_conf.get("rdkit_mol")
    if mol is None:
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    torsion_index = np.array(get_torsions_idx_mol(mol)[0] + get_torsions_idx_mol(mol)[2]).reshape(-1, 4)
    torsion_index_noH = get_torsion_index_noH(torsion_index, mol)

    ##############################
    # Conformer-stage metrics
    ##############################
    ref_coords = []
    gen_coords = []
    for conf in ref_conf.get("conformers", []):
        ref_coords.append(conf["coordinates"] * (0.1 if args.conformer_coord_angstrom else 1.0))
    for conf in gen_conf.get("conformers", []):
        gen_coords.append(conf["coordinates"] * (0.1 if args.conformer_coord_angstrom else 1.0))
    if ref_coords and gen_coords:
        ref_coords = np.stack(ref_coords)
        gen_coords = np.stack(gen_coords)
        ref_tensor = torch.from_numpy(ref_coords).float()
        gen_tensor = torch.from_numpy(gen_coords).float()
        torsion_ref = get_torsions_in_gen(ref_tensor, torsion_index)
        torsion_gen = get_torsions_in_gen(gen_tensor, torsion_index_noH)
        bond_len_ref = get_bond_lengths_in_conformer_ref(ref_tensor, mol)
        bond_len_gen = get_bond_lengths_in_gen(gen_tensor, mol)
        bond_ang_ref = get_bond_angles_in_gen(ref_tensor, mol)
        bond_ang_gen = get_bond_angles_in_gen(gen_tensor, mol)
        out["conformer"] = {
            "torsion": pairwise_hist(torsion_ref, torsion_gen, bins=100, value_range=(-np.pi, np.pi)),
            "bond_length": pairwise_hist(bond_len_ref, bond_len_gen, bins=80, value_range=(0.05, 0.3)),
            "bond_angle": pairwise_hist(bond_ang_ref, bond_ang_gen, bins=80, value_range=(0.0, np.pi)),
        }
        # backbone/sidechain distributions for conformers
        ref_traj_conf = md.Trajectory(ref_coords, topology)
        gen_traj_conf = md.Trajectory(gen_coords, topology)
        ref_bb, ref_sc = compute_backbone_sidechain(ref_traj_conf)
        gen_bb, gen_sc = compute_backbone_sidechain(gen_traj_conf)
        out["conformer"]["backbone"] = dihedral_summary(ref_bb, gen_bb, bins=args.backbone_bins)
        out["conformer"]["sidechain"] = dihedral_summary(ref_sc, gen_sc, bins=args.backbone_bins)

    ##############################
    # Trajectory-stage metrics
    ##############################
    gen_payload = traj_gen.get(peptide_key)
    if gen_payload is None or "coords" not in gen_payload:
        return peptide_key, out

    gen_trajs = []
    coords_field = gen_payload["coords"]
    if isinstance(coords_field, dict) and "coords" in coords_field:
        coords_field = coords_field["coords"]
    if isinstance(coords_field, np.ndarray) and coords_field.ndim == 4:
        traj_list = [coords_field[i] for i in range(coords_field.shape[0])]
    else:
        traj_list = coords_field
    for arr in traj_list:
        if arr.shape[2] == 3:
            frames_atoms = arr
        elif arr.shape[0] == 3:
            frames_atoms = np.transpose(arr, (1, 2, 0))
        else:
            frames_atoms = np.transpose(arr, (2, 0, 1))
        gen_trajs.append(frames_atoms * (0.1 if args.traj_coord_angstrom else 1.0))

    # Convert to tensors
    gen_tensors = [torch.from_numpy(traj).permute(0, 2, 1).contiguous().permute(0, 2, 1) for traj in gen_trajs]  # ensure (T,N,3)
    gen_tensors = [t.float() for t in gen_tensors]
    torsion_ref_traj = get_torsions_in_gen(torch.from_numpy(ref_traj.xyz).float(), torsion_index)
    torsion_gen_traj = [get_torsions_in_gen(t, torsion_index_noH) for t in gen_tensors]
    bond_len_ref_traj = get_bond_lengths_in_traj(ref_traj, mol)
    bond_ang_ref_traj = get_bond_angles_in_traj(ref_traj, mol)
    bond_len_gen_traj = np.concatenate([get_bond_lengths_in_gen(t, mol) for t in gen_tensors], axis=0)
    bond_ang_gen_traj = np.concatenate([get_bond_angles_in_gen(t, mol) for t in gen_tensors], axis=0)

    out["trajectory"] = {
        "torsion": pairwise_hist(torsion_ref_traj, np.concatenate(torsion_gen_traj, axis=0), bins=100, value_range=(-np.pi, np.pi)),
        "bond_length": pairwise_hist(bond_len_ref_traj, bond_len_gen_traj, bins=100, value_range=(0.05, 0.3)),
        "bond_angle": pairwise_hist(bond_ang_ref_traj, bond_ang_gen_traj, bins=100, value_range=(0.0, np.pi)),
    }

    # Backbone / sidechain distributions and decorrelation
    ref_bb, ref_sc = compute_backbone_sidechain(ref_traj)
    gen_bb_all = {}
    gen_sc_all = {}
    backbone_trajs = []
    sidechain_trajs = []
    for traj in gen_trajs:
        mdt = md.Trajectory(traj, topology)
        bb, sc = compute_backbone_sidechain(mdt)
        for k, v in bb.items():
            gen_bb_all.setdefault(k, []).append(v)
        for k, v in sc.items():
            gen_sc_all.setdefault(k, []).append(v)
    for k, arrs in gen_bb_all.items():
        gen_bb_all[k] = np.concatenate(arrs, axis=0)
        backbone_trajs.extend(arrs)
    for k, arrs in gen_sc_all.items():
        gen_sc_all[k] = np.concatenate(arrs, axis=0)
        sidechain_trajs.extend(arrs)

    out["trajectory"]["backbone"] = dihedral_summary(ref_bb, gen_bb_all, bins=args.backbone_bins)
    out["trajectory"]["sidechain"] = dihedral_summary(ref_sc, gen_sc_all, bins=args.backbone_bins)
    out["trajectory"]["decorrelation_backbone"] = compute_decorrelation(backbone_trajs, time_step, max_lag=args.decorrelation_max_lag)
    out["trajectory"]["decorrelation_sidechain"] = compute_decorrelation(sidechain_trajs, time_step, max_lag=args.decorrelation_max_lag)

    # MSM on combined backbone+sidechain cosine/sine features
    try:
        ref_blocks = []
        for arr in ref_bb.values():
            ref_blocks.append(np.concatenate([np.cos(arr), np.sin(arr)], axis=1))
        for arr in ref_sc.values():
            ref_blocks.append(np.concatenate([np.cos(arr), np.sin(arr)], axis=1))
        if ref_blocks:
            kmeans = pyemma.coordinates.cluster_kmeans(ref_blocks, k=args.msm_k, max_iter=50)
            gen_blocks = []
            for arr in backbone_trajs + sidechain_trajs:
                gen_blocks.append(np.concatenate([np.cos(arr), np.sin(arr)], axis=1))
            ref_assign = kmeans.dtrajs
            gen_assign = kmeans.transform(gen_blocks)
            msm = pyemma.msm.estimate_markov_model(ref_assign, lag=args.msm_lag)
            msm.pcca(args.msm_states)
            ref_p, gen_p = get_metastate_prob(msm, gen_assign, args.msm_k, nstates=args.msm_states)
            out["trajectory"]["msm"] = {
                "ref_metastable_p": ref_p,
                "gen_metastable_p": gen_p,
                "jsd": float(jensenshannon(ref_p, gen_p)),
                "w1": float(wasserstein_distance(np.arange(len(ref_p)), np.arange(len(gen_p)), ref_p, gen_p)),
            }
    except Exception as e:  # pylint: disable=broad-except
        out["trajectory"]["msm_error"] = str(e)

    return peptide_key, out


##############################
# Argument parsing and driver
##############################
parser = argparse.ArgumentParser(description="Evaluate 4AA peptide conformer + trajectory stages.")
parser.add_argument("--conformer_ref", type=str, required=True)
parser.add_argument("--conformer_gen", type=str, required=True)
parser.add_argument("--traj_ref_root", type=str, required=True)
parser.add_argument("--traj_gen", type=str, required=True)
parser.add_argument("--keys_file", type=str, default=None)
parser.add_argument("--conformer_coord_angstrom", action="store_true")
parser.add_argument("--traj_coord_angstrom", action="store_true")
parser.add_argument("--backbone_bins", type=int, default=100)
parser.add_argument("--decorrelation_max_lag", type=int, default=498)
parser.add_argument("--msm_k", type=int, default=100)
parser.add_argument("--msm_states", type=int, default=10)
parser.add_argument("--msm_lag", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--save", action="store_true")
parser.add_argument("--save_name", type=str, default="eval_4aa_results.pkl")
args = parser.parse_args()

with open(args.conformer_ref, "rb") as f:
    ref_conformers = pickle.load(f)
with open(args.conformer_gen, "rb") as f:
    gen_conformers = pickle.load(f)
with open(args.traj_gen, "rb") as f:
    traj_gen = pickle.load(f)

if args.keys_file:
    with open(args.keys_file, "r") as f:
        all_keys = [line.strip() for line in f if line.strip()]
else:
    all_keys = sorted(set(ref_conformers.keys()) & set(gen_conformers.keys()) & set(traj_gen.keys()))

if args.num_workers > 1:
    p = Pool(args.num_workers)
    p.__enter__()
    __map__ = p.imap
else:
    __map__ = map
out = dict(tqdm.tqdm(__map__(main, all_keys), total=len(all_keys)))
if args.num_workers > 1:
    p.__exit__(None, None, None)

if args.save:
    with open(args.save_name, "wb") as f:
        pickle.dump({"per_peptide": out, "summary": {}}, f)