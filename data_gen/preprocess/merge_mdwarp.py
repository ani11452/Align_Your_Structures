"""
Produce the combined "mdwarp" (MDGen + TimeWarp) tetrapeptide conformer split
consumed by configs_official/conformer/tetrapeptide/
mdwarp_tetrapeptide_pretrain_noH_1000_kabsch_conf_basic_es_order_3.yaml.

Inputs (all are outputs of the upstream sampler scripts in this folder):
  ${MD_DATA_ROOT}/tetrapeptide_conformers_mdgen.pkl     <- sample_conformer_mdgen.py
  ${MD_DATA_ROOT}/tetrapeptide_conformers_large.pkl     <- sample_conformer_AA.py,
                                                            run with splits =
                                                            ['train','val','test']
                                                            and OUTPUT_PICKLE_FILE =
                                                            'tetrapeptide_conformers_large.pkl'
  ${MD_DATA_ROOT}/tetrapeptide_conformers_val.pkl       <- sample_conformer_AA.py,
                                                            default run (splits =
                                                            ['val'])

Plus the MDGen split CSVs at:
  ${MDGEN_DATA_ROOT}/splits/4AA_{train,val,test}.csv

Outputs (per-split pickles the mdwarp config reads):
  ${MD_DATA_ROOT}/tetrapeptide_conformers_final_train.pkl
  ${MD_DATA_ROOT}/tetrapeptide_conformers_final_val.pkl

Merge semantics (with explicit leak prevention):
  1. MDGen conformer entries are partitioned by MDGen's own train/val/test CSVs.
     MDGen test peptides are HELD OUT entirely.
  2. TimeWarp conformer entries use their split prefix in the key "<split>/<peptide>":
       train/val -> final_train,  test -> final_val.
  3. Any TimeWarp peptide that also appears in ANY MDGen split is SKIPPED
     (so an MDGen test peptide never sneaks in via TimeWarp).
  4. Post-merge assertions verify:
       (a) no peptide appears in both final_train and final_val
       (b) no MDGen test peptide appears anywhere in train/val

Usage:
  bash data_gen/preprocess/merge_mdwarp.sh       # if you add a runner wrapper
  OR
  python -m data_gen.preprocess.merge_mdwarp
"""

import os
import pickle

import pandas as pd


def _abort(msg: str) -> None:
    raise RuntimeError(msg)


def load_mdgen_splits(mdgen_root: str):
    """Return (train_set, val_set, test_set) of 4-letter MDGen peptide names."""
    splits_dir = os.path.join(mdgen_root, 'splits')
    train_df = pd.read_csv(os.path.join(splits_dir, '4AA_train.csv'))
    val_df   = pd.read_csv(os.path.join(splits_dir, '4AA_val.csv'))
    test_df  = pd.read_csv(os.path.join(splits_dir, '4AA_test.csv'))
    return (
        set(train_df['name']),
        set(val_df['name']),
        set(test_df['name']),
    )


def _load_pkl(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _add_timewarp_source(
    source: dict,
    source_label: str,
    mdgen_all: set,
    final_train_confs: dict,
    final_val_confs: dict,
) -> tuple[int, int]:
    """
    Merge a TimeWarp-source conformer dict into the running final splits.

    TimeWarp keys are expected to be "<split>/<peptide>" where split is one of
    {train, val, test}.  train/val entries land in final_train; test entries
    land in final_val.  Any peptide already present in any MDGen split is
    skipped (no leakage back from TimeWarp).  Entries already present in the
    target final dict are left untouched (first source wins).
    """
    added_train = added_val = 0
    for key, value in source.items():
        try:
            split_prefix, peptide = key.split('/', 1)
        except ValueError:
            _abort(f"[{source_label}] malformed key: {key!r}")

        if peptide in mdgen_all:
            continue

        if split_prefix in ('train', 'val'):
            if peptide not in final_train_confs:
                final_train_confs[peptide] = value
                added_train += 1
        elif split_prefix == 'test':
            if peptide not in final_val_confs:
                final_val_confs[peptide] = value
                added_val += 1
    return added_train, added_val


def main() -> None:
    md_data_root  = os.environ.get('MD_DATA_ROOT', '')
    mdgen_root    = os.environ.get('MDGEN_DATA_ROOT', '')
    if not md_data_root:
        _abort('MD_DATA_ROOT is not set')
    if not mdgen_root:
        _abort('MDGEN_DATA_ROOT is not set')

    mdgen_train, mdgen_val, mdgen_test = load_mdgen_splits(mdgen_root)
    mdgen_all = mdgen_train | mdgen_val | mdgen_test
    print(
        f'MDGen splits — train: {len(mdgen_train)}, '
        f'val: {len(mdgen_val)}, test: {len(mdgen_test)}  '
        f'(test will be held out)'
    )

    # 1. MDGen conformer entries -> partition by MDGen split CSV (test held out).
    mdgen_confs = _load_pkl(os.path.join(md_data_root, 'tetrapeptide_conformers_mdgen.pkl'))
    final_train_confs: dict = {}
    final_val_confs: dict = {}
    for peptide, payload in mdgen_confs.items():
        if peptide in mdgen_train:
            final_train_confs[peptide] = payload
        elif peptide in mdgen_val:
            final_val_confs[peptide] = payload
        # mdgen_test peptides are intentionally dropped
    print(f'After MDGen — train: {len(final_train_confs)}, val: {len(final_val_confs)}')

    # 2. TimeWarp (_large) -> first-wins merge, skipping any MDGen-split overlap.
    tw_large = _load_pkl(os.path.join(md_data_root, 'tetrapeptide_conformers_large.pkl'))
    tr, va = _add_timewarp_source(tw_large, 'timewarp_large', mdgen_all, final_train_confs, final_val_confs)
    print(
        f'After TimeWarp (_large) — added {tr} to train, {va} to val — '
        f'totals: train={len(final_train_confs)}, val={len(final_val_confs)}'
    )

    # 3. TimeWarp (_val) -> same policy; fills any gaps the _large source missed.
    tw_val = _load_pkl(os.path.join(md_data_root, 'tetrapeptide_conformers_val.pkl'))
    tr, va = _add_timewarp_source(tw_val, 'timewarp_val', mdgen_all, final_train_confs, final_val_confs)
    print(
        f'After TimeWarp (_val) — added {tr} to train, {va} to val — '
        f'totals: train={len(final_train_confs)}, val={len(final_val_confs)}'
    )

    # 4. Leak checks.
    train_peptides = set(final_train_confs.keys())
    val_peptides   = set(final_val_confs.keys())
    assert not (train_peptides & val_peptides), 'peptide appears in both train and val'
    assert not (train_peptides & mdgen_test), 'MDGen test peptide leaked into train'
    assert not (val_peptides   & mdgen_test), 'MDGen test peptide leaked into val'
    print('\n✓ leak checks passed')

    # 5. Write outputs.
    train_out = os.path.join(md_data_root, 'tetrapeptide_conformers_final_train.pkl')
    val_out   = os.path.join(md_data_root, 'tetrapeptide_conformers_final_val.pkl')
    with open(train_out, 'wb') as f:
        pickle.dump(final_train_confs, f)
    with open(val_out, 'wb') as f:
        pickle.dump(final_val_confs, f)
    print(f'Wrote {len(final_train_confs)} peptides -> {train_out}')
    print(f'Wrote {len(final_val_confs)} peptides   -> {val_out}')
    print(f'Held out MDGen test ({len(mdgen_test)} peptides) for evaluation')


if __name__ == '__main__':
    main()
