# eval/pipelines/

```
eval/pipelines/
├── utils.py              (shared helpers, imported by scripts in subdirs)
├── smiles/               (SMILES test-set text files)
├── trajectory/           (main QM9 + DRUGS trajectory eval + plot variants)
├── conformer/            (conformer model eval)
├── mdbaseline/           (MD baseline comparisons)
├── interpolation/        (interpolation task eval + inference)
├── tetrapeptide/         (tetrapeptide / timewarp / mdgen eval)
└── energy/               (energy-based eval)
```

Scripts below are runnable from the repo root. Each script in a subdir
pre-pends `eval/pipelines/` onto `sys.path` so that `from utils import *`
resolves. SMILES txt paths default to the `smiles/` subdir via
`_PIPELINE_DIR`.

## 1. Trajectory eval

### 1.1 QM9
- `trajectory/eval_traj_pipeline.py` — run eval and save `.pkl` result:
  ```
  python eval/pipelines/trajectory/eval_traj_pipeline.py --no_msm --plot \
      --num_workers 32 --save --save_name 20250509_2344 \
      --gener_dir model_outputs_official/qm9_trajectory_official/qm9_noH_1000_kabsch_traj_interpolator_pretrain_final/epoch=199-step=138400-gen.pkl
  ```
  Produces `20250509_2344_evalout.pkl`.
- `trajectory/read_out_pkl_traj.ipynb` — load an `_evalout.pkl` and emit a CSV.
- MSM analysis takes ~1.5 hr; pass `--no_msm` for only bond-angle / length /
  torsion / TICA metrics.
- QM9 test-set SMILES default to `smiles/eval_qm9_unconditional_generation_smiles.txt`.
  Two SMILES are dropped from the test set (`OC[C@H]1CO[C@@H](CO)O1`,
  `CCC#CCC#CC=O`). To run on a subset, put SMILES in
  `smiles/small_batch_test_smiles.txt` and pass
  `--gener_smiles smiles/small_batch_test_smiles.txt`.

### 1.2 DRUGS
```
python eval/pipelines/trajectory/eval_traj_pipeline_drugs.py --no_msm \
    --gener_dir model_outputs_official/drugs_trajectory_official/drugs_noH_1000_kabsch_traj_interpolator_naive_fs_25/epoch=199-step=35600-gen.pkl \
    --num_workers 2
```

## 2. Conformer eval

- `conformer/eval_conformer_model.py` — conformer eval:
  ```
  python eval/pipelines/conformer/eval_conformer_model.py --use_drug --plot --num_workers 32
  ```
  Edit `pkl_file_path` / `drug_pkl_file_path` in the script to point at the
  evaluated generation pkl; edit the save dir at the end of the file.
- `conformer/read_out_pkl_conformer.ipynb` — load `.pkl` eval output and emit CSV.
