import yaml, pickle
from easydict import EasyDict
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from experiments.train import Model
from experiments.data_load.data_loader import mol_2d, TRANSFORMS
from utils.data_filter import filter_data, get_smiles_dict
from rdkit import Chem
import random
import torch
import lightning as L
import yaml
import os
import glob
import time
import argparse
from easydict import EasyDict
from torch_geometric.data import DataLoader
import torch.distributed as dist

from experiments.diffusion import OurDiffusion
from experiments.data_load.data_loader import get_datasets, get_test_data
from experiments.models import EGTN, BasicES, EGInterpolator, EGInterpolatorSimple, EGStack, Embedding

from utils.visualize_mols import plot_ours

class InferenceModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser = self._init_denoiser()
        self.diffusion = OurDiffusion(**config.diffusion)

    def _init_denoiser(self):
        if self.config.denoiser.type == 'egtn':
            denoiser = EGTN(**self.config.denoiser)
        elif self.config.denoiser.type == 'basic_es':
            denoiser = BasicES(**self.config.denoiser)
        elif self.config.denoiser.type == 'interpolator':
            denoiser = EGInterpolator(**self.config.denoiser)
        elif self.config.denoiser.type == 'interpolator_simple':
            denoiser = EGInterpolatorSimple(**self.config.denoiser)
        elif self.config.denoiser.type == 'stack':
            denoiser = EGStack(**self.config.denoiser)
        else:
            raise NotImplementedError(f"Unknown denoiser type: {self.config.denoiser.type}")
        return denoiser

    def predict_step(self, batch):
        model_kwargs={
            "h":          batch.x,
            "f":          batch.x_features,
            "edge_index": batch.edge_index,
            "edge_attr":  batch.edge_attr,
            "batch":      batch.batch,
        }

        if self.config.dataset.type in ['trajectory', 'both_trajectory'] and ('interpolator' in self.config.denoiser.type or self.config.denoiser.type == 'stack'):
            conditioning = torch.zeros(self.config.dataset.expected_time_dim, dtype=torch.bool)
            if self.config.denoiser.conditioning != 'none':
                conditioning[0] = True
                if self.config.denoiser.conditioning == 'interpolation':
                    conditioning[-1] = True
            model_kwargs['cond_mask'] = conditioning
            model_kwargs['original_frames'] = batch.original_frames

        elif self.config.dataset.type in ['trajectory', 'both_trajectory'] and self.config.denoiser.type == 'egtn':
            conditioning = torch.zeros(self.config.dataset.expected_time_dim, dtype=torch.bool)
            if self.config.denoiser.conditioning != 'none':
                conditioning[0] = True
                if self.config.denoiser.conditioning == 'interpolation':
                    conditioning[-1] = True
                model_kwargs['cond_mask'] = conditioning
                model_kwargs['original_frames'] = batch.original_frames

        # Time the sampling
        t_sample_start = time.time()
        if self.config.denoiser.conditioning == 'none':
            samples = self.diffusion.p_sample_loop(
                model=self.denoiser,
                shape=list(batch.pos.shape),           # [total_nodes,3,1]
                model_kwargs=model_kwargs,
                progress=False
            ).squeeze(-1)  # -> [total_nodes,3]
        elif self.config.denoiser.conditioning == 'forward':
            # Get num_blocks from config (default to 4 if not specified)
            num_blocks = getattr(self.config, 'num_blocks', 4)
            samples = self.diffusion.ar_block_diffusion(
                model=self.denoiser,
                shape=list(batch.pos.shape),           # [total_nodes,3,1]
                model_kwargs=model_kwargs,
                ddim=self.config.diffusion.num_timesteps != 1000,
                ddim_steps=self.config.diffusion.num_timesteps,
                progress=False,
                num_blocks=num_blocks
            ).squeeze(-1)  # -> [total_nodes,3]
        elif self.config.denoiser.conditioning == 'unconditional_forward':
            # Get num_blocks from config (default to 4 if not specified)
            num_blocks = getattr(self.config, 'num_blocks_for_uncond', 4)
            samples = self.diffusion.uncond_block_diffusion(
                model=self.denoiser,
                shape=list(batch.pos.shape),           # [total_nodes,3,1]
                model_kwargs=model_kwargs,
                progress=False,
                num_blocks=num_blocks
            ).squeeze(-1)  # -> [total_nodes,3]
        elif self.config.denoiser.conditioning == 'unconditional_forward_autoreg':
            samples = self.diffusion.uncond_autoreg_diffusion(
                model=self.denoiser,
                shape=list(batch.pos.shape),           # [total_nodes,3,1]
                model_kwargs=model_kwargs,
                progress=False
            ).squeeze(-1)  # -> [total_nodes,3]
        elif self.config.denoiser.conditioning == 'forward_autoreg':
            samples = self.diffusion.autoreg_diffusion(
                model=self.denoiser,
                shape=list(batch.pos.shape),           # [total_nodes,3,1]
                model_kwargs=model_kwargs,
                progress=False
            ).squeeze(-1)  # -> [total_nodes,3]
        elif self.config.denoiser.conditioning == 'interpolation':
            samples = self.diffusion.interpolation_diffusion(
                model=self.denoiser,
                shape=list(batch.pos.shape),           # [total_nodes,3,1]
                model_kwargs=model_kwargs,
                progress=False
            ).squeeze(-1)  # -> [total_nodes,3]
        else:
            raise NotImplementedError()
        t_sample_end = time.time()
        sample_time = t_sample_end - t_sample_start

        if torch.isnan(samples).any():
            num_nans = torch.isnan(samples).sum().item()
            total_elements = samples.numel()
            nan_ratio = num_nans / total_elements
            print(f"WARNING: Nan Generation!!  {nan_ratio}\n")
    
        # Handle batch attributes with fallbacks for compatibility
        smiles = getattr(batch, 'smiles', ['UNKNOWN'] * len(samples))
        rdmol = getattr(batch, 'rdmol', [None] * len(samples))
        conf_idx = getattr(batch, 'conf_idx', [0] * len(samples))
        
        return {
            "smiles": smiles,
            "rdmol":  rdmol,
            "coords": samples,
            "batch":  batch.batch,
            "conf_idx": conf_idx,
            "sample_time": sample_time,  # new field
        }
    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     type=str, required=True,
                   help='path to yaml config (see configs_official/)')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='path to ckpt')
    p.add_argument('--gen_path', type=str,
                   default='test_conformers_multi.pkl',
                   help='path to generated conformers pickle file')
    p.add_argument('--lmbda', type=float,
                   default=1.0,
                   help='How much weight we assign to the traj model alphas')

    '''
    alphas_new = lambda * alpha + (1 - alpha) * 1

    When lambda is 0, we recover alphas_new is 1 and therefore just get the normal conformer model
    When lambda is 1, we recover alphas_new is alpha and there just get the normal trajectory model on one frame
    '''
    return p.parse_args()

def main():
    start = time.time()
    print("START TIME: ", start)
    args = parse_args()
    cfg = EasyDict(yaml.safe_load(open(args.config, 'r')))
    lmbda = args.lmbda
    print(args)
    print(cfg)

    # Override conditioning based on test_on_drugs for both_trajectory
    # DRUGS uses 'forward', QM9 uses 'unconditional_forward'
    # Store num_blocks separately (not in cfg.diffusion to avoid passing to OurDiffusion.__init__)
    num_blocks_for_uncond = None
    if cfg.dataset.type == 'both_trajectory':
        test_on_drugs = getattr(cfg.dataset, 'test_on_drugs', True)
        if test_on_drugs:
            print("Overriding conditioning to 'forward' for DRUGS testing (4 blocks with first-frame conditioning)")
            cfg.denoiser.conditioning = 'forward'
            # DRUGS: 4 blocks for ~250 frames total (251 time steps)
            num_blocks_for_uncond = 4
        else:
            print("Overriding conditioning to 'unconditional_forward' for QM9 testing (2 unconditional blocks)")
            cfg.denoiser.conditioning = 'unconditional_forward'
            # QM9: 2 blocks × 250 frames + 1 ref = 501 frames total (500 time steps)
            num_blocks_for_uncond = 2
    
    # Store num_blocks in a separate variable accessible by InferenceModel
    if num_blocks_for_uncond is not None:
        cfg.num_blocks_for_uncond = num_blocks_for_uncond

    # Check to see if using DDPM or DDIM
    if cfg.diffusion.num_timesteps < 1000:
        print("USING DDIM SAMPLER: Number of Diffusion Steps Less Than 1000")
    else:
        print("USING DDPM at 1000 denoising steps")
    
    # Initialize the Model  
    print("Initializing the Model")
    model = InferenceModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    # Set the alpha logits to the correct values
    if lmbda < 1.0:
        for key in ckpt['state_dict']:
            if "alpha" not in key:
                continue
            curr_log = ckpt['state_dict'][key]
            curr_frac = torch.sigmoid(curr_log)
            new_frac = lmbda * curr_frac + (1 - lmbda) * 1.0
            new_log = torch.special.logit(new_frac)
            print(f"{key}\n  logit: {curr_log}\n  sigmoid: {curr_frac}\n  new_frac: {new_frac}\n  new_logit: {new_log}\n")
            ckpt['state_dict'][key] = new_log

    # Load the modle weights
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print("Getting the Datasets")
    test_dataset = get_test_data(cfg)
    # test_dataset.data = test_dataset.data[:10]
    print("TEST DATASET: ", len(test_dataset))


    # with open('qm9_test_dataset.pkl', 'wb') as f:
    #     pickle.dump(test_dataset, f)

    # Initialize the DataLoader
    print("Initializing the DataLoaders")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=4
    )

    print("Initializing the Trainer")
    trainer = L.Trainer(
        accelerator='cuda',
        # devices=torch.cuda.device_count(),
    )

    print("Evaluating the Model")
    start_of_preds = time.time()
    predictions = trainer.predict(model, dataloaders=test_dataloader)
    prediction_loop_time = time.time() - start_of_preds
    print(f"Total trainer.predict(...) duration: {prediction_loop_time:.3f} seconds")

    # aggregate per-graph coords
    all_results = {}
    sample_times = []

    for out in predictions:
        if 'sample_time' in out:
            sample_times.append(out['sample_time'])

        smiles_list = out['smiles']
        rdmol_list  = out['rdmol']
        batch_idx   = out['batch']
        if 'is_real' in out:
            is_real     = out['is_real']
            batch_idx   = out['batch'][is_real]     # [total_nodes]
        coords      = out['coords']    # [total_nodes,3]
        conf_idx    = out['conf_idx'] 

        num_graphs  = len(smiles_list)
        for i in range(num_graphs):
            smi   = smiles_list[i]
            rdmol = rdmol_list[i]
            conf_idx_i = conf_idx[i]
            mask  = (batch_idx == i)
            coords_i = coords[mask].cpu().numpy()
            rec = all_results.setdefault((smi, conf_idx_i), {'rdmol': None, 'coords': []})
            if rec['rdmol'] is None:
                rec['rdmol'] = rdmol
            rec['coords'].append(coords_i)

    # report averages
    if sample_times:
        avg_sample_time = sum(sample_times) / len(sample_times)
        print(f"Average sampling time per batch: {avg_sample_time:.3f} seconds")

    try:
        # Determine rank
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # Safeguard: store per-rank results before any reduction
        per_rank_results = dict(all_results)

        # Safe-guard: save per-rank pickle
        base, ext = os.path.splitext(args.gen_path)
        per_rank_path = f"{base}_rank{rank}{ext}"
        with open(per_rank_path, 'wb') as f:
            pickle.dump(per_rank_results, f)
        print(f"Rank {rank} writing {per_rank_path}", flush=True)
    except Exception as e:
        pass

    # Wait for all processes to finish
    if dist.is_initialized():        
        # Gather results from all processes
        world_size = dist.get_world_size()
        all_process_results = [None] * world_size
        dist.all_gather_object(all_process_results, all_results)

        # On rank 0, merge all dictionaries
        if dist.get_rank() == 0:
            merged_results = {}
            for process_dict in all_process_results:
                for (smi, conf_idx), rec in process_dict.items():
                    if smi not in merged_results:
                        merged_results[smi] = {'rdmol': rec['rdmol'], 'coords': []}
                    merged_results[smi]['coords'].extend(rec['coords'])
            all_results = merged_results

    if not dist.is_initialized() or dist.get_rank() == 0:
        end = time.time()
        total_time = end - start
        print("END TIME: ", end)
        print("WALL CLOCK TIME: ", total_time)
        all_results['WALL_CLOCK'] = total_time
        with open(args.gen_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Done → {args.gen_path}")


def merge_results(base_path):
    merged = {}
    for fpath in glob.glob(base_path.replace('.pkl', '_rank*.pkl')):
        with open(fpath, 'rb') as f:
            results = pickle.load(f)
        for (smi, conf_idx), rec in results.items():
            if smi not in merged:
                merged[smi] = {'rdmol': rec['rdmol'], 'coords': []}
            merged[smi]['coords'].extend(rec['coords'])
    with open(base_path.replace('.pkl', '_merged.pkl'), 'wb') as f:
        pickle.dump(merged, f)
    print("Merged all ranks into:", f.name)


if __name__ == '__main__':
    main()
    # args = parse_args()
    # merge_results(args.gen_path)







