import torch
import lightning as L
import yaml
import os
import argparse
from easydict import EasyDict
from torch_geometric.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from experiments.diffusion import OurDiffusion
from experiments.data_load.data_loader import get_datasets
from experiments.models import EGTN, BasicES, EGInterpolator, EGInterpolatorSimple, EGStack, Embedding
from utils.visualize_mols import plot_ours
import random


class Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser = self._init_denoiser()
        print(self.denoiser)
        self.diffusion = OurDiffusion(**config.diffusion)
        self.validation_losses_drugs = []
        self.validation_losses_qm9 = []

    def _init_denoiser(self):
        if self.config.denoiser.type == 'egtn':
            return EGTN(**self.config.denoiser)
        elif self.config.denoiser.type == 'basic_es':
            return BasicES(**self.config.denoiser)
        elif self.config.denoiser.type == 'interpolator':
            return EGInterpolator(**self.config.denoiser)
        elif self.config.denoiser.type == 'interpolator_simple':
            return EGInterpolatorSimple(**self.config.denoiser)
        elif self.config.denoiser.type == 'stack':
            return EGStack(**self.config.denoiser)
        else:
            raise NotImplementedError(f"Unknown denoiser type: {self.config.denoiser.type}")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.denoiser.parameters(), 
            lr=self.config.optim.lr, 
            weight_decay=self.config.optim.weight_decay
        )

    def on_train_epoch_start(self):
        self.denoiser.train()

    def _compute_loss(self, batch):
        batch = batch.to(self.device)
        x_start = batch.pos
        model_kwargs = {
            "h": batch.x,
            "f": batch.x_features,
            "edge_index": batch.edge_index,
            "edge_attr": batch.edge_attr,
            "batch": batch.batch,
        }

        if self.config.dataset.type in ['trajectory', 'both_trajectory'] and ('interpolator' in self.config.denoiser.type or self.config.denoiser.type == 'stack'):
            conditioning = torch.zeros(self.config.dataset.expected_time_dim, dtype=torch.bool)
            if self.config.denoiser.conditioning != 'none':
                if self.config.denoiser.conditioning == 'unconditional_forward':
                    if random.choice([True, False]):
                        conditioning[0] = True
                        x_start = x_start[..., 1:]
                else:   
                    conditioning[0] = True
                if self.config.denoiser.conditioning == 'interpolation':
                    conditioning[-1] = True
            model_kwargs['cond_mask'] = conditioning
            model_kwargs['original_frames'] = batch.original_frames

        elif self.config.dataset.type in ['trajectory', 'both_trajectory'] and self.config.denoiser.type == 'egtn':
            conditioning = torch.zeros(self.config.dataset.expected_time_dim, dtype=torch.bool)
            if self.config.denoiser.conditioning != 'none':
                if self.config.denoiser.conditioning == 'unconditional_forward':
                    if random.choice([True, False]):
                        conditioning[0] = True
                        x_start = x_start[..., 1:]
                else:   
                    conditioning[0] = True
                if self.config.denoiser.conditioning == 'interpolation':
                    conditioning[-1] = True
                model_kwargs['cond_mask'] = conditioning
                model_kwargs['original_frames'] = batch.original_frames

        loss = self.diffusion.training_losses(
            model=self.denoiser,
            x_start=x_start,
            t=None,
            model_kwargs=model_kwargs,
        )["loss"]

        return loss

    def training_step(self, batch, batch_idx):
        oom_flag = torch.tensor([0], device=self.device)

        try:
            loss = self._compute_loss(batch).mean()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[OOM] Rank {self.global_rank} had OOM at step {batch_idx}")
                torch.cuda.empty_cache()
                oom_flag[0] = 1
            else:
                raise

            if self.trainer.world_size > 1:
                torch.distributed.all_reduce(oom_flag, op=torch.distributed.ReduceOp.SUM)

            if oom_flag.item() > 0:
                dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                self.log("train_loss", torch.nan, on_step=True, prog_bar=False)
                return dummy_loss

        # Lightning automatically handles gradient accumulation and batch size weighting
        # batch_size parameter ensures proper averaging when accumulate_grad_batches > 1
        num_graphs = batch.batch.max().item() + 1
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                batch_size=num_graphs, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.denoiser.eval()
        if self.config.dataset.type in ['both', 'both_trajectory', 'tetrapeptide_drugs']:
            self.validation_losses_drugs = []
            self.validation_losses_qm9 = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_loss = self._compute_loss(batch).mean()
        num_graphs = batch.batch.max().item() + 1
        
        if self.config.dataset.type in ['both', 'both_trajectory', 'tetrapeptide_drugs']:
            if dataloader_idx == 0:
                # First validation dataset (DRUGS for all mixed types)
                dataset_name = "drugs"
                self.validation_losses_drugs.append((val_loss.item(), num_graphs))
            elif dataloader_idx == 1:
                # Second validation dataset (QM9 for 'both'/'both_trajectory', tetrapeptide for 'tetrapeptide_drugs')
                if self.config.dataset.type in ['both', 'both_trajectory']:
                    dataset_name = "qm9"
                else:  # tetrapeptide_drugs
                    dataset_name = "tetrapeptide"
                self.validation_losses_qm9.append((val_loss.item(), num_graphs))
            else:
                dataset_name = f"unknown_{dataloader_idx}"
                print(f"Warning: Unexpected dataloader_idx={dataloader_idx}")
            
            # Let Lightning handle the averaging with proper batch_size weighting
            self.log(f"val_loss_{dataset_name}", val_loss,
                     on_step=False, on_epoch=True, prog_bar=False,
                     batch_size=num_graphs, sync_dist=True, add_dataloader_idx=False)
        else:
            # For single dataset types, use dataset-specific names for clarity in wandb
            if self.config.dataset.type == 'tetrapeptide':
                # Log both val_loss_tetrapeptide (for separate panel) and val_loss (for checkpoint monitoring)
                self.log("val_loss_tetrapeptide", val_loss,
                         on_step=False, on_epoch=True, prog_bar=True,
                         batch_size=num_graphs, sync_dist=True)
                self.log("val_loss", val_loss,
                         on_step=False, on_epoch=True, prog_bar=False,
                         batch_size=num_graphs, sync_dist=True)
            else:
                self.log("val_loss", val_loss,
                         on_step=False, on_epoch=True, prog_bar=True,
                         batch_size=num_graphs, sync_dist=True)

        if batch_idx < self.config.eval.num_sample_batches and self.config.dataset.type in ['conformer', 'both', 'tetrapeptide', 'tetrapeptide_drugs']:
            model_kwargs = {
                "h": batch.x,
                "f": batch.x_features,
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "batch": batch.batch,
            }

            samples = self.diffusion.p_sample_loop(
                model=self.denoiser,
                shape=list(batch.pos.shape),
                model_kwargs=model_kwargs,
                progress=True,
            )
                        
            if self.trainer.global_rank == 0:
                cur_samples_output_dir = os.path.join(
                    self.config.eval.samples_output_dir,
                    self.config.wandb.name,
                    f"epoch_{self.current_epoch}",
                )
                os.makedirs(cur_samples_output_dir, exist_ok=True)
                image_files = []
                
                for sample_idx in range(batch.batch.max().item() + 1):
                    cur_position = samples[batch.batch == sample_idx]
                    cur_atom_number = batch.x[batch.batch == sample_idx]
                    
                    frame_position = cur_position[..., 0] if len(cur_position.shape) > 2 else cur_position
                    frame_file = os.path.join(
                        cur_samples_output_dir,
                        f"sample_{batch_idx}_{sample_idx}.png"
                    )
                    molecule = (
                        cur_atom_number.detach().cpu(),
                        frame_position.detach().cpu(),
                    )
                    
                    # Determine dataset name for visualization
                    if hasattr(self.config.dataset, 'dataset'):
                        dataset_name = self.config.dataset.dataset.lower()
                    elif hasattr(self.config.dataset, 'datasets') and len(self.config.dataset.datasets) > 0:
                        dataset_name = self.config.dataset.datasets[0].lower()
                    elif self.config.dataset.type == 'tetrapeptide':
                        dataset_name = 'tetrapeptide'
                    elif self.config.dataset.type == 'tetrapeptide_drugs':
                        dataset_name = 'drugs'  # Use drugs visualization for mixed datasets
                    else:
                        dataset_name = "drugs"  # Default to drugs (GEOM) visualization
                    
                    plot_ours(
                        molecule=molecule,
                        output_path=frame_file,
                        dataset_name=dataset_name,
                        remove_h=False,
                    )
                    image_files.append(frame_file)
                
                if image_files:
                    print(f'Logging images to wandb on rank {self.trainer.global_rank}')
                    self.logger.log_image(
                        key="validation_samples",
                        images=image_files,
                        caption=[f'Sample {i}' for i in range(len(image_files))],
                    )

    def on_validation_epoch_end(self):
        if self.config.dataset.type in ['both', 'both_trajectory', 'tetrapeptide_drugs']:
            # Compute weighted average of validation losses from both datasets
            # using the collected (loss, batch_size) tuples for proper weighting
            if hasattr(self, 'validation_losses_drugs') and hasattr(self, 'validation_losses_qm9'):
                has_drugs = len(self.validation_losses_drugs) > 0
                has_qm9 = len(self.validation_losses_qm9) > 0
                
                if has_drugs and has_qm9:
                    # Compute batch-size weighted average for each dataset
                    total_loss_drugs = sum(loss * bs for loss, bs in self.validation_losses_drugs)
                    total_bs_drugs = sum(bs for _, bs in self.validation_losses_drugs)
                    mean_val_loss_drugs = total_loss_drugs / total_bs_drugs if total_bs_drugs > 0 else 0
                    
                    total_loss_qm9 = sum(loss * bs for loss, bs in self.validation_losses_qm9)
                    total_bs_qm9 = sum(bs for _, bs in self.validation_losses_qm9)
                    mean_val_loss_qm9 = total_loss_qm9 / total_bs_qm9 if total_bs_qm9 > 0 else 0
                    
                    # Apply mixing ratio weights
                    mix_ratio = self.config.dataset.mixing.ratio
                    weight_drugs = mix_ratio.get('DRUGS', 0.4)
                    # For 'both' and 'both_trajectory' types, second dataset is QM9; for 'tetrapeptide_drugs', it's TETRAPEPTIDE
                    if self.config.dataset.type in ['both', 'both_trajectory']:
                        weight_qm9 = mix_ratio.get('QM9', 0.6)
                    else:  # tetrapeptide_drugs
                        weight_qm9 = mix_ratio.get('TETRAPEPTIDE', 0.7)
                    
                    aggregated_val_loss = weight_drugs * mean_val_loss_drugs + weight_qm9 * mean_val_loss_qm9
                    self.log('val_loss', aggregated_val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                elif has_drugs:
                    total_loss_drugs = sum(loss * bs for loss, bs in self.validation_losses_drugs)
                    total_bs_drugs = sum(bs for _, bs in self.validation_losses_drugs)
                    mean_val_loss_drugs = total_loss_drugs / total_bs_drugs if total_bs_drugs > 0 else 0
                    self.log('val_loss', mean_val_loss_drugs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                elif has_qm9:
                    total_loss_qm9 = sum(loss * bs for loss, bs in self.validation_losses_qm9)
                    total_bs_qm9 = sum(bs for _, bs in self.validation_losses_qm9)
                    mean_val_loss_qm9 = total_loss_qm9 / total_bs_qm9 if total_bs_qm9 > 0 else 0
                    self.log('val_loss', mean_val_loss_qm9, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config (see configs_official/)')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    config = EasyDict(config)
    print(config)
    
    # Validate configuration for both_trajectory
    if config.dataset.type == 'both_trajectory':
        valid_conditioning = ['forward', 'unconditional_forward']
        if config.denoiser.conditioning not in valid_conditioning:
            raise ValueError(
                f"Invalid conditioning '{config.denoiser.conditioning}' for both_trajectory. "
                f"Only {valid_conditioning} are supported (no interpolation due to different time dimensions)."
            )
    
    logger = L.pytorch.loggers.WandbLogger(**config.wandb)
    
    model = Model(config)

    if hasattr(config.denoiser, 'pretrain_ckpt') and config.denoiser.pretrain_ckpt:
        print(f"Loading pretrained checkpoint from {config.denoiser.pretrain_ckpt}")
        ckpt = torch.load(config.denoiser.pretrain_ckpt)
        ckpt_state_dict = ckpt["state_dict"]

        original_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        incompat = model.load_state_dict(ckpt_state_dict, strict=False)
        print("Missing keys:   ", incompat.missing_keys)
        print("Unexpected keys:", incompat.unexpected_keys)

        updated_layers = []
        for name, param in model.named_parameters():
            if name in ckpt_state_dict:
                if not torch.allclose(param.detach(), original_params[name], atol=1e-6):
                    updated_layers.append(name)

        if updated_layers:
            print("\n✅ The following layers were updated from checkpoint:")
            for name in updated_layers:
                print(f"  • {name}")
        else:
            print("\n⚠️  No parameters changed (check key names or values).")

        if hasattr(config.denoiser, 'freeze_spatial') and config.denoiser.freeze_spatial:
            frozen_layers = []
            for name, param in model.named_parameters():
                if name in ckpt_state_dict:
                    param.requires_grad = False
                    frozen_layers.append(name)
            print("\n🧊 The following layers have been frozen from gradient updates:")
            for name in frozen_layers:
                print(f"  • {name}")

    datasets = get_datasets(config)
    
    if len(datasets) == 3:
        train_dataset, val_dataset_1, val_dataset_2 = datasets
        print(f"Dataset sizes - Train: {len(train_dataset)}, Val Dataset 1: {len(val_dataset_1)}, Val Dataset 2: {len(val_dataset_2)}")
        is_both_type = True
    else:
        train_dataset, val_dataset = datasets
        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        is_both_type = False

    if hasattr(train_dataset, 'has_mixing') and train_dataset.has_mixing:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            sampler=train_sampler,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
        )
    
    if is_both_type:
        val_dataloader_1 = DataLoader(
            val_dataset_1,
            batch_size=config.eval.batch_size,
            shuffle=False,
        )
        val_dataloader_2 = DataLoader(
            val_dataset_2,
            batch_size=config.eval.batch_size,
            shuffle=False,
        )
        val_dataloaders = [val_dataloader_1, val_dataloader_2]
    else:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.eval.batch_size,
            shuffle=False,
        )
        val_dataloaders = val_dataloader

    checkpoint_dir = f'checkpoints/{config.wandb.project}/{config.wandb.name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        filename='{epoch}-{step}',
        save_top_k=-1,
        every_n_epochs=4,
        save_last=False,
        dirpath=checkpoint_dir,
        monitor='val_loss',
        save_on_train_epoch_end=False,
    )

    trainer = L.Trainer(
        accumulate_grad_batches=config.train.accum_grad,
        max_epochs=config.train.max_epochs,
        accelerator='cuda',
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloader,
        val_dataloaders,
        ckpt_path=config.train.resume_from_checkpoint if config.train.resume_from_checkpoint else None
    )


if __name__ == '__main__':
    main()