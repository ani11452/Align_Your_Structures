from .models import *
from .train import *
import numpy as np

# Initalize the conformer model with the respective checkpoint
conf_config = "configs_official/conformer/qm9/qm9_noH_1000_kabsch_conf_basic_es_order_3.yaml"
config = yaml.safe_load(open(conf_config, 'r'))
config = EasyDict(config)
model = Model(config)
ckpt = torch.load(
    "checkpoints/qm9_conformer_official_fixed/qm9_noH_1000_kabsch_conf_basic_es_order_3/epoch=99-step=147400.ckpt"
)
model.load_state_dict(ckpt["state_dict"])
conformer = model.denoiser

# Initialize the trajectoru model with respective checkpoint
traj_config = "configs_official/trajectory/qm9/qm9_noH_1000_kabsch_traj_interpolator_pretrain_final.yaml"
config = yaml.safe_load(open(traj_config, 'r'))
config = EasyDict(config)
model = Model(config)
ckpt = torch.load(
    "checkpoints/qm9_trajectory_official/qm9_noH_1000_kabsch_traj_interpolator_pretrain_final/epoch=199-step=138400.ckpt"
)
model.load_state_dict(ckpt["state_dict"])
trajectory = model.denoiser

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
BN = 5  # Number of nodes
B = 2   # Number of graphs
Hh = 128
He = 4
H = 30
F = 42
T = 10  # Number of time steps

# Graph structure
batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long).to(device)
row = torch.tensor([0, 0, 1, 3], dtype=torch.long).to(device)
col = torch.tensor([1, 2, 2, 4], dtype=torch.long).to(device)
edge_index = torch.stack((row, col), dim=0).to(device)  # [2, BM]
BM = edge_index.size(-1)

# Node & edge features
f = torch.randint(0, 10, size=(BN, F)).float().to(device)  # node features [BN, F]
h = torch.randint(0, 10, size=(BN,)).to(device)             # e.g., categorical node features
x_conf = torch.rand(BN, 3, 1).to(device)                    # conformer input [BN, 3, 1]
x_traj = x_conf.expand(-1, -1, T).to(device)                # trajectory input [BN, 3, T]
edge_attr = torch.randint(0, 30, size=(BM,)).to(device)     # edge types [BM]
t = torch.randint(0, 1000, size=(B,), dtype=torch.long).to(device)[batch]  # time per node

# Move models to device
conformer = conformer.to(device)
trajectory = trajectory.to(device)

# Print input shapes
print(f"batch shape: {batch.shape}")
print(f"row shape: {row.shape}")
print(f"col shape: {col.shape}")
print(f"edge_index shape: {edge_index.shape}")
print(f"h shape: {h.shape}")
print(f"x_conf shape: {x_conf.shape}")
print(f"x_traj shape: {x_traj.shape}")
print(f"f shape: {f.shape}")
print(f"edge_attr shape: {edge_attr.shape}")
print(f"t shape: {t.shape}")

# Run through the models
trajectory.set_zero = False
conf_x_out, conf_h_out = conformer(t, x_conf, h, f, edge_index, edge_attr, batch)

conditioning = torch.zeros(10, dtype=torch.bool)
model_kwargs = {
    'cond_mask': conditioning,
    'original_frames': x_traj
}

traj_x_out, traj_h_out = trajectory(t, x_traj, h, f, edge_index, edge_attr, batch, **model_kwargs)

# Print output shapes
print(f"conf_x_out shape: {conf_x_out.shape}")
print(f"conf_h_out shape: {conf_h_out.shape}")
print(f"traj_x_out shape: {traj_x_out.shape}")
print(f"traj_h_out shape: {traj_h_out.shape}")

conf_x_rep = conf_x_out.expand(-1, -1, traj_x_out.size(-1))   # [BN, 3, T]
conf_h_rep = conf_h_out.expand(-1, -1, traj_h_out.size(-1))   # [BN, Hh, T]

print("Max Abs Difference using No Set Zero")
print(torch.max(torch.abs(conf_x_rep - traj_x_out)))
print(torch.max(torch.abs(conf_h_rep - traj_h_out)))

# Run through the models
trajectory.set_zero = True
conf_x_out, conf_h_out = conformer(t, x_conf, h, f, edge_index, edge_attr, batch)

conditioning = torch.zeros(10, dtype=torch.bool)
model_kwargs = {
    'cond_mask': conditioning,
    'original_frames': x_traj
}

traj_x_out, traj_h_out = trajectory(t, x_traj, h, f, edge_index, edge_attr, batch, **model_kwargs)

print("Max Abs Difference using Set Zero")
print(torch.max(torch.abs(conf_x_rep - traj_x_out)))
print(torch.max(torch.abs(conf_h_rep - traj_h_out)))













