import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geometric_vector_perceptron_local import GVP, VLinear

# Basic FF layer
class FF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, a, x=None):
        if x:
            return self.out(a), x
        else:
            return self.out(a)

# GVP Based FF Layer
class GV_FF(nn.Module):
    def __init__(
        self, 
        input_dim, hidden_dim, output_dim,
        v_input_dim, v_hidden_dim, v_output_dim
    ):
        super().__init__()
        self.gvp1 = GVP(
            dim_vectors_in=v_input_dim, dim_vectors_out=v_hidden_dim,
            dim_feats_in=input_dim, dim_feats_out=hidden_dim
        )
        self.gvp2 = GVP(
            dim_vectors_in=v_hidden_dim, dim_vectors_out=v_output_dim,
            dim_feats_in=hidden_dim, dim_feats_out=output_dim,
            feats_activation=nn.Identity()
        )

    def forward(self, a, x):
        a_n, x_n = self.gvp1((a, x))
        a, x = self.gvp2((a_n, x_n))
        return a, x
