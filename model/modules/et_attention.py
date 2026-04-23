'''
Code based on https://github.com/hanjq17/GeoTDM

MIT License

Copyright (c) 2024 Jiaqi Han

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import torch.nn as nn
from einops import rearrange, einsum
from torch_scatter import scatter
import torch.nn.functional as F
import math
from rotary_embedding_torch import RotaryEmbedding
from geometric_vector_perceptron_local import GVP, VLinear

import numpy as np
from scipy.linalg import qr

eps = 1e-8

def merge_time_dim(x):
    return rearrange(x, 'm d t -> (m t) d')

def separate_time_dim(x, t):
    return rearrange(x, '(m t) d -> m d t', t=t)


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layer, act_fn, last_act=False):
        super().__init__()
        assert n_layer >= 2
        actions = nn.ModuleList()
        actions.append(nn.Linear(in_dim, hidden_dim))
        actions.append(act_fn)
        for i in range(n_layer - 2):
            actions.append(nn.Linear(hidden_dim, hidden_dim))
            actions.append(act_fn)
        actions.append(nn.Linear(hidden_dim, out_dim))
        if last_act:
            actions.append(act_fn)
        self.actions = nn.Sequential(*actions)

    def forward(self, x):
        x = self.actions(x)
        return x


class ETAttention(nn.Module):
    def __init__(
        self, 
        node_dim, 
        kq_dim=16, 
        v_dim=16,
        n_heads=8,
        act_fn=nn.SiLU(), 
    ):
        super().__init__()
        self.n_heads = n_heads
        self.node_dim = node_dim
        self.kq_dim = kq_dim
        self.v_dim = v_dim

        # Define the QKV
        self.q_mlp = MLP(in_dim=node_dim, hidden_dim=kq_dim * n_heads, out_dim=kq_dim * n_heads, n_layer=2, act_fn=act_fn)
        self.k_mlp = MLP(in_dim=node_dim, hidden_dim=kq_dim * n_heads, out_dim=kq_dim * n_heads, n_layer=2, act_fn=act_fn)
        self.v_mlp = MLP(in_dim=node_dim, hidden_dim=v_dim * n_heads, out_dim=v_dim * n_heads, n_layer=2, act_fn=act_fn)
        self.x_mlp = MLP(in_dim=v_dim * n_heads, hidden_dim=v_dim * n_heads, out_dim=n_heads, n_layer=2, act_fn=act_fn)
        
        # Instantiate RoPE on the per-head dimension
        self.rotary_emb = RotaryEmbedding(dim=self.kq_dim)

        # Define the out layers
        self.h_out = nn.Linear(v_dim * n_heads, node_dim)
        self.v_out = GVP(
            dim_vectors_in=n_heads, dim_vectors_out=1, 
            dim_feats_in=node_dim, dim_feats_out=node_dim
        )

    def forward(self, x, h):
        """
        Args:
            x: coordinates, shape [B, 3, T] 
            h: features,   shape [B, node_dim, T]
        Returns:
            x_out: updated coordinates, shape [B, 3, T]
            h_out: updated features,    shape [B, node_dim, T]
        """
        B, _, T = x.shape

        # Merge dims
        x = x.reshape(B, 3, T)
        h = h.reshape(B, self.node_dim, T)

        # Compute Q, K, V, and scalar for coords
        h_merged = merge_time_dim(h)     # [B*T, node_dim]
        q   = self.q_mlp(h_merged)       # [B*T, kq_dim*n_heads]
        k   = self.k_mlp(h_merged)
        v   = self.v_mlp(h_merged)       # [B*T, v_dim*n_heads]
        v_x = self.x_mlp(v)              # [B*T, n_heads]

        # Restore time dimension
        q   = separate_time_dim(q,   t=T).transpose(1,2)  # [B, T, kq_dim*n_heads]
        k   = separate_time_dim(k,   t=T).transpose(1,2)
        v   = separate_time_dim(v,   t=T).transpose(1,2)
        v_x = separate_time_dim(v_x, t=T)                 # [B, n_heads, T]

        # Split heads and apply RoPE
        q = q.reshape(B, T, self.n_heads, self.kq_dim).permute(0,2,1,3)  # [B, H, T, kq_dim]
        k = k.reshape(B, T, self.n_heads, self.kq_dim).permute(0,2,1,3)
        v = v.reshape(B, T, self.n_heads, self.v_dim).permute(0,2,1,3)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Attention weights
        scale = math.sqrt(self.kq_dim)
        attn_scores = torch.einsum('b h t d, b h s d -> b h t s', q, k) / scale  # [B, H, T, T]
        attn = F.softmax(attn_scores, dim=-1)                                   # [B, H, T, T]

        # Feature update with residual connection
        h_update = torch.einsum('b h t s, b h s d -> b h t d', attn, v)          # [B, H, T, v_dim]
        h_update = h_update.permute(0,2,1,3).reshape(B, T, self.n_heads*self.v_dim)
        h_update = self.h_out(h_update).transpose(1,2)                           # [B, node_dim, T]
        h_out = h + h_update  # Residual connection as in previous versions

        # Coordinate update matching previous versions (v_x per key position)
        attn_w = attn * v_x.unsqueeze(-2)  # [B, H, T, T] * [B, H, 1, T] -> [B, H, T, T]
        # attn_w[b, h, i, j] = attn[b, h, i, j] * v_x[b, h, j]
        gamma = attn_w.sum(dim=-1)         # [B, H, T], gamma[b, h, i] = sum_j attn[b, h, i, j] * v_x[b, h, j]
        weighted_x = torch.einsum(
            'b h i j, b d j -> b h d i',
            attn_w, x
        )                                   # [B, H, 3, T], weighted_x[b, h, d, i] = sum_j attn[b, h, i, j] * v_x[b, h, j] * x[b, d, j]
        x_head_updates = gamma.unsqueeze(2) * x.unsqueeze(1) - weighted_x  # [B, H, 3, T]
        x_update = x_head_updates.permute(0,3,1,2).reshape(B * T, -1, 3)                       # [B, T, H, 3]
        delta    = self.v_out((merge_time_dim(h_out), x_update))[1]
        delta = separate_time_dim(delta.squeeze(1), T)
        x_out    = x + delta  # [B, 3, T]

        return x_out, h_out


class ESLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, act_fn, normalize=True, pre_norm=False):
        super().__init__()
        self.edge_mlp = MLP(in_dim=node_dim * 2 + 1 + edge_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                            n_layer=2, act_fn=act_fn, last_act=True)
        self.coord_mlp = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=1,
                             n_layer=2, act_fn=act_fn)
        self.node_mlp = MLP(in_dim=node_dim + hidden_dim, hidden_dim=hidden_dim, out_dim=node_dim,
                            n_layer=2, act_fn=act_fn)

        self.normalize = normalize
        self.pre_norm = pre_norm

    def forward(self, x, h, edge_index, edge_attr, batch, **model_kwargs):
        """
        :param x: shape [BN, 3, T]
        :param h: shape [BN, D, T]
        :param edge_index: [2, 2BM]
        :param edge_attr: [2BM, He, T]
        :param batch: [BN]
        :return:
        """
        row, col = edge_index[0], edge_index[1]
        T = x.size(-1)

        x_ij = x[row] - x[col]  # [BM, 3, T]

        if self.pre_norm:
            x_ij_norm2 = torch.norm(x_ij, dim=1, keepdim=True)
        else:
            x_ij_norm2 = (x_ij ** 2).sum(dim=1, keepdim=True)  # [BM, 1, T]
        if edge_attr is not None:
            m_ij = torch.cat((h[row], h[col], x_ij_norm2, edge_attr), dim=1)  # [BM, Hh+Hh+1+He, T]
        else:
            m_ij = torch.cat((h[row], h[col], x_ij_norm2), dim=1)  # [BM, Hh+Hh+1+He, T]
        m_ij = self.edge_mlp(merge_time_dim(m_ij))  # [BM*T, H]
        m_i = scatter(separate_time_dim(m_ij, T), row, dim=0, dim_size=x.size(0), reduce='sum')  # [BN, H, T]
        h = torch.cat((merge_time_dim(h), merge_time_dim(m_i)), dim=-1)  # [BN*T, Hh+H]
        h = separate_time_dim(self.node_mlp(h), t=T)  # [BN, Hh, T]
        coord_m_ij = self.coord_mlp(m_ij)  # [BM*T, 1]
        if self.normalize and not self.pre_norm:
            coord_m_ij = coord_m_ij / ((merge_time_dim(x_ij_norm2) + eps).sqrt() + 1)
        x = x + scatter(separate_time_dim(coord_m_ij, t=T) * x_ij, row, dim=0, dim_size=x.size(0), reduce='mean')

        return x, h


if __name__ == '__main__':
    def rand_rot_trans():
        Q = np.random.randn(3, 3)
        Q = qr(Q)[0]
        Q = Q / np.linalg.det(Q)
        Q = torch.from_numpy(np.array(Q)).float()
        return Q

    # Create dummy inputs
    atom_rep = torch.randn((32, 50, 128, 25), dtype=torch.float32)
    pair_rep = torch.randn((32, 50, 50, 64), dtype=torch.float32)
    atom_coord = torch.randn((32, 50, 25, 3), dtype=torch.float32)
    batch_r = torch.stack([rand_rot_trans() for _ in range(32)])
    batch_r_inv = batch_r.transpose(1, 2)
    t = torch.rand(32, 3)[:, None, None, :]

    atom_coord_transfomed = torch.matmul(atom_coord, batch_r.unsqueeze(1)) + t

    atom_coord = atom_coord.transpose(2, 3)
    atom_coord_transfomed = atom_coord_transfomed.transpose(2, 3)

    # Create the ETA object
    eta = ETAttention(node_dim=128)

    # Do the forward with and without transformation
    print(atom_rep.shape, atom_coord.shape)
    a_out, x_out = eta(atom_rep, atom_coord)
    a_out_t, x_out_t = eta(atom_rep, atom_coord_transfomed)

    x_out = x_out.transpose(2, 3)
    x_out_t = x_out_t.transpose(2, 3)
    print(batch_r_inv.unsqueeze(0).shape)

    # Check shapes
    print(x_out.shape, x_out_t.shape)
    print(torch.max(torch.abs(a_out - a_out_t)))
    print(torch.max(torch.abs(x_out - torch.matmul(x_out_t - t, batch_r_inv.unsqueeze(1)))))



