import torch
import torch.nn as nn
from einops import rearrange, einsum
from torch_scatter import scatter
import torch.nn.functional as F
import math

import os, sys
sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'model', 'modules',
))
from feed_forward import GV_FF
from et_attention import ETAttention

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

def SE3_equivariant_layer_norm(vectors, eps=1e-5):
    """
    vectors: Tensor of shape [BN, 3, T]
    Returns: normalized tensor of shape [BN, 3, T]
    """
    # Subtract mean over T → translation equivariant
    mean = vectors.mean(dim=-1, keepdim=True)  # shape [BN, 3, 1]
    centered = vectors - mean

    # Normalize by norm over all spatial dimensions → rotation equivariant
    norm = centered.norm(dim=(-1, -2), keepdim=True)  # scalar per sample [BN, 1, 1]
    normalized = centered / (norm + eps)

    return normalized

class BasicESLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, act_fn, normalize=True, pre_norm=False):
        super().__init__()
        self.edge_mlp = MLP(in_dim=node_dim * 2 + 1 + edge_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                            n_layer=2, act_fn=act_fn, last_act=True)
        self.coord_mlp = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=1,
                             n_layer=2, act_fn=act_fn)
        self.node_mlp = MLP(in_dim=node_dim + hidden_dim, hidden_dim=hidden_dim, out_dim=node_dim,
                            n_layer=2, act_fn=act_fn)
        self.gv_ff = GV_FF(
            input_dim=node_dim, hidden_dim=hidden_dim, output_dim=node_dim,
            v_input_dim=1, v_hidden_dim=4, v_output_dim=1
        )

        self.normalize = normalize
        self.pre_norm = pre_norm

    def forward(self, x, h, edge_index, edge_attr, batch, **model_kwargs):
        """
        :param x: shape [BN, 3, T]
        :param h: shape [BN, Hh, T]
        :param edge_index: [2, BM]
        :param edge_attr: [BM, He, T]
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
        x_upt = scatter(separate_time_dim(coord_m_ij, t=T) * x_ij, row, dim=0, dim_size=x.size(0), reduce='mean')

        # Transition layer
        h = merge_time_dim(h)
        x_upt = merge_time_dim(x_upt).unsqueeze(1)
        h_new, x_upt = self.gv_ff(h, x_upt)

        h = h + h_new
        h = separate_time_dim(h, t=T)

        x_upt = separate_time_dim(x_upt.squeeze(1), t=T)
        x = x + x_upt

        return x, h


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
        :param h: shape [BN, Hh, T]
        :param edge_index: [2, BM]
        :param edge_attr: [BM, He, T]
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

        x_upt = scatter(separate_time_dim(coord_m_ij, t=T) * x_ij, row, dim=0, dim_size=x.size(0), reduce='mean')
        x_upt = SE3_equivariant_layer_norm(x_upt)
        x = x + x_upt
        return x, h


class ETLayer(nn.Module):
    def __init__(self, node_dim, hidden_dim, act_fn, time_emb_dim):
        super().__init__()

        assert node_dim == hidden_dim

        self.q_mlp = MLP(in_dim=node_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, n_layer=2, act_fn=act_fn)
        self.k_mlp = MLP(in_dim=node_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, n_layer=2, act_fn=act_fn)
        self.v_mlp = MLP(in_dim=node_dim, hidden_dim=hidden_dim, out_dim=node_dim, n_layer=2, act_fn=act_fn)
        self.x_mlp = MLP(in_dim=node_dim, hidden_dim=hidden_dim, out_dim=1, n_layer=2, act_fn=act_fn)

        self.time_emb = nn.Linear(time_emb_dim, hidden_dim)
        self.time_emb_dim = time_emb_dim

    def forward(self, x, h):
        """
        :param x: shape [BN, 3, T]
        :param h: shape [BN, Hh, T]
        :return:
        """

        # Memory efficient implementation of ETLayer
        B, C, T = x.shape
        device  = x.device

        # 1) make the small [2T-1,H] table
        rel_range = torch.arange(-T+1, T, device=device)  # [2T-1]
        emb_rel    = self.time_emb(get_timestep_embedding(rel_range, self.time_emb_dim))  # [2T-1,H]

        # 2) index matrix: idx[i,j] = (i-j)+(T-1)
        idx        = (torch.arange(T, device=device)[:,None] 
                    - torch.arange(T, device=device)[None,:]) + (T-1)  # [T,T]

        # 3) gather into a single [H,T,T] for biases
        rel_pos = emb_rel[idx]            # [T,T,H]
        rel_pos = rel_pos.permute(2,0,1)  # [H,T,T]

        q = self.q_mlp(merge_time_dim(h)); k = self.k_mlp(merge_time_dim(h))
        v = self.v_mlp(merge_time_dim(h)); v_x = self.x_mlp(v)

        v_x_s = separate_time_dim(v_x, T)           # [B,1,T]
        qt    = separate_time_dim(q,   T).transpose(-1,-2)  # [B,T,H]
        ks    = separate_time_dim(k,   T)           # [B,H,T]
        vs    = separate_time_dim(v,   T)           # [B,H,T]

        # base logits
        base_logits = torch.einsum('bth,bhs->bts', qt, ks)    # [B,T,T]

        # relative bias logits without any big [B,H,T,T]
        q_flat     = qt.reshape(B*T, -1)     # [B*T,H]
        rel_logits = q_flat @ emb_rel.T      # [B*T,2T-1]
        rel_logits = rel_logits.view(B, T, -1)  # [B,T,2T-1]
        bias_logits= rel_logits.gather(2, idx.unsqueeze(0).expand(B,-1,-1))  # [B,T,T]

        alpha = F.softmax(base_logits + bias_logits, dim=-1)  # [B,T,T]
        
        # correct standard vs‐term
        std_vs = torch.einsum('bts,bhs->bht', alpha, vs)      # [B, H, T]

        # correct relative‐bias term (unchanged)
        bias_h = torch.einsum('bts,hts->bht', alpha, rel_pos) # [B, H, T]

        h_new = h + std_vs + bias_h

        # x‐update stays the same
        x_diff = x.unsqueeze(-1) - x.unsqueeze(-2)            # [B, C, T, T]
        
        x_update = (alpha.unsqueeze(1)                         # [B,1,T,T]
                   * x_diff                                   # [B,C,T,T]
                   * v_x_s.unsqueeze(-2)                      # [B,1,1,T]->[B,1,T,T]
                   ).sum(dim=-1)                              # → [B, C, T]
        
        # Apply SE3_equivariant_layer_norm to the coordinate update
        # x_update_normalized = SE3_equivariant_layer_norm(x_update)  # [B, C, T]
        
        x_new = x + x_update

        return x_new, h_new

        # time_index = torch.arange(T).to(x)  # [T]
        # rel_time_index = time_index.unsqueeze(-1) - time_index.unsqueeze(-2)  # [T, S]
        # rel_time_emb = get_timestep_embedding(rel_time_index.view(-1), embedding_dim=self.time_emb_dim)  # [T*S, Ht]
        # rel_time_emb = self.time_emb(rel_time_emb).view(T, T, -1)  # [T*S, H]
        # rel_time_emb = rel_time_emb.permute(2, 0, 1).unsqueeze(0).repeat(x.size(0), 1, 1, 1)  # [BN, H, T, S]

        # q = self.q_mlp(merge_time_dim(h))  # [BN*T, H]
        # k = self.k_mlp(merge_time_dim(h))  # [BN*T, H]
        # v = self.v_mlp(merge_time_dim(h))  # [BN*T, Hh]
        # v_x = self.x_mlp(v)  # [BN*T, 1]
        # v_x_s = separate_time_dim(v_x, t=T)  # [BN, 1, S]
        # qt = separate_time_dim(q, t=T).transpose(-1, -2)  # [BN, H, T] -> [BN, T, H]
        # ks = separate_time_dim(k, t=T)  # [BN, H, S]
        # k_ts = ks.unsqueeze(-2).repeat(1, 1, T, 1) + rel_time_emb  # [BN, H, T, S]
        # vs = separate_time_dim(v, t=T)  # [BN, Hh, S]
        # v_ts = vs.unsqueeze(-2).repeat(1, 1, T, 1) + rel_time_emb  # [BN, H, T, S]
        # alpha_ts = F.softmax(einsum(qt, k_ts, 'n t h, n h t s-> n t s'), dim=-1)  # [BN, T, S]
        # h = h + einsum(alpha_ts, v_ts, 'n t s, n h t s-> n h t')  # [BN, BH, T]
        # x_ts = x.unsqueeze(-1) - x.unsqueeze(-2)  # [BN, 3, T, S]
        # alpha_x_ts = alpha_ts.unsqueeze(1) * x_ts  # [BN, 3, T, S]
        # x = x + (alpha_x_ts * v_x_s.unsqueeze(-2)).sum(dim=-1)  # [BN, 3, T]
        # return x, h


if __name__ == '__main__':
    import numpy as np
    from scipy.linalg import qr

    def rand_rot_trans():
        Q = np.random.randn(3, 3)
        Q = qr(Q)[0]
        Q = Q / np.linalg.det(Q)
        Q = torch.from_numpy(np.array(Q)).float()
        return Q

    BN = 5
    B = 2
    Hh = 32
    He = 2
    H = 32
    T = 10
    # eslayer = BasicESLayer(node_dim=Hh, edge_dim=He, hidden_dim=H, act_fn=nn.SiLU())
    eslayer = ESLayer(node_dim=Hh, edge_dim=He, hidden_dim=H, act_fn=nn.SiLU())
    # etlayer = ETAttention(node_dim=Hh)
    etlayer = ETLayer(node_dim=Hh, hidden_dim=H, act_fn=nn.SiLU(), time_emb_dim=32)
    batch = torch.from_numpy(np.array([0, 0, 0, 1, 1])).long()
    row = [0, 0, 1, 3]
    col = [1, 2, 2, 4]
    row = torch.from_numpy(np.array(row)).long()
    col = torch.from_numpy(np.array(col)).long()
    h = torch.rand(BN, Hh, T)
    x = torch.rand(BN, 3, T)
    edge_index = torch.stack((row, col), dim=0)  # [2, BM]
    BM = edge_index.size(-1)
    edge_attr = torch.rand(BM, He, T)
    _x, _h = eslayer(x, h, edge_index, edge_attr, batch)
    _x, _h = etlayer(_x, _h)

    # Test rotation/translation equivariance
    R = rand_rot_trans()
    t = torch.rand(B, 3)[batch].unsqueeze(-1)

    x_R = (x.transpose(-1, -2) @ R).transpose(-1, -2) + t
    _x_R, _h_R = eslayer(x_R, h, edge_index, edge_attr, batch)
    _x_R, _h_R = etlayer(_x_R, _h_R)
    _R_x = (_x.transpose(-1, -2) @ R).transpose(-1, -2) + t
    print((_R_x - _x_R).abs().sum())
    print((_h - _h_R).abs().sum())