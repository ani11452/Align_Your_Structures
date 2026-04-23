'''
Code based on https://github.com/lucidrains/geometric-vector-perceptron

MIT License

Copyright (c) 2021 Phil Wang

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
from torch import nn, einsum
from torch_geometric.nn import MessagePassing

from typing import Optional, List, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, Tensor

def exists(val):
    return val is not None

# Custom VLinear Layer
class VLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.vector_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, vectors):
        # vectors: [B, N, D, 3]
        return self.vector_proj(vectors.transpose(-2, -1)).transpose(-2, -1)

class GVP(nn.Module):
    def __init__(
        self,
        *,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        feats_activation=nn.SiLU(),
        vectors_activation=nn.Sigmoid(),
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in
        self.dim_vectors_out = dim_vectors_out

        dim_h = max(dim_vectors_in, dim_vectors_out)
        self.Wh = VLinear(dim_vectors_in, dim_h)
        self.Wu = VLinear(dim_h, dim_vectors_out)

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_feats_in + dim_h, dim_feats_out),
            feats_activation
        )

        self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out)

    def forward(self, data):
        feats, vectors = data
        f_in = feats.shape[-1]
        v_in, c = vectors.shape[-2:]

        assert c == 3
        assert v_in == self.dim_vectors_in
        assert f_in == self.dim_feats_in

        Vh = self.Wh(vectors)
        Vu = self.Wu(Vh)

        sh = torch.norm(Vh, p=2, dim=-1)
        s = torch.cat([feats, sh], dim=-1)
        feats_out = self.to_feats_out(s)

        if self.scalar_to_vector_gates is not None:
            gating = self.scalar_to_vector_gates(feats_out).unsqueeze(-1)
        else:
            gating = torch.norm(Vu, p=2, dim=-1, keepdim=True)

        activated_gate = self.vectors_activation(gating)
        vectors_out = activated_gate * Vu

        return feats_out, vectors_out

class GVPDropout(nn.Module):
    """ Separate dropout for scalars and vectors. """
    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = nn.Dropout2d(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)

class GVPLayerNorm(nn.Module):
    """ Normal layer norm for scalars, nontrainable norm for vectors. """
    def __init__(self, feats_h_size, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, feats, vectors):
        vector_norm = vectors.norm(dim=(-1,-2), keepdim=True)
        normed_feats = self.feat_norm(feats)
        normed_vectors = vectors / (vector_norm + self.eps)
        return normed_feats, normed_vectors