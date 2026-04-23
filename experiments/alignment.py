import torch
import math
import numpy as np

def kabsch_torch(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    if A.ndim == 3:  # (N, D, T)
        T = A.size(-1)
        A = A.permute(0, 2, 1).flatten(0, 1)  # [NT, D]
        B = B.permute(0, 2, 1).flatten(0, 1)
    else:
        T = None

    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = (t.T).squeeze()
    A_rot = R.mm(A.T).T + t

    if T is not None:
        A_rot = A_rot.reshape(-1, T, A_rot.shape[-1]).permute(0, 2, 1)

    return A_rot, R, t

import torch
import math
import numpy as np

def batched_frame_kabsch(x: torch.Tensor, batch: torch.LongTensor, eps: float = 1e-6):
    """
    Align every frame x[..., f] → x[..., 0] per graph in one pass,
    with per-group degeneracy handling.
    """
    BN, D, T = x.shape
    device = x.device
    B = int(batch.max().item()) + 1

    # Flatten nodes×frames → P = BN*T points
    Xf = x.permute(0, 2, 1).reshape(-1, D)            # [P, D]
    Yf = x[:, :, 0].repeat_interleave(T, dim=0)       # [P, D]

    # Group index for each point: graph_id * T + frame
    batch_rep = batch.unsqueeze(1).repeat(1, T).reshape(-1)  # [P]
    frames    = torch.arange(T, device=device).repeat(BN)    # [P]
    gf        = batch_rep * T + frames                       # [P]
    Gf        = B * T

    # Centroids μX, μY per (graph,frame)
    ones   = torch.ones_like(batch_rep, dtype=Xf.dtype)
    counts = torch.zeros(Gf, device=device).index_add_(0, gf, ones)      # [Gf]
    sumX   = torch.zeros(Gf, D, device=device).index_add_(0, gf, Xf)     # [Gf, D]
    sumY   = torch.zeros(Gf, D, device=device).index_add_(0, gf, Yf)     # [Gf, D]
    muX    = sumX / counts.unsqueeze(-1)                                 # [Gf, D]
    muY    = sumY / counts.unsqueeze(-1)                                 # [Gf, D]

    # Covariance H[g,f] = Σ_i (Xc ⊗ Yc)
    Xc = Xf - muX[gf]             # [P, D]
    Yc = Yf - muY[gf]             # [P, D]
    H = torch.zeros(Gf, D, D, device=device)
    H.index_add_(0, gf, Xc[:, :, None] * Yc[:, None, :])  # [Gf, D, D]

    # Batched SVD → U, S, Vt
    U, S, Vt = torch.linalg.svd(H)                        # U:[Gf,D,D], S:[Gf], Vt:[Gf,D,D]
    V = Vt.transpose(-2, -1)                              # [Gf, D, D]

    # Preliminary rotation and fix reflections
    R = V @ U.transpose(-2, -1)                           # [Gf, D, D]
    detR = torch.det(R)
    refl = detR < 0
    V[refl, :, -1] *= -1
    R = V @ U.transpose(-2, -1)                           # corrected [Gf, D, D]
    R = R.detach()

    # Translation t = μY - R μX
    t = (muY - torch.bmm(R, muX.unsqueeze(-1)).squeeze(-1)).detach()  # [Gf, D]

    # Handle degenerate cases per group: if S[g] < eps, use identity+centroid shift
    deg = S < eps
    if deg.any():
        eye = torch.eye(D, device=device).unsqueeze(0).expand(Gf, D, D)  # [Gf, D, D]
        R[deg] = eye[deg]
        t[deg] = (muY - muX)[deg]

    # Re-apply to every point: X' = R[g,f] @ Xf + t[g,f]
    Rg = R[gf]               # [P, D, D]
    tg = t[gf]               # [P, D]
    Xf_col = Xf.unsqueeze(-1)                          # [P, D, 1]
    Xr_col = torch.bmm(Rg, Xf_col).squeeze(-1)         # [P, D]
    Xr     = Xr_col + tg                               # [P, D]

    # Unflatten → [BN, D, T]
    aligned = Xr.view(BN, T, D).permute(0, 2, 1)
    return aligned


def batched_pairwise_frame_kabsch(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: torch.LongTensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Align each frame of x to the corresponding frame of y per graph,
    with per-group degeneracy handling.
    """
    N, D, T = x.shape
    device = x.device
    B = int(batch.max().item()) + 1

    # Flatten
    Xf = x.permute(0, 2, 1).reshape(-1, D)  # [P, D]
    Yf = y.permute(0, 2, 1).reshape(-1, D)  # [P, D]

    # Group index
    batch_rep = batch.unsqueeze(1).expand(-1, T).reshape(-1)  # [P]
    frames    = torch.arange(T, device=device).repeat(N)      # [P]
    gf        = batch_rep * T + frames                       # [P]
    Gf        = B * T

    # Centroids
    ones   = torch.ones_like(batch_rep, dtype=Xf.dtype)
    counts = torch.zeros(Gf, device=device).index_add_(0, gf, ones)
    sumX   = torch.zeros(Gf, D, device=device).index_add_(0, gf, Xf)
    sumY   = torch.zeros(Gf, D, device=device).index_add_(0, gf, Yf)
    muX    = sumX / counts.unsqueeze(-1)
    muY    = sumY / counts.unsqueeze(-1)

    # Covariance
    Xc = Xf - muX[gf]
    Yc = Yf - muY[gf]
    H = torch.zeros(Gf, D, D, device=device)
    H.index_add_(0, gf, Xc[:, :, None] * Yc[:, None, :])

    # SVD
    U, S, Vt = torch.linalg.svd(H)
    V        = Vt.transpose(-2, -1)

    # Rotation + reflection fix
    R = V @ U.transpose(-2, -1)
    detR = torch.det(R)
    mask = detR < 0
    V[mask, :, -1] *= -1
    R = (V @ U.transpose(-2, -1)).detach()

    # Translation
    t = (muY - torch.bmm(R, muX.unsqueeze(-1)).squeeze(-1)).detach()

    # Degeneracy override
    deg = S < eps
    if deg.any():
        eye = torch.eye(D, device=device).unsqueeze(0).expand(Gf, D, D)
        R[deg] = eye[deg]
        t[deg] = (muY - muX)[deg]

    # Re-apply
    Rg = R[gf]
    tg = t[gf]
    Xf_col = Xf.unsqueeze(-1)
    Xr_col = torch.bmm(Rg, Xf_col).squeeze(-1)
    Xr     = Xr_col + tg

    aligned = Xr.view(N, T, D).permute(0, 2, 1)
    return aligned


# ——————— Unit‐Test Suite ———————

def random_rotation_matrix():
    axis = torch.randn(3); axis /= axis.norm()
    angle = torch.rand(1).item() * 2 * math.pi
    K = torch.tensor([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],   0]
    ])
    return torch.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)

def make_test_batch(num_graphs=2, nodes_per_graph=3, T=4):
    B = num_graphs
    N = B * nodes_per_graph
    x = torch.zeros(N, 3, T)
    batch = torch.zeros(N, dtype=torch.long)
    for g in range(B):
        base = torch.randn(nodes_per_graph, 3)
        idx = slice(g*nodes_per_graph, (g+1)*nodes_per_graph)
        batch[idx] = g
        x[idx, :, 0] = base
        for f in range(1, T):
            R = random_rotation_matrix()
            t = torch.randn(3)
            x[idx, :, f] = (R @ base.T).T + t
    return x, batch

def make_pairwise_test_batch(num_graphs=2, nodes_per_graph=3, T=4):
    B = num_graphs
    N = B * nodes_per_graph
    x = torch.randn(N, 3, T)
    y = torch.zeros_like(x)
    batch = torch.zeros(N, dtype=torch.long)
    for g in range(B):
        idx = slice(g*nodes_per_graph, (g+1)*nodes_per_graph)
        batch[idx] = g
        for f in range(T):
            Xgf = x[idx, :, f]
            R = random_rotation_matrix()
            t = torch.randn(3)
            y[idx, :, f] = (R @ Xgf.T).T + t
    return x, y, batch

def test_shape_and_identity():
    x, batch = make_test_batch(3, 5, 6)
    aligned = batched_frame_kabsch(x, batch)
    assert aligned.shape == x.shape
    assert torch.allclose(aligned[..., 0], x[..., 0], atol=1e-6)

def test_all_frames_align_to_frame0():
    x, batch = make_test_batch(2, 4, 5)
    aligned = batched_frame_kabsch(x, batch)
    for g in batch.unique():
        mask = (batch == g)
        for f in range(1, x.size(-1)):
            assert torch.allclose(
                aligned[mask, :, f],
                aligned[mask, :, 0],
                atol=1e-5
            ), f"Graph {g} frame {f} misaligned"

def test_grad_flow():
    x, batch = make_test_batch(1, 3, 3)
    x = x.clone().requires_grad_(True)
    aligned = batched_frame_kabsch(x, batch)
    loss = (aligned**2).sum()
    loss.backward()
    assert x.grad is not None

def test_pairwise_shape():
    x, y, batch = make_pairwise_test_batch(3, 4, 5)
    aligned = batched_pairwise_frame_kabsch(x, y, batch)
    assert aligned.shape == x.shape

def test_pairwise_alignment():
    x, y, batch = make_pairwise_test_batch(2, 6, 4)
    aligned = batched_pairwise_frame_kabsch(x, y, batch)
    for g in batch.unique():
        mask = (batch == g)
        for f in range(x.size(-1)):
            assert torch.allclose(
                aligned[mask, :, f],
                y[mask, :, f],
                atol=1e-5
            ), f"Graph {g} frame {f} not aligned"

def test_pairwise_grad_flow():
    x, y, batch = make_pairwise_test_batch(1, 3, 3)
    x = x.clone().requires_grad_(True)
    aligned = batched_pairwise_frame_kabsch(x, y, batch)
    loss = (aligned - y).pow(2).sum()
    loss.backward()
    assert x.grad is not None

def test_collinear_points():
    # Collinear along x-axis for one graph (nodes=3, T=3)
    base = torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    x = base.unsqueeze(-1).repeat(1,1,3)          # [3,3,3]
    x = x  # BN=3 nodes-per-graph, but only one graph => batch zeros
    batch = torch.zeros(x.size(0), dtype=torch.long)
    aligned = batched_frame_kabsch(x, batch)
    assert not torch.isnan(aligned).any()
    assert torch.allclose(aligned[:,:,1], aligned[:,:,0], atol=1e-6)

    # Pairwise with collinear
    y = x + torch.tensor([1.,0.,0.]).view(1,3,1)
    aligned2 = batched_pairwise_frame_kabsch(x, y, batch)
    assert not torch.isnan(aligned2).any()
    assert torch.allclose(aligned2, y, atol=1e-6)

def run_all_tests():
    torch.manual_seed(42)
    test_shape_and_identity()
    test_all_frames_align_to_frame0()
    test_grad_flow()
    test_pairwise_shape()
    test_pairwise_alignment()
    test_pairwise_grad_flow()
    test_collinear_points()
    print("✅ All unit tests passed.")

if __name__ == '__main__':
    run_all_tests()
