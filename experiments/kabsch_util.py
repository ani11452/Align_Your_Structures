import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .alignment import kabsch_torch, batched_frame_kabsch, batched_pairwise_frame_kabsch
from .layers import ESLayer, ETLayer, BasicESLayer, ETAttention, merge_time_dim, separate_time_dim

'''
Frame 0 Alignment Kabsch
'''
def align_traj_kabsch_naive(X: torch.Tensor) -> torch.Tensor:
    n_pts, D, T = X.shape
    assert D == 3, "Only 3D supported"

    P = X.permute(2, 0, 1)

    a_mean = P.mean(dim=1, keepdim=True)
    B0 = P[0:1]
    b_mean = B0.mean(dim=1, keepdim=True).detach()

    P_cent = P - a_mean
    B0_cent = B0 - b_mean

    H = P_cent.transpose(1, 2) @ B0_cent

    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(-2, -1)

    R = V @ U.transpose(-2, -1).detach()

    P_rot = (P_cent @ R.transpose(-2, -1)) + b_mean

    return P_rot.permute(1, 2, 0)

def naive_batched_frame_kabsch(x: torch.Tensor, batch: torch.LongTensor):
    out = x.clone()
    for b in torch.unique(batch, sorted=True):
        mask = (batch == b)
        out[mask] = align_traj_kabsch_naive(x[mask])
    return out

'''
Global Trajectory Alignment Kabsch
'''
def kabsch_torch_local(A, B):
    n_points, D, T = A.shape
    assert D == 3, "Expected 3D coordinates"

    A = A.permute(0, 2, 1).reshape(-1, 3)
    B = B.permute(0, 2, 1).reshape(-1, 3)

    a_mean = A.mean(dim=0)
    b_mean = B.mean(dim=0)
    A_c = A - a_mean
    B_c = B - b_mean

    H = A_c.T @ B_c
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    t = b_mean - (R @ a_mean)
    return R.detach(), t.detach()

def align_globally(Xb, Ref):
    R, t = kabsch_torch_local(Xb, Ref)
    Xb_flat = Xb.permute(0, 2, 1).reshape(-1, 3).T
    Xb_al_flat = (R @ Xb_flat).T + t
    return Xb_al_flat.reshape(Xb.shape[0], Xb.shape[2], 3).permute(0, 2, 1)

def naive_batched_global_kabsch(x_t: torch.Tensor, x_r: torch.Tensor, batch: torch.LongTensor):
    out = x_t.clone()
    for b in torch.unique(batch, sorted=True):
        mask = (batch == b)
        out[mask] = align_globally(x_t[mask], x_r[mask])
    return out

'''
Pairwise Kabsch Alignment
'''
def align_traj_kabsch_pairwise_naive(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    n_pts, D, T = X.shape
    assert D == 3, "Only 3D supported"

    P = X.permute(2, 0, 1)
    Q = Y.permute(2, 0, 1)

    a_mean = P.mean(dim=1, keepdim=True).detach()
    b_mean = Q.mean(dim=1, keepdim=True).detach()

    P_cent = P - a_mean
    Q_cent = Q - b_mean

    H = P_cent.transpose(1, 2) @ Q_cent
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(-2, -1)

    R = (V @ U.transpose(-2, -1)).detach()
    P_rot = (P_cent @ R.transpose(-2, -1)) + b_mean
    return P_rot.permute(1, 2, 0)

def naive_batched_pairwise_kabsch(x_t: torch.Tensor, x_r: torch.Tensor, batch: torch.LongTensor):
    out = x_t.clone()
    for b in torch.unique(batch, sorted=True):
        mask = (batch == b)
        out[mask] = align_traj_kabsch_pairwise_naive(x_t[mask], x_r[mask])
    return out

if __name__ == '__main__':
    import sys

    def random_rotation_matrix():
        M = torch.randn(3, 3)
        Q, R = torch.linalg.qr(M)
        if torch.det(Q) < 0:
            Q[:, -1] *= -1
        return Q

    # --- test functions ---
    def test_align_traj_kabsch_naive_identity():
        torch.manual_seed(0)
        n_pts, T = 5, 3
        X0 = torch.randn(n_pts, 3)
        X = X0.unsqueeze(2).repeat(1, 1, T)
        out = align_traj_kabsch_naive(X)
        assert out.shape == X.shape
        torch.testing.assert_close(out, X, rtol=1e-5, atol=1e-6)

    def test_align_traj_kabsch_naive_rotation():
        torch.manual_seed(1)
        n_pts, T = 5, 3
        X0 = torch.randn(n_pts, 3)
        X = X0.unsqueeze(2).repeat(1, 1, T)
        R1, R2 = random_rotation_matrix(), random_rotation_matrix()
        X[:, :, 1] = (R1 @ X[:, :, 1].T).T + torch.randn(3)
        X[:, :, 2] = (R2 @ X[:, :, 2].T).T + torch.randn(3)
        out = align_traj_kabsch_naive(X)
        for f in range(T):
            torch.testing.assert_close(out[:, :, f], X0, rtol=1e-4, atol=1e-5)

    def test_align_traj_kabsch_naive_wrong_dim():
        threw = False
        try:
            align_traj_kabsch_naive(torch.randn(5, 2, 3))
        except AssertionError:
            threw = True
        assert threw

    def test_align_traj_kabsch_pairwise_naive_identity():
        torch.manual_seed(2)
        n_pts, T = 6, 4
        X = torch.randn(n_pts, 3, T)
        Y = X.clone()
        out = align_traj_kabsch_pairwise_naive(X, Y)
        torch.testing.assert_close(out, Y, rtol=1e-5, atol=1e-6)

    def test_align_traj_kabsch_pairwise_naive_rotation_translation():
        torch.manual_seed(3)
        n_pts, T = 6, 4
        X = torch.randn(n_pts, 3, T)
        R = random_rotation_matrix()
        t = torch.randn(3)
        Y = torch.zeros_like(X)
        for f in range(T):
            Y[:, :, f] = (R @ X[:, :, f].T).T + t
        out = align_traj_kabsch_pairwise_naive(X, Y)
        torch.testing.assert_close(out, Y, rtol=1e-4, atol=1e-5)

    def test_align_traj_kabsch_pairwise_naive_wrong_dim():
        threw = False
        try:
            align_traj_kabsch_pairwise_naive(torch.randn(5, 2, 3), torch.randn(5, 2, 3))
        except AssertionError:
            threw = True
        assert threw

    def test_kabsch_globally_identity():
        torch.manual_seed(4)
        n_pts, T = 7, 5
        X = torch.randn(n_pts, 3, T)
        out = align_globally(X, X)
        torch.testing.assert_close(out, X, rtol=1e-5, atol=1e-6)

    def test_kabsch_globally_rotation_translation():
        torch.manual_seed(5)
        n_pts, T = 7, 5
        X = torch.randn(n_pts, 3, T)
        R = random_rotation_matrix()
        t = torch.randn(3)
        X_rt = torch.zeros_like(X)
        for i in range(n_pts):
            for f in range(T):
                X_rt[i, :, f] = R @ X[i, :, f] + t
        out = align_globally(X_rt, X)
        torch.testing.assert_close(out, X, rtol=1e-4, atol=1e-5)

    def test_batched_functions():
        torch.manual_seed(6)
        n1, n2, T = 4, 5, 3
        X1 = torch.randn(n1, 3, T)
        X2 = torch.randn(n2, 3, T)
        R1, R2 = random_rotation_matrix(), random_rotation_matrix()
        Y1 = torch.einsum('ij,pjf->pif', R1, X1)
        Y2 = torch.einsum('ij,pjf->pif', R2, X2)
        batch = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)])
        x_t = torch.cat([X1, X2])
        x_r = torch.cat([Y1, Y2])

        out_pair = naive_batched_pairwise_kabsch(x_t, x_r, batch)
        torch.testing.assert_close(out_pair, x_r, rtol=1e-4, atol=1e-5)

        out_global = naive_batched_global_kabsch(x_t, x_r, batch)
        torch.testing.assert_close(out_global, x_r, rtol=1e-4, atol=1e-5)

        out_frame = naive_batched_frame_kabsch(x_t, batch)
        assert out_frame.shape == x_t.shape

    def test_gradient_flow_pairwise():
        torch.manual_seed(7)
        n_pts, T = 6, 3
        X = torch.randn(n_pts, 3, T, requires_grad=True)
        Y = torch.randn(n_pts, 3, T, requires_grad=True)
        batch = torch.zeros(n_pts, dtype=torch.long)
        out = naive_batched_pairwise_kabsch(X, Y, batch)
        loss = out.sum()
        loss.backward()
        assert X.grad is not None
        assert Y.grad is None

    def test_gradient_flow_global():
        torch.manual_seed(8)
        n_pts, T = 6, 3
        X = torch.randn(n_pts, 3, T, requires_grad=True)
        Y = torch.randn(n_pts, 3, T, requires_grad=True)
        batch = torch.zeros(n_pts, dtype=torch.long)
        out = naive_batched_global_kabsch(X, Y, batch)
        loss = out.mean()
        loss.backward()
        assert X.grad is not None
        assert Y.grad is None

    # --- simple runner ---
    tests = [
        test_align_traj_kabsch_naive_identity,
        test_align_traj_kabsch_naive_rotation,
        test_align_traj_kabsch_naive_wrong_dim,
        test_align_traj_kabsch_pairwise_naive_identity,
        test_align_traj_kabsch_pairwise_naive_rotation_translation,
        test_align_traj_kabsch_pairwise_naive_wrong_dim,
        test_kabsch_globally_identity,
        test_kabsch_globally_rotation_translation,
        test_batched_functions,
        test_gradient_flow_pairwise,
        test_gradient_flow_global,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {fn.__name__}: {e}")

    if failures:
        sys.exit(1)
    else:
        print("All tests passed!")


