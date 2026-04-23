# Adapted from GeoDiff Transforms page
import copy
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import coalesce

from rdkit.Chem.rdchem import HybridizationType, BondType
BOND_TYPES = {t: i + 1 for i, t in enumerate(BondType.names.values())}

# Should also add a class AddVirtualNodes(object) class
class AddVirtualNodes(object):
    def __init__(self, num_virtual, virtual_edge_type=0):
        """
        Args:
            num_virtual: how many virtual nodes to add
            virtual_edge_type: integer label for the new virtual edges.
                If None, will pick max(edge_attr)+1 (or 0 if no edge_attr exists).
        """
        self.num_virtual = num_virtual
        self.virtual_edge_type = virtual_edge_type

    def __call__(self, data: Data) -> Data:
        orig_N = data.num_nodes
        new_N = orig_N + self.num_virtual
        device = data.x.device

        # --- node features -----------------------------------------------
        x = data.x
        virt_x = torch.zeros(self.num_virtual, dtype=x.dtype, device=device)
        data.x = torch.cat([x, virt_x], dim=0)

        # --- positions: place each virtual at the *average* of all atom coords
        if getattr(data, "pos", None) is not None:
            pos = data.pos                        # (orig_N, 3, T)
            centroid = pos.mean(dim=0, keepdim=True)   # (1, 3, T)
            virt_pos = centroid.repeat(self.num_virtual, 1, 1)  # (num_virtual, 3, T)
            data.pos = torch.cat([pos, virt_pos], dim=0)       # (new_N, 3, T)

        # --- any extra node-features -------------------------------------
        if getattr(data, "x_features", None) is not None:
            xf = data.x_features              # (orig_N, F)
            centroid_xf = xf.mean(dim=0, keepdim=True)    # (1, F)
            virt_xf = centroid_xf.repeat(self.num_virtual, 1)  # (num_virtual, F)
            data.x_features = torch.cat([xf, virt_xf], dim=0)

        # --- mark which are virtual --------------------------------------
        data.is_real = torch.cat([
            torch.ones(orig_N, dtype=torch.bool, device=device),
            torch.zeros(self.num_virtual, dtype=torch.bool, device=device)
        ], dim=0)

        # --- pick virtual‐edge type --------------------------------------
        if self.virtual_edge_type is None and getattr(data, "edge_attr", None) is not None:
            virt_edge_type = int(data.edge_attr.max().item() + 1)
        else:
            virt_edge_type = self.virtual_edge_type or 0

        # --- build virtual ↔ real edges ---------------------------------
        real_idx = torch.arange(orig_N, device=device)
        virt_idx = torch.arange(orig_N, new_N, device=device)

        # virtual→real
        src1 = virt_idx.repeat_interleave(orig_N)
        tgt1 = real_idx.repeat(self.num_virtual)
        # real→virtual
        src2 = real_idx.repeat(self.num_virtual)
        tgt2 = virt_idx.repeat_interleave(orig_N)

        virt_edges = torch.stack([
            torch.cat([src1, src2], dim=0),
            torch.cat([tgt1, tgt2], dim=0)
        ], dim=0)

        data.edge_index = torch.cat([data.edge_index.to(device), virt_edges], dim=1)

        # --- edge_attr ---------------------------------------------------
        if getattr(data, "edge_attr", None) is not None:
            orig_attr = data.edge_attr
            M = virt_edges.size(1)
            if orig_attr.dim() == 1:
                virt_attr = torch.full((M,), virt_edge_type,
                                       dtype=orig_attr.dtype, device=device)
            else:
                feat_dim = orig_attr.size(1)
                virt_attr = torch.full((M, feat_dim), virt_edge_type,
                                       dtype=orig_attr.dtype, device=device)
            data.edge_attr = torch.cat([orig_attr.to(device), virt_attr], dim=0)

        # --- is_bond mask -----------------------------------------------
        if getattr(data, "is_bond", None) is not None:
            bond = data.is_bond
            virt_bond = torch.zeros(virt_edges.size(1), dtype=bond.dtype, device=device)
            data.is_bond = torch.cat([bond.to(device), virt_bond], dim=0)

        # --- edge_order (if any) ----------------------------------------
        if getattr(data, "edge_order", None) is not None:
            order = data.edge_order
            virt_order = torch.zeros(virt_edges.size(1), dtype=order.dtype, device=device) - 1
            data.edge_order = torch.cat([order.to(device), virt_order], dim=0)

        # --- update count ------------------------------------------------
        data.num_nodes = new_N
        return data


class RemoveHydrogens(object):
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, data: Data):
        device = data.x.device

        keep_mask = data.one_hot_keep.to(device)

        # --- build old→new index map ----------------------------------------
        old_idx = torch.arange(keep_mask.size(0), device=device)
        new_idx = torch.cumsum(keep_mask.long(), dim=0) - 1
        idx_map = -torch.ones_like(old_idx)
        idx_map[keep_mask] = new_idx[keep_mask]

        # --- node-level tensors ---------------------------------------------
        data.x = data.x[keep_mask]
        data.pos = data.pos[keep_mask]
        if getattr(data, "x_features", None) is not None:
            data.x_features = data.x_features[keep_mask]
        if getattr(data, "batch", None) is not None:
            data.batch = data.batch[keep_mask]

        # --- edge-level tensors ---------------------------------------------
        def _filter_and_reindex(eidx, attr=None):
            if eidx is None:
                return None, None
            eidx = eidx.to(device)
            emask = keep_mask[eidx[0]] & keep_mask[eidx[1]]
            eidx  = eidx[:, emask]
            eidx  = idx_map[eidx]
            eattr = attr[emask].to(device) if attr is not None else None
            return eidx, eattr

        data.edge_index, data.edge_attr = _filter_and_reindex(
            getattr(data, "edge_index", None),
            getattr(data, "edge_attr", None)
        )
        data.bond_edge_index, _ = _filter_and_reindex(
            getattr(data, "bond_edge_index", None), None
        )

        if getattr(data, "is_bond", None) is not None:
            ib_mask = keep_mask[data.edge_index[0]] & keep_mask[data.edge_index[1]]
            data.is_bond = data.is_bond[ib_mask].to(device)

        data.num_nodes = int(keep_mask.sum().item())
        return data


class AddHigherOrderEdges(object):

    def __init__(self, order, num_types=len(BOND_TYPES)):
        super().__init__()
        self.order = order
        self.num_types = num_types

    def binarize(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order+1):
            adj_mats.append(self.binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i

        return order_mat

    def __call__(self, data: Data):
        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = self.get_higher_order_adj_matrix(adj, self.order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, self.num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_attr = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_attr = coalesce(new_edge_index, new_edge_attr.long(), N, N) # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
        data.is_bond = (data.edge_attr < self.num_types)
        assert (data.edge_index == edge_index_1).all()

        return data

class AddEdgeLength(object):

    def __call__(self, data: Data):

        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data    


# Add attribute placeholder for data object, so that we can use batch.to_data_list
class AddPlaceHolder(object):
    def __call__(self, data: Data):
        data.pos_gen = -1. * torch.ones_like(data.pos)
        data.d_gen = -1. * torch.ones_like(data.edge_length)
        data.d_recover = -1. * torch.ones_like(data.edge_length)
        return data


class AddEdgeName(object):

    def __init__(self, asymmetric=True):
        super().__init__()
        self.bonds = copy.deepcopy(BOND_NAMES)
        self.bonds[len(BOND_NAMES) + 1] = 'Angle'
        self.bonds[len(BOND_NAMES) + 2] = 'Dihedral'
        self.asymmetric = asymmetric

    def __call__(self, data:Data):
        data.edge_name = []
        for i in range(data.edge_index.size(1)):
            tail = data.edge_index[0, i]
            head = data.edge_index[1, i]
            if self.asymmetric and tail >= head:
                data.edge_name.append('')
                continue
            tail_name = get_atom_symbol(data.atom_type[tail].item())
            head_name = get_atom_symbol(data.atom_type[head].item())
            name = '%s_%s_%s_%d_%d' % (
                self.bonds[data.edge_type[i].item()] if data.edge_type[i].item() in self.bonds else 'E'+str(data.edge_type[i].item()),
                tail_name,
                head_name,
                tail,
                head,
            )
            if hasattr(data, 'edge_length'):
                name += '_%.3f' % (data.edge_length[i].item())
            data.edge_name.append(name)
        return data


class AddAngleDihedral(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def iter_angle_triplet(bond_mat):
        n_atoms = bond_mat.size(0)
        for j in range(n_atoms):
            for k in range(n_atoms):
                for l in range(n_atoms):
                    if bond_mat[j, k].item() == 0 or bond_mat[k, l].item() == 0: continue
                    if (j == k) or (k == l) or (j >= l): continue
                    yield(j, k, l)

    @staticmethod
    def iter_dihedral_quartet(bond_mat):
        n_atoms = bond_mat.size(0)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i >= j: continue
                if bond_mat[i,j].item() == 0:continue
                for k in range(n_atoms):
                    for l in range(n_atoms):
                        if (k in (i,j)) or (l in (i,j)): continue
                        if bond_mat[k,i].item() == 0 or bond_mat[l,j].item() == 0: continue
                        yield(k, i, j, l)

    def __call__(self, data:Data):
        N = data.num_nodes
        if 'is_bond' in data:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.is_bond).long().squeeze(0) > 0
        else:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).long().squeeze(0) > 0

        # Note: if the name of attribute contains `index`, it will automatically
        #       increases during batching.
        data.angle_index = torch.LongTensor(list(self.iter_angle_triplet(bond_mat))).t()
        data.dihedral_index = torch.LongTensor(list(self.iter_dihedral_quartet(bond_mat))).t()

        return data


class CountNodesPerGraph(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data


TRANSFORMS = {
    "edge_order": AddHigherOrderEdges,
    "remove_hs": RemoveHydrogens,
    "add_virtual": AddVirtualNodes
}   