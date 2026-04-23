import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .alignment import kabsch_torch, batched_frame_kabsch, batched_pairwise_frame_kabsch
from .layers import ESLayer, ETLayer, BasicESLayer, ETAttention, merge_time_dim, separate_time_dim
from .kabsch_util import *


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

class Embedding(nn.Module):
    def __init__(self, node_dim, ft_dim, edge_dim):
        super().__init__()
        # Do the atom embedding
        self.atom_embedding = nn.Embedding(100, node_dim)
        # Do the node feature embedding
        self.input_linear = nn.Linear(node_dim + ft_dim, node_dim)
        # Do the edge embedding     
        self.edge_embedding = nn.Embedding(50, edge_dim)

    def forward(self, x_features, atoms, edge_attr, pos):
        # Get the T
        T = pos.size(-1)
        # Do the atom embedding
        atom_embed = self.atom_embedding(atoms)
        # Do the node feature embedding
        x_features = self.input_linear(torch.cat((atom_embed, x_features), dim=-1))
        # Do the edge embedding
        edge_embed = self.edge_embedding(edge_attr)
        # Do the output
        return x_features, edge_embed

    
class EGTN(nn.Module):
    def __init__(self, n_layers, node_dim, ft_dim, edge_dim, hidden_dim, time_emb_dim, act_fn,
                 scale=1, conditioning='none', pre_norm=False, **kwargs):
        super().__init__()
        print(f"EGTN Initialized with {conditioning}")
        self.s_modules = nn.ModuleList()
        self.t_modules = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.n_layers = n_layers
        self.time_emb_dim = time_emb_dim
        self.input_linear = nn.Linear(node_dim + time_emb_dim, hidden_dim)
        self.scale = scale
        self.embedding = Embedding(node_dim, ft_dim, edge_dim)
        self.conditioning = conditioning

        # Parse activation
        if act_fn == 'silu':
            act_fn = nn.SiLU()
        else:
            raise NotImplementedError(act_fn)

        for i in range(n_layers):
            self.s_modules.append(
                ESLayer(node_dim=hidden_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, act_fn=act_fn,
                        normalize=True, pre_norm=pre_norm)
            )

            self.t_modules.append(
                ETLayer(node_dim=hidden_dim, hidden_dim=hidden_dim, act_fn=act_fn, time_emb_dim=time_emb_dim)
            )

            self.norm1.append(nn.LayerNorm(hidden_dim))
            if i < n_layers - 1:
                self.norm2.append(nn.LayerNorm(hidden_dim))

    def forward(self, diffusion_t, x, h, f, edge_index, edge_attr, batch, **model_kwargs):
        """
        :param diffusion_t: The diffusion time step, shape [BN,]
        :param x: shape [BN, 3, T]
        :param h: shape [BN] 
        :param f: shape [BN, 10]
        :param edge_index: shape [2, BM]
        :param edge_attr: shape [BM]
        :param batch: shape [BN]
        """
        # Run through the embedding layer
        h, edge_attr = self.embedding(f, h, edge_attr, x)

        # Get condition mask and concat the condition frames
        cond_mask = model_kwargs.get('cond_mask', None)  # [1, 1, T]
        if self.conditioning != "none":
            c = int(cond_mask.sum().item())
            if self.conditioning == "forward":
                assert c == 1 
            elif self.conditioning == "forward_autoreg":
                assert c == 1
            elif self.conditioning == "unconditional_forward":
                assert c == 1 or c == 0
                if model_kwargs.get('uncond', False):
                    cond_mask = cond_mask[1:]
                    model_kwargs['original_frames'] = model_kwargs['original_frames'][..., 1:]
                    T = cond_mask.shape[0]
                    c = 0
            elif self.conditioning == "unconditional_forward_autoreg":
                assert c == 1 or c == 0
                if model_kwargs.get('uncond', False):
                    cond_mask = cond_mask[1:]
                    model_kwargs['original_frames'] = model_kwargs['original_frames'][..., 1:]
                    T = cond_mask.shape[0]
                    c = 0
            elif self.conditioning == "interpolation":
                assert c == 2
            else:
                raise NotImplementedError()
            cond_mask = cond_mask.view(-1).bool()
            x_given = model_kwargs['original_frames']
            assert x.shape[-1] + c == x_given.shape[-1] 
            x_input = x  # Record x in order to subtract it in the end for translation invariance
            x = x_given.clone()
            x[:, :, ~cond_mask] = x_input
        else:
            assert self.conditioning == 'none'
            x_input = x

        x = x * self.scale

        T = x.size(-1)
        diffusion_t = get_timestep_embedding(diffusion_t, embedding_dim=self.time_emb_dim)  # [BN, Ht]
        diffusion_t = diffusion_t.unsqueeze(-1).repeat(1, 1, T)  # [BN, Ht, T]
        t_emb = diffusion_t

        if h.dim() == 2:
            h = h.unsqueeze(-1).repeat(1, 1, T)
        else:
            pass
        
        h = torch.cat((h, t_emb), dim=1)  # [BN, Hh+Ht+Ht, T]
        h = separate_time_dim(self.input_linear(merge_time_dim(h)), t=T)  # [BN, H, T]
        if edge_attr is not None:
            edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, T)  # [BM, He, T]

        for i in range(self.n_layers):
            x, h = self.s_modules[i](x, h, edge_index, edge_attr, batch, **model_kwargs)
            h = h.transpose(1, 2)
            h = self.norm1[i](h)                 
            h = h.transpose(1, 2) 
            x, h = self.t_modules[i](x, h)
            if i < self.n_layers -1:
                h = h.transpose(1, 2)
                h = self.norm2[i](h)                 
                h = h.transpose(1, 2)
                
        # Clip the output through the conditional mask
        if cond_mask is not None:
            x = x[..., ~cond_mask]
            h = h[..., ~cond_mask]

        # Let x be translation invariant
        x = x - x_input

        x = x / self.scale

        return x, h
    
class BasicES(nn.Module):
    def __init__(
            self, 
            n_layers, node_dim, 
            ft_dim, edge_dim, 
            hidden_dim, time_emb_dim, 
            act_fn,
            scale=1, pre_norm=False, **kwargs):
        super().__init__()
        self.s_modules = nn.ModuleList()
        self.n_layers = n_layers
        self.time_emb_dim = time_emb_dim
        self.input_linear = nn.Linear(node_dim + time_emb_dim, hidden_dim)
        self.scale = scale
        self.embedding = Embedding(node_dim, ft_dim, edge_dim)

        # Parse activation
        if act_fn == 'silu':
            act_fn = nn.SiLU()
        else:
            raise NotImplementedError(act_fn)
        
        # Initialize the Spatial Layers
        for _ in range(n_layers):
            self.s_modules.append(
                BasicESLayer(node_dim=hidden_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, act_fn=act_fn,
                        normalize=True, pre_norm=pre_norm)
            )

    def forward(self, diffusion_t, x, h, f, edge_index, edge_attr, batch, **model_kwargs):
        """
        :param diffusion_t: The diffusion time step, shape [BN,]
        :param x: shape [BN, 3, T]
        :param h: shape [BN] 
        :param f: shape [BN, 10]
        :param edge_index: shape [2, BM]
        :param edge_attr: shape [BM]
        :param batch: shape [BN]
        """

        # Run through the embedding layer
        h, edge_attr = self.embedding(f, h, edge_attr, x)

        # Get condition mask and concat the condition frames
        x_input = x

        x = x * self.scale

        T = x.size(-1)
        diffusion_t = get_timestep_embedding(diffusion_t, embedding_dim=self.time_emb_dim)  # [BN, Ht]
        diffusion_t = diffusion_t.unsqueeze(-1).repeat(1, 1, T)  # [BN, Ht, T]
        t_emb = diffusion_t

        if h.dim() == 2:
            h = h.unsqueeze(-1).repeat(1, 1, T)
        else:
            pass
        
        h = torch.cat((h, t_emb), dim=1)  # [BN, Hh+Ht+Ht, T]
        h = separate_time_dim(self.input_linear(merge_time_dim(h)), t=T)  # [BN, H, T]
        if edge_attr is not None:
            edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, T)  # [BM, He, T]

        for i in range(self.n_layers):
            x, h = self.s_modules[i](x, h, edge_index, edge_attr, batch, **model_kwargs)

        # Let x be translation invariant
        x = x - x_input

        x = x / self.scale

        return x, h
    
class EGInterpolator(nn.Module):
    def __init__(
        self,
        n_layers: int,
        node_dim: int,
        ft_dim: int,
        edge_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        act_fn: str,
        scale: float = 1.0,
        pre_norm: bool = False,
        set_zero: bool = False,
        use_mha: bool = True,
        num_heads: int = 8,
        conditioning: str = 'none',
        use_kabsch: bool = False,
        use_extra: bool = False,
        use_double: bool = False,
        use_norm: bool = False,
        sigmoid_init: bool = False,
        residual: bool = False,
        **kwargs
    ):
        super().__init__()
        # Initialization Parameters
        self.n_layers = n_layers
        self.scale = scale
        self.set_zero = set_zero
        self.time_emb_dim = time_emb_dim
        self.conditioning = conditioning
        self.use_kabsch = use_kabsch # Refers to using the internal kabsch alignment --> will need to change the loss for this
        self.use_extra = use_extra # Refers to extra spatial layer
        self.sigmoid_init = sigmoid_init # If true, will set alpha = 0.5
        self.use_double = use_double # Refers to double temporal layer
        self.use_norm = use_norm
        self.residual = residual
        # Embeddings and input projection
        self.input_linear = nn.Linear(node_dim + time_emb_dim, hidden_dim)
        self.embedding = Embedding(node_dim, ft_dim, edge_dim)
        if conditioning == 'none':
            self.cond_embedding = nn.Embedding(1, hidden_dim)
        else:
            self.cond_embedding = nn.Embedding(2, hidden_dim)

        # Activation
        act = nn.SiLU() if act_fn == 'silu' else None
        if act is None:
            raise NotImplementedError(f"act_fn={act_fn}")

        # Spatial layers
        self.s_modules = nn.ModuleList([
            BasicESLayer(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                act_fn=act,
                normalize=True,
                pre_norm=pre_norm
            )
            for _ in range(n_layers)
        ])

        # Interpolation parameters
        if not self.residual:
            if self.sigmoid_init:
                self.alpha_h_t = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_layers - 1)]) # n - 1 to avoid DDP issue
                self.alpha_x_t = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_layers)])
                if use_extra:
                    self.alpha_h_s = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_layers)])
                    self.alpha_x_s = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_layers)])
                if use_double:
                    self.alpha_h_t_2 = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_layers)])
                    self.alpha_x_t_2 = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_layers)])
            else:
                self.alpha_h_t = nn.ParameterList([nn.Parameter(torch.tensor(5.0)) for _ in range(n_layers - 1)]) # n - 1 to avoid DDP issue
                self.alpha_x_t = nn.ParameterList([nn.Parameter(torch.tensor(5.0)) for _ in range(n_layers)])
                if use_extra:
                    self.alpha_h_s = nn.ParameterList([nn.Parameter(torch.tensor(5.0)) for _ in range(n_layers)])
                    self.alpha_x_s = nn.ParameterList([nn.Parameter(torch.tensor(5.0)) for _ in range(n_layers)])
                if use_double:
                    self.alpha_h_t_2 = nn.ParameterList([nn.Parameter(torch.tensor(5.0)) for _ in range(n_layers)])
                    self.alpha_x_t_2 = nn.ParameterList([nn.Parameter(torch.tensor(5.0)) for _ in range(n_layers)])

            # … after self.alpha_x_t / self.alpha_h_t are assigned …
            print("alpha_x_t values:", [p.item() for p in self.alpha_x_t]) # sanity check on init
            print("alpha_h_t values:", [p.item() for p in self.alpha_h_t])

        # This is the norm for the final temporal layer mixing
        if self.use_norm:
            self.norm3 = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(n_layers - 1)
            ])
        # Initialize the second spatial layer
        if use_extra:
            self.s_modules_new = nn.ModuleList()
            # Get a norm for this mixing as well
            if self.use_norm:
                self.norm2 = nn.ModuleList([
                    nn.LayerNorm(hidden_dim) for _ in range(n_layers)
                ])
        # Create the second temporal layer
        if use_double:
            self.t_modules_2 = nn.ModuleList()
            # Get the norm for this mixing as well
            if self.use_norm:
                self.norm1 = nn.ModuleList([
                    nn.LayerNorm(hidden_dim) for _ in range(n_layers)
                ])

        # Set the layers
        self.t_modules = nn.ModuleList()
        for _ in range(n_layers):
            if use_extra:
                self.s_modules_new.append(
                    BasicESLayer(
                        node_dim=hidden_dim,
                        edge_dim=edge_dim,
                        hidden_dim=hidden_dim,
                        act_fn=act,
                        normalize=True,
                        pre_norm=pre_norm
                    )
                )
            if use_double:
                self.t_modules_2.append(
                    ETLayer(
                        node_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        act_fn=act,
                        time_emb_dim=time_emb_dim
                    )
                )
            if use_mha:
                kq_dim = hidden_dim // num_heads
                v_dim = hidden_dim // num_heads
                self.t_modules.append(
                    ETAttention(
                        node_dim=hidden_dim,
                        kq_dim=kq_dim,
                        v_dim=v_dim,
                        n_heads=num_heads,
                        act_fn=act
                    )
                )
            else:
                self.t_modules.append(
                    ETLayer(
                        node_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        act_fn=act,
                        time_emb_dim=time_emb_dim
                    )
                )

    def forward(self, diffusion_t, x, h, f, edge_index, edge_attr, batch, **model_kwargs):
        # Embedding
        h, edge_attr = self.embedding(f, h, edge_attr, x)

        # Get the conditioning
        device = x.device
        BN, Tc = x.size(0), x.size(-1)

        if 'cond_mask' in model_kwargs:
            cond_mask = model_kwargs['cond_mask'].to(device)
            T = cond_mask.shape[0]
            c = int(cond_mask.sum().item())
        else:
            cond_mask = torch.zeros(Tc, device=device)
            c = 0
            T = cond_mask.shape[0]

        if self.conditioning == 'none':
            assert c == 0
        elif self.conditioning == 'forward':
            assert c == 1
        elif self.conditioning == 'interpolation':
            assert c == 2
        elif self.conditioning == 'unconditional_forward':
            assert c == 1 or c == 0
            if model_kwargs.get('uncond', False):
                cond_mask = cond_mask[1:]
                model_kwargs['original_frames'] = model_kwargs['original_frames'][..., 1:]
                T = cond_mask.shape[0]
                c = 0
        else:
            raise NotImplementedError("Invalid Conditioning Setting")
        assert T == Tc + c
        B = int(batch.max().item()) + 1
        basic = torch.tensor(1.0, device=device)

        # Get the original frames for conditioning
        if 'original_frames' in model_kwargs:
            x_given = model_kwargs['original_frames'].to(device)
        else:
            x_given = x
        cond_mask = cond_mask.bool()
        cond_mask_batched = cond_mask.view(1, -1).expand(BN, -1)
        cond_emb = self.cond_embedding(cond_mask_batched.long()).permute(0, 2, 1)

        # Switch out the information --> Conditioning + Noise
        x_input = x
        x = x_given.clone()
        x[:, :, ~cond_mask] = x_input

        # Time embedding
        t_emb = get_timestep_embedding(diffusion_t, embedding_dim=self.time_emb_dim)
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, T)

        # Add time embedding and smear across T
        if h.dim() == 2:
            h = h.unsqueeze(-1).repeat(1, 1, T)
        h = torch.cat((h, t_emb), dim=1)
        h = separate_time_dim(self.input_linear(merge_time_dim(h)), t=T)
        if edge_attr is not None:
            edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, T)

        # Ensure A-invariant subspace
        if self.use_kabsch:
            x = naive_batched_frame_kabsch(x, batch) # This puts all the x into the orbit of frame 0

        # Iterate the blocks
        for i in range(self.n_layers):
            # Spatial update from pretrained layer
            # Only pass in the unconditonal frames here
            x_s = x[:, :, ~cond_mask] 
            h_s = h[:, :, ~cond_mask]
            if edge_attr is not None:
                edge_attr_s = edge_attr[:, :, ~cond_mask]
            else:
                edge_attr_s = None
            assert x_s.shape[-1] == Tc
            assert h_s.shape[-1] == Tc
            assert edge_attr_s.shape[-1] == Tc
            x_s, h_s = self.s_modules[i](x_s, h_s, edge_index, edge_attr_s, batch, **model_kwargs)

            # Update the global trajectory state
            x[:, :, ~cond_mask] = x_s
            h[:, :, ~cond_mask] = h_s

            # Add conditional embedding
            if i == 0:
                h = h + cond_emb

            # Assert dim and pass through
            assert x.shape[-1] == T
            assert h.shape[-1] == T

            # Ensure A-invariant subspace
            if self.use_kabsch:
                x = naive_batched_frame_kabsch(x, batch) # This puts all the x into the orbit of frame 0

            # First temporal layer if needed
            if self.use_double:
                # Extract out temporal outputs
                x_t, h = self.t_modules_2[i](x, h)

                # By default update all the frame h's but only update non-conditioning frame x's
                x[:, :, ~cond_mask] = x_t[:, :, ~cond_mask]

                # Add the layer norm on the invariant features
                if self.use_norm:
                    h = h.transpose(1, 2)
                    h = self.norm1[i](h)                 
                    h = h.transpose(1, 2) 

                # Extract out the non-conditional frames for interpolation
                x_t = x[:, :, ~cond_mask]
                h_t = h[:, :, ~cond_mask]
                assert h_t.shape[-1] == Tc
                assert x_t.shape[-1] == Tc

                # Get the mixing factors
                if not self.residual:
                    alpha_x_i_t = basic if self.set_zero else torch.sigmoid(self.alpha_x_t_2[i])
                    alpha_h_i_t = basic if self.set_zero else torch.sigmoid(self.alpha_h_t_2[i])

                # Use framewise pairwise Kabsch alignment
                # This is a strong condition, such that we do not require x_t
                # to be in the A-invariant subspace, since x_t will be rotated to match x_s anyways
                if self.use_kabsch:
                    x_t = naive_batched_pairwise_kabsch(x_t, x_s, batch)

                # Interpolation between spatial and temporal
                if not self.residual:
                    x_s_t = alpha_x_i_t * x_s + (1 - alpha_x_i_t) * x_t
                    h_s_t = alpha_h_i_t * h_s + (1 - alpha_h_i_t) * h_t
                else:
                    x_s_t = 0.5 * x_s + 0.5 * x_t
                    h_s_t = 0.5 * h_s + 0.5 * h_t

                # Update the global trajectory state
                # This will retain x_s and h_s if alpha = 1
                x[:, :, ~cond_mask] = x_s_t
                h[:, :, ~cond_mask] = h_s_t

                # Ensure A-invariant subspace
                if self.use_kabsch:
                    x = naive_batched_frame_kabsch(x, batch)

            # Second spatial layer if needed
            if self.use_extra:
                # Spatial Update on all the frames
                x_s_n, h = self.s_modules_new[i](x, h, edge_index, edge_attr, batch, **model_kwargs)

                # Only keep the updates to the non-conditional frame x's
                x[:, :, ~cond_mask] = x_s_n[:, :, ~cond_mask]

                # Second Norm
                if self.use_norm:
                    h = h.transpose(1, 2)
                    h = self.norm2[i](h)                 
                    h = h.transpose(1, 2) 

                # Extract out the non-conditional frames for interpolation
                x_s_n = x[:, :, ~cond_mask]
                h_s_n = h[:, :, ~cond_mask]
                assert h_s_n.shape[-1] == Tc
                assert x_s_n.shape[-1] == Tc

                # Get the mixing factors 1
                if not self.residual:
                    alpha_x_i_s = basic if self.set_zero else torch.sigmoid(self.alpha_x_s[i])
                    alpha_h_i_s = basic if self.set_zero else torch.sigmoid(self.alpha_h_s[i])

                # Use framewise pairwise Kabsch alignment
                # This is a strong condition, such that we do not require x_s_n
                # to be in the A-invariant subspace, since x_s_n will be rotated to match x_s anyways
                if self.use_kabsch:
                    x_s_n = naive_batched_pairwise_kabsch(x_s_n, x_s, batch)

                # Interpolation number 1
                if not self.residual:
                    x_s_n = alpha_x_i_s * x_s + (1 - alpha_x_i_s) * x_s_n
                    h_s_n = alpha_h_i_s * h_s + (1 - alpha_h_i_s) * h_s_n
                else:
                    x_s_n = 0.5 * x_s + 0.5 * x_s_n
                    h_s_n = 0.5 * h_s + 0.5 * h_s_n

                # Update global trajectory state
                # This will retain x_s and h_s if alpha = 1
                x[:, :, ~cond_mask] = x_s_n
                h[:, :, ~cond_mask] = h_s_n

                # Ensure A-invariant subspace
                if self.use_kabsch:
                    x = naive_batched_frame_kabsch(x, batch)

            # Temporal update
            assert x.shape[-1] == T
            assert h.shape[-1] == T
            x_t, h = self.t_modules[i](x, h)
            x[:, :, ~cond_mask] = x_t[:, :, ~cond_mask]

            # Do the layer norm
            if self.use_norm and i < self.n_layers - 1:
                h = h.transpose(1, 2)
                h = self.norm3[i](h)                 
                h = h.transpose(1, 2) 

            # Extract out the non-conditional frames for interpolation
            x_t = x[:, :, ~cond_mask]
            h_t = h[:, :, ~cond_mask]
            assert h_t.shape[-1] == Tc
            assert x_t.shape[-1] == Tc

            # Get the mixing factors
            if not self.residual:
                alpha_x_i_t = basic if self.set_zero else torch.sigmoid(self.alpha_x_t[i])
                alpha_h_i_t = basic if (self.set_zero or i == self.n_layers - 1) else torch.sigmoid(self.alpha_h_t[i]) # DDP aware
            
            # Framewise alignment before interpolation
            if self.use_kabsch:
                x_t = naive_batched_pairwise_kabsch(x_t, x_s, batch)

            # Interpolation between spatial and temporal
            if not self.residual:
                x_i = alpha_x_i_t * x_s + (1 - alpha_x_i_t) * x_t
                h_i = alpha_h_i_t * h_s + (1 - alpha_h_i_t) * h_t
            else:
                x_i = 0.5 * x_s + 0.5 * x_t
                h_i = 0.5 * h_s + 0.5 * h_t

            # Update the global trajectory state
            x = x.clone()
            h = h.clone()
            x[:, :, ~cond_mask] = x_i
            h[:, :, ~cond_mask] = h_i

            # Check shapes
            assert x.shape[-1] == T
            assert h.shape[-1] == T
            
            # Project back to A-invariant subspace
            if self.use_kabsch:
                x = naive_batched_frame_kabsch(x, batch)

        # Final extraction
        h = h[:, :, ~cond_mask]
        x_pred = x[:, :, ~cond_mask]
        assert x_pred.shape == x_input.shape

        # Do a final global alignment with the world frame 
        if self.use_kabsch:
            x_pred = naive_batched_pairwise_kabsch(x_pred, x_input, batch)

        # Substract and Scale
        x_out = (x_pred - x_input) / self.scale

        return x_out, h


class EGStack(nn.Module):
    def __init__(
        self,
        n_layers: int,
        node_dim: int,
        ft_dim: int,
        edge_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        act_fn: str,
        scale: float = 1.0,
        pre_norm: bool = False,
        num_heads: int = 8,
        conditioning: str = 'none',
        use_kabsch: bool = False,
        use_norm: bool = True,
        **kwargs
    ):
        super().__init__()
        # Initialization Parameters
        self.n_layers = n_layers
        self.scale = scale
        self.time_emb_dim = time_emb_dim
        self.conditioning = conditioning
        self.use_kabsch = use_kabsch
        self.use_norm = use_norm
        
        # Embeddings and input projection
        self.input_linear = nn.Linear(node_dim + time_emb_dim, hidden_dim)
        self.embedding = Embedding(node_dim, ft_dim, edge_dim)
        
        if conditioning == 'none':
            self.cond_embedding = nn.Embedding(1, hidden_dim)
        else:
            self.cond_embedding = nn.Embedding(2, hidden_dim)

        # Activation
        act = nn.SiLU() if act_fn == 'silu' else None
        if act is None:
            raise NotImplementedError(f"act_fn={act_fn}")

        # PHASE 1: Pretrained Spatial layers (loaded from BasicES checkpoint)
        self.s_modules = nn.ModuleList([
            BasicESLayer(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                act_fn=act,
                normalize=True,
                pre_norm=pre_norm
            )
            for _ in range(n_layers)
        ])

        # PHASE 2: Temporal-Spatial-Temporal blocks (uses ETLayer for fair comparison)
        self.tst_blocks = nn.ModuleList()
        
        for _ in range(n_layers):
            block = nn.ModuleDict()
            
            # Temporal layer 1
            block['temporal_1'] = ETLayer(
                node_dim=hidden_dim,
                hidden_dim=hidden_dim,
                act_fn=act,
                time_emb_dim=time_emb_dim
            )
            
            # Spatial layer
            block['spatial'] = BasicESLayer(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                act_fn=act,
                normalize=True,
                pre_norm=pre_norm
            )
            
            # Temporal layer 2
            block['temporal_2'] = ETLayer(
                node_dim=hidden_dim,
                hidden_dim=hidden_dim,
                act_fn=act,
                time_emb_dim=time_emb_dim
            )
            
            self.tst_blocks.append(block)
        
        # Layer norms for stability (after each sub-layer)
        # Note: norm_t2 is not used in the last layer since h is not used in the loss
        if self.use_norm:
            self.norms = nn.ModuleList()
            for i in range(n_layers):
                norm_dict = nn.ModuleDict({
                    'norm_t1': nn.LayerNorm(hidden_dim),
                    'norm_s': nn.LayerNorm(hidden_dim),
                })
                if i < n_layers - 1:
                    norm_dict['norm_t2'] = nn.LayerNorm(hidden_dim)
                self.norms.append(norm_dict)

    def forward(self, diffusion_t, x, h, f, edge_index, edge_attr, batch, **model_kwargs):
        # Embedding
        h, edge_attr = self.embedding(f, h, edge_attr, x)

        # Get the conditioning
        device = x.device
        BN, Tc = x.size(0), x.size(-1)

        if 'cond_mask' in model_kwargs:
            cond_mask = model_kwargs['cond_mask'].to(device)
            T = cond_mask.shape[0]
            c = int(cond_mask.sum().item())
        else:
            cond_mask = torch.zeros(Tc, device=device)
            c = 0
            T = cond_mask.shape[0]

        # Conditioning validation
        if self.conditioning == 'none':
            assert c == 0
        elif self.conditioning == 'forward':
            assert c == 1
        elif self.conditioning == 'interpolation':
            assert c == 2
        elif self.conditioning == 'unconditional_forward':
            assert c == 1 or c == 0
            if model_kwargs.get('uncond', False):
                cond_mask = cond_mask[1:]
                model_kwargs['original_frames'] = model_kwargs['original_frames'][..., 1:]
                T = cond_mask.shape[0]
                c = 0
        else:
            raise NotImplementedError("Invalid Conditioning Setting")
        
        assert T == Tc + c
        B = int(batch.max().item()) + 1

        # Get the original frames for conditioning
        if 'original_frames' in model_kwargs:
            x_given = model_kwargs['original_frames'].to(device)
        else:
            x_given = x
        
        cond_mask = cond_mask.bool()
        cond_mask_batched = cond_mask.view(1, -1).expand(BN, -1)
        cond_emb = self.cond_embedding(cond_mask_batched.long()).permute(0, 2, 1)

        # Switch out the information --> Conditioning + Noise
        x_input = x
        x = x_given.clone()
        x[:, :, ~cond_mask] = x_input

        # Time embedding
        t_emb = get_timestep_embedding(diffusion_t, embedding_dim=self.time_emb_dim)
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, T)

        # Add time embedding and smear across T
        if h.dim() == 2:
            h = h.unsqueeze(-1).repeat(1, 1, T)
        h = torch.cat((h, t_emb), dim=1)
        h = separate_time_dim(self.input_linear(merge_time_dim(h)), t=T)
        if edge_attr is not None:
            edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, T)

        # Ensure A-invariant subspace
        if self.use_kabsch:
            x = naive_batched_frame_kabsch(x, batch)

        # ========================================
        # PHASE 1: Run all pretrained spatial layers first
        # ========================================
        for i in range(self.n_layers):
            # Only process unconditional frames
            x_s = x[:, :, ~cond_mask] 
            h_s = h[:, :, ~cond_mask]
            if edge_attr is not None:
                edge_attr_s = edge_attr[:, :, ~cond_mask]
            else:
                edge_attr_s = None
            
            assert x_s.shape[-1] == Tc
            assert h_s.shape[-1] == Tc
            if edge_attr_s is not None:
                assert edge_attr_s.shape[-1] == Tc
            
            x_s, h_s = self.s_modules[i](x_s, h_s, edge_index, edge_attr_s, batch, **model_kwargs)
            
            # Update only unconditional frames
            x[:, :, ~cond_mask] = x_s
            h[:, :, ~cond_mask] = h_s
            
            # Add conditional embedding (only once)
            if i == 0:
                h = h + cond_emb
            
            # Ensure A-invariant subspace
            if self.use_kabsch:
                x = naive_batched_frame_kabsch(x, batch)

        # ========================================
        # PHASE 2: Temporal-Spatial-Temporal blocks
        # ========================================
        for i in range(self.n_layers):
            block = self.tst_blocks[i]
            is_last_layer = (i == self.n_layers - 1)
            
            # Temporal layer 1 with residual connection
            h_residual = h.clone()
            x_t, h_delta = block['temporal_1'](x, h)
            # Avoid in-place operation - clone and update
            x = x.clone()
            x[:, :, ~cond_mask] = x_t[:, :, ~cond_mask]
            h = h_residual + h_delta  # Residual connection
            
            if self.use_norm:
                h = h.transpose(1, 2)
                h = self.norms[i]['norm_t1'](h)
                h = h.transpose(1, 2)
            
            if self.use_kabsch:
                x = naive_batched_frame_kabsch(x, batch)
            
            # Spatial layer with residual connection
            h_residual = h.clone()
            x_s = x[:, :, ~cond_mask] 
            h_s = h[:, :, ~cond_mask]
            if edge_attr is not None:
                edge_attr_s = edge_attr[:, :, ~cond_mask]
            else:
                edge_attr_s = None
            
            x_s, h_s_delta = block['spatial'](x_s, h_s, edge_index, edge_attr_s, batch, **model_kwargs)
            # Avoid in-place operation - clone and update
            x = x.clone()
            h = h.clone()
            x[:, :, ~cond_mask] = x_s
            h[:, :, ~cond_mask] = h_residual[:, :, ~cond_mask] + h_s_delta
            
            if self.use_norm:
                h = h.transpose(1, 2)
                h = self.norms[i]['norm_s'](h)
                h = h.transpose(1, 2)
            
            if self.use_kabsch:
                x = naive_batched_frame_kabsch(x, batch)
            
            # Temporal layer 2 with residual connection
            h_residual = h.clone()
            x_t, h_delta = block['temporal_2'](x, h)
            # Avoid in-place operation - clone and update
            x = x.clone()
            x[:, :, ~cond_mask] = x_t[:, :, ~cond_mask]
            h = h_residual + h_delta  # Residual connection
            
            # Skip norm on the last layer's final temporal output since h is not used in loss
            if self.use_norm and not is_last_layer:
                h = h.transpose(1, 2)
                h = self.norms[i]['norm_t2'](h)
                h = h.transpose(1, 2)
            
            if self.use_kabsch:
                x = naive_batched_frame_kabsch(x, batch)

        # Final extraction
        h = h[:, :, ~cond_mask]
        x_pred = x[:, :, ~cond_mask]
        assert x_pred.shape == x_input.shape

        # Do a final global alignment with the world frame 
        if self.use_kabsch:
            x_pred = naive_batched_pairwise_kabsch(x_pred, x_input, batch)

        # Subtract and Scale
        x_out = (x_pred - x_input) / self.scale

        return x_out, h


class EGInterpolatorSimple(nn.Module):
    def __init__(
        self,
        n_layers: int,
        node_dim: int,
        ft_dim: int,
        edge_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        act_fn: str,
        scale: float = 1.0,
        pre_norm: bool = False,
        set_zero: bool = False,
        use_mha: bool = True,
        num_heads: int = 8,
        conditioning: str = 'none',
        use_kabsch: bool = False,
        use_extra: bool = False,
        use_double: bool = False,
        use_norm: bool = False,
        sigmoid_init: bool = False,
        **kwargs
    ):
        super().__init__()
        # Initialization Parameters
        self.n_layers = n_layers
        self.scale = scale
        self.set_zero = set_zero
        self.time_emb_dim = time_emb_dim
        self.conditioning = conditioning
        self.use_kabsch = use_kabsch # Refers to using the internal kabsch alignment --> will need to change the loss for this
        self.use_extra = use_extra # Refers to extra spatial layer
        self.sigmoid_init = sigmoid_init # If true, will set alpha = 0.5
        self.use_double = use_double # Refers to double temporal layer
        self.use_norm = use_norm

        # Embeddings and input projection
        self.input_linear = nn.Linear(node_dim + time_emb_dim, hidden_dim)
        self.embedding = Embedding(node_dim, ft_dim, edge_dim)
        if conditioning == 'none':
            self.cond_embedding = nn.Embedding(1, hidden_dim)
        else:
            self.cond_embedding = nn.Embedding(2, hidden_dim)

        # Activation
        act = nn.SiLU() if act_fn == 'silu' else None
        if act is None:
            raise NotImplementedError(f"act_fn={act_fn}")

        # Spatial layers
        self.s_modules = nn.ModuleList([
            BasicESLayer(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                act_fn=act,
                normalize=True,
                pre_norm=pre_norm
            )
            for _ in range(n_layers)
        ])

        # Interpolation parameters
        if self.sigmoid_init:
            self.alpha_h_t = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_layers - 1)]) # n - 1 to avoid DDP issue
            self.alpha_x_t = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(n_layers)])
        else:
            self.alpha_h_t = nn.ParameterList([nn.Parameter(torch.tensor(5.0)) for _ in range(n_layers - 1)]) # n - 1 to avoid DDP issue
            self.alpha_x_t = nn.ParameterList([nn.Parameter(torch.tensor(5.0)) for _ in range(n_layers)])

        # … after self.alpha_x_t / self.alpha_h_t are assigned …
        print("alpha_x_t values:", [p.item() for p in self.alpha_x_t]) # sanity check on init
        print("alpha_h_t values:", [p.item() for p in self.alpha_h_t])

        # This is the norm for the final temporal layer mixing
        self.norm3 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers - 1)
        ])
        # Initialize the second spatial layer
        self.s_modules_new = nn.ModuleList()
        # Get a norm for this mixing as well
        self.norm2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        # Create the second temporal layer
        self.t_modules_2 = nn.ModuleList()
        # Get the norm for this mixing as well
        self.norm1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Set the layers
        self.t_modules = nn.ModuleList()
        for _ in range(n_layers):
            if use_extra:
                self.s_modules_new.append(
                    BasicESLayer(
                        node_dim=hidden_dim,
                        edge_dim=edge_dim,
                        hidden_dim=hidden_dim,
                        act_fn=act,
                        normalize=True,
                        pre_norm=pre_norm
                    )
                )
            if use_double:
                self.t_modules_2.append(
                    ETLayer(
                        node_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        act_fn=act,
                        time_emb_dim=time_emb_dim
                    )
                )
            if use_mha:
                kq_dim = hidden_dim // num_heads
                v_dim = hidden_dim // num_heads
                self.t_modules.append(
                    ETAttention(
                        node_dim=hidden_dim,
                        kq_dim=kq_dim,
                        v_dim=v_dim,
                        n_heads=num_heads,
                        act_fn=act
                    )
                )
            else:
                self.t_modules.append(
                    ETLayer(
                        node_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        act_fn=act,
                        time_emb_dim=time_emb_dim
                    )
                )

    def forward(self, diffusion_t, x, h, f, edge_index, edge_attr, batch, **model_kwargs):
        # Embedding
        h, edge_attr = self.embedding(f, h, edge_attr, x)

        # Get the conditioning
        device = x.device
        BN, Tc = x.size(0), x.size(-1)

        if 'cond_mask' in model_kwargs:
            cond_mask = model_kwargs['cond_mask'].to(device)
            T = cond_mask.shape[0]
            c = int(cond_mask.sum().item())
        else:
            cond_mask = torch.zeros(Tc, device=device)
            c = 0
            T = cond_mask.shape[0]

        if self.conditioning == 'none':
            assert c == 0
        elif self.conditioning == 'forward':
            assert c == 1
        elif self.conditioning == 'interpolation':
            assert c == 2
        elif self.conditioning == 'forward_uncond':
            assert c == 1
            if model_kwargs['uncond']:
                cond_mask = cond_mask[1:]
                model_kwargs['original_frames'] = model_kwargs['original_frames'][..., 1:]
                T = cond_mask.shape[0]
                c = 0
        else:
            raise NotImplementedError("Invalid Conditioning Setting")
        assert T == Tc + c
        B = int(batch.max().item()) + 1
        basic = torch.tensor(1.0, device=device)

        # Get the original frames for conditioning
        if 'original_frames' in model_kwargs:
            x_given = model_kwargs['original_frames'].to(device)
        else:
            x_given = x
        cond_mask = cond_mask.bool()
        cond_mask_batched = cond_mask.view(1, -1).expand(BN, -1)
        cond_emb = self.cond_embedding(cond_mask_batched.long()).permute(0, 2, 1)

        # Switch out the information --> Conditioning + Noise
        x_input = x
        x = x_given.clone()
        x[:, :, ~cond_mask] = x_input

        # Time embedding
        t_emb = get_timestep_embedding(diffusion_t, embedding_dim=self.time_emb_dim)
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, T)

        # Add time embedding and smear across T
        if h.dim() == 2:
            h = h.unsqueeze(-1).repeat(1, 1, T)
        h = torch.cat((h, t_emb), dim=1)
        h = separate_time_dim(self.input_linear(merge_time_dim(h)), t=T)
        if edge_attr is not None:
            edge_attr = edge_attr.unsqueeze(-1).repeat(1, 1, T)

        # Iterate the blocks
        for i in range(self.n_layers):

            # Part 1 Initial Spatial Update
            # -----------------------------
            # Spatial update from pretrained layer
            # Only pass in the unconditonal frames here
            x_s = x[:, :, ~cond_mask] 
            h_s = h[:, :, ~cond_mask]
            if edge_attr is not None:
                edge_attr_s = edge_attr[:, :, ~cond_mask]
            else:
                edge_attr_s = None
            assert x_s.shape[-1] == Tc
            assert h_s.shape[-1] == Tc
            assert edge_attr_s.shape[-1] == Tc
            x_s, h_s = self.s_modules[i](x_s, h_s, edge_index, edge_attr_s, batch, **model_kwargs)

            # Update the global trajectory state
            x[:, :, ~cond_mask] = x_s
            h[:, :, ~cond_mask] = h_s

            # Add conditional embedding
            if i == 0:
                h = h + cond_emb

            # Assert dim and pass through
            assert x.shape[-1] == T
            assert h.shape[-1] == T

            # Part 2 Interpolation Updates
            # -----------------------------
            # Make a clone for the interpolation
            x_temp = x.clone()
            h_temp = h.clone()

            # First temporal layer
            # Extract out temporal outputs
            x_t, h_temp = self.t_modules_2[i](x_temp, h_temp)

            # By default update all the frame h's but only update non-conditioning frame x's
            x_temp[:, :, ~cond_mask] = x_t[:, :, ~cond_mask]

            # Add the layer norm on the invariant features
            if self.use_norm:
                h_temp = h_temp.transpose(1, 2)
                h_temp = self.norm1[i](h_temp)                 
                h_temp = h_temp.transpose(1, 2) 

            # Second spatial layer if needed
            # Spatial Update on all the frames
            x_s_n, h_temp = self.s_modules_new[i](x_temp, h_temp, edge_index, edge_attr, batch, **model_kwargs)
            # Only keep the updates to the non-conditional frame x's
            x_temp[:, :, ~cond_mask] = x_s_n[:, :, ~cond_mask]

            # Second Norm
            if self.use_norm:
                h_temp = h_temp.transpose(1, 2)
                h_temp = self.norm2[i](h_temp)                 
                h_temp = h_temp.transpose(1, 2) 

            # Third temporal layer
            assert x_temp.shape[-1] == T
            assert h_temp.shape[-1] == T
            x_t, h_temp = self.t_modules[i](x_temp, h_temp)
            x_temp[:, :, ~cond_mask] = x_t[:, :, ~cond_mask]

            # Do the layer norm
            if self.use_norm and i < self.n_layers - 1:
                h_temp = h_temp.transpose(1, 2)
                h_temp = self.norm3[i](h_temp)                 
                h_temp = h_temp.transpose(1, 2) 

            # Extract out the non-conditional frames for interpolation
            x_t = x_temp[:, :, ~cond_mask]
            h_t = h_temp[:, :, ~cond_mask]
            assert h_t.shape[-1] == Tc
            assert x_t.shape[-1] == Tc

            # Get the mixing factors
            alpha_x_i_t = basic if self.set_zero else torch.sigmoid(self.alpha_x_t[i])
            alpha_h_i_t = basic if (self.set_zero or i == self.n_layers - 1) else torch.sigmoid(self.alpha_h_t[i]) # DDP aware

            # Interpolation between spatial and temporal
            x_i = alpha_x_i_t * x_s + (1 - alpha_x_i_t) * x_t
            h_i = alpha_h_i_t * h_s + (1 - alpha_h_i_t) * h_t

            # Update the global trajectory state
            x = x.clone()
            h = h.clone()
            x[:, :, ~cond_mask] = x_i
            h[:, :, ~cond_mask] = h_i

            # Check shapes
            assert x.shape[-1] == T
            assert h.shape[-1] == T

        # Final extraction
        h = h[:, :, ~cond_mask]
        x_pred = x[:, :, ~cond_mask]
        assert x_pred.shape == x_input.shape

        # Substract and Scale
        x_out = (x_pred - x_input) / self.scale

        return x_out, h


if __name__ == '__main__':
    import numpy as np

    BN = 5
    B = 2
    Hh = 16
    He = 2
    H = 32
    T = 10

    model = EGTN(n_layers=3, node_dim=Hh, ft_dim=10, edge_dim=He, hidden_dim=H, time_emb_dim=64, act_fn='silu',
                 scale=1, pre_norm=False)
    # model = EGInterpolator(n_layers=3, node_dim=Hh, ft_dim=10, edge_dim=He, hidden_dim=H, time_emb_dim=64, num_heads=4, act_fn='silu', scale=1, pre_norm=False)
    
    print(model)

    batch = torch.from_numpy(np.array([0, 0, 0, 1, 1])).long()
    row = [0, 0, 1, 3]
    col = [1, 2, 2, 4]
    row = torch.from_numpy(np.array(row)).long()
    col = torch.from_numpy(np.array(col)).long()
    f = torch.randint(0, 10, size=(BN, 10))
    h = torch.randint(0, 10, size=(BN,))
    x = torch.rand(BN, 3, T)
    edge_index = torch.stack((row, col), dim=0)  # [2, BM]
    BM = edge_index.size(-1)
    edge_attr = torch.randint(0, 30, size=(BM,))

    t = torch.randint(0, 1000, size=(BN,)).to(x)[batch]
    print(t.shape, x.shape, h.shape, f.shape, edge_index.shape, edge_attr.shape, batch.shape)
    x_out, h_out = model(t, x, h, f, edge_index, edge_attr, batch)
    assert x_out.size() == x.size()
    assert h_out.size(0) == x.size(0)
    assert h_out.size(1) == H
    print('Test successful')
