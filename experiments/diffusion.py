"""
Code adapted from OpenAI guided diffusion repo:
https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
"""
import enum
import math
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool, global_add_pool

from .alignment import kabsch_torch
from .kabsch_util import align_traj_kabsch_pairwise_naive


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def gaussian_log_likelihood(x, *, means, log_scales, batch, num_nodes, subspace_dim_reduce=0):
    """
    Compute the log-likelihood of a Gaussian distribution
    """
    assert x.shape == means.shape == log_scales.shape
    n_dim = num_nodes * x.size(1) * x.size(2) - subspace_dim_reduce  # [B]
    n_dim = n_dim[batch]  # [BN]
    constants = n_dim * (log_scales[:, 0, 0] + 0.5 * np.log(2 * np.pi))  # [BN]
    constants = constants / num_nodes[batch]  # [BN], divide by the number of nodes to avoid repetitive compute
    term = 0.5 * ((x - means) ** 2) / torch.exp(2 * log_scales)  # [BN, 3, T]
    return constants, term


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)


class OurDiffusion(object):
    """
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        beta_schedule_name,
        num_timesteps,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
        use_kabsch_alignment=False,
    ):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.use_kabsch_alignment = use_kabsch_alignment

        # Use float64 for accuracy.
        betas = get_named_beta_schedule(schedule_name=beta_schedule_name, num_diffusion_timesteps=num_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def project_to_subspace(self, x, batch):  # [BN, 3, T]
        x1 = global_mean_pool(x.mean(dim=-1), batch)  # [B, 3]
        x1 = x1[batch].unsqueeze(-1)  # [BN, 3, 1]
        return x - x1  # [BN, 3, T] in the subspace with dim (TN-1)D

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def compute_noise(self, x_start, x_t, t):
        """
        Compute the noise from the model.

        :param x_start: the initial data batch.
        :param x_t: the noisy version of x_start.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A noisy version of x_start.
        """
        assert x_start.shape == x_t.shape
        return (
            (x_t - _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start) 
            / _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        x = self.project_to_subspace(x, model_kwargs['batch'])

        model_output, _ = model(x=x, diffusion_t=self._scale_timesteps(t), **model_kwargs)

        model_output = self.project_to_subspace(model_output, model_kwargs['batch'])

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = model_output
            else:
                pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t))
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self._scale_timesteps(t))

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(self, model, x, t, cond_fn=None, model_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """

        out = self.p_mean_variance(model, x, t, model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, noise=None, cond_fn=None, progress=False, model_kwargs=None):
        """
        Generate samples from the model.

        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            cond_fn=cond_fn,
            progress=False,
            model_kwargs=model_kwargs
        ):
            final = sample

        return final["sample"]

    def interpolation_diffusion(self, model, shape, noise=None, cond_fn=None, progress=False, model_kwargs=None):
        device = model_kwargs['original_frames'].device
        orig_frames = model_kwargs['original_frames']
        out = self.p_sample_loop(
                model, 
                shape,
                noise=noise, 
                cond_fn=cond_fn,
                progress=progress, 
                model_kwargs=model_kwargs
            )
        orig_frames[..., 1:-1] = out
        return orig_frames
    

    def ar_block_diffusion(
        self, 
        model, shape, 
        noise=None, cond_fn=None, 
        progress=False, model_kwargs=None,
        ddim=False, ddim_steps=100, eta=0.0,  # was 500; large step counts can destabilize DDIM
        num_blocks=4  # Configurable number of blocks (default=4)
    ):
        print(f"AR BLOCK DIFFUSION with {num_blocks} blocks")
        device = model_kwargs['original_frames'].device
        orig_ref_frame = model_kwargs['original_frames'][..., 0].unsqueeze(-1)
        blocks = [orig_ref_frame]
        for i in range(num_blocks):
            print(f"Block: {i}")
            # fresh noise each block to avoid propagating a bad state
            block_noise = None if noise is None else torch.randn_like(noise)

            if not ddim:
                out = self.p_sample_loop(
                    model, 
                    shape,
                    noise=block_noise, 
                    cond_fn=cond_fn,
                    progress=progress, 
                    model_kwargs=model_kwargs
                )
            else:
                out = self.ddim_sample_loop(
                    model,
                    shape,
                    steps=ddim_steps,
                    eta=eta,              # keep 0.0 for deterministic/stable; increase later if stable
                    noise=block_noise,
                    cond_fn=cond_fn,
                    progress=progress,
                    model_kwargs=model_kwargs,
                )
                    
            blocks.append(out)

            last_frame = out[..., -1]
            new_given  = torch.zeros_like(model_kwargs['original_frames']).to(device)
            new_given[..., 0] = torch.nan_to_num(last_frame, nan=0.0, posinf=1e6, neginf=-1e6)
            model_kwargs['original_frames'] = new_given

        return torch.cat(blocks, dim=-1)

        # def fake_ar_block_diffusion(self, model, shape, noise=None, cond_fn=None, progress=False, model_kwargs=None):
        #     num_blocks = 4
        #     device = model_kwargs['original_frames'].device
        #     orig_ref_frame = model_kwargs['original_frames'][..., 0].unsqueeze(-1)
        #     blocks = [orig_ref_frame]
        #     for i in range(num_blocks):
        #         print(f"Block: {i}")
        #         if i == 0:
        #             model_kwargs['cond_mask'][0] = False
        #             print(f"Block {i} is unconditioned on sampled conformer")
        #         else:
        #             model_kwargs['cond_mask'][0] = True
        #             print(f"Block {i} is conditioned normally")
        #         out = self.p_sample_loop(
        #             model, 
        #             shape,
        #             noise=noise, 
        #             cond_fn=cond_fn,
        #             progress=progress, 
        #             model_kwargs=model_kwargs
        #         )
        #         blocks.append(out)

        #         last_frame = out[..., -1]
        #         new_given  = torch.zeros_like(model_kwargs['original_frames']).to(device)
        #         new_given[..., 0] = last_frame
        #         model_kwargs['original_frames'] = new_given

        #     # concat [ B×C×block_T ] × num_blocks → B×C×total_T + 1
        #     return torch.cat(blocks, dim=-1)

    def uncond_block_diffusion(self, model, shape, noise=None, cond_fn=None, progress=False, model_kwargs=None, num_blocks=4):
        print("UNCOND BLOCK DIFFUSION -- num_blocks: ", num_blocks)
        device = model_kwargs['original_frames'].device
        orig_ref_frame = model_kwargs['original_frames'][..., 0].unsqueeze(-1)
        blocks = [orig_ref_frame]
        print(f"Generating {num_blocks} blocks with unconditional forward simulation")
        for i in range(num_blocks):
            if i == 0:
                model_kwargs['uncond'] = True
                print(f"Block {i} is unconditioned")
            else:
                model_kwargs['uncond'] = False
                print(f"Block {i} is conditioned on block {i-1}")
            out = self.p_sample_loop(
                model, 
                shape,
                noise=noise, 
                cond_fn=cond_fn,
                progress=progress, 
                model_kwargs=model_kwargs
            )
            blocks.append(out)

            last_frame = out[..., -1]
            new_given  = torch.zeros_like(model_kwargs['original_frames']).to(device)
            new_given[..., 0] = last_frame
            model_kwargs['original_frames'] = new_given

        # concat [ B×C×block_T ] × num_blocks → B×C×total_T + 1
        return torch.cat(blocks, dim=-1)


    def uncond_autoreg_diffusion(self, model, shape, noise=None, cond_fn=None, progress=False, model_kwargs=None):
        num_blocks = 20
        device = model_kwargs['original_frames'].device
        orig_ref_frame = model_kwargs['original_frames'][..., 0].unsqueeze(-1)
        blocks = [orig_ref_frame]
        for i in tqdm(range(num_blocks)):
            if i == 0:
                model_kwargs['uncond'] = True
                print(f"Block {i} is unconditioned")
            else:
                model_kwargs['uncond'] = False
            out = self.p_sample_loop(
                model, 
                shape,
                noise=noise, 
                cond_fn=cond_fn,
                progress=progress, 
                model_kwargs=model_kwargs
            )
            blocks.append(out[..., -1].unsqueeze(-1))

            last_frame = out[..., -1]
            new_given  = torch.zeros_like(model_kwargs['original_frames']).to(device)
            new_given[..., 0] = last_frame
            model_kwargs['original_frames'] = new_given

        # concat [ B×C×block_T ] × num_blocks → B×C×total_T + 1
        return torch.cat(blocks, dim=-1)
    
    def autoreg_diffusion(self, model, shape, noise=None, cond_fn=None, progress=False, model_kwargs=None):
        num_blocks = 500
        device = model_kwargs['original_frames'].device
        orig_ref_frame = model_kwargs['original_frames'][..., 0].unsqueeze(-1)
        blocks = [orig_ref_frame]
        for i in tqdm(range(num_blocks)):
            out = self.p_sample_loop(
                model, 
                shape,
                noise=noise, 
                cond_fn=cond_fn,
                progress=progress, 
                model_kwargs=model_kwargs
            )
            blocks.append(out[..., -1].unsqueeze(-1))

            last_frame = out[..., -1]
            new_given  = torch.zeros_like(model_kwargs['original_frames']).to(device)
            new_given[..., 0] = last_frame
            model_kwargs['original_frames'] = new_given

        # concat [ B×C×block_T ] × num_blocks → B×C×total_T + 1
        return torch.cat(blocks, dim=-1)

    def p_sample_loop_progressive(self, model, shape, noise=None, cond_fn=None, progress=False, model_kwargs=None):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        device = model_kwargs['batch'].device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)  # Feed the same time step for all BN nodes
            with torch.no_grad():
                out = self.p_sample(model, img, t, cond_fn=cond_fn, model_kwargs=model_kwargs)
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t=None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param x_start: the [N x C x ...] tensor of inputs. shape: [BN, 3, T]
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x_start)
        if t is None:
            assert 'batch' in model_kwargs
            num_batch = int(model_kwargs['batch'].max() + 1)
            t = torch.randint(0, self.num_timesteps, size=(num_batch,)).to(x_start.device)  # [B,]
            t = t[model_kwargs['batch']]  # [BN,]

        x_start = self.project_to_subspace(x_start, model_kwargs['batch'])
        noise = self.project_to_subspace(noise, model_kwargs['batch'])

        x_t = self.q_sample(x_start, t, noise=noise)  # [BN, 3, T_p]

        # if self.use_kabsch_alignment:
        #     # loop through each batch and align the x_t with the x_start
        #     # using Kabsch algorithm
        #     with torch.no_grad():
        #         for batch_idx in range(model_kwargs['batch'].max() + 1):
        #             batch_mask = model_kwargs['batch'] == batch_idx
        #             cur_x_t = x_t[batch_mask]
        #             cur_x_start = x_start[batch_mask]
        #             # currently kabsch_torch only works for [N, 3]
        #             aligned_x_t, _, _ = kabsch_torch(cur_x_t[..., 0], cur_x_start[..., 0])
        #             aligned_x_t = aligned_x_t[..., None]  # [N, 3, 1]
        #             x_t[batch_mask] = aligned_x_t
        #             noise[batch_mask] = self.compute_noise(cur_x_start, aligned_x_t, t[batch_mask])

        if self.use_kabsch_alignment:
            # loop through each batch and align the x_t with the x_start
            # using Kabsch algorithm
            with torch.no_grad():
                for batch_idx in range(model_kwargs['batch'].max() + 1):
                    batch_mask = model_kwargs['batch'] == batch_idx
                    cur_x_t = x_t[batch_mask]
                    cur_x_start = x_start[batch_mask]
                    # Now kabsch works for all [N, 3, T]
                    if hasattr(model, 'use_kabsch') and not model.use_kabsch:
                        aligned_x_t, _, _ = kabsch_torch(cur_x_t, cur_x_start)
                    else:
                        aligned_x_t = align_traj_kabsch_pairwise_naive(cur_x_t, cur_x_start)
                    # currently kabsch_torch only works for [N, 3]
                    # aligned_x_t, _, _ = kabsch_torch(cur_x_t[..., 0], cur_x_start[..., 0])
                    # aligned_x_t = aligned_x_t[..., None]  # [N, 3, 1]
                    x_t[batch_mask] = aligned_x_t
                    noise[batch_mask] = self.compute_noise(cur_x_start, aligned_x_t, t[batch_mask])

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output, _ = model(x=x_t, diffusion_t=self._scale_timesteps(t), **model_kwargs)

            model_output = self.project_to_subspace(model_output, model_kwargs['batch'])
            
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)  # [BN]
            terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        nats.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in nats), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return sum_flat(kl_prior)

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are nats. (I modified here from bits to nats.)
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )

        kl = sum_flat(kl)  # Here I compute total nll summed over all dimensions, instead of per dimension nll

        decoder_nll_constants, decoder_nll_term = gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"],
            batch=model_kwargs['batch'], num_nodes=model_kwargs['num_nodes'],
            subspace_dim_reduce=x_t.size(1)
        )

        assert decoder_nll_term.shape == x_start.shape

        decoder_nll = decoder_nll_constants + sum_flat(decoder_nll_term)
        # decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        # output = kl
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def calc_bpd_loop(self, model, x_start, model_kwargs=None, progress=False):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param x_start: the [N x C x ...] tensor of inputs.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []

        x_start = self.project_to_subspace(x_start, model_kwargs['batch'])

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        # Calc num_nodes
        batch = model_kwargs['batch']  # [BN]
        temp = torch.ones_like(batch)
        num_nodes = global_add_pool(temp, batch)  # [B]
        model_kwargs['num_nodes'] = num_nodes

        # for t in list(range(self.num_timesteps))[::-1]:
        for t in indices:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)

            noise = self.project_to_subspace(noise, model_kwargs['batch'])

            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    model=model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd

        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
    
    def _make_ddim_timesteps(self, num_ddim_steps: int, device):
        assert 2 <= num_ddim_steps <= self.num_timesteps
        # Decreasing linspace, then enforce strict monotonicity
        ts = np.round(np.linspace(self.num_timesteps - 1, 0, num=num_ddim_steps)).astype(np.int64)
        # Make strictly decreasing with length preserved
        for i in range(1, len(ts)):
            if ts[i] >= ts[i - 1]:
                ts[i] = ts[i - 1] - 1
        ts = np.clip(ts, 0, self.num_timesteps - 1)
        ts[0]  = self.num_timesteps - 1
        ts[-1] = 0
        return torch.from_numpy(ts).to(device=device, dtype=torch.long)


    def _ddim_step(self, model, x_t, t, t_prev, eta=0.0, cond_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        B = x_t.shape[0]
        if t.ndim == 0:
            t = torch.full((B,), int(t.item()), device=x_t.device, dtype=torch.long)
        if t_prev.ndim == 0:
            t_prev = torch.full((B,), int(t_prev.item()), device=x_t.device, dtype=torch.long)

        # Project once and use consistently
        x_proj = self.project_to_subspace(x_t, model_kwargs['batch'])

        # Predict model output at time t
        with torch.cuda.amp.autocast(enabled=False):  # avoid half-precision NaNs
            x_proj = x_proj.float()
            model_out, _ = model(x=x_proj, diffusion_t=self._scale_timesteps(t), **model_kwargs)
            model_out = self.project_to_subspace(model_out, model_kwargs['batch']).float()

        model_out = torch.nan_to_num(model_out, nan=0.0, posinf=1e6, neginf=-1e6)

        # Convert to eps prediction if needed (all in projected space)
        if self.model_mean_type == ModelMeanType.EPSILON:
            eps = model_out
        elif self.model_mean_type == ModelMeanType.START_X:
            eps = self._predict_eps_from_xstart(x_proj, t, model_out)
        elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
            x0 = self._predict_xstart_from_xprev(x_proj, t, model_out)
            eps = self._predict_eps_from_xstart(x_proj, t, x0)
        else:
            raise NotImplementedError(self.model_mean_type)

        eps = torch.nan_to_num(eps, nan=0.0, posinf=1e6, neginf=-1e6)

        # Optional score conditioning should also use projected x
        if cond_fn is not None:
            alpha_bar_t = _extract_into_tensor(self.alphas_cumprod, t, x_proj.shape).float()
            score = cond_fn(x_proj, self._scale_timesteps(t))
            score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
            eps = eps - (1.0 - alpha_bar_t).sqrt() * score

        # Recover x0_pred from (x_proj, eps) at time t
        x0_pred = self._predict_xstart_from_eps(x_t=x_proj, t=t, eps=eps).float()
        x0_pred = torch.nan_to_num(x0_pred, nan=0.0, posinf=1e6, neginf=-1e6)

        # DDIM coefficients between t and t_prev (add tiny eps for safety)
        eps_den = 1e-12
        alpha_bar_t    = _extract_into_tensor(self.alphas_cumprod, t, x_proj.shape).float()
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod, t_prev, x_proj.shape).float()

        denom = (1 - alpha_bar_t).clamp_min(eps_den)
        ratio = (1 - alpha_bar_prev).clamp_min(eps_den) / denom
        inner = (1 - alpha_bar_t / alpha_bar_prev.clamp_min(eps_den)).clamp_min(0.0)
        sigma = eta * torch.sqrt((ratio * inner).clamp_min(0.0))
        dir_coef = torch.sqrt((1 - alpha_bar_prev - sigma**2).clamp_min(0.0))

        noise = torch.randn_like(x_proj)
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_coef * eps + sigma * noise

        # Optional per-sample RMS clip to prevent explosion in AR blocks
        # (tuned threshold; set lower if you still see NaNs)
        def _rms_clip(z, max_rms=50.0):
            # compute RMS over (C, T, ...) dims
            dims = tuple(range(1, z.ndim))
            rms = z.pow(2).mean(dim=dims, keepdim=True).sqrt()
            scale = torch.clamp(max_rms / (rms + 1e-8), max=1.0)
            return z * scale

        x_prev = torch.nan_to_num(_rms_clip(x_prev), nan=0.0, posinf=1e6, neginf=-1e6)

        return x_prev, x0_pred

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, steps=100, eta=0.0, noise=None,
                        cond_fn=None, progress=False, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        device = model_kwargs['batch'].device
        assert isinstance(shape, (tuple, list))

        with torch.cuda.amp.autocast(enabled=False):
            img = (noise if noise is not None else torch.randn(*shape, device=device)).float()

            schedule = self._make_ddim_timesteps(steps, device=device)  # length = steps
            it = range(len(schedule) - 1)
            if progress:
                from tqdm.auto import tqdm
                it = tqdm(it)

            for i in it:
                t      = schedule[i]
                t_prev = schedule[i+1]
                t_b     = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
                tprev_b = torch.full((shape[0],), int(t_prev.item()), device=device, dtype=torch.long)

                img, _ = self._ddim_step(
                    model=model,
                    x_t=img,
                    t=t_b,
                    t_prev=tprev_b,
                    eta=eta,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )

                # Repair if anything went non-finite
                if not torch.isfinite(img).all():
                    img = torch.nan_to_num(img, nan=0.0, posinf=1e6, neginf=-1e6)

        return img

    @torch.no_grad()
    def ddim_sample_loop_progressive(self, model, shape, steps=100, eta=0.1, noise=None,
                                    cond_fn=None, progress=False, model_kwargs=None):
        """
        Same as ddim_sample_loop, but yields intermediate outputs after each step.
        """
        if model_kwargs is None:
            model_kwargs = {}
        device = model_kwargs['batch'].device
        assert isinstance(shape, (tuple, list))
        img = noise if noise is not None else torch.randn(*shape, device=device)

        schedule = self._make_ddim_timesteps(steps, device=device)
        iterator = range(len(schedule) - 1)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator)

        for i in iterator:
            t      = schedule[i]
            t_prev = schedule[i+1]
            t_b      = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
            tprev_b  = torch.full((shape[0],), int(t_prev.item()), device=device, dtype=torch.long)
            img, x0_pred = self._ddim_step(
                model=model,
                x_t=img,
                t=t_b,
                t_prev=tprev_b,
                eta=eta,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
            yield {"sample": img, "pred_xstart": x0_pred}


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



