import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


def get_beta_from_alpha_bar(num_steps: int, alpha_bar_fn, beta_max: int=0.999):
    """
    obtain beta schedule given alpha_bar function
    """
    betas = []
    for i in range(num_steps):
        t1 = i / num_steps
        t2 = (i+1) / num_steps
        betas.append(min(1 - alpha_bar_fn(t2)/alpha_bar_fn(t1)), beta_max)
    return np.array(betas, dtype=np.float64)

def define_schedule(num_steps: int, name: str="linear"):
    """
    Define the noise schedule given the number of steps. 
    """
    if name == "linear":
        scale = 1000 / num_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    
    elif name == "cosine":   # we pass f(t) instead of alpha_bar(t) = f(t)/f(0) to get_beta_from_alpha_bar
        betas = get_beta_from_alpha_bar(num_steps, 
                                        lambda x: math.cos((x + 0.008) / 1.008) * math.pi / 2)
        return betas
    
    elif name == "sqrt":
        betas = get_beta_from_alpha_bar(num_steps, lambda x: math.sqrt(1 - x + 0.008))
        return betas
    
    else:
        raise NotImplementedError(f"Unknown noise schedule: {name}")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    -> arr: the 1-D numpy array.
    -> timesteps: a tensor of indices into the array to extract.
    -> broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class Diffusion:
    """
    Defines diffusion model training and sampling procedure
    """

    def __init__(self,
                 *,
                 betas,
                 predict_xstart: bool,
                 learn_sigmas: bool,
                 ) -> None:
        self.betas = betas
        self.predict_xstart = predict_xstart
        self.learn_sigmas = learn_sigmas

        assert isinstance(betas, np.ndarray), "betas must be np.array."
        assert len(betas.shape) == 1, "betas must be 1D"
        assert (betas > 0).all() and (betas < 1).all()

        self.num_steps = int(betas.shape[0])

        alphas = 1 - betas
        self.alpha_bar = np.cumprod(alphas, axis=0)
        self.alpha_bar_prev = np.append(1.0, self.alpha_bar[:-1])
        self.alpha_bar_next = np.append(self.alpha_bar[1:], 0.0)
        assert self.alpha_bar_next.shape == (self.num_steps,)
        assert self.alpha_bar_prev.shape == (self.num_steps,)

        """
        begin posterior calculation q(x_t-1| x_t, x_0)
        """
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar)
        self.one_upon_sqrt_alpha_bar = 1.0 / self.sqrt_alpha_bar
        self.log_one_minus_alpha_bar = np.log(1.0 - self.alpha_bar)

        self.post_var = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        # avoiding log(0) at the beginning (t = 0)
        self.post_log_var = np.log(np.append(self.post_var[1], self.post_var[1:]))  

        self.mean_x0_coeff = np.sqrt(self.alpha_bar_prev) * self.betas / (1.0 - self.alpha_bar)
        self.mean_xt_coeff = self.sqrt_alpha_bar * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def predict_x0_from_eps(self,
                            x_t: torch.Tensor,
                            t: torch.Tensor, 
                            eps: torch.Tensor):
        """
        Obtain original data x_0 from given noisy version x_t

        -> x_0: [B X seq_len X ....]
        ->   t: the number of diffusion steps (minus 1)
        """
        assert x_t.shape == eps.shape, "eps and x must have the same shape"
        x_0 = _extract_into_tensor(self.one_upon_sqrt_alpha_bar, t, x_t.shape) * \
              (x_t - _extract_into_tensor(self.sqrt_one_minus_alpha_bar) * eps) 
        return x_0
    
    def predict_eps_from_x0(self, 
                            x_t: torch.Tensor, 
                            t: torch.Tensor, 
                            x0_pred: torch.Tensor):
        """
        Obtain added noise in data x_0 given noisy version x_t

        -> x_0: [B X seq_len X ....]
        ->   t: the number of diffusion steps (minus 1)
        """
        eps = _extract_into_tensor(1.0 / self.sqrt_one_minus_alpha_bar, t, x_t.shape) * \
              (x_t - _extract_into_tensor(self.sqrt_alpha_bar, t, x_t.shape) * x0_pred)
        return eps
    
    def q_mean_var(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Get the mean, variance and log variance of the forward process q(x_t | x_0)

        -> x_0: [B X seq_len X ....]
        ->   t: the number of diffusion steps (minus 1)
        """
        mean = _extract_into_tensor(self.sqrt_alpha_bar, t, x_0.shape) * x_0
        var = _extract_into_tensor(1 - self.alpha_bar, t, x_0.shape)
        log_var = _extract_into_tensor(self.log_one_minus_alpha_bar, t, x_0.shape)

        return mean, var, log_var
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, eps=None, mask=None):
        """
        Get x_t given x_0 and t

        -> mask: input_ids mask of shape [B X seq_len], used to mask the sentence to be kept in-context
        """
        if eps == None:
            eps = torch.randn_like(x_0)
        
        x_t = _extract_into_tensor(self.sqrt_alpha_bar, t, x_0.shape) * x_0 + \
              _extract_into_tensor(x_0 + self.sqrt_one_minus_alpha_bar, t, x_0.shape) * eps
        
        if mask is not None: 
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_0.shape)
            x_t = torch.where(mask==0, x_0, x_t)  # keep x_0 wherever mask is zero, else make it x_t
        
        return x_t
    
    def q_posterior_mean_var(self, x_0, x_t, t):
        """
        compute the parameters of the distribution q(x_t-1 | x_t, x_0)
        """
        assert x_t.shape == x_0.shape
        post_mean = _extract_into_tensor(self.mean_x0_coeff, t, x_0.shape) * x_0 +\
                     _extract_into_tensor(self.mean_xt_coeff, t, x_0.shape) * x_t
        post_var = _extract_into_tensor(self.post_var, t, x_0.shape)
        post_log_var_clipped = _extract_into_tensor(self.post_log_var, t, x_0.shape)

        assert post_mean.shape[0] == post_var.shape[0] == post_log_var_clipped.shape[0] == x_0.shape[0]

        return post_mean, post_var, post_log_var_clipped

    def p_mean_var(self, model, x_t, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        compute the predicted posterior p(x_t-1 | x_t) and the prediction of x_0
        -> model: the model, which takes a x_t and a corresponding batch of timesteps
                      as input.
        -> x_t: the [N x seq_len x ...] tensor at time t.
        -> t: a 1-D Tensor of timesteps.
        -> clip_denoised: if True, clip the denoised signal into [-1, 1].
        -> denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        -> model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'var': the model variance output.
                 - 'log_var': the log of 'variance'.
                 - 'pred_x0': the prediction for x_0. 
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        assert t.shape == (x_t.size(0),)
        out = model(x_t, self._scale_timesteps(t), **model_kwargs)

        # We set the variance at t = 0 equal to the variance at t = 1 instead of zero.
        model_var = _extract_into_tensor(
            np.append(self.post_var[1], self.post_var[1:]), t, x_t.shape
            )
        model_log_var = _extract_into_tensor(
            np.append(self.post_log_var[1], self.post_log_var[1:]), t, x_t.shape
            )
        
        def process_x0(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                x = torch.clamp(-1, 1)
            return x
        
        if self.predict_xstart:
            pred_x0 = process_x0(out)
        else: # model predicts eps instead of x_start
            pred_x0 = process_x0(
                self.predict_x0_from_eps(x_t, t, eps=out)
            )
        
        """
        using predicted x_0 to calculate posterior essentially gives us
        the estimated reverse distribution p.
        """
        model_mean, _, _ = self.q_posterior_mean_var(pred_x0, x_t, t)

        assert model_mean.shape == model_var.shape == pred_x0.shape == x_t.shape

        return {
            "mean": model_mean,
            "var": model_var,
            "log_var": model_log_var,
            "pred_x0": pred_x0
        }
    
    def p_sample(
        self, model, x_t, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None, mask=None, x_start=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        -> model: the model to sample from.
        -> x: the current tensor (from timestep t).
        -> t: the value of t, starting at 0 for the first diffusion step.
        -> clip_denoised: if True, clip the x_start prediction to [-1, 1].
        -> denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        -> mask: anchoring masked position to x_start
        -> model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        """
        out = self.p_mean_var(x_t, t, 
                              clip_denoised=clip_denoised, 
                              denoised_fn=denoised_fn, 
                              model_kwargs=model_kwargs)
        
        if top_p is not None and top_p > 0:
            # Ensure magnitude of all eps values to be greater than top_p
            eps = torch.randn_like(x_t)
            replace_mask = torch.abs(eps) > top_p
            while replace_mask.any():
                eps[replace_mask] = torch.randn_like(eps[replace_mask])
                replace_mask = torch.abs(eps) > top_p
            assert (torch.abs(eps) <= top_p).all()
        else:
            eps = torch.randn_like(x_t)

        """
        no noise to be added at t = 0
        matching number of dimensions with x_t
        """
        time_zero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))  
        sample = out["mean"] + time_zero_mask * torch.exp(0.5 * out["log_var"]) * eps

        return {
            "sample": sample,
            "pred_x0": out["pred_x0"],
            "greedy_mean": out["mean"]
        }
    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_0=None,
        gap=1,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            mask=mask,
            x_0=x_0
        ):
            final.append(sample['sample'])
        return final
    
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_0=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            sample_x = noise
        else:
            sample_x = torch.randn(*shape, device=device)
        indices = list(range(self.num_steps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur = denoised_fn
                else:
                    denoised_fn_cur = None
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn_cur,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                    mask=mask,
                    x_start=x_0
                )
                yield out
                sample_x = out["sample"]    


    def _get_x_start(self, x_start_mean, std):
        '''
        Word embedding projection from {Emb(w)} to {x_0}
        -> x_start_mean: word embedding
        :return: x_0
        '''
        noise = torch.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return x_start_mean + std * noise
    
    def discrete_token_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):
        """
        enforces the predicted x_0 to be closer to actual word embeddings
        -log(w | x_0)
        """
        logits = get_logits(x_t)
        ce_loss = nn.CrossEntropyLoss(reduction="none")
        nll = ce_loss(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        # nll is now simply a 1D tensor
        if mask is not None:
            nll *= mask
            nll = nll.sum(dim=-1) / mask.sum(dim=-1)
        else:
            nll = nll.mean(dim=-1)
        
        return nll

    def training_loss(self, model, x_0, t, model_kwargs=None, noise=None):
        """
        Compute training loss for a single pass
        -> model: the model to evaluate loss on.
        -> x_start: the [N x seq_len x ...] tensor of inputs. # not used unless fixing the input embeddings
        -> t: a batch of timestep indices.
        -> model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        -> noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        pass

    def ddim_sample(
        self,
        model,
        x_t,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None
    ):
        """
        Similar to p_sample, sample x_t-1 using DDIM
        -> eta: Controls the stochasticity of the sampling process. 
                Setting it to zero makes the process deterministic.
        """
        out_dict = self.p_mean_var(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        eps = self.predict_eps_from_x0(x_t, t, out_dict["pred_x0"])
        alpha_bar = _extract_into_tensor(self.alpha_bar, t, x_t.shape)
        alpha_bar_prev = _extract_into_tensor(self.alpha_bar_prev, t, x_t.shape)

        """
        Refer to section 4.1 from the "Denoising Diffusion Implicit Models" paper
        for obtaining sample (x_t-1)
        """
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))

        """
        noise is different from eps, 
        noise adds stochasticity to the sampling process while eps is the predicted noise.
        """
        noise = torch.randn_like(x_t)  

        pred_mean = (
            torch.sqrt(alpha_bar_prev) * out_dict["pred_x0"] +
            torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        """
        no noise to be added at t = 0
        matching number of dimensions with x_t
        """
        time_zero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))  
        sample = pred_mean + time_zero_mask * sigma * noise

        #ADD LANGEVIN FUNCTION HERE

        if mask is not None:
            sample = torch.where(mask==0, x_start, sample)
        
        return {
            "sample": sample,
            "pred_x0": out_dict["pred_x0"]
        }
    
    def ddim_reverse_sample(
        self,
        model,
        x_t,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Obtain x_t+1 from x_t using the DDIM equation used in ddim_sample.
        To do this, eta must be zero, which makes the process deterministic and allows us to exactly
        calculate x_t+1
        """
        assert eta == 0.0, "Non-zero eta value won't give exact x_t+1 computation."
        out_dict = self.p_mean_var(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        eps = self.predict_eps_from_x0(x_t, t, out_dict["pred_x0"])
        alpha_bar_next = _extract_into_tensor(self.alpha_bar_next, t, x_t.shape)

        pred_mean = (
            out_dict["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {
            "sample":pred_mean, 
            "pred_x0": out_dict["pred_x0"]
        }

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        Generate samples using DDIM sampling (similar to p_sample_loop).
        -> gap: skipping steps for DDIM sampling
        """
        final = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            mask=mask,
            x_start=x_start,
            gap = gap
        ):
            final.append(sample['sample'])
        return final
    
    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None,
        gap=1
    ):
        """
         Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device()
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1][::gap]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for idx in indices:
            t = torch.tensor([idx] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]

   
def spaced_timesteps(num_steps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    -> num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    -> section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        # special case for ddimN
        if section_counts.startswith("ddim"):
            count = int(section_counts[len("ddim"):])
            for i in range(1, num_steps):
                if len(range(0, num_steps, i)) == count:
                    return set(range(0, num_steps, i))
            raise ValueError(
                f"cannot create exactly {num_steps} steps with an integer stride"
            )
        
        section_counts = [int(x) for x in section_counts.split(",")]
    # divide into equal sections
    size_per_section = num_steps // len(section_counts)
    extra = num_steps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per_section + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        #define a fractional value as stride
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        # generate timesteps by striding starting from 0.0
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class _WrappedModel:
    """
    Wrapper class to wrap NN to adjust to resampled timesteps
    """
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


class SpacedDiffusion(Diffusion):
    """
    Create a diffusion process with spaced timesteps
    -> spaced_timesteps: a collection (sequence or set) of timesteps from the
                    original diffusion process to retain.
    -> kwargs: the kwargs to create the base diffusion process.
    """
    def __init__(self, space_timesteps, **kwargs):
        self.use_timesteps = set(space_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = Diffusion(**kwargs)
        last_alpha_bar = 1.0
        new_betas = []
        """
        Use the resampled timesteps to obtain new betas from alphas.
        Create a timestep map that maps timesteps to the sampled ones from original diffusion.
        """
        for i, alpha_bar in enumerate(base_diffusion.alpha_bar):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_bar / last_alpha_bar)
                last_alpha_bar = alpha_bar
                self.timestep_map.append(i)
        
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        else:
            return _WrappedModel(model,
                                self.timestep_map, 
                                self.rescale_timesteps, 
                                self.original_num_steps)
    
    def p_mean_var(self, model, x_t, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        return super().p_mean_var(self._wrap_model(model), x_t, t, clip_denoised, denoised_fn, model_kwargs)
    
    def training_loss(self, model, x_0, t, model_kwargs=None, noise=None):
        return super().training_loss(self._wrap_model(model), x_0, t, model_kwargs, noise)
