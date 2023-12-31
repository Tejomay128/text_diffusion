import torch
import numpy as np

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    -> name: the name of the sampler.
    -> diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "lossaware":
        return LossSecondMomentReSampler(diffusion)
    elif name == "fixstep":
        return FixSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        -> batch_size: the number of timesteps.
        -> device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class FixSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion

        ###############################################################
        ###  You can custom your own sampling weight of steps here. ###
        ###############################################################
        self._weights = np.concatenate([np.ones([diffusion.num_timesteps//2]), np.zeros([diffusion.num_timesteps//2]) + 0.5])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts: torch.tensor, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        -> local_ts: an integer Tensor of timesteps.
        -> local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [torch.tensor([0], dtype=torch.int32, device=local_ts.device) 
                       for _ in range(dist.get_world_size())
                       ]
        dist.all_gather(batch_sizes, 
                        torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device)
                        )
        
        max_batch_sz = max(batch_sizes)

        timestep_batches = [torch.zeros(max_batch_sz).to(local_ts) for _ in len(max_batch_sz)]
        loss_batches = [torch.zeros(max_batch_sz).to(local_ts) for _ in len(max_batch_sz)]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_ts)

        timesteps = [x.item() for t, b in zip(timestep_batches, batch_sizes) for x in t[:b]]
        losses = [x.item() for t, b in zip(loss_batches, batch_sizes) for x in t[:b]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, timesteps, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        -> ts: a list of int timesteps.
        -> losses: a list of float losses, one per timestep.
        """


class LossSecondMomentReSampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        """
        diffusion -> instance of the class that defines the Diffusion
        history_per_term -> How many steps to look behind for re-weighting
        uniform_prob -> defines the weight of uniform probability while reweighting
            Eg: new_weights = new_weights * (1 - uniform_prob) + 1/(num_weights) * uniform_prob
            1/num_weights gives us the unifrom probabilty over all timesteps (num_weights = num_timesteps)
        """
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([diffusion.num_steps, history_per_term], dtype=torch.float64)
        self._loss_counts = np.zeros([diffusion.num_steps], dtype=np.int)

    def _warmed_up(self):
        """
        checking if the history is built till the "history_per_term" argument
        """
        return (self._loss_counts == self.history_per_term).all()

    def weights(self):
        """
        Weighting with loss as defined in "Improved Denoising Diffusion Probabilistic Models"
        paper section 3.3
        """
        if not self._warmed_up():
            return np.ones([self.diffusion.num_steps], dtype=torch.float64)
        
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, timesteps, losses):
        for t, loss in zip(timesteps, losses):
            if self._loss_counts[t] == self.history_per_term:
                #shifting the oldest loss term out
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1
                