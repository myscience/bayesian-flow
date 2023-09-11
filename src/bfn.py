import torch
import torch.nn as nn

from torch import sqrt
from einops import reduce
from einops import rearrange
from itertools import pairwise

from typing import Tuple
from jaxtyping import Float
from torch import Tensor as Tensor

from .utils import default
from .utils import enlarge_as

class BayesianFlowNetwork(nn.Module):
    '''
        This class implements a Bayesian Flow Network as introduced
        in Graves et al. (2023).
    '''

    def __init__(
        self,
        backbone : nn.Module,
        loss_kind : str = 'continuous',
        data_kind : str = 'continuous',
        data_shape : Tuple[int, ...] = (3, 32, 32),
    ) -> None:
        '''
        
        '''
        super().__init__()

        self.backbone = backbone

        self.data_kind = data_kind
        self.loss_kind = loss_kind

        self.data_shape = data_shape

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    torch.inference_mode()
    def forward(
        self,
        num_samples : int,
        num_steps : int = 100,
        sigma_1 : float = 1e-3,
        cond : Float[Tensor, 'b ...'] | None = None,
    ) -> Float[Tensor, 'b ...']:
        '''
            We reserve the forward call of the model to the posterior sampling,
            that is used with a fully trained model to generate samples, hence
            the torch.inference_mode() decorator.
        '''

        sigma = torch.tensor([sigma_1], device=self.device)
        sigma_n = sigma ** (2 / num_steps)

        m_t = torch.zeros((num_samples, *self.data_shape), device=self.device)
        rho = 1.

        time = torch.linspace(0, 1, num_steps + 1)

        for t, tp1 in pairwise(time):
            t = enlarge_as(t, x_0)

            gamma = 1 - sigma ** (2. * t)

            alpha = sigma ** (-2. * tp1) * (1 - sigma_n)

            x_0 = self.predict(m_t, t, gamma, cond=cond)
            x_0 = x_0 + torch.randn_like(x_0) / alpha
            m_t = (rho * m_t + alpha * x_0) / (rho + alpha)
            rho = rho + alpha

        # Once the flow got the correct mean, sample the data
        t_1 = torch.ones((num_samples,), device=self.device)

        x_0 = self.predict(m_t, t_1, 1 - sigma ** 2, cond=cond)

        return x_0

    def predict(
        self,
        mu_t : Float[Tensor, 'b ...'],
        time : Float[Tensor, 'b'],
        gamma : Float[Tensor, 'b'],
        cond : Float[Tensor, 'b ...'] | None = None,
        x_bound : Tuple[float, float] = (-1., +1.),
        t_min : float = 1e-10,
    ) -> Float[Tensor, '...']:
        '''
            This method leverages the backbone model to obtain a prediction
            for the current noise at time t and then based on the noise
            estimates reconstructs the original data using the gamma factor.
        '''

        z_0 = torch.zeros_like(x_t)

        # Get the backbone prediction for the noise in the data
        # with optional external conditioning
        eps : Float[Tensor, 'b ...'] = self.backbone(mu_t, time, cond=cond)

        # Recompose data estimate following Eq. (84) in the main paper
        x_t = mu_t / gamma - sqrt((1 - gamma) / gamma) * eps

        # Clamp x_t in valid range and ensure no backflow beyond time limit
        x_t = torch.where(time < t_min, z_0, x_t.clip(*x_bound))

        return x_t
    
    def compute_loss(
        self,
        loss_kind : str = 'continuous',
        data_kind : str = 'continuous',
    ) -> Float[Tensor, 'b']:
        '''
        '''

        loss_kind = default(loss_kind, self.loss_kind)
        data_kind = default(data_kind, self.data_kind)

        err_msg = f'Invalid loss|data combination. Loss kind must be one of (continuous, discrete), while data kind must be one of (continuous, discretized, discrete). Got ({loss_kind}, {data_kind})'

        match (loss_kind, data_kind):
            case ('continuous', 'continuous'):
                loss_fn = self._continuous_loss_inf
            case ('continuous', 'discretized'):
                loss_fn = self._discretized_loss_inf
            case ('continuous', 'discrete'):
                loss_fn = self._discrete_loss_inf
            case ('discrete', 'continuous'):
                loss_fn = self._continuous_loss_n
            case ('discrete', 'discretized'):
                loss_fn = self._discretized_loss_n
            case ('discrete', 'discrete'):
                loss_fn = self._discrete_loss_n

            case _: raise ValueError(err_msg)

        loss = loss_fn()

        return loss

    def _continuous_loss_inf(
        self,
        x_0 : Float[Tensor, 'b ...'],
        sigma : float = 2e-3,
    ) -> Float[Tensor, 'b']:
        '''
            Compute the continuous-time limit loss L_∞ for continuos data.
            This loss is analogous to the Variational Diffusion Model and
            can be derived as L_∞ = -VLB.
        '''
        bs, _ = x_0.shape
        sigma : Float[Tensor, '1'] = torch.tensor([sigma], device=self.device)

        # Sample times for estimating average in t ~ U[0, 1]
        times = self._get_times(bs)
        times = enlarge_as(x_0)

        gamma = 1 - sigma ** (2. * times)

        # Corrupt the data via forward diffusion
        eps = torch.randn_like(x_0)
        x_t = gamma * x_0 + sqrt(gamma * (1 - gamma)) * eps

        # Get network reconstruction for the corrupted data
        x_hat = self.predict(x_t, times, gamma, cond = None)

        # Compute the loss as the average over the batch
        loss = -sigma.log() * (sigma ** (-2. * times)) * (x_0 - x_hat) ** 2

        return reduce(loss, 'b ... -> b', 'mean')
    
    def _continuous_loss_n(
        self,
    ) -> Float[Tensor, 'b']:
        raise NotImplementedError()
    
    def _discretized_loss_inf(
        self,
    ) -> Float[Tensor, 'b']:
        raise NotImplementedError()
    
    def _discretized_loss_inf(
        self,
    ) -> Float[Tensor, 'b']:
        raise NotImplementedError()
    
    def _discrete_loss_inf(
        self,
    ) -> Float[Tensor, 'b']:
        raise NotImplementedError()

    def _discrete_loss_n(
        self,
    ) -> Float[Tensor, 'b']:
        raise NotImplementedError()
    
    def _get_times(
        self,
        batch_size : int,
        sampler : str = 'low-var'
    ) -> Float[Tensor, 'b']:
        '''
            Sample the diffusion time steps. We can choose the sampler to
            be either are low-variance or naive.
        '''

        samplers = ('low-var', 'naive')

        match sampler:
            case 'low-var':
                t_0 = torch.rand(1).item() / batch_size
                ts = torch.arange(t_0, 1., 1 / batch_size, device=self.device)

                # Add single channel dimension
                return rearrange(ts, 'b -> b 1')
            
            case 'naive':
                return torch.rand((batch_size, 1), device=self.device)
            
        raise ValueError(f'Unknown sampler: {sampler}. Available samplers are: {samplers}')