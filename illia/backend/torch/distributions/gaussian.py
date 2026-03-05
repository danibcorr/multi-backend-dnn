"""PyTorch learnable Gaussian distribution."""

# 3pps
import torch
from torch import nn

# Own modules
from illia.backend.base.distributions.base import DistributionBase


class GaussianDistribution(DistributionBase, nn.Module):
    """Learnable Gaussian with diagonal covariance.

    The standard deviation is derived from *rho* via softplus to ensure
    positivity.  KL divergence can be estimated from ``log_prob`` differences.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        nn.Module.__init__(self)

        self.shape = shape

        self.register_buffer("mu_prior", torch.tensor([mu_prior]))
        self.register_buffer("std_prior", torch.tensor([std_prior]))

        self.mu = nn.Parameter(torch.empty(shape).normal_(mu_init, 0.1))
        self.rho = nn.Parameter(torch.empty(shape).normal_(rho_init, 0.1))

    @torch.jit.export
    def sample(self) -> torch.Tensor:
        """Draw a reparameterised sample."""
        eps = torch.randn_like(self.rho)
        sigma = torch.log1p(torch.exp(self.rho))
        return self.mu + sigma * eps

    @torch.jit.export
    def log_prob(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Log-probability of *x* (sampled internally when *None*)."""
        if x is None:
            x = self.sample()

        pi = torch.acos(torch.zeros(1)) * 2
        log_norm = -torch.log(torch.sqrt(2 * pi))

        log_prior = (
            log_norm.to(x.device)
            - torch.log(self.std_prior)
            - ((x - self.mu_prior) ** 2) / (2 * self.std_prior**2)
            - 0.5
        )

        sigma = torch.log1p(torch.exp(self.rho)).to(x.device)
        log_posterior = (
            log_norm.to(x.device)
            - torch.log(sigma)
            - ((x - self.mu) ** 2) / (2 * sigma**2)
            - 0.5
        )

        return log_posterior.sum() - log_prior.sum()

    @property
    @torch.jit.export
    @torch.no_grad()
    def num_params(self) -> int:
        """Total number of learnable parameters."""
        return len(self.mu.view(-1))
