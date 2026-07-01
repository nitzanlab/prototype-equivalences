import torch
from torch import nn
import pysindy as psi
import numpy as np
from tqdm import tqdm
from typing import Callable


class MLP(nn.Module):
    """A simple feed-forward network, mapping `in_dim` to `out_dim`."""

    def __init__(self, in_dim: int, out_dim: int, hidden: tuple=(32, 32), activation=nn.SiLU):
        super().__init__()

        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.f = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.f(x)


def _torch_library(library: str, degree: int, dim: int) -> tuple:
    """
    Build a differentiable torch evaluation of a pysindy feature library on the `dim`-dimensional latent space.
    The library structure (which candidate terms to include) is taken from pysindy - for consistency with `fit_SINDy`
    - but the features are evaluated in torch so the SINDy coefficients can be learned jointly with the autoencoder.
    :return: a tuple (theta, n_features) where theta maps a [N, dim] tensor to the [N, n_features] feature matrix
    """
    if library.lower() == 'poly':
        lib = psi.feature_library.PolynomialLibrary(degree=degree)
        lib.fit(np.zeros((1, dim)))
        # powers_ has shape [n_features, dim]: the exponent of each latent variable in each candidate term
        powers = torch.from_numpy(lib.powers_.astype(np.float32))

        def theta(z: torch.Tensor) -> torch.Tensor:
            p = powers.to(z.device)
            return torch.prod(z[:, None, :] ** p[None, :, :], dim=-1)

        return theta, powers.shape[0]

    elif library.lower() == 'fourier':
        # match pysindy's FourierLibrary: sin/cos of each latent variable at frequencies 1..degree
        freqs = torch.arange(1, degree + 1, dtype=torch.float32)

        def theta(z: torch.Tensor) -> torch.Tensor:
            f = freqs.to(z.device)
            arg = z[:, :, None] * f[None, None, :]                       # [N, dim, degree]
            feats = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # [N, dim, 2*degree]
            return feats.reshape(z.shape[0], -1)

        return theta, dim * 2 * degree

    else:
        raise NotImplementedError


class SINDyAutoencoder(nn.Module):
    """
    A bare-bones SINDy autoencoder (Champion et al., 2019). An encoder maps the high-dimensional input x to a
    low-dimensional latent z, a decoder reconstructs x from z, and a SINDy model with learnable coefficients Xi
    captures the latent dynamics zdot = Theta(z) @ Xi. After fitting, the module acts as a vector field in the
    original space, x -> xdot, analogous to `get_NODE`.
    """

    def __init__(self, dim: int, latent: int, library: str='poly', degree: int=3,
                 hidden: tuple=(32, 32), activation=nn.SiLU):
        super().__init__()
        self.encoder = MLP(dim, latent, hidden=hidden, activation=activation)
        self.decoder = MLP(latent, dim, hidden=hidden, activation=activation)
        self.theta, n_features = _torch_library(library, degree, latent)
        self.Xi = nn.Parameter(torch.randn(n_features, latent) * 0.1)
        # mask used for sequential thresholding (a proxy for L0 sparsity)
        self.register_buffer('mask', torch.ones(n_features, latent))

    def coefficients(self) -> torch.Tensor:
        return self.mask * self.Xi

    def latent_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """The SINDy estimate of zdot at latent points z."""
        return self.theta(z) @ self.coefficients()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate xdot in the original space: decode the latent dynamics through the decoder Jacobian."""
        z = self.encoder(x)
        zdot = self.latent_dynamics(z)
        return torch.func.jvp(self.decoder, (z,), (zdot,))[1]


def get_SINDy_AE(x: torch.Tensor, xdot: torch.Tensor, latent: int=3, library: str='poly', degree: int=3,
                 its: int=1000, lr: float=1e-3, hidden: tuple=None, width: int=32, depth: int=2,
                 weight_decay: float=1e-3, activation=nn.SiLU, l1: float=1e-3, l2: float=1e-3, l_reg: float=1e-5,
                 threshold: float=0.1, thresh_freq: int=100, verbose: bool=False) -> Callable:
    """
    Fit a SINDy autoencoder (Champion et al., 2019) to high-dimensional data and its derivatives, following the same
    conventions as `get_NODE`. Returns a frozen Callable mapping x to the estimated xdot in the original space.
    :param x: the positions of the inputs, a torch tensor with shape [N, dim]
    :param xdot: the velocities of the inputs, a torch tensor with shape [N, dim]
    :param latent: the dimension of the discovered latent coordinates z (d << dim)
    :param library: the feature library for the latent dynamics - 'poly' or 'fourier' (as in `fit_SINDy`)
    :param degree: the degree (poly) / number of frequencies (fourier) of the feature library
    :param l1: weight of the SINDy-in-x loss (reconstruction of xdot through the decoder)
    :param l2: weight of the SINDy-in-z loss (prediction of the latent velocities)
    :param l_reg: weight of the L1 regularization on the SINDy coefficients Xi
    :param threshold: coefficients below this magnitude are zeroed during sequential thresholding
    :param thresh_freq: how often (in iterations) to apply sequential thresholding
    :return: a Callable estimating the velocity in the original space at each input position
    """
    if hidden is None:
        hidden = [width]*depth

    model = SINDyAutoencoder(dim=x.shape[-1], latent=latent, library=library, degree=degree,
                             hidden=tuple(hidden), activation=activation)
    model.to(x.device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(range(its), disable=not verbose)
    for i in pbar:
        optim.zero_grad()

        # encoder coordinates and their velocities zdot = grad(phi) @ xdot, via a Jacobian-vector product
        z, zdot_true = torch.func.jvp(model.encoder, (x,), (xdot,))
        xhat = model.decoder(z)

        # SINDy prediction of the latent dynamics, and its decoding back to the original space
        zdot_pred = model.latent_dynamics(z)
        xdot_pred = torch.func.jvp(model.decoder, (z,), (zdot_pred,))[1]

        recon = torch.mean((x - xhat)**2)            # autoencoder reconstruction
        dxdt = torch.mean((xdot - xdot_pred)**2)     # SINDy loss in x
        dzdt = torch.mean((zdot_true - zdot_pred)**2)  # SINDy loss in z
        reg = torch.mean(torch.abs(model.Xi))        # L1 sparsity on coefficients
        loss = recon + l1*dxdt + l2*dzdt + l_reg*reg

        loss.backward()
        optim.step()

        # sequential thresholding as a proxy for L0 sparsity
        if threshold > 0 and (i + 1) % thresh_freq == 0:
            with torch.no_grad():
                model.mask *= (model.Xi.abs() >= threshold).float()

        pbar.set_postfix_str(f'loss={loss.item():.4f}')

    model.requires_grad_(False)
    model.eval()
    return model
