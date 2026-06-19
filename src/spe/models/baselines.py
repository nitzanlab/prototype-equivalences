import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from typing import Tuple
from .NFDiffeo import ActNorm, LogTransf
from scipy import interpolate
from typing import Callable, Union
import pysindy as psi
import numpy as np
from tqdm import tqdm


class ODENet(nn.Module):

    def __init__(self, dim: int, hidden: tuple=(32, 32), activation=nn.SiLU):
        super().__init__()

        layers = []
        last = dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers += [nn.Linear(last, dim)]
        self.f = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return self.f(x)


def get_NODE(x: torch.Tensor, xdot: torch.Tensor, its: int=1000, lr: float=1e-3, hidden: tuple=None,
             width: int=32, depth: int=2, weight_decay: float=1e-3, activation=nn.SiLU, verbose: bool=False):
    if hidden is None:
        hidden = [width]*depth

    model = ODENet(dim=x.shape[-1], hidden=hidden, activation=activation)
    model.to(x.device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(range(its), disable=not verbose)
    for i in pbar:
        optim.zero_grad()

        loss = torch.mean((xdot - model(0, x))**2)
        loss.backward()
        optim.step()
        pbar.set_postfix_str(f'loss={loss.item():.4f}')

    model.requires_grad_(False)
    return lambda x: model(0, x)


def kNN_vectors(y: torch.Tensor, ydot: torch.Tensor, k: int=5) -> Callable:
    """
    A naive k-NN reconstruction of a vector field. Estimates the velocity at each point as the mean of the k-NN from an
    initial training set given during construction
    :param y: the positions of the observed training points, a torch tensor
    :param ydot: the velocities of the observed training points, a torch tensor
    :param k: the number of nearest neighbors to use (default: 5)
    :return: a function that estimates the velocity for each input point; recieves as input a time t and
             position tensor x, and returns a tensor with same shape as x which is the estimated velocity
    """
    yy = torch.sum(y*y, dim=-1)[None, :]

    def f(x: torch.Tensor) -> torch.Tensor:
        D = yy - 2*x@y.T + torch.sum(x*x, dim=1)[:, None]
        D = torch.argsort(D, dim=1)[:, :k]
        vecs = torch.stack([ydot[D[:, i]] for i in range(k)])
        return torch.mean(vecs, dim=0)

    return f


def fit_SINDy(x: torch.Tensor, xdot: torch.Tensor, library: str='poly', degree: int=3):
    """
    Wrapper function to fit SINDy
    :param x: the positions of the inputs, a torch tensor with shape [N, dim]
    :param xdot: the velocities of the inputs, a torch tensor with shape [N, dim]
    :param library: the feature library to use - either 'poly' for polynomial or 'fourier' for Fourier features
    :param degree: the number of features to use (in 'poly' this is directly the degree of the polynomial)
    :return: a Callable function which is the estimate velocity according to SINDy at each position
    """
    # create data for sindy
    times = np.random.rand(x.shape[0]) * 3
    x_multi, xdot_multi, t_multi = [], [], []
    for i in range(x.shape[0]):
        x_multi.append(x[i:i + 1])
        xdot_multi.append(xdot[i:i + 1])
        t_multi.append(times[i:i + 1])

    # create feature library
    if library.lower() == 'poly': library = psi.feature_library.PolynomialLibrary(degree=degree)
    elif library.lower() == 'fourier': library = psi.feature_library.FourierLibrary(n_frequencies=degree)
    else: raise NotImplementedError

    # fit SINDy
    sindy_model = psi.SINDy(feature_library=library)
    sindy_model = sindy_model.fit(x=x_multi, x_dot=xdot_multi, t=t_multi, multiple_trajectories=True)

    def sindy_func(v):
        v = torch.clamp(v, -100, 100)
        try:
            return torch.from_numpy(sindy_model.predict(v.detach().numpy())).float()
        except:
            return torch.zeros_like(v)
    return sindy_func
