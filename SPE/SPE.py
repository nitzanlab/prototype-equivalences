import torch
from torch import nn
from dynamics.utils import simulate_trajectory
from models.NFDiffeo import Diffeo
from dynamics.prototypes import Prototype, SOPrototype
from typing import Iterable, Union
from torch.optim import Adam
from tqdm import tqdm
import numpy as np


class SPEModel(nn.Module):

    def __init__(self): super().__init__()

    def loss_terms(self, x: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor: raise NotImplementedError

    def logdet(self, x: torch.Tensor) -> torch.Tensor: raise NotImplementedError

    def get_invariant(self, N: int) -> torch.Tensor:
        """
        Get points from the invariant set of the prototype, useful for plotting
        :param N: number of points on the attractor to return
        :return: around N points on the attractor, a torch tensor with shape [~N, dim]
        """
        raise NotImplementedError

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the points x onto the invariant set of the prototype. In essence, this finds the closest point in the
        invariant set for each of the points x and allows for better evaluation
        :param x: the positions x, as a torch tensor with shape [N, dim]
        :return: the points after projecting them onto the invariant set, a torch tensor with shape [N, dim]
        """
        raise NotImplementedError

    @torch.no_grad()
    def trajectories(self, init: torch.Tensor, T: float = 10., step: float = 5e-2, euler: bool = False):
        return simulate_trajectory(self.forward, init, T=T, step=step, euler=euler)


class NFSmoothOrbital(SPEModel):

    def __init__(self, dim: int, g: Union[Prototype, tuple, dict], **NF_kwargs):
        super().__init__()
        if isinstance(g, tuple): g = SOPrototype(*g)
        if isinstance(g, dict): g = SOPrototype(**g)
        self.g = g
        self.H = Diffeo(dim=dim, **NF_kwargs)
        self.dim = dim
        if 'latent_dim' in NF_kwargs: self.dim = NF_kwargs['latent_dim']

    def logdet(self, x: torch.Tensor):
        return self.H.logdet(x)

    def loss_terms(self, x: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, jvp, ldet = self.H.forward_jvp_logdet(x, f)
        return self.g(y), jvp, ldet

    def forward(self, x: torch.Tensor):
        y = self.H.forward(x)
        return self.H.inv_jvp(y, self.g(y))

    def get_invariant(self, N: int):
        return self.H.reverse(self.g.get_invariant(N, self.dim))

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        return self.H.reverse(self.g.project_onto_invariant(x))


# a default set of archetypes that can be used; these are the archetypes used for all 2D tests
_default_prototypes = [
    [-.25, .5, .5],
    [.25, .5, .5],
    [-.25, -.5, .5],
    [.25, -.5, .5],
]


def equiv_err(xdot, pred):
    # define the MSE error for the smooth orbital equivalence
    err = (xdot / (torch.norm(xdot, dim=-1, keepdim=True) + 1e-6)
           - pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-6))**2
    return err


def proj_loss(x: torch.Tensor, model: SPEModel, proj_var: torch.Tensor):
    # define the loss related to how far the 2D projection is from the true points
    if proj_var is None: return torch.zeros_like(x[0, 0])
    projLL = .5*x.shape[1]*proj_var
    y = model.project_onto_invariant(x)
    return torch.exp(proj_var)*torch.mean((y - x)**2) - 2*projLL/(x.shape[0]*x.shape[1])


def fit_prototype(model: SPEModel, x: torch.Tensor, xdot: torch.Tensor, its: int=300, lr: float=5e-3,
                  verbose=False, det_reg: float=.0, weight_decay: float=1e-3, proj_reg: float=None):
    """
    Fits the supplied observations to a prototype using the DFORM loss
    :param model: the diffeomorphism-wrapped prototype to be trained, as a prototypes.DiffeoWrapper object
    :param x: the positions of the observations, a torch tensor with shape [N, dim]
    :param xdot: the vectors associated with the above positions, a torch tensor with shape [N, dim]
    :param its: number of optimization iterations
    :param lr: the learning rate
    :param verbose: a boolean indicating whether progress should be printed (True) or not (False)
    :param det_reg: how much regularization should be added on the absolute value of the determinant
    :param weight_decay: amount of weight decay to use during training
    :param proj_reg: initial strength of the projection regularization used during fitting; this value is in log-scale,
                     so for no regularization use proj_reg=None (proj_reg=0 is like a regularization strength of 1)
    :return: - the fitted model
             - the average loss over all observed vectors
             - the score, which is just the equivalence loss of the observed vectors
    """
    # ========================== initialize things before fitting ======================================================
    # initialize parameters list
    params = list(model.parameters())

    if x.shape[-1]==2: proj_reg = None

    # define optimizer
    if proj_reg is not None:
        proj_reg = torch.ones(1, device=x.device)*proj_reg
        proj_reg.requires_grad = True
        optim = Adam(params + [proj_reg], weight_decay=weight_decay)
    else:
        optim = Adam(params, lr=lr, weight_decay=weight_decay)

    # ==================================================================================================================

    pbar = tqdm(range(its), disable=not verbose)
    # fitting process
    for i in pbar:
        optim.zero_grad()

        ldet = model.logdet(x)
        pred = model.forward(x)
        mseloss = torch.mean(equiv_err(pred, xdot))  # calculate the error according to the equivalence

        dloss = torch.mean(torch.abs(ldet))  # adds the loss over the determinant (for regularization)
        ploss = proj_loss(x, model, proj_reg)  # adds loss for projection
        loss = mseloss + det_reg*dloss + ploss  # put full loss together

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)  # clip gradients for stable training
        optim.step()

        loss = loss.item()
        pbar.set_postfix_str(f'loss={loss:.4f}'+
                             (f'; projvar={torch.exp(proj_reg).item():.3f}' if proj_reg is not None else ''))

    # ========================== calculate final losses for everything =================================================
    with torch.no_grad():
        ploss = proj_loss(x, model, proj_reg)  # adds loss for projection
        ldet = model.logdet(x)
        pred = model.forward(x)
        err = torch.mean(equiv_err(pred, xdot))
    dloss = torch.mean(torch.abs(ldet))  # adds the loss over the determinant (for regularization)

    score = torch.mean(err).item()
    loss = torch.mean(err).item() + det_reg*dloss.item() + ploss.item()
    # ==================================================================================================================

    return model, loss, score


def fit_all_prototypes(x: torch.Tensor, xdot: torch.Tensor, prototypes: Iterable=_default_prototypes,
                       diffeo_args: dict=dict(), **fitting_args: dict) -> dict:
    """
    A wrapper function that fits the given observations, x and xdot, to the list of supplied dynamics
    :param x: the positions of the observations, a torch tensor with shape [N, dim]
    :param xdot: the vectors associated with the above positions, a torch tensor with shape [N, dim]
    :param prototypes: a list of Prototype classes, which are the dynamics to use
    :param diffeo_args: dictionary of named arguments to pass on when initializing the diffeomorphism (all except
                        dimension, which is automatically set); these arguments are:
                            - rank: the rank used in the Affine transformations
                            - n_layers: number of Affine-Coupling-ReverseCoupling layers to use
                            - K: number of hidden units to use, either for the transformations
                            - actnorm: a boolean determinig if the first layer is an ActNorm
                            - n_householder: number of Householder transformations to use
                            - logtransf: a boolean indicating if to log-transform in the first layer or not

    :param fitting_args: any other named argument that is supplied will be in this dictionary, which is passed on to
                         the fitting algorithm; these arguments are:
                            - its: number of optimization iterations
                            - lr: the learning rate
                            - verbose: a boolean indicating whether progress should be printed
                            - det_reg: regularization on the absolute value of the determinant
                            - weight_decay: amount of weight decay to use during training

    :return: a dictionary with the following keys:
            - dynamics: the list of dynamics supplied as input to this function
            - losses:     a list of the losses for each prototype, in the same order as the above list of dynamics
            - ldets:      a list of the average log determinants for each prototype, over all samples x
            - scores:     a list of the scores for each prototype, which are the statistics used to classify
            - Hs:         a list containing all of the fitted models, in the same order as all of the above
                          these Hs are returned as SPE.models.SPEModel objects
    """
    results = {
        'dynamics': prototypes,
        'losses': [],
        'ldets': [],
        'scores': [],
        'Hs': []
    }

    for (a, omega, decay) in prototypes:
        g = SOPrototype(a=a, omega=omega, decay=decay, optimize=False)
        H = NFSmoothOrbital(dim=x.shape[1], g=g, **diffeo_args)
        H, loss, ldet, score = fit_prototype(H, x, xdot, **fitting_args)

        results['losses'].append(loss)
        results['ldets'].append(ldet)
        results['scores'].append(score)
        results['Hs'].append(H)

    return results
