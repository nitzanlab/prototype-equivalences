import torch
from torch import nn
from ..dynamics.utils import simulate_trajectory
from ..models.NFDiffeo import Diffeo
from ..dynamics.prototypes import Prototype, SOPrototype
from typing import Iterable, Union
from torch.optim import Adam
from tqdm.autonotebook import tqdm
import numpy as np


class SPEModel(nn.Module):

    def __init__(self, dim: int, prototype: Union[Prototype, tuple, dict], **NF_kwargs):
        super().__init__()
        if isinstance(prototype, tuple): prototype = SOPrototype(*prototype)
        if isinstance(prototype, dict): prototype = SOPrototype(**prototype)
        self.proto = prototype
        self.H = Diffeo(dim=dim, **NF_kwargs)
        self.dim = dim
        self.latent_dim = dim
        if 'latent_dim' in NF_kwargs: self.latent_dim = NF_kwargs['latent_dim']

    def freeze_scale(self, freeze: bool):
        self.H.freeze_scale(freeze)

    def logdet(self, x: torch.Tensor):
        return self.H.logdet(x)

    def loss_terms(self, x: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, jvp, ldet = self.H.jvp_forward(x, f)
        return self.proto(y), jvp, ldet

    def forward(self, x: torch.Tensor):
        return self.H.forward(x)

    def reverse(self, y: torch.Tensor):
        return self.H.reverse(y)

    def get_invariant(self, N: int, **kwargs) -> torch.Tensor:
        """
        Get points from the invariant set of the prototype, useful for plotting
        :param N: number of points on the attractor to return
        :return: around N points on the attractor, a torch tensor with shape [~N, dim]
        """
        return self.H.reverse(self.proto.get_invariant(N, self.latent_dim, **kwargs))

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the points x onto the invariant set of the prototype. In essence, this finds the closest point in the
        invariant set for each of the points x and allows for better evaluation
        :param x: the positions x, as a torch tensor with shape [N, dim]
        :return: the points after projecting them onto the invariant set, a torch tensor with shape [N, dim]
        """
        return self.H.reverse(self.proto.project_onto_invariant(x))

    @torch.no_grad()
    def trajectories(self, init: torch.Tensor, T: float = 10., step: float = 5e-2, euler: bool = False):
        init = self.H.forward(init)
        traj = self.proto.trajectories(init, T=T, step=step, euler=euler)
        return self.H.reverse(traj.reshape(-1, self.latent_dim)).reshape(traj.shape[0], traj.shape[1], -1)


# a default set of archetypes that can be used; these are the archetypes used for all 2D tests
_default_prototypes = [
    [-.25, .5, .5],
    [.25, .5, .5],
    [-.25, -.5, .5],
    [.25, -.5, .5],
]


def equiv_err(xdot, jvp, ydot, noise: float=0.):
    # define the MSE error for the smooth orbital equivalence
    err = (jvp / (torch.norm(jvp, dim=-1, keepdim=True) + 1e-6)
           - ydot / (torch.norm(ydot, dim=-1, keepdim=True) + 1e-6))**2
    if noise == 0: return err

    # if there is expected to be noise in the observations, down-weight points with small norms relative to noise
    norms = torch.norm(xdot, dim=-1, keepdim=True)**2
    nvar = noise*noise
    return err/(1+nvar/norms)


def proj_loss(x, model, proj_var, proj_dim: int=2):
    # define the loss related to how far the projection is from the true points
    if proj_var is None: return 0
    projLL = .5*x.shape[1]*proj_var
    y = model.forward(x)
    yproj = torch.cat([y[:, :proj_dim], torch.zeros(y.shape[0], y.shape[1]-proj_dim, device=y.device)], dim=-1)
    # diff = torch.mean((model.reverse(yproj) - x)**2)
    diff = torch.mean((model.project_onto_invariant(x) - x)**2)
    return torch.exp(proj_var)*diff - proj_dim*projLL/(x.shape[0]*x.shape[1])


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

        gHx, dHxdot, ldet = model.loss_terms(x, xdot)
        mseloss = torch.mean(equiv_err(gHx, dHxdot, xdot))  # calculate the error according to the equivalence

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
        gHx, dHxdot, ldet = model.loss_terms(x, xdot)
        err = torch.mean(equiv_err(gHx, dHxdot, xdot))
        ploss = proj_loss(x, model, proj_reg)
    dloss = torch.mean(torch.abs(ldet))  # adds the loss over the determinant (for regularization)

    score = torch.mean(err).item()
    loss = torch.mean(err).item() + det_reg*dloss.item() + ploss.item()
    # ==================================================================================================================

    return model, loss, score


def fit_all_prototypes(x: torch.Tensor, xdot: torch.Tensor, prototypes: Iterable=_default_prototypes, to_cpu: bool=True,
                       diffeo_args: dict=dict(), fitting_args: dict=dict()) -> dict:
    """
    A wrapper function that fits the given observations, x and xdot, to the list of supplied dynamics
    :param x: the positions of the observations, a torch tensor with shape [N, dim]
    :param xdot: the vectors associated with the above positions, a torch tensor with shape [N, dim]
    :param prototypes: a list of Prototype classes, which are the dynamics to use
    :param to_cpu: a boolean indicating whether all models should be returned to CPU after training
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
        'prototypes': prototypes,
        'losses': [],
        'scores': [],
        'Hs': []
    }

    device = x.device

    for proto in prototypes:
        if not isinstance(proto, Prototype):
            g = SOPrototype(**proto)
        else:
            g = proto
        H = SPEModel(dim=x.shape[1], prototype=g, **diffeo_args)

        H.to(device)
        H, loss, score = fit_prototype(H, x.clone(), xdot.clone(), **fitting_args)
        H.requires_grad_(False)
        if to_cpu: H.to('cpu')

        results['losses'].append(loss)
        results['scores'].append(score)
        results['Hs'].append(H)

    return results
