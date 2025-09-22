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

    def freeze_scale(self, freeze: bool): raise NotImplementedError

    def loss_terms(self, x: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor: raise NotImplementedError

    def reverse(self, y: torch.Tensor) -> torch.Tensor: raise NotImplementedError

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
        self.latent_dim = dim
        if 'latent_dim' in NF_kwargs: self.latent_dim = NF_kwargs['latent_dim']

    def freeze_scale(self, freeze: bool):
        self.H.freeze_scale(freeze)

    def logdet(self, x: torch.Tensor):
        return self.H.logdet(x)

    def loss_terms(self, x: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, jvp, ldet = self.H.jvp_forward(x, f)
        return self.g(y), jvp, ldet

    def forward(self, x: torch.Tensor):
        return self.H.forward(x)

    def reverse(self, y: torch.Tensor):
        return self.H.reverse(y)

    def get_invariant(self, N: int):
        return self.H.reverse(self.g.get_invariant(N, self.latent_dim))

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        return self.H.reverse(self.g.project_onto_invariant(x))

    def trajectories(self, init: torch.Tensor, T: float = 10., step: float = 5e-2, euler: bool = False):
        init = self.H.forward(init)
        traj = self.g.trajectories(init, T=T, step=step, euler=euler)
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
    return torch.exp(proj_var)*torch.mean((model.reverse(yproj) - x)**2) - proj_dim*projLL/(x.shape[0]*x.shape[1])


def fit_prototype(model: SPEModel, x: torch.Tensor, xdot: torch.Tensor, its: int=300,
                  lr: float=5e-3, verbose=False, freeze_frac: float=.0, det_reg: float=.0, center_reg: float=.0,
                  weight_decay: float=1e-3, proj_reg: float=None, noise: float=0.):
    """
    Fits the supplied observations to the given prototype g using a variant of the smooth-equivalence loss defined in
    DFORM under the diffeomorophism H
    :param model: the model to fit, an SPEModel class (see above)
    :param x: the positions of the observations, a torch tensor with shape [N, dim]
    :param xdot: the vectors associated with the above positions, a torch tensor with shape [N, dim]
    :param g: the prototype; a Callable that gets a position y as input and returns the velocity ydot at that position
    :param its: number of GD iterations
    :param lr: the learning rate
    :param verbose: a boolean indicating whether progress should be printed (True) or not (False)
    :param freeze_frac: fraction of iterations to freeze non-linear transformations in H (between 0 and 1)
    :param det_reg: how much regularization should be added on the absolute value of the determinant
    :param center_reg: a regularization coefficient that penalizes transformations whose stationary point is very far
                       from the mean of the data
    :param weight_decay: amount of weight decay to use during training
    :param proj_reg: initial strength of the projection regularization used during fitting; this value is in log-scale,
                     so for no regularization use proj_reg=None (proj_reg=0 is like a regularization strength of 1)
    :param noise: the standard deviation of the noise assumed to exist in the vector data, used for weighting loss
    :return: - the fitted network, H
             - the average loss over all observed vectors
             - the average log-determinant over all observed vectors
             - the score, which is the loss of the first two dimensions of the prototype, of the observed vectors
    """
    # ========================== initialize things before fitting ======================================================
    if freeze_frac > 0: model.freeze_scale(True)  # freeze weights if needed
    if freeze_frac > 1: freeze_frac = freeze_frac/its

    # calculate data mean for center regularization
    center = torch.mean(x, dim=0, keepdim=True)

    # initialize parameters list
    params = list(model.parameters())

    # setup all of the needed options for projection regularization
    if x.shape[-1]==2: proj_reg = None
    if proj_reg is not None:
        proj_reg = torch.ones(1, device=x.device)*proj_reg
        proj_reg.requires_grad = True
        params += [proj_reg]


    # define optimizer
    optim = Adam(params, lr=lr, weight_decay=weight_decay)
    # ==================================================================================================================

    unfrozen = False
    pbar = tqdm(range(its), disable=not verbose)
    # fitting process
    for i in pbar:
        optim.zero_grad()

        # if enough iterations have passed, unfreeze weights
        if freeze_frac > 0 and i > its*freeze_frac and not unfrozen:
            model.freeze_scale(False)  # unfreeze weights to do with determinant
            if proj_reg is not None: optim = Adam(list(model.parameters()) + [proj_reg], weight_decay=weight_decay)
            else: optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            unfrozen = True

        ydot, jvp, ldet = model.loss_terms(x, xdot)   # calculate transformed inputs

        err = equiv_err(xdot, jvp, ydot, noise=noise)  # calculate the error according to the equivalence
        mseloss = torch.mean(err)

        dloss = torch.mean(torch.abs(ldet))  # adds the loss over the determinant (for regularization)
        closs = torch.mean((center-model.reverse(torch.zeros_like(center)))**2)  # adds loss for transformation of center (regularization)
        ploss = proj_loss(x, model, proj_reg)  # adds loss for projection
        loss = mseloss + det_reg*dloss + center_reg*closs + ploss  # put full loss together

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.)  # clip gradients for stable training
        optim.step()

        loss = loss.item()
        pbar.set_postfix_str(f'loss={loss:.4f}'+
                             (f'; proj_reg={torch.exp(proj_reg).item():.3f}' if proj_reg is not None else ''))

    # ========================== calculate final losses for everything =================================================
    model.eval()

    with torch.no_grad():
        ydot, jvp, ldet = model.loss_terms(x, xdot)  # calculate transformed inputs

        err = equiv_err(xdot, jvp, ydot, noise=noise)  # calculate the error according to the equivalence
        mseloss = torch.mean(err)

        dloss = torch.mean(torch.abs(ldet))  # adds the loss over the determinant (for regularization)
        closs = torch.mean((center - model.reverse(
            torch.zeros_like(center))) ** 2)  # adds loss for transformation of center (regularization)
        ploss = proj_loss(x, model, proj_reg)  # adds loss for projection
        loss = mseloss + det_reg * dloss + center_reg * closs + ploss  # put full loss together

    score = torch.mean(err[:, :2]).item()
    loss = loss.item()
    # ==================================================================================================================

    return model, loss, score


def fit_cleaner(model: SPEModel, x: torch.Tensor, xdot: torch.Tensor, its: int=300, lr: float=5e-3,
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
        'ldets': [],
        'scores': [],
        'Hs': []
    }

    device = x.device

    for proto in prototypes:
        if not isinstance(proto, Prototype):
            g = SOPrototype(**proto)
        else:
            g = proto
        H = NFSmoothOrbital(dim=x.shape[1], g=g, **diffeo_args)

        H.to(device)
        H, loss, score = fit_prototype(H, x.clone(), xdot.clone(), **fitting_args)
        H.requires_grad_(False)
        if to_cpu: H.to('cpu')

        results['losses'].append(loss)
        results['scores'].append(score)
        results['Hs'].append(H)

    return results
