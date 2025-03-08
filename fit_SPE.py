import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from typing import Callable
from Hutils import get_oscillator
from NFDiffeo import Diffeo


# a default set of archetypes that can be used; these are the archetypes used for all 2D tests
_default_archetypes = [
    [-.25, .5, .5],
    [.25, .5, .5],
    [-.25, -.5, .5],
    [.25, -.5, .5],
]


def equiv_err(xdot, jvp, ydot, soequiv: bool=True, noise: float=0.):
    # define the MSE error for the equivalence used
    if soequiv:
        # smooth orbital equivalence
        err = (jvp / torch.norm(jvp, dim=-1, keepdim=True) - ydot / torch.norm(ydot, dim=-1, keepdim=True))**2
    else:
        # dynamical orbital equivalence
        err = (xdot / torch.norm(xdot, dim=-1, keepdim=True) - ydot / torch.norm(ydot, dim=-1, keepdim=True))**2

    # if no noise, return unweighted samples
    if noise == 0: return err
    else:
        norms = torch.norm(ydot, dim=-1, keepdim=True)**2
        nvar = noise*noise
        return err/(1+nvar/norms)


def proj_loss(x, y, H, proj_var):
    # define the loss related to how far the 2D projection is from the true points
    if proj_var is None: return 0
    projLL = .5*x.shape[1]*proj_var
    yproj = torch.cat([y[:, :2], torch.zeros(y.shape[0], y.shape[1]-2, device=y.device)], dim=-1)
    return torch.exp(proj_var)*torch.mean((H.reverse(yproj) - x)**2) - 2*projLL/(x.shape[0]*x.shape[1])


def fit_prototype(H: nn.Module, x: torch.Tensor, xdot: torch.Tensor, g: Callable, its: int=300, lr: float=5e-3,
                  verbose=False, freeze_frac: float=.0, det_reg: float=.0, center_reg: float=.0, weight_decay: float=1e-3,
                  proj_reg: float=None, soequiv: bool=True, dim2_weight: float=None, test_fr: float=0,
                  noise: float=0.):
    """
    Fits the supplied observations to the given archetype g using a variant of the smooth-equivalence loss defined in
    DFORM under the diffeomorophism H
    :param H: the diffeomorphism to be trained, usually an NFDiffeo.Diffeo object
    :param x: the positions of the observations, a torch tensor with shape [N, dim]
    :param xdot: the vectors associated with the above positions, a torch tensor with shape [N, dim]
    :param g: the archetype; a Callable that gets a position y as input and returns the velocity ydot at that position
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
    :param soequiv: a boolean indicating whether to use smooth-orbital equivalence or not; this should always be set to
                    to True, but if not then dynamical equivalence (see appendix of TWA) is used
    :param dim2_weight: the relative weight of the loss to the first two dimensions of the archetypes; if None, then the
                        weight defaults to 2/dim, the natural choice
    :param test_fr: if this value is bigger than 0, then the loss/score/ldet are all calculated with respect to a test
                    set that is held-out during training; test_fr is the fraction of points to test (between 0 and 1)
    :param noise: the standard deviation of the noise assumed to exist in the vector data, used for weighting loss
    :return: - the fitted network, H
             - the average loss over all observed vectors
             - the average log-determinant over all observed vectors
             - the score, which is the loss of the first two dimensions of the archetype, of the observed vectors
    """
    # ========================== initialize things before fitting ======================================================
    if freeze_frac > 0: H.freeze_scale(True)  # freeze weights if needed

    if x.shape[-1]==2: proj_reg = None

    if proj_reg is not None:
        proj_reg = torch.ones(1, device=x.device)*proj_reg
        proj_reg.requires_grad = True
        optim = Adam(list(H.parameters()) + [proj_reg], weight_decay=weight_decay)
    else:
        optim = Adam(list(H.parameters()), lr=lr, weight_decay=weight_decay)

    center = torch.mean(x, dim=0, keepdim=True)

    # if the loss is calculated on a held-out set, create this set
    if test_fr > 0:
        amnt = int(x.shape[0]*test_fr)
        inds = np.random.choice(x.shape[0], amnt, replace=False)
        x_test, xdot_test = x[inds].clone(), xdot[inds].clone()
        uninds = [i for i in range(x.shape[0]) if i not in inds]
        x, xdot = x[uninds], xdot[uninds]
    else:
        x_test = xdot_test = None
    # ==================================================================================================================

    unfrozen = False
    pbar = tqdm(range(its), disable=not verbose)
    # fitting process
    for i in pbar:
        optim.zero_grad()

        # if enough iterations have passed, unfreeze weights
        if freeze_frac > 0 and i > its*freeze_frac and not unfrozen:
            H.freeze_scale(False)  # unfreeze weights to do with determinant
            if proj_reg is not None: optim = Adam(list(H.parameters()) + [proj_reg], weight_decay=weight_decay)
            else: optim = Adam(H.parameters(), lr=lr, weight_decay=weight_decay)
            unfrozen = True

        y, jvp, ldet = H.jvp_forward(x, xdot)   # calculate transformed inputs
        ydot = g(y)  # velocities of vectors of archetypes

        err = equiv_err(xdot, jvp, ydot, soequiv, noise=noise)  # calculate the error according to the equivalence
        if dim2_weight is None:  # assumes all dimensions are the same
            mseloss = torch.mean(err)
        else:  # gives a different weight to the first two dimensions than all the other (0 < dim2_weight < 1)
            dim = err.shape[-1]
            loss2 = torch.mean(err[:, :2])*2/dim
            loss_other = torch.mean(err[:, 2:])*(dim-2)/dim
            mseloss = dim2_weight*loss2 + (1-dim2_weight)*loss_other

        dloss = torch.mean(torch.abs(ldet))  # adds the loss over the determinant (for regularization)
        closs = torch.mean((center-H.reverse(torch.zeros_like(center)))**2)  # adds loss for transformation of center (regularization)
        ploss = proj_loss(x, y, H, proj_reg)  # adds loss for projection
        loss = mseloss + det_reg*dloss + center_reg*closs + ploss  # put full loss together

        loss.backward()
        nn.utils.clip_grad_norm_(H.parameters(), 10)  # clip gradients for stable training
        optim.step()

        loss = loss.item()
        pbar.set_postfix_str(f'loss={loss:.4f}'+
                             (f'; projvar={torch.exp(proj_reg).item():.3f}' if proj_reg is not None else ''))

    # ========================== calculate final losses for everything =================================================
    H.requires_grad = False
    with torch.no_grad():
        if x_test is not None:
            y, jvp, ldet = H.jvp_forward(x_test.to(x.device), xdot_test.to(x.device))
            ploss = proj_loss(x_test, y, H, proj_reg).item() if proj_reg is not None else 0
        y, jvp, ldet = H.jvp_forward(x.clone(), xdot.clone())
        ploss = proj_loss(x, y, H, proj_reg).item() if proj_reg is not None else 0

    ydot = g(y)
    err = equiv_err(xdot, jvp, ydot, noise=noise) if x_test is None else equiv_err(xdot_test, jvp, ydot, noise=noise)
    score = torch.mean(err[:, :2]).item()
    loss = torch.mean(err).item() + ploss
    ldet = torch.mean(ldet.detach()).item()
    # ==================================================================================================================

    return H, loss, ldet, score


def fit_all_prototypes(x: torch.Tensor, xdot: torch.Tensor, archetypes: list=_default_archetypes,
                       diffeo_args: dict=dict(), **fitting_args: dict) -> dict:
    """
    A wrapper function that fits the given observations, x and xdot, to the list of supplied archetypes
    :param x: the positions of the observations, a torch tensor with shape [N, dim]
    :param xdot: the vectors associated with the above positions, a torch tensor with shape [N, dim]
    :param archetypes: a list of the parameters for the SO archetypes. Each item in the list is a sublist with three
                       values (a, omega, decay) that dictate the parameters of the simple oscillator
    :param diffeo_args: dictionary of named arguments to pass on when initializing the diffeomorphism (all except
                        dimension, which is automatically set); see documentation in NFDiffeo.Diffeo for possible args
    :param fitting_args: any other named argument that is supplied will be in this dictionary, which is passed on to
                         the fitting algorithm; see the documentation for fit_DFORM above for possible values
    :return: a dictionary with the following keys:
            - archetypes: the list of archetypes supplied as input to this function
            - losses:     a list of the losses for each archetype, in the same order as the above list of archetypes
            - ldets:      a list of the average log determinants for each archetype, over all samples x
            - scores:     a list of the scores for each archetype, which are the statistics used to classify
            - Hs:         a list containing all of the fitted diffeomorphisms, in the same order as all of the above
    """
    results = {
        'archetypes': archetypes,
        'losses': [],
        'ldets': [],
        'scores': [],
        'Hs': []
    }

    for (a, omega, decay) in archetypes:
        g = get_oscillator(a=a, omega=omega, decay=decay)
        H = Diffeo(dim=x.shape[1], **diffeo_args)
        H, loss, ldet, score = fit_prototype(H, x, xdot, g, **fitting_args)
        H.eval()
        H.requires_grad_(False)

        results['losses'].append(loss)
        results['ldets'].append(ldet)
        results['scores'].append(score)
        results['Hs'].append(H)

    return results
