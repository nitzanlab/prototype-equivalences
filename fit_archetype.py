import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from typing import Callable


def equiv_err(xdot, jvp, ydot, soequiv: bool=True):
    # define the MSE error for the equivalence used
    if soequiv:
        # smooth orbital equivalence
        return jvp / torch.norm(jvp, dim=-1, keepdim=True) - ydot / torch.norm(ydot, dim=-1, keepdim=True)
    else:
        # dynamical orbital equivalence
        return xdot / torch.norm(xdot, dim=-1, keepdim=True) - ydot / torch.norm(ydot, dim=-1, keepdim=True)


def proj_loss(x, y, H, proj_var):
    # theta = std_var*std_var
    # alpha = mean_var/(std_var*std_var)

    # define the loss related to how far the 2D projection is from the true points
    if proj_var is None: return 0
    # projLL = .5*x.shape[1]*proj_var + (alpha-1)*proj_var - torch.exp(proj_var)/theta
    projLL = .5*x.shape[1]*proj_var
    yproj = torch.cat([y[:, :2], torch.zeros(y.shape[0], y.shape[1]-2, device=y.device)], dim=-1)
    return torch.exp(proj_var)*torch.mean((H.reverse(yproj) - x)**2) - 2*projLL/(x.shape[0]*x.shape[1])


def fit_DFORM(H: nn.Module, x: torch.Tensor, xdot: torch.Tensor, g: Callable, its: int=300, lr: float=5e-3,
              verbose=False, freeze_frac: float=.0, det_reg: float=.0, center_reg: float=.0, weight_decay: float=1e-3,
              proj_reg: float=None, soequiv: bool=True, dim2_weight: float=None, test_fr: float=0):

    # ========================== initialize things before fitting ======================================================
    if freeze_frac > 0: H.freeze_scale(True)  # freeze weights if needed

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

        err = equiv_err(xdot, jvp, ydot, soequiv)  # calculate the error according to the equivalence
        if dim2_weight is None:  # assumes all dimensions are the same
            mseloss = torch.mean(err**2)
        else:  # gives a different weight to the first two dimensions than all the other (0 < dim2_weight < 1)
            dim = err.shape[-1]
            loss2 = torch.mean(err[:, :2]**2)*2/dim
            loss_other = torch.mean(err[:, 2:]**2)*(dim-2)/dim
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
    err = equiv_err(xdot, jvp, ydot) if x_test is None else equiv_err(xdot_test, jvp, ydot)
    score = torch.mean(err[:, :2]**2).item()
    loss = torch.mean(err**2).item() + ploss
    ldet = torch.mean(ldet.detach()).item()
    # ==================================================================================================================

    return H, loss, ldet, score
