from fit_archetype import fit_DFORM
from NFDiffeo import Diffeo
import numpy as np
import torch
import pickle
import click
from pathlib import Path
from Hutils import get_oscillator, simulate_trajectory, interp_vectors
from systems import SO, PhaseSpace
import matplotlib
from tqdm import tqdm

from matplotlib import pyplot as plt

plt.rcParams.update({
    # 'text.usetex': True,
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'figure.dpi': 90,
    'figure.autolayout': True,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'font.size': 16,
    'axes.linewidth': 2,
    'lines.linewidth': 3,
    'legend.handlelength': 2.,
    'legend.handletextpad': .4,
    'legend.labelspacing': .0,
    'legend.borderpad': .4,
    'legend.fontsize': 14,
    'font.weight': 'normal',
    'axes.edgecolor': '#303030',
    'savefig.facecolor': 'white',
    'text.latex.preamble': r'\usepackage{amsfonts}',
})


def traj_in_vecs(x: torch.Tensor, xdot: torch.Tensor, arch: PhaseSpace, H: Diffeo,
                 T: float=40, extra_str: str='', color: str='k'):

    # choose points as initial points from which to iterate
    inds = np.random.choice(x.shape[0], 3, replace=False)

    # get fitted trajectories
    y = H.forward(x[inds]).detach()
    traj = arch.trajectories(y, T=T)
    traj = H.reverse(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape).detach()

    x, xdot = x[..., :2].detach(), xdot[..., :2].detach()
    # traj = H.forward(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape).detach()[..., :2]

    traj = traj.cpu().numpy()

    # normalize all velocities to have a norm of 1 for readable plots
    norms = torch.clamp(torch.norm(xdot, dim=1, keepdim=True), 1)
    xdot = xdot/norms

    x, xdot = x.cpu().numpy(), xdot.cpu().numpy()

    # scatter half of vectors as gray
    inds = np.random.choice(x.shape[0], int(x.shape[0]*.5), replace=False)
    plt.scatter(x[inds, 0], x[inds, 1], 25, 'gray', alpha=.25)
    plt.quiver(x[inds, 0], x[inds, 1], xdot[inds, 0], xdot[inds, 1], scale=3, color='gray', alpha=.25, width=.012)

    # scatter chosen vectors from the dynamics
    inds = np.random.choice(x.shape[0], 50, replace=False)
    plt.scatter(x[inds, 0], x[inds, 1], 25, 'k', alpha=.5)
    plt.quiver(x[inds, 0], x[inds, 1], xdot[inds, 0], xdot[inds, 1], scale=3, color='k', alpha=.5, width=.012)

    # plot fitted and true trajectories (all in 2D space)
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], lw=2, alpha=.7, color='tab:red')

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title(extra_str, color=color)

# ================================================= load data ==========================================================
data_root = 'data/pancreas_ductal/'
xdot = np.load(data_root + 'X_test.npy')
x = np.load(data_root + 'coords_test.npy')

x, xdot = torch.from_numpy(x).float(), torch.from_numpy(xdot).float()
device = 'cpu'
x, xdot = x.to(device), xdot.to(device)

# ================================================= parameters =========================================================
omega = .8
decay = .25

archetypes = [
        # [-.25, -omega, decay],
        # [.25, -omega, decay],
        [-.25, omega, decay],
        [.25, omega, decay],
    ]

# ================================================= fitting ============================================================
its = 1000
proj = -1
lr = 1e-3
n_layers = 3
n_freqs = 5
dim = 2
x, xdot = x[:, :dim], xdot[:, :dim]
# xdot = 2*xdot

x, xdot = interp_vectors(x, xdot, smoothing=.01)

plt.figure(figsize=(8, 4*len(archetypes)//2))
subplot = 1
for (a, omega, decay) in archetypes:
    # ============================================ fit archetype ==========================================
    archetype = get_oscillator(a=a, omega=omega, decay=decay)
    H = Diffeo(dim=dim, n_layers=n_layers, K=n_freqs, rank=2, affine_init=False).to(device)
    H, loss, ldet, score = fit_DFORM(H, x.clone(), xdot.clone(), archetype, its=its,
                                     verbose=True, lr=lr, freeze_frac=.0, proj_reg=proj if dim>2 else None,
                                     weight_decay=1e-3, center_reg=0)

    print(f'archetype a={a:.2f} om={omega:.2f}: loss={loss:.4f}; ldet={ldet:.2f}, 2D loss={score:.3f}\n')
    # ============================================ plots ==========================================
    plt.subplot(len(archetypes)//2, 2, subplot)
    traj_in_vecs(x.clone(), xdot.clone(), SO(a=a, omega=omega, decay=decay),
                 H, T=20, extra_str=f'{"cycle" if a>0 else "point"}\n'
                                    f'omega={omega:.2f} ; a={a:.2f}\n'
                                    f'loss={loss:.2f} ; ldet={ldet:.1f}\n2D-loss={score:.3f}')
    subplot += 1
plt.tight_layout()
plt.show()
