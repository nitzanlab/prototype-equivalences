from fit_archetype import fit_DFORM
from NFDiffeo import Diffeo
import numpy as np
import torch
import pickle
import click
from pathlib import Path
from Hutils import get_oscillator, simulate_trajectory
from multidim_benchmark import traj_in_vecs
from systems import SO
import matplotlib
from tqdm import tqdm
matplotlib.use('pgf')

from matplotlib import pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
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


@torch.no_grad()
def evaluate_cycle(H: Diffeo, archetype: SO, aug: Diffeo, gt_system: SO,
                   M: int=500) -> [float, torch.Tensor, torch.Tensor]:
    """
    Assuming that gt_system -> aug -> x <- H <- archetype
    :param H: fitted diffeomorphism to archetype
    :param archetype: the archetype used to fit
    :param aug: the augmentation applied to the ground-truth system
    :param gt_system: the ground-truth system
    :param M: the number of points used to evaluate the models
    :return:
        - a float that is a sense of distance of the fitted cycle to the ground truth cycle
        - GT cycle points, a tensor with shape [M, 2]
        - fitted cycle points, a tensor with shape [M, 2]
    """
    device = next(H.parameters()).device
    angles = torch.linspace(0, 2*np.pi, M, device=device)

    true_rad = np.sqrt(gt_system.parameters['a'])
    true_pts = torch.stack([
        true_rad*torch.cos(angles),
        true_rad*torch.sin(angles),
    ]).T
    true_pts = aug(true_pts)

    fit_rad = np.sqrt(archetype.parameters['a'])
    fit_pts = torch.stack([
        fit_rad*torch.cos(angles),
        fit_rad*torch.sin(angles),
    ]).T
    fit_pts = H.reverse(fit_pts)

    norms = torch.norm(aug.reverse(fit_pts), dim=1)

    return torch.mean((norms - true_rad)**2).item(), true_pts, fit_pts


def plot_cycles(x: torch.Tensor, xdot: torch.Tensor, true_cycle: torch.Tensor, fitted_cycle: torch.Tensor,
                color: str='k', title: str='', legend: bool=True):

    # plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], 15, 'gray', alpha=.2)

    # normalize all velocities to have a norm of 1 for readable plots
    norms = torch.clamp(torch.norm(xdot, dim=1, keepdim=True), 1)
    xdot = xdot/norms
    x, xdot = x.cpu().numpy(), xdot.cpu().numpy()

    # scatter the position of all points
    plt.scatter(x[:, 0], x[:, 1], 15, 'gray', alpha=.2)
    plt.quiver(x[:, 0], x[:, 1], xdot[:, 0], xdot[:, 1], scale=5, color='gray', alpha=.2, width=.012)

    # scatter chosen vectors from the dynamics
    inds = np.random.choice(x.shape[0], 50)
    x, xdot = x[inds], xdot[inds]
    plt.scatter(x[:, 0], x[:, 1], 25, 'k', alpha=.5)
    plt.quiver(x[:, 0], x[:, 1], xdot[:, 0], xdot[:, 1], scale=5, color='k', alpha=.5, width=.012)

    true_cycle = true_cycle.cpu().numpy()
    plt.plot(true_cycle[:, 0], true_cycle[:, 1], lw=3, alpha=.7, color='tab:blue', label='true cycle')
    fitted_cycle = fitted_cycle.cpu().numpy()
    plt.plot(fitted_cycle[:, 0], fitted_cycle[:, 1], lw=3, alpha=.7, color='tab:red', label='fitted')

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title(title, color=color)
    if legend: plt.legend()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
root_path = 'results/invariant_set/'

all_results = {}

torch.manual_seed(0)
np.random.seed(0)

n_systems = 100
n_points = [50, 100, 250, 500, 1000]
its = 2000

archetype = get_oscillator(a=.25, omega=.5)
arch_system = SO(a=.25, omega=.5)

for npts in n_points:
    print(f'\nNumber of points: {npts}\n', flush=True)
    all_results[npts] = {'dists': [], 'true': [], 'fitted': [], 'loss': []}

    pts_path = root_path + f'{npts}/'
    Path(pts_path).mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(n_systems)):
        # sample dynamics
        a = np.random.rand()*.4
        omega = np.random.rand()*.8
        system = SO(a=a, omega=omega)

        # sample points from system
        init = torch.rand(npts, 2)*(system.position_lims[1]-system.position_lims[0]) + system.position_lims[0]
        x = system.rand_on_traj(init, T=.5)
        dx = system(0, x)
        x, dx = x.to(device), dx.to(device)

        # augment the system
        init_amnt = np.random.rand()*.75
        aug = Diffeo(dim=2, n_layers=3, K=5, actnorm=False).to(device)
        for param in aug.parameters():
            param.data = torch.randn_like(param) * init_amnt / np.prod(param.shape)
        aug.requires_grad_(False)
        aug.eval()
        x = aug(x)

        # fit the system
        H = Diffeo(dim=2, n_layers=4, K=5).to(device)
        H, loss, ldet, score = fit_DFORM(H, x.clone(), dx.clone(), archetype, its=its,
                                         verbose=False, lr=1e-3, freeze_frac=.2, det_reg=.0,
                                         center_reg=0, proj_reg=None, weight_decay=1e-3,
                                         dim2_weight=None)

        # evaluate
        dist, true_pts, fitted = evaluate_cycle(H, arch_system, aug.to(device), system)
        all_results[npts]['dists'].append(dist)
        all_results[npts]['true'].append(true_pts)
        all_results[npts]['fitted'].append(fitted)
        all_results[npts]['loss'].append(loss)

        # plot result
        plt.figure(figsize=(3.5, 3.5))
        plot_cycles(x, dx, true_pts, fitted, title=f'error={dist:.3f}', legend=i==0)
        plt.tight_layout()
        plt.savefig(f'{pts_path}{i}.png')
        plt.close('all')

        # save everything
        with open(f'{root_path}all_results.pkl', 'wb') as f: pickle.dump(all_results, f)

dists = [np.mean(all_results[npts]['dists']) for npts in n_points]
plt.figure(figsize=(3.5, 3.5))
plt.plot(n_points, dists, lw=3, alpha=.7, color='k')
plt.xlim(n_points[0], n_points[-1])
plt.xlabel('number of observations')
plt.ylabel('average distance')
plt.yscale('log')
plt.tight_layout()
plt.savefig(f'{root_path}distances.png')