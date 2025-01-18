from fit_archetype import fit_DFORM
from Hutils import get_oscillator
from NFDiffeo import Diffeo
from systems import PhaseSpace, SO
from tqdm import tqdm
import numpy as np
import torch
import pickle
import click
from pathlib import Path
import matplotlib
matplotlib.use('pgf')
from matplotlib import pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'figure.dpi': 90,
    'figure.autolayout': True,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    # 'font.weight': 'bold',
    'font.size': 14,
    'axes.linewidth': 2,
    # 'axes.labelweight': 'bold',
    'lines.linewidth': 3,
    'legend.handlelength': 1.,
    'legend.handletextpad': .4,
    'legend.labelspacing': .0,
    'legend.borderpad': .4,
    'axes.edgecolor': '#303030',
    'savefig.facecolor': 'white',
    'text.latex.preamble': r'\usepackage{amsfonts}',
})


root = '/cs/labs/yweiss/roy.friedmam/collabs/time-warp/results/'


archetypes = [
    [-.25, -.5],
    [.25, -.5],
    [-.25, .5],
    [.25, .5]
]


def traj_in_vecs(x: torch.Tensor, xdot: torch.Tensor, arch: PhaseSpace, H: Diffeo,
                 T: float=40, title_str: str='', color: str='k'):
    # choose points as initial points from which to iterate
    inds = np.random.choice(x.shape[0], 3, replace=False)

    # get fitted trajectories
    y = H.forward(x[inds].to(next(H.parameters()).device)).detach()
    traj = arch.trajectories(y, T=T)
    traj = H.reverse(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape).detach().cpu().numpy()

    # normalize all velocities to have a norm of 1 for readable plots
    norms = torch.clamp(torch.norm(xdot, dim=1, keepdim=True), 1)
    xdot = xdot/norms
    x, xdot = x.cpu().numpy(), xdot.cpu().numpy()

    # scatter the position of all points
    # plt.scatter(x[:, 0], x[:, 1], 15, 'gray', alpha=.1)
    # plt.quiver(x[:, 0], x[:, 1], xdot[:, 0], xdot[:, 1], scale=5, color='gray', alpha=.2, width=.012)

    # scatter chosen vectors from the dynamics
    inds = np.random.choice(x.shape[0], 50)
    x, xdot = x[inds], xdot[inds]
    plt.scatter(x[:, 0], x[:, 1], 25, 'k', alpha=.5)
    plt.quiver(x[:, 0], x[:, 1], xdot[:, 0], xdot[:, 1], scale=5, color='k', alpha=.5, width=.012)

    # plot fitted (all in 2D space)
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], lw=2, alpha=.7, color='tab:red')

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title(title_str, color=color)


def DFORM_classify(x, xdot, its, lr, n_layers, n_freqs, freeze_frac, path, true_sign):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {
        'losses': [],
        'logdets': [],
        '2Dlosses': []
    }

    plt.figure(figsize=(8, 8))
    subplot = 1
    for (a, omega) in archetypes:
        # ============================================ fit archetype ==========================================
        archetype = get_oscillator(a=a, omega=omega)
        H = Diffeo(dim=2, n_layers=n_layers, K=n_freqs, rank=2).to(device)
        H, loss, ldet, score = fit_DFORM(H, x.clone().to(device), xdot.clone().to(device), archetype, its=its,
                                         verbose=False, lr=lr, freeze_frac=freeze_frac, weight_decay=1e-4)
        results['losses'].append(loss)
        results['logdets'].append(ldet)
        results['2Dlosses'].append(score)

        plt.subplot(2, 2, subplot)
        traj_in_vecs(x.clone(), xdot.clone(), SO(a=a, omega=omega),
                     H, T=10, title_str=f'a={a:.2f}; omega={omega:.2f}'
                                        f'\n2D-loss={score:.3f}',
                     color='blue' if np.sign(a) == true_sign else 'red')

        subplot += 1

    plt.tight_layout()
    plt.savefig(path)
    plt.close('all')

    return results


@click.command()
@click.option('--type',     help='which data to classify', type=str, default='simple_oscillator_nsfcl')
@click.option('--n',        help='number of points to classify', type=int, default=10)
@click.option('--job',      help='job number (used for parallelization)', type=int, default=0)
@click.option('--lr',       help='learning rate used in training', type=float, default=1e-3)
@click.option('--its',      help='number of iterations', type=int, default=2000)
@click.option('--n_layers', help='number of layers in models', type=int, default=4)
@click.option('--n_freqs',  help='number of frequencies in coupling', type=int, default=5)
@click.option('--fr_rat',   help='ratio of iterations during which scale params are frozen', type=float, default=.2)
@click.option('--noise',    help='amount of noise to add to velocities', type=float, default=.0)
def classify_all(type: str, n: int, job: int, lr: float, its: int, n_layers: int,
                 n_freqs: int, fr_rat: float, noise: float):
    sz = 1
    yy, xx = np.meshgrid(np.linspace(-sz, sz, 64), np.linspace(-sz, sz, 64))
    pos = torch.from_numpy(np.stack([xx, yy]).T).float().reshape(-1, 2)

    path = root + (f'{type}_TWAgrid/its={its}_lr={lr:.0E}_layers={n_layers}_freqs={n_freqs}_'
                   f'fr_rat={fr_rat:.2f}_noise={noise:.2f}/')
    Path(path).mkdir(parents=True, exist_ok=True)

    data = np.load(f'/cs/labs/yweiss/roy.friedmam/collabs/time-warp/data/{type}/X_test.npy')
    true_params = np.load(f'/cs/labs/yweiss/roy.friedmam/collabs/time-warp/data/{type}/sysp_test.npy')
    true_labels = np.load(f'/cs/labs/yweiss/roy.friedmam/collabs/time-warp/data/{type}/topo_test.npy')

    data = data[job*n:job*n+n]
    true_params = true_params[job*n:job*n+n]
    true_labels = true_labels[job*n:job*n+n]

    res_dict = {'archetypes': archetypes,
        'params': [],
        'true_labels': [],
        'losses': [],
        'logdets': [],
        '2Dlosses': [],
    }

    for i in tqdm(range(n)):
        x = pos.clone().reshape(-1, 2)
        xdot = torch.from_numpy(data[i]).float().reshape(-1, 2)
        xdot = xdot + torch.randn_like(xdot)*noise

        results = DFORM_classify(x, xdot, its=its, lr=lr, n_layers=n_layers, n_freqs=n_freqs,
                                 freeze_frac=fr_rat, path=f'{path}{job}_{i}.png',
                                 true_sign=-1 if true_labels[i][0]==4 else 1)
        res_dict['params'].append(true_params[i])
        res_dict['true_labels'].append(true_labels[i])
        res_dict['losses'].append(results['losses'])
        res_dict['logdets'].append(results['logdets'])
        res_dict['2Dlosses'].append(results['2Dlosses'])

        with open(path + f'{job}.pkl', 'wb') as f: pickle.dump(res_dict, f)


if __name__ == '__main__':
    classify_all()
