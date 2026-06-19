from dynamics.systems import SO, VanDerPol, LienardPoly, LienardSigmoid, BZreaction, Selkov, AffineLifting
from SPE.SPE import SPEModel
from dynamics.prototypes import SOPrototype
from SPE.SPE import fit_prototype, fit_all_prototypes
from dynamics.utils import invariant_distribution_error, simulate_trajectory
from models.baselines import get_NODE, kNN_vectors, fit_SINDy

import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import logging
import pandas as pd
import click

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'figure.dpi': 90,
    'figure.autolayout': True,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'font.weight': 'bold',
    'font.size': 18,
    'axes.linewidth': 1.5,
    'axes.labelweight': 'bold',
    'lines.linewidth': 3,
    'legend.handlelength': 2.,
    'legend.handletextpad': .4,
    'legend.labelspacing': .0,
    'legend.borderpad': .4,
    'legend.fontsize': 13,
    'axes.edgecolor': '#303030',
    'savefig.facecolor': 'white',
    'text.latex.preamble': r'\usepackage{amsfonts}',
})

SYSTEMS = [SO, LienardPoly, LienardSigmoid, VanDerPol, BZreaction, Selkov]
PROTOS = [
    {'a': .25, 'omega': .5},
    {'a': .25, 'omega': -.5},
    # {'a': -.25, 'omega': .5},
    # {'a': -.25, 'omega': -.5},
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = ''

config = {
    'SOproto': True,
    'save-plots': False,
    'det-reg': 1e-3,
    'weight-decay': 1e-3,
}

EVALS = [
    'SWD',
    'W2',
    'W1',
]

METHODS = [
    'SPE',
    'kNN',
    'SINDy-poly',
    'SINDy-fourier',
    'NeuralODE',
]
HEIGHT, WIDTH = 2, 4

def eval_estimate(x, xdot, system, model, eval_t, eval_n, dim):
    torch.manual_seed(0)
    np.random.seed(0)
    # distribution of points on invariant set that the methods will be compared to
    inits = system.random_x(eval_n, dim=dim)
    sys_sim = simulate_trajectory(system, inits, T=eval_t)[-1]

    # get distribution of points from the model
    if isinstance(model, SPEModel): model_sim = model.trajectories(inits, T=eval_t)[-1]
    else: model_sim = simulate_trajectory(model, inits, T=eval_t)[-1]

    model_evs = [
        invariant_distribution_error(sys_sim.clone(), model_sim.clone(), distance='swd'),
        invariant_distribution_error(sys_sim.clone(), model_sim.clone(), distance='emd'),
        invariant_distribution_error(sys_sim.clone(), model_sim.clone(), distance='emd', p=1),
    ]

    norms = torch.clamp(torch.norm(xdot, dim=1, keepdim=True), 1)
    dy = xdot / norms
    x_np, dy = x.detach().numpy(), dy.detach().numpy()
    plt.scatter(x_np[:, 0], x_np[:, 1], 25, 'dimgray', alpha=.3, label='observed data')
    plt.quiver(x_np[:, 0], x_np[:, 1], dy[:, 0], dy[:, 1], scale=5, color='dimgray', alpha=.3, width=.012)

    plt.scatter(sys_sim[:100, 0].numpy(), sys_sim[:100, 1].numpy(), 25, 'tab:blue', alpha=.6, marker='P',
                label='true inv.', zorder=10)
    plt.scatter(model_sim[:100, 0].detach().cpu().numpy(), model_sim[:100, 1].detach().cpu().numpy(), 25, 'tab:red',
                alpha=.6, marker='x', label='pred. inv.', zorder=11)

    traj = system.trajectories(inits[:1], T=eval_t)[:, -1].cpu().detach().numpy()
    plt.plot(traj[:, 0], traj[:, 1], alpha=.7, label='true', color='k', lw=3)
    traj = simulate_trajectory(model, inits[:1], T=eval_t)[:, -1].cpu().detach().numpy()
    plt.plot(traj[:, 0], traj[:, 1], alpha=.7, label='predicted', color='tab:red', lw=3, linestyle='--')

    return model_evs


def fit_SPE(x, xdot):
    protos = PROTOS
    # if config['nodes'] == 0: protos = protos[:2]
    res = fit_all_prototypes(x.to(device), xdot.to(device), protos,
                             diffeo_args={'n_layers': config['n-layers'],
                                          'K': config['n-freqs'], 'RFF': config['RFF']},
                             fitting_args={
                                 'its': config['SPE-its'],
                                 'lr': config['lr'],
                                 'det_reg': config['det-reg'],
                                 'weight_decay': config['weight-decay']
                             }
                             )
    ind = np.argmin(res['scores'])
    return res['Hs'][ind]


@click.command()
@click.option('--n',           help='number of systems to fit', type=int, default=5)
@click.option('--job',         help='job number (used for parallelization)', type=int, default=0)
@click.option('--its',         help='number of iterations to fit SPE', type=int, default=500)
@click.option('--lr',          help='learning rate for fitting SPE', type=float, default=1e-3)
@click.option('--n_points',    help='number of points to sample', type=int, default=100)
@click.option('--dim',         help='dimension of the data', type=int, default=2)
@click.option('--n_layers',    help='number of layers in models', type=int, default=4)
@click.option('--n_freqs',     help='number of frequencies in coupling', type=int, default=5)
@click.option('--snr',         help='Signal to noise ratio in observed velocities', type=float, default=5.)
@click.option('--t_max',       help='max integration time for simulation', type=float, default=3.)
@click.option('--eval_t',      help='integration time to use for evaluation', type=float, default=100.)
@click.option('--eval_n',      help='number of points to use during evaluation', type=int, default=1000)
@click.option('--rff',         help='whether to use RFFCoupling instead of FFCoupling', type=int, default=0)
def classify_all(n: int, job: int, its: int, lr: float, n_points: int, dim: int, n_layers: int,
                 n_freqs: int, snr: float, t_max: float, eval_t: float, eval_n: int, rff: int):

    # ============================= define paths ==================================================#
    path_root = root + f'results/invariant_dim={dim}/'
    name = (
        f'npoints={n_points}_'
        f'SNR={snr:.2f}_'
        f'layers={n_layers}_'
        f'freqs={n_freqs}_'
        f'its={its}_'
        f'T={t_max:.2f}'
        f'{"_RFF" if rff==1 else ""}'
    )
    path = path_root + name + '/'
    Path(path).mkdir(parents=True, exist_ok=True)

    global config
    config['n-layers'] = n_layers
    config['n-freqs'] = n_freqs
    config['nodes'] = 0
    config['RFF'] = rff==1
    config['SPE-its'] = its
    config['lr'] = lr

    # ============================= logging =======================================================#
    # write all hyperparameters to a file
    open(path + f'{job}.log', 'w').close()
    handlers = [logging.FileHandler(path + f'{job}.log')]
    logging.basicConfig(format='',
                        level=logging.INFO,
                        handlers=handlers
                        )

    logging.info('\n'.join(f'{k}={v}' for k, v in locals().items()))

    # ============================= create save dict ==============================================#
    res_dict = {
        'system': [],
        'params': []
    }
    for method in METHODS:
        res_dict[method] = {e: [] for e in EVALS}

    # ============================= load systems ==================================================#
    torch.manual_seed(job)
    np.random.seed(job)
    ks = np.random.choice(len(SYSTEMS), n)
    systems = [
        SYSTEMS[i](**SYSTEMS[i].random_cycle_params()) for i in ks
    ]

    if dim > 2:
        systems = [
            AffineLifting(system, dim=dim, decay=1.) for system in systems
        ]

    xs = [
        system.rand_on_traj(system.random_x(n_points, dim=dim), T=t_max) for system in systems
    ]
    xdots = [
        systems[i](x) for i, x in enumerate(xs)
    ]

    # ============================= evaluations ======================================================#
    pbar = tqdm(range(n))
    for sys_ind in pbar:
        res_dict['system'].append(str(systems[sys_ind]))
        res_dict['params'].append(systems[sys_ind].parameters)

        # observed training data
        x = systems[sys_ind].rand_on_traj(systems[sys_ind].random_x(n_points, dim=dim), T=t_max)
        xdot = systems[sys_ind](x)

        noise = torch.mean(torch.norm(xdot, dim=-1)).item()/snr

        xdot = xdot + torch.randn_like(xdot)*noise

        plt.figure(figsize=(WIDTH*4, HEIGHT*4))

        # ================================= kNN
        pbar.set_postfix_str('kNN')
        plt.subplot(HEIGHT, WIDTH, 1)
        res = eval_estimate(x, xdot, systems[sys_ind], kNN_vectors(x.clone(), xdot.clone()), eval_t, eval_n, dim)
        for i, e in enumerate(EVALS): res_dict['kNN'][e].append(res[i])
        plt.title(f'kNN, W1={res[-1]:.4f}')
        plt.legend()

        # ================================= SINDy
        pbar.set_postfix_str('SINDy-poly')
        plt.subplot(HEIGHT, WIDTH, 2)
        res = eval_estimate(x, xdot, systems[sys_ind], fit_SINDy(x.clone(), xdot.clone(), 'poly', degree=3), eval_t,
                            eval_n, dim)
        for i, e in enumerate(EVALS): res_dict['SINDy-poly'][e].append(res[i])
        plt.title(f'SINDy-poly, W1={res[-1]:.4f}')

        pbar.set_postfix_str('SINDy-fourier')
        plt.subplot(HEIGHT, WIDTH, 3)
        res = eval_estimate(x, xdot, systems[sys_ind], fit_SINDy(x.clone(), xdot.clone(), 'fourier', degree=10), eval_t,
                            eval_n, dim)
        for i, e in enumerate(EVALS): res_dict['SINDy-fourier'][e].append(res[i])
        plt.title(f'SINDy-fourier, W1={res[-1]:.4f}')

        # ================================= NeuralODE
        pbar.set_postfix_str('NODE')
        plt.subplot(HEIGHT, WIDTH, 4)
        node = get_NODE(x.clone(), xdot.clone(), its=5000, width=128)
        res = eval_estimate(x, xdot, systems[sys_ind], node, eval_t, eval_n, dim)
        for i, e in enumerate(EVALS): res_dict['NeuralODE'][e].append(res[i])
        plt.title(f'NODE, W1={res[-1]:.4f}')

        # ================================= SPE
        pbar.set_postfix_str('SPE')
        plt.subplot(HEIGHT, WIDTH, 5)
        res = eval_estimate(x, xdot, systems[sys_ind], fit_SPE(x.clone(), xdot.clone()), eval_t, eval_n, dim)
        for i, e in enumerate(EVALS): res_dict['SPE'][e].append(res[i])
        plt.title(f'SPE, W1={res[-1]:.4f}')

        # ================================= save everything
        if config['save-plots']:
            try:
                plt.tight_layout()
                plt.savefig(path + f'{job}-{sys_ind}.png')
            except:
                print('\n\nmatplotlib failed\n\n')

        plt.close('all')

        with open(path + f'{job}.pkl', 'wb') as f: pickle.dump(res_dict, f)

    # ============================= save everything ======================================================#

    full_d = {
        'system': [],
        'params': [],
    }
    for method in METHODS: full_d[method] = {e: [] for e in EVALS}

    for i in range(500):
        try:
            with open(path + f'{i}.pkl', 'rb') as f:
                d = pickle.load(f)
            full_d['system'] += d['system']
            full_d['params'] += d['params']
            for method in METHODS:
                for e in EVALS:
                    full_d[method][e] += d[method][e]
        except:
            pass

    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(full_d, f)

    # ============================= save csv of results ======================================================#

    table = {
        'metric': [],
        'method': [],
        'distance': [],
        'system': []
    }
    for method in METHODS:
        for eval in EVALS:
            r = full_d[method][eval]
            table['distance'] += r
            table['method'] += [method]*len(r)
            table['metric'] += [eval]*len(r)
            table['system'] += full_d['system']
    table['SNR'] = [snr]*len(table['distance'])
    table['npoints'] = [n_points]*len(table['distance'])
    table['T'] = [t_max]*len(table['distance'])
    pd.DataFrame(table).to_csv(path + name + '.csv')


if __name__ == '__main__':
    classify_all()