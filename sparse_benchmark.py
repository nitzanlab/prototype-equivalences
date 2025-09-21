from dynamics.systems import SO, VanDerPol, LienardPoly, LienardSigmoid, BZreaction, Selkov, AffineLifting
from dynamics.prototypes import SOPrototype
from SPE.SPE import fit_all_prototypes
from DSA import DSA

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
    {'a': -.25, 'omega': .5},
    {'a': -.25, 'omega': -.5},
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = ''

config = {
    'SPE-its': 500,
    'lr': 1e-3,
    'det-reg': 1e-3,
    'weight-decay': 1e-3,
    'DSA-its': 5000,
    'DSA-step': 1e-3,

}


METHODS = [
    'SPE',
    'DSA',
]
HEIGHT, WIDTH = 2, 4


def SPE_classify(x, xdot):
    results = fit_all_prototypes(x=x, xdot=xdot, prototypes=PROTOS,
                                 diffeo_args={'n_layers': config['n-layers'],
                                              'K': config['n-freqs'],
                                              },
                                 fitting_args={
                                     'its': config['SPE-its'],
                                     'lr': config['lr'],
                                     'det_reg': config['det-reg'],
                                     'weight_decay': config['weight-decay'],
                                 }
                                 )
    return np.array([[proto['a'], proto['omega'], results['scores'][i]] for i, proto in enumerate(PROTOS)])


def DSA_classify(x, xdot):
    X = torch.cat([x[:, None], x[:, None] + config['DSA-step']*xdot[:, None]], dim=1)
    rets = []
    for proto in PROTOS:
        s = SOPrototype(a=proto['a'], omega=proto['omega'])
        Y = s.trajectories(x, T=5).transpose(0, 1)
        score = DSA(X, Y, iters=config['DSA-its']).fit_score()
        rets.append([proto['a'], proto['omega'], score])

    return np.array(rets)


@click.command()
@click.option('--n',           help='number of systems to fit', type=int, default=5)
@click.option('--its',         help='number of iterations to train SPE', type=int, default=500)
@click.option('--job',         help='job number (used for parallelization)', type=int, default=0)
@click.option('--n_points',    help='number of points to sample', type=int, default=100)
@click.option('--dim',         help='dimension of the data', type=int, default=2)
@click.option('--n_layers',    help='number of layers in models', type=int, default=1)
@click.option('--n_freqs',     help='number of frequencies in coupling', type=int, default=5)
@click.option('--snr',         help='Signal to noise ratio in observed velocities', type=float, default=5.)
@click.option('--t_max',       help='max integration time for simulation', type=float, default=3.)
def classify_all(n: int, its: int, job: int, n_points: int, dim: int, n_layers: int,
                 n_freqs: int, snr: float, t_max: float):

    # ============================= define paths ==================================================#
    path_root = root + f'results/sparse_classify_dim={dim}/'
    name = (
        f'npoints={n_points}_'
        f'SNR={snr:.2f}_'
        f'layers={n_layers}_'
        f'freqs={n_freqs}_'
        f'its={its}_'
        f'T={t_max:.2f}'
    )
    path = path_root + name + '/'
    Path(path).mkdir(parents=True, exist_ok=True)

    global config
    config['n-layers'] = n_layers
    config['n-freqs'] = n_freqs
    config['SPE-its'] = its

    # ============================= logging =======================================================#
    # # write all hyperparameters to a file
    # open(path + f'{job}.log', 'w').close()
    # handlers = [logging.FileHandler(path + f'{job}.log')]
    # logging.basicConfig(format='',
    #                     level=logging.INFO,
    #                     handlers=handlers
    #                     )
    #
    # logging.info('\n'.join(f'{k}={v}' for k, v in locals().items()))

    # ============================= create save dict ==============================================#
    res_dict = {
        'system': [],
        'true_label': [],
        'method': [],
        'predicted': [],
        'SNR': [],
        'npoints': [],
        'T': [],
    }

    # ============================= load systems ==================================================#
    torch.manual_seed(job)
    np.random.seed(job)
    ks = np.random.choice(len(SYSTEMS), n)
    systems = [
        SYSTEMS[i]() for i in ks
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

        # observed training data
        x = systems[sys_ind].rand_on_traj(systems[sys_ind].random_x(n_points, dim=dim), T=t_max)
        xdot = systems[sys_ind](x)

        noise = torch.mean(torch.norm(xdot, dim=-1)).item()/snr

        xdot = xdot + torch.randn_like(xdot)*noise

        x, xdot = x.to(device), xdot.to(device)

        # ================================= SPE
        pbar.set_postfix_str('SPE')
        res = SPE_classify(x, xdot)
        pred = np.sign(res[np.argmin(res[:, 2]), 0])
        res_dict['system'].append(str(systems[sys_ind]))
        res_dict['true_label'].append(np.sign(systems[sys_ind].dist_from_bifur()))
        res_dict['method'].append('SPE')
        res_dict['predicted'].append(pred)
        res_dict['SNR'].append(snr)
        res_dict['npoints'].append(n_points)
        res_dict['T'].append(t_max)

        # ================================= DSA
        pbar.set_postfix_str('DSA')
        res = DSA_classify(x, xdot)
        pred = np.sign(res[np.argmin(res[:, 2]), 0])
        res_dict['system'].append(str(systems[sys_ind]))
        res_dict['true_label'].append(np.sign(systems[sys_ind].dist_from_bifur()))
        res_dict['method'].append('DSA')
        res_dict['predicted'].append(pred)
        res_dict['SNR'].append(snr)
        res_dict['npoints'].append(n_points)
        res_dict['T'].append(t_max)

        with open(path + f'{job}.pkl', 'wb') as f: pickle.dump(res_dict, f)

    # ============================= save everything ======================================================#

    full_d = {
        'system': [],
        'true_label': [],
        'method': [],
        'predicted': [],
        'SNR': [],
        'npoints': [],
        'T': [],
    }

    for i in range(500):
        try:
            with open(path + f'{i}.pkl', 'rb') as f:
                d = pickle.load(f)
            for k in full_d.keys():
                full_d[k] += d[k]
        except:
            pass

    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(full_d, f)

    pd.DataFrame(full_d).to_csv(path + name + '.csv')


if __name__ == '__main__':
    classify_all()