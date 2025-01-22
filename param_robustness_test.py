from fit_archetype import fit_DFORM
from NFDiffeo import Diffeo
import numpy as np
import torch
import pickle
import click
from pathlib import Path
from Hutils import get_oscillator, simulate_trajectory, cycle_error
from systems import DiffeoAug, ComposeAug, Augmentation
from systems import SO
from matplotlib import pyplot as plt
import sys, logging
from multidim_benchmark import iterate_trajectory, param_str, traj_in_vecs, plot_trajectory, compile_results

root = 'results/param_robustness/'


@click.command()
@click.option('--factor',   help='multiplicative factor of change from archetypes', type=float, default=1.)
@click.option('--n_points', help='number of points in the system', type=int, default=1000)
@click.option('--job',      help='job number (used for parallelization)', type=int, default=0)
@click.option('--lr',       help='learning rate used in training', type=float, default=1e-3)
@click.option('--its',      help='number of iterations', type=int, default=2000)
@click.option('--n_layers', help='number of layers in models', type=int, default=4)
@click.option('--n_freqs',  help='number of frequencies in coupling', type=int, default=5)
@click.option('--fr_rat',   help='ratio of iterations during which scale params are frozen', type=float, default=.25)
@click.option('--dim',      help='the dimensionality of the data', type=int, default=2)
@click.option('--time',     help='amount of time for discretization', type=float, default=.5)
@click.option('--det_reg',  help='regularization of the determinant ', type=float, default=1e-3)
@click.option('--cen_reg',  help='regularization of the origin ', type=float, default=1e-6)
@click.option('--proj_reg', help='regularization for the 2D projections', type=float, default=-1)
@click.option('--verbose',  help='whether to print progress or not', type=int, default=-1)
@click.option('--noise',    help='amount of noise to add to phase space', type=float, default=0)
@click.option('--w_decay',  help='weight decay used', type=float, default=1e-3)
@click.option('--rem_small', help='if 1, removes velocities with norms that are small compared to the observed noise (as preprocessing)', type=int, default=1)
def classify_all(factor: float, n_points: int, job: int, lr: float, its: int, n_layers: int, n_freqs: int,
                 fr_rat: float, dim: int, time: float, det_reg: float, cen_reg: float, proj_reg: float, verbose: int,
                 noise: float, w_decay: float, rem_small: int):
    # ============================================ create directory ====================================================
    path = root + f'factor={factor:.1f}/' +\
                  f'dim={dim}_its={its}_npts={n_points}_freqs={n_freqs}_layers={n_layers}' +\
                  (f'_proj={proj_reg:.1f}' if dim > 2 else '') +\
                  (f'_noise={noise:.2f}' if noise > 0 else '') +\
                  f'_time={time:.1f}' +\
                  f'/'
    Path(path).mkdir(parents=True, exist_ok=True)

    if dim == 2 or proj_reg <= -10: proj_reg = None
    # ============================================ create directory ====================================================

    # ============================================ create log ==========================================================
    open(path + f'{job}.log', 'w').close()
    handlers = [logging.FileHandler(path + f'{job}.log'), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format='',
                        level=logging.INFO,
                        handlers=handlers
                        )

    logging.info('\n'.join(f'{k}={v}' for k, v in locals().items()))
    # ============================================ create log ==========================================================

    torch.manual_seed(job)
    np.random.seed(job)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    archetypes = [
        [-.25, -.5, .25],
        [.25, -.5, .25],
        [-.25, .5, .25],
        [.25, .5, .25]
    ]

    a, omega, _ = archetypes[np.random.choice(1, 4)[0]]
    system = SO(a=a*factor, omega=omega*factor)
    parameters = system.parameters

    logging.info('\n\n')
    logging.info('; '.join(f'{k}={v:.2f}' for k, v in parameters.items()))

    # iterate points on trajectory for a bit, so they'll be closer to the invariant set
    init = torch.rand(n_points, dim)*(system.position_lims[1]-system.position_lims[0]) + system.position_lims[0]
    x = system.rand_on_traj(init, T=time)
    dx = system(0, x)

    # augmentation to add on top of data
    init_amnt = np.random.rand() * .75
    aug = Diffeo(dim=2, n_layers=3, K=5, actnorm=False).to(x.device)
    for param in aug.parameters():
        param.data = torch.randn_like(param) * init_amnt / np.prod(param.shape)
    aug.requires_grad_(False)
    aug.eval()
    x = aug(x)

    # add noise
    dx = dx + torch.randn_like(dx)*noise

    # as a preprocessing step for our method, removes all vectors with norms that are too small compared to noise
    if rem_small == 1 and noise > 0:
        norms = torch.norm(dx, dim=1)
        x, dx = x[norms > dim*noise], dx[norms > dim*noise]

    res_dict = {
        'archetypes': archetypes,
        'losses': [],
        'logdets': [],
        '2Dlosses': [],
        'cycle-errors': [],
        'data_mean': torch.mean(x).item(),
        'data_std': torch.std(x).item(),
        'params': parameters,
    }

    x, dx = x.to(device), dx.to(device)

    # calculate trajectory for calculating the cycle errors
    aug.to(x.device)
    cycle_traj = system.trajectories(x[:1], T=50)
    cycle_traj = aug(cycle_traj[cycle_traj.shape[0]//2:][:, 0])

    plt.figure(figsize=(7, 4*len(archetypes)//2))
    subplot = 1
    for (a, omega, decay) in res_dict['archetypes']:
        # ============================================ fit archetype ===================================================
        archetype = get_oscillator(a=a, omega=omega, decay=decay)
        H = Diffeo(dim=dim, n_layers=n_layers, K=n_freqs, rank=2).to(device)
        H, loss, ldet, score = fit_DFORM(H, x.clone(), dx.clone(), archetype, its=its,
                                         verbose=verbose>0, lr=lr, freeze_frac=fr_rat, det_reg=det_reg,
                                         center_reg=cen_reg, proj_reg=proj_reg, weight_decay=w_decay)

        # ============================================ save stats on fit ===============================================
        res_dict['losses'].append(loss)
        res_dict['logdets'].append(ldet)
        res_dict['2Dlosses'].append(score)
        cerr = cycle_error(H, cycle_traj.to(device), a)
        res_dict['cycle-errors'].append(cerr)

        logging.info(f'archetype a={a:.2f} om={omega:.2f}: loss={loss:.4f}; ldet={ldet:.2f}, 2D loss={score:.3f}\n')
        # ============================================ plots ==========================================
        plt.subplot(len(archetypes)//2, 2, subplot)
        traj_in_vecs(x.clone(), dx.clone(), system, SO(a=a, omega=omega, decay=decay),
                     H, T=20, extra_str=f'loss={loss:.3f} ; ldet={ldet:.3f}\ncycle={cerr:.2f} ; 2D-loss={score:.3f}',
                     color='blue' if np.sign(a)==np.sign(system.parameters['bif']) else 'red')
        subplot += 1
    plt.tight_layout()
    plt.savefig(path + f'{job}_projections.png')
    with open(path + f'{job}.pkl', 'wb') as f: pickle.dump(res_dict, f)

    compile_results(path, dim)


if __name__ == '__main__':
    classify_all()
