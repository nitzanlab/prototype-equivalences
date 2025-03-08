from fit_SPE import fit_prototype
from NFDiffeo import Diffeo
import numpy as np
import torch
import pickle
import click
from pathlib import Path
from Hutils import get_oscillator
from systems import Repressilator, SO, PhaseSpace
from multidim_benchmark import compile_results, plot_trajectory, param_str
from matplotlib import pyplot as plt
import sys, logging


def traj_in_vecs(true_traj: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor, arch: PhaseSpace, H: Diffeo,
                 T: float=40, extra_str: str='', color: str='k'):
    # choose points as initial points from which to iterate
    inds = np.random.choice(x.shape[0], 3, replace=False)
    dim = x.shape[-1]

    # get fitted trajectories
    y = H.forward(x[inds]).detach()
    traj = arch.trajectories(y, T=T)
    traj = H.reverse(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape).detach()
    true_traj = true_traj.to(traj.device)

    if dim > 2:
        # move all observables to 2D archetype space
        x, xdot, _ = H.jvp_forward(x, xdot)
        x, xdot = x[..., :2].detach(), xdot[..., :2].detach()

        traj = H.forward(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape).detach()[..., :2]
        true_traj = H.forward(true_traj.reshape(-1, true_traj.shape[-1])).reshape(true_traj.shape).detach()[..., :2]

    traj = traj.cpu().numpy()
    true_traj = true_traj.cpu().numpy()

    # scatter the position of all points
    # plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], 15, 'gray', alpha=.25)

    # normalize all velocities to have a norm of 1 for readable plots
    norms = torch.clamp(torch.norm(xdot, dim=1, keepdim=True), 1)
    xdot = xdot/norms

    # scatter chosen vectors from the dynamics
    inds = np.random.choice(x.shape[0], 50)
    x, xdot = x[inds].cpu().numpy(), xdot[inds].cpu().numpy()
    plt.scatter(x[:, 0], x[:, 1], 25, 'k', alpha=.5)
    plt.quiver(x[:, 0], x[:, 1], xdot[:, 0], xdot[:, 1], scale=5, color='k', alpha=.5, width=.012)

    # plot fitted 2D space
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], lw=2, alpha=.7, color='tab:red')
    for i in range(true_traj.shape[1]):
        plt.plot(true_traj[:, i, 0], true_traj[:, i, 1], lw=2, alpha=.7, color='tab:blue')

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    title_str = param_str(arch)
    plt.title(title_str+extra_str, color=color)

root = 'results/repressilator'

@click.command()
@click.option('--n_points', help='number of points in the system', type=int, default=1000)
@click.option('--job',      help='job number (used for parallelization)', type=int, default=0)
@click.option('--lr',       help='learning rate used in training', type=float, default=1e-3)
@click.option('--its',      help='number of iterations', type=int, default=2000)
@click.option('--n_layers', help='number of layers in models', type=int, default=4)
@click.option('--n_freqs',  help='number of frequencies in coupling', type=int, default=5)
@click.option('--fr_rat',   help='ratio of iterations during which scale params are frozen', type=float, default=.0)
@click.option('--dim',      help='the dimensionality of the data', type=int, default=2)
@click.option('--time',     help='amount of time for discretization', type=float, default=10.)
@click.option('--det_reg',  help='regularization of the determinant ', type=float, default=.0)
@click.option('--cen_reg',  help='regularization of the origin ', type=float, default=.0)
@click.option('--proj_reg', help='regularization for the 2D projections', type=float, default=-1)
@click.option('--verbose',  help='whether to print progress or not', type=int, default=-1)
@click.option('--noise',    help='amount of noise to add to phase space', type=float, default=0)
@click.option('--dim2_weight', help='relative weight of the first 2 dimensions', type=float, default=-1)
@click.option('--test_fr', help='fraction of test points', type=float, default=0)
@click.option('--save_h',   help='whether to save the model', type=int, default=0)
@click.option('--decay',   help='delay amount in archetype', type=float, default=.25)
@click.option('--single_traj',   help='whether to use a single trajectory or many', type=int, default=1)
@click.option('--omega',   help='the value of omega to use', type=float, default=.5)
def classify_all(n_points: int, job: int, lr: float, its: int, n_layers: int, n_freqs: int, fr_rat: float, dim: int,
                 time: float, det_reg: float, cen_reg: float, proj_reg: float, verbose: int, noise: float,
                 dim2_weight: float, save_h: int, test_fr, decay: float, single_traj: int, omega: float):
    # ============================================ create directory ====================================

    path = root + f'_dim={dim}/' +\
                  f'its={its}_npts={n_points}_freqs={n_freqs}_layers={n_layers}' +\
                  (f'_proj={proj_reg:.1f}' if dim > 2 else '') +\
                  (f'_noise={noise:.2f}' if noise > 0 else '') +\
                  (f'_time={time:.1f}' if time != 10. else '') +\
                  (f'_tst={test_fr:.1f}' if test_fr > 0 else '') +\
                  (f'_multitraj' if single_traj == 0 else '') +\
                  (f'_omega={omega:.1f}' if omega != .5 else '') +\
                  f'/'
    Path(path).mkdir(parents=True, exist_ok=True)

    if dim == 2 or proj_reg <= -10: proj_reg = None
    # ============================================ create directory ====================================

    # ============================================ create log ==========================================
    open(path + f'{job}.log', 'w').close()
    handlers = [logging.FileHandler(path + f'{job}.log'), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format='',
                        level=logging.INFO,
                        handlers=handlers
                        )

    logging.info('\n'.join(f'{k}={v}' for k, v in locals().items()))
    # ============================================ create log ==========================================

    torch.manual_seed(job)
    np.random.seed(job)

    system = Repressilator()
    parameters = system.parameters

    logging.info('\n\n')
    logging.info('; '.join(f'{k}={v:.2f}' for k, v in parameters.items()))

    # plot the underlying latent system
    plot_trajectory(dim, system, 10, path + f'{job}_original_trajectories.png')

    if single_traj > 0:
        # sample points along trajectory
        init = torch.rand(1, 6)*(system.position_lims[1]-system.position_lims[0]) + system.position_lims[0]
        traj = system.trajectories(init, T=time, step=min(1e-2, time/n_points))
        inds = np.random.choice(traj.shape[0], n_points, replace=False)
        x = traj[inds, 0]
    else:
        # points after burn-in on trajectories
        init = torch.rand(n_points, 6)*(system.position_lims[1]-system.position_lims[0]) + system.position_lims[0]
        x = system.rand_on_traj(init, T=time, step=1e-3, min_time=3)
        # create trajectory for plotting
        init = torch.rand(1, 6) * (system.position_lims[1] - system.position_lims[0]) + system.position_lims[0]
        traj = system.trajectories(init, T=time*3, step=1e-3)[int(3/1e-3):]

    # get vectors
    dx = system(0, x)

    if dim == 2: x, dx, traj = x[:, [1, 3]], dx[:, [1, 3]], traj[:, :, [1, 3]]
    elif dim == 3: x, dx, traj = x[:, [1, 3, 5]], dx[:, [1, 3, 5]], traj[:, :, [1, 3, 5]]
    elif dim == 4: x, dx, traj = x[:, [0, 1, 3, 5]], dx[:, [0, 1, 3, 5]], traj[:, :, [0, 1, 3, 5]]
    elif dim == 5: x, dx, traj = x[:, [0, 1, 2, 3, 5]], dx[:, [0, 1, 2, 3, 5]], traj[:, :, [0, 1, 2, 3, 5]]

    # add noise, if needed
    dx = dx + torch.randn_like(dx)*noise

    archetypes = [
        [-.25, -omega, decay],
        [.25, -omega, decay],
        [-.25, omega, decay],
        [.25, omega, decay],
    ]

    res_dict = {
        'archetypes': archetypes,
        'losses': [],
        'logdets': [],
        '2Dlosses': [],
        'data_mean': torch.mean(x).item(),
        'data_std': torch.std(x).item(),
        'params': parameters,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x, dx = x.to(device), dx.to(device)

    plt.figure(figsize=(7, 4*len(archetypes)//2))
    subplot = 1
    for (a, omega, decay) in res_dict['archetypes']:
        # ============================================ fit archetype ==========================================
        archetype = get_oscillator(a=a, omega=omega, decay=decay)
        H = Diffeo(dim=dim, n_layers=n_layers, K=n_freqs, rank=2, affine_init=True).to(device)
        H, loss, ldet, score = fit_prototype(H, x.clone(), dx.clone(), archetype, its=its, lr=lr, verbose=verbose > 0,
                                             freeze_frac=fr_rat, det_reg=det_reg, center_reg=cen_reg, weight_decay=1e-3,
                                             proj_reg=proj_reg, dim2_weight=None if dim2_weight < 0 else dim2_weight)

        # ============================================ save stats on fit ==========================================
        res_dict['losses'].append(loss)
        res_dict['logdets'].append(ldet)
        res_dict['2Dlosses'].append(score)

        logging.info(f'\narchetype a={a:.2f} om={omega:.2f}: loss={loss:.4f}; ldet={ldet:.2f}, 2D loss={score:.3f}')
        # ============================================ plots ==========================================
        plt.subplot(len(archetypes)//2, 2, subplot)
        traj_in_vecs(traj.clone(), x.clone(), dx.clone(), SO(a=a, omega=omega, decay=decay),
                     H, T=40, extra_str=f'loss={loss:.3f} ; ldet={ldet:.3f}\n2D-loss={score:.3f}',
                     color='blue' if np.sign(a)==np.sign(system.parameters['bif']) else 'red')
        subplot += 1
        if save_h > 0: torch.save(H, path + f'{job}_a={a:.2f}_om={omega:.2f}.pth')
    plt.tight_layout()
    plt.savefig(path + f'{job}_projections.png')
    with open(path + f'{job}.pkl', 'wb') as f: pickle.dump(res_dict, f)

    compile_results(path, dim)


if __name__ == '__main__':
    classify_all()
