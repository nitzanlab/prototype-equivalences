from fit_archetype import fit_DFORM
from NFDiffeo import Diffeo
import numpy as np
import torch
import pickle
import click
from pathlib import Path
from Hutils import get_oscillator, simulate_trajectory, cycle_error
from systems import LinearAug, QuadAug, DiffeoAug, ComposeAug, PhaseSpace, Augmentation
from systems import SO, Selkov, BZreaction, Repressilator, VanDerPol, LienardSigmoid, LienardPoly, SubcriticalHopf, \
    SupercriticalHopf
from matplotlib import pyplot as plt
import sys, logging
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


SYSTEMS = {
    'SO': SO,
    'repressilator': Repressilator,
    'selkov': Selkov,
    'bzreaction': BZreaction,
    'vanderpol': VanDerPol,
    'lienardsigmoid': LienardSigmoid,
    'lienardpoly': LienardPoly,
    'subhopf': SubcriticalHopf,
    'suphopf': SupercriticalHopf,
}

root = 'results/'


def iterate_trajectory(inits, grad, T: float=5):
    """
    Iterates the initialized points along the trajectory defined by grad and chooses random time points. This is
    simulates more real-world data, which has usually had time to progress according to the system
    :param inits: initial positions of the points, a tensor with shape [N, dim]
    :param grad: the gradient defining the dynamics of the system, a function f(t, x) where x has the shape [N, dim]
    :return: the points after a bit of iterating according to the system, a tensor with shape [N, dim]
    """
    traj = simulate_trajectory(grad, inits, T, step=1e-2)
    # return traj[-1]
    T = traj.shape[0]
    pts = []
    for i in range(inits.shape[0]):
        pts.append(traj[np.random.choice(T, 1)[0], i])
    return torch.stack(pts)


def param_str(system: PhaseSpace):
    param_tuples = [[k, system.parameters[k]] for k in system.parameters]
    title_str = ''
    for i in range(len(param_tuples)):
        name = system.param_display[param_tuples[i][0]] if param_tuples[i][0] in \
                                                           list(system.param_display.keys()) else param_tuples[i][0]
        title_str += f'{name}={param_tuples[i][1]:.2f}'
        if not (i + 1) % 2: title_str += '\n'
        elif i < len(param_tuples) - 1: title_str += ' ; '
    return title_str


def traj_in_vecs(x: torch.Tensor, xdot: torch.Tensor, gt_sys: PhaseSpace, arch: PhaseSpace, H: Diffeo,
                 T: float=40, extra_str: str='', aug: Augmentation=None, color: str='k'):
    # choose points as initial points from which to iterate
    inds = np.random.choice(x.shape[0], 3, replace=False)
    dim = x.shape[-1]

    # get ground-truth trajectories
    gt_traj = gt_sys.trajectories(x[inds], T=T).detach()
    if aug is not None:
        traj, _ = aug(0, gt_traj.reshape(-1, x.shape[-1]), gt_traj.reshape(-1, x.shape[-1]))
        gt_traj = traj.reshape(gt_traj.shape)

    # get fitted trajectories
    y = H.forward(x[inds]).detach()
    traj = arch.trajectories(y, T=T)
    traj = H.reverse(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape).detach()

    if dim > 2:
        # move all observables to 2D archetype space
        x, xdot, _ = H.jvp_forward(x, xdot)
        x, xdot = x[..., :2].detach(), xdot[..., :2].detach()

        gt_traj = H.forward(gt_traj.reshape(-1, gt_traj.shape[-1])).reshape(gt_traj.shape).detach()[..., :2]
        traj = H.forward(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape).detach()[..., :2]

    gt_traj, traj = gt_traj.cpu().numpy(), traj.cpu().numpy()

    # plot fitted and true trajectories (all in 2D space)
    for i in range(traj.shape[1]):
        plt.plot(gt_traj[:, i, 0], gt_traj[:, i, 1], lw=2, alpha=.7, color='tab:blue')
        plt.plot(traj[:, i, 0], traj[:, i, 1], lw=2, alpha=.7, color='tab:red')

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

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    title_str = param_str(arch)
    plt.title(title_str+extra_str, color=color)


def plot_trajectory(dim: int, system: PhaseSpace, T: float, save_path: str, aug: Augmentation=None):
    init = torch.rand(7, dim)*(system.position_lims[1]-system.position_lims[0]) + system.position_lims[0]
    traj = system.trajectories(init, T=T)
    t = np.linspace(0, T, traj.shape[0])

    if aug is not None:
        atraj, _ = aug.forward(0, traj.reshape(-1, dim), traj.reshape(-1, dim))
        traj = atraj.reshape(traj.shape)

    traj = traj.numpy()

    title_str = param_str(system)

    plt.figure(figsize=(4, 2*dim))
    for i in range(dim):
        plt.subplot(dim, 1, i+1)
        if i==0: plt.title(title_str)
        for j in range(traj.shape[1]):
            plt.plot(t, traj[:, j, i], alpha=.5)
        plt.ylabel('position')
        plt.xlim(t[0], t[-1])
    plt.subplots_adjust(left=None, bottom=.01, right=None, top=None, wspace=None, hspace=None)

    plt.xlabel('time')
    plt.tight_layout()
    plt.savefig(save_path)


def compile_results(path: str, dim: int):
    archetypes = [
        [-.25, -.5, .25],
        [.25, -.5, .25],
        [-.25, .5, .25],
        [.25, .5, .25]
    ]

    full_dict = {
        'archetypes': archetypes
    }
    fields = ['losses', 'logdets', '2Dlosses', 'data_mean', 'data_std', 'params']
    for field in fields: full_dict[field] = []
    for i in range(1000):
        try:
            with open(path + f'{i}.pkl', 'rb') as f:
                res = pickle.load(f)
            for field in fields:
                full_dict[field].append(res[field])
        except: pass

    for field in fields: full_dict[field] = np.array(full_dict[field])
    try:
        with open(path + 'all_results.pkl', 'wb') as f: pickle.dump(full_dict, f)
    except: pass


@click.command()
@click.option('--exp_type', help='which data to classify', type=str, default='SO')
@click.option('--n_points', help='number of points in the system', type=int, default=1000)
@click.option('--job',      help='job number (used for parallelization)', type=int, default=0)
@click.option('--lr',       help='learning rate used in training', type=float, default=1e-3)
@click.option('--its',      help='number of iterations', type=int, default=2000)
@click.option('--n_layers', help='number of layers in models', type=int, default=4)
@click.option('--n_freqs',  help='number of frequencies in coupling', type=int, default=5)
@click.option('--fr_rat',   help='ratio of iterations during which scale params are frozen', type=float, default=.25)
@click.option('--dim',      help='the dimensionality of the data', type=int, default=2)
@click.option('--time',     help='amount of time for discretization', type=float, default=20.)
@click.option('--det_reg',  help='regularization of the determinant ', type=float, default=1e-3)
@click.option('--cen_reg',  help='regularization of the origin ', type=float, default=1e-6)
@click.option('--proj_reg', help='regularization for the 2D projections', type=float, default=-1)
@click.option('--linaug',   help='whether to add a linear augmentation', type=float, default=0)
@click.option('--quadaug',  help='strength of the quadratic augmentation', type=float, default=0)
@click.option('--diffaug',  help='strength of the diffeomorphism augmentation', type=float, default=0)
@click.option('--verbose',  help='whether to print progress or not', type=int, default=-1)
@click.option('--noise',    help='amount of noise to add to phase space', type=float, default=0)
@click.option('--dim2_weight', help='relative weight of the first 2 dimensions', type=float, default=-1)
@click.option('--save_h',   help='whether to save the model', type=int, default=0)
@click.option('--rep',      help='repitition of the experiment (basically just adds a number to the start of the path', type=int, default=0)
@click.option('--w_decay',  help='weight decay used', type=float, default=1e-3)
@click.option('--rem_small', help='if 1, removes velocities with norms that are small compared to the observed noise (as preprocessing)', type=int, default=0)
def classify_all(exp_type: str, n_points: int, job: int, lr: float, its: int, n_layers: int,
                 n_freqs: int, fr_rat: float, dim: int, time: float,
                 det_reg: float, cen_reg: float, proj_reg: float,
                 linaug: float, quadaug: float, diffaug: float, verbose: int, noise: float,
                 dim2_weight: float, save_h: int, rep, w_decay, rem_small):
    # ============================================ create directory ====================================================
    assert exp_type in list(SYSTEMS.keys())

    path = root + f'{exp_type}_dim={dim}/' +\
                  (f'lin{linaug}quad{quadaug}diff{diffaug}_' if dim > 2 else '') +\
                  f'its={its}_npts={n_points}_freqs={n_freqs}_layers={n_layers}' +\
                  (f'_proj={proj_reg:.1f}' if dim > 2 else '') +\
                  (f'_noise={noise:.2f}' if noise > 0 else '') +\
                  f'_time={time:.1f}' +\
                  f'/rep={rep}/'
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

    system = SYSTEMS[exp_type]()

    # ============================================ add augmentations ===================================================
    augmentaions = []
    if linaug > 0: augmentaions.append(LinearAug(dim=dim))
    if quadaug > 0: augmentaions.append(QuadAug(dim=dim, amnt=quadaug))
    if diffaug > 0: augmentaions.append(DiffeoAug(dim=dim, init_amnt=diffaug))
    augment = ComposeAug(*augmentaions)
    parameters = system.parameters
    # ============================================ add augmentations ===================================================

    logging.info('\n\n')
    logging.info('; '.join(f'{k}={v:.2f}' for k, v in parameters.items()))

    # plot the underlying latent system
    if dim > 2: plot_trajectory(dim, system, time, path + f'{job}_original_trajectories.png')

    # plot the observed (augmented) system
    if (linaug > 0 or quadaug > 0 or diffaug > 0) and dim > 2:
        plot_trajectory(dim, system, time, path + f'{job}_augmented_trajectories.png', aug=augment)

    # iterate points on trajectory for a bit, so they'll be closer to the invariant set
    init = torch.rand(n_points, dim)*(system.position_lims[1]-system.position_lims[0]) + system.position_lims[0]
    x = system.rand_on_traj(init, T=time)
    dx = system(0, x)

    # augment points
    x, dx = augment(0, x, dx)
    dx = dx + torch.randn_like(dx)*noise

    # as a preprocessing step for our method, removes all vectors with norms that are too small compared to noise
    if rem_small == 1 and noise > 0:
        norms = torch.norm(dx, dim=1)
        x, dx = x[norms > dim*noise], dx[norms > dim*noise]

    archetypes = [
        [-.25, -.5, .25],
        [.25, -.5, .25],
        [-.25, .5, .25],
        [.25, .5, .25]
    ]

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x, dx = x.to(device), dx.to(device)

    # allow for different repititions to initialize different network parameters
    torch.manual_seed(job + 1000*rep)
    np.random.seed(job + 1000*rep)

    # calculate trajectory for calculating the cycle errors
    cycle_traj = system.trajectories(x[:1], T=50)
    cycle_traj = cycle_traj[cycle_traj.shape[0]//2:][:, 0]

    plt.figure(figsize=(7, 4*len(archetypes)//2))
    subplot = 1
    for (a, omega, decay) in res_dict['archetypes']:
        # ============================================ fit archetype ===================================================
        archetype = get_oscillator(a=a, omega=omega, decay=decay)
        H = Diffeo(dim=dim, n_layers=n_layers, K=n_freqs, rank=2).to(device)
        H, loss, ldet, score = fit_DFORM(H, x.clone(), dx.clone(), archetype, its=its,
                                         verbose=verbose>0, lr=lr, freeze_frac=fr_rat, det_reg=det_reg,
                                         center_reg=cen_reg, proj_reg=proj_reg, weight_decay=w_decay,
                                         dim2_weight=None if dim2_weight < 0 else dim2_weight)

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
                     aug=augment, color='blue' if np.sign(a)==np.sign(system.parameters['bif']) else 'red')
        subplot += 1
        if save_h > 0: torch.save(H, path + f'{job}_a={a:.2f}_om={omega:.2f}.pth')
    plt.tight_layout()
    plt.savefig(path + f'{job}_projections.png')
    with open(path + f'{job}.pkl', 'wb') as f: pickle.dump(res_dict, f)

    compile_results(path, dim)


if __name__ == '__main__':
    classify_all()
