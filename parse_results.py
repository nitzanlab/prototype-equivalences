import numpy as np
from systems import SO, Selkov, BZreaction, Repressilator, ComposeAug, DiffeoAug, VanDerPol, LienardPoly, LienardSigmoid
import pickle
import torch
from scipy.special import logsumexp
import matplotlib
matplotlib.use('pgf')

from matplotlib.lines import Line2D
from matplotlib import pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'figure.dpi': 90,
    'figure.autolayout': True,
    'axes.labelsize': 20,
    'axes.titlesize': 21,
    # 'font.weight': 'bold',
    'font.size': 21,
    'axes.linewidth': 2,
    # 'axes.labelweight': 'bold',
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

SYSTEMS = {
    'SO': SO,
    'selkov': Selkov,
    'bzreaction': BZreaction,
    'repressilator': Repressilator,
    'vanderpol': VanDerPol,
    'lienardpoly': LienardPoly,
    'lienardsigmoid': LienardSigmoid,
}


def make_dictionary(typ: str, dim: int=2, npts: int=2500, its: int=2500, diff: float=0, nfreqs: int=15,
                    lin: bool=False, noise: float=0, proj: float=0, time: float=10., tst: float=0,
                    layers: int=4, path: str=None):
    if path is None:
        repr = typ=='repressilator'
        path = f'results/{typ}_dim={dim}/'
        if not repr: path = path+ f'lin{"1.0" if lin else "0.0"}quad0.0diff{diff:.1f}_'
        path = path + f'its={its}_npts={npts}_' \
               f'freqs={nfreqs}_layers={layers}' +\
               (f'_proj={proj:.1f}' if dim > 2 else '') +\
               (f'_noise={noise:.2f}' if noise > 0 else '') +\
               (f'_time={time:.1f}' if time != 10 else '') +\
               (f'_tst={tst:.1f}' if tst > 0 else '') +\
               f'/'
    print(path)
    full_dict = {
        'losses': [],
        'logdets': [],
        'params': [],
        'preds': [],
        'probs': [],
        'pred_archetypes': [],
        'true_labels': [],
        'param_display': SYSTEMS[typ].param_display,
        'param_ranges': SYSTEMS[typ].param_ranges
    }
    display_params = list(SYSTEMS[typ].param_display.keys())

    with open(path + 'all_results.pkl', 'rb') as f:
        res_dict = pickle.load(f)

    full_dict['archetypes'] = res_dict['archetypes']
    full_dict['losses'] = res_dict['2Dlosses']
    full_dict['logdets'] = res_dict['logdets'] - 2*np.log(res_dict['data_std'])[:, None]
    full_dict['pred_archetypes'] = np.argmin(res_dict['2Dlosses'], axis=1)
    full_dict['preds'] = np.array([np.sign(res_dict['archetypes'][a][0]) for a in full_dict['pred_archetypes']])
    full_dict['true_labels'] = np.array([np.sign(a['bif']) for a in res_dict['params']])
    full_dict['params'] = np.array([[a[b] for b in display_params] for a in res_dict['params']])
    full_dict['probs'] = np.exp(-res_dict['losses'] - logsumexp(-res_dict['losses'], axis=1)[:, None])
    if full_dict['probs'].shape[1] == 4:
        full_dict['cycle_probs'] = full_dict['probs'][:, 1] + full_dict['probs'][:, 3]
    elif full_dict['probs'].shape[1] ==2:
        full_dict['cycle_probs'] = full_dict['probs'][:, 1]
    else:
        full_dict['cycle_probs'] = np.sum([full_dict['probs'][:, i] for i in [1, 3, 5, -1]], axis=0)

    print(f'{typ} acc={np.mean(1-np.abs(full_dict["preds"] - full_dict["true_labels"])/2):.3f}')
    return full_dict


def plot_boundary(typ):
    par_keys = list(SYSTEMS[typ].param_display.keys())
    ranges = [
        SYSTEMS[typ].param_ranges[k] for k in par_keys
    ]
    xx, yy = np.meshgrid(np.linspace(ranges[0][0], ranges[0][1], 100),
                         np.linspace(ranges[1][0], ranges[1][1], 100))
    zz = np.zeros_like(xx)
    for i in range(zz.shape[0]):
        for j in range(zz.shape[0]):
            syst = SYSTEMS[typ](**{par_keys[0]: xx[i, j], par_keys[1]: yy[i, j]})
            zz[i, j] = syst.dist_from_bifur()
    plt.contour(xx, yy, zz, levels=[0], alpha=.8, colors='k', linestyles='dashed', linewidths=4)


archetypes = [
    ['CW node', 'tab:orange'],
    ['CW cycle', 'tab:blue'],
    ['CCW node', 'tab:red'],
    ['CCW cycle', 'tab:green'],
    ['CW node', 'tab:orange'],
    ['CW cycle', 'tab:blue'],
    ['CCW node', 'tab:red'],
    ['CCW cycle', 'tab:green']
]


# ============================================ 2D results 1000 points ==================================================
# dim = 4
# npts = 1000
# its = 2000
# noise = 0
# proj = 0
#
# base = {
#     'npts': npts, 'dim': dim, 'lin': dim>2, 'nfreqs': 5, 'its': its, 'noise': noise, 'proj': proj,
#     'time': 10,
# }
#
# experiments = [
#     # [{'typ': 'SO', }, 'SO'],
#     # [{'typ': 'SO', }, 'Aug. SO'],
#     [{'typ': 'bzreaction',}, 'BZReaction'],
#     [{'typ': 'selkov', }, "Selkov"],
#     # [{'typ': 'vanderpol', }, "Vand der Pol"],
#     # [{'typ': 'lienardsigmoid', }, "Lienard Sigmoid"],
#     # [{'typ': 'lienardpoly', }, "Lienard Poly"],
# ]
#
# plt.figure(figsize=(3.5*len(experiments), 8))
# for i, exp in enumerate(experiments):
#     plt.subplot(2, len(experiments), i+1)
#     typ = exp[0]['typ']
#     res_dict = make_dictionary(**exp[0], **base)
#     keys = list(res_dict['param_display'].keys())
#     params = res_dict['params']
#
#     preds = res_dict['preds']
#     probs = res_dict['probs']
#     ldets = np.array([res_dict['logdets'][i, res_dict['pred_archetypes'][i]] for i in range(len(probs))])
#     szprobs = np.array([probs[i, res_dict['pred_archetypes'][i]] for i in range(len(probs))])
#     # szprobs = 1/(np.abs(ldets)+1)
#
#     for j in range(len(res_dict['archetypes'])):
#         inds = np.array(res_dict['pred_archetypes']) == j
#         plt.scatter(params[inds, 0], params[inds, 1], 60, archetypes[j][1],
#                     label=archetypes[j][0], alpha=.6)
#     plot_boundary(typ)
#
#     plt.xlabel(res_dict['param_display'][keys[0]])
#     plt.xlim(res_dict['param_ranges'][keys[0]])
#     plt.ylabel(res_dict['param_display'][keys[1]])
#     plt.ylim(res_dict['param_ranges'][keys[1]])
#     acc = np.mean(1-np.abs(res_dict['preds'] - res_dict['true_labels'])/2)
#     plt.title(exp[1] + f'\nacc={acc:.2f}')
#
# lines_labels = [plt.gca().get_legend_handles_labels()]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# plt.gcf().legend(lines, labels, loc='center', ncol=4, fancybox=True, fontsize=16, bbox_to_anchor=(.5, .475))
#
# for i, exp in enumerate(experiments):
#     plt.subplot(2, len(experiments), len(experiments)+i+1)
#     typ = exp[0]['typ']
#     res_dict = make_dictionary(**exp[0], **base)
#     keys = list(res_dict['param_display'].keys())
#     params = res_dict['params']
#
#     pred_archetypes = np.argmin(res_dict['losses'], axis=1)
#     losses = np.sum(res_dict['probs']*res_dict['logdets'], axis=1)
#     # losses = res_dict['cycle_probs']
#
#     plot_boundary(typ)
#     plt.scatter(params[:, 0], params[:, 1], c=losses, cmap='coolwarm')
#     # plt.colorbar()
#     plt.xlabel(res_dict['param_display'][keys[0]])
#     plt.xlim(res_dict['param_ranges'][keys[0]])
#     plt.ylabel(res_dict['param_display'][keys[1]])
#     plt.ylim(res_dict['param_ranges'][keys[1]])
#
# plt.tight_layout(h_pad=2, w_pad=-1)
# plt.savefig(f'graphics/{dim}D_results{npts}.pdf')
# plt.savefig(f'graphics/{dim}D_results{npts}.png')

# ============================================ 2D results vs noise and points ==========================================
# kwargs = {'its': 2000, 'dim': 2, 'nfreqs': 5}
# avg_typs = ['SO', 'bzreaction', 'selkov', 'vanderpol', 'lienardsigmoid', 'lienardpoly']
# display_typs = [['SO', 'SO', 'tab:blue'], ['bzreaction', 'BZ Reaction', 'tab:orange'], ['selkov', 'Selkov', 'tab:green']]
#
# plt.figure(figsize=(9, 4))
#
# plt.subplot(121)
# n_points = [50, 100, 250, 500, 1000]
# pnt_dict = {}
# for typ in avg_typs:
#     accs = []
#     for pnts in n_points:
#         res_dict = make_dictionary(typ=typ, npts=pnts, **kwargs)
#         accs.append(np.mean(1-np.abs(res_dict['preds'] - res_dict['true_labels'])/2))
#     pnt_dict[typ] = np.array(accs)
#
# avg_accs = np.mean([pnt_dict[a] for a in avg_typs], axis=0)
#
# plt.plot(n_points, avg_accs, lw=3, color='k', alpha=.7, label='average')
# for (typ, name, color) in display_typs:
#     plt.plot(n_points, pnt_dict[typ], lw=3, alpha=.7, label=name, color=color)
# plt.xlim(n_points[0], n_points[-1])
# plt.legend()
# plt.xlabel('number of observations')
# plt.ylabel('accuracy')
# plt.ylim(.5, 1)
# # plt.xscale('log')
#
# plt.subplot(122)
# noises = [0.05, 0.10, 0.25, 0.5]
# kwargs = {'its': 2000, 'dim': 2, 'nfreqs': 5, 'time': .5}
#
# noise_dict = {}
# for typ in avg_typs:
#     accs = {
#         1000: []
#     }
#     for noise in noises:
#         for n in accs.keys():
#             res_dict = make_dictionary(typ=typ, npts=n, noise=noise, **kwargs)
#             accs[n].append(np.mean(1-np.abs(res_dict['preds'] - res_dict['true_labels'])/2))
#     for n in accs.keys():
#         accs[n] = np.array(accs[n])
#     noise_dict[typ] = accs
#
# avg_accs = {}
# for n in noise_dict['SO'].keys():
#     avg_accs[n] = np.mean([noise_dict[a][n] for a in avg_typs], axis=0)
#
# lines = {500: 'dashed', 1000: None}
# for n in avg_accs.keys():
#     plt.plot(noises, avg_accs[n], lw=3, color='k', alpha=.7, label='average', linestyle=lines[n])
#     for (typ, name, color) in display_typs:
#         plt.plot(noises, noise_dict[typ][n], lw=3, alpha=.7, label=name, color=color, linestyle=lines[n])
# plt.xlim(noises[0], noises[-1])
# plt.ylim(.49, 1)
# # plt.legend([Line2D([0], [.6], color='k', linestyle=lines[n]) for n in lines.keys()],
# #            [f'{n} points' for n in lines.keys()], loc='lower left')
#
# plt.xlabel(r'noise $\sigma$')
# plt.ylabel('accuracy')
#
# plt.savefig(f'graphics/2D_vs_pnts.png')

# ============================================ grid results ============================================================
nfreqs = 5
layers = 3
fr_rat = .2
noise = 0.1
its = 2000

systems = [
    'simple_oscillator_noaug_TWAgrid',
    'simple_oscillator_nsfcl_TWAgrid',
    'bzreaction_TWAgrid',
    'selkov_TWAgrid',
    'lienard_poly_TWAgrid',
    'lienard_sigmoid_TWAgrid',
    'vanderpol_TWAgrid',
    ]

for sys in systems:
    path = f'results/{sys}/its={its}_lr=1E-03_layers={layers}_freqs={nfreqs}_fr_rat={fr_rat:.2f}_noise={noise:.2f}/'
    full_res = {
        'losses': [],
        'logdets': [],
        'labels': [],
        'preds': [],
    }
    for i in range(100):
        try:
            with open(path + f'{i}.pkl', 'rb') as f: res = pickle.load(f)
            full_res['losses'] += res['2Dlosses']
            full_res['logdets'] += res['logdets']
            full_res['preds'] += [np.sign(res['archetypes'][a][0]) for a in np.argmin(res['2Dlosses'], axis=1)]
            full_res['labels'] += [-1 if a[0]==4 else 1 for a in res['true_labels']]
        except: pass
    acc = np.mean(1-np.abs(np.array(full_res['preds']) - np.array(full_res['labels']))/2)
    print(f'{sys} accuracy={acc:.3f}')

# ============================================ invariant set ===========================================================
# with open('results/invariant_set/all_results.pkl', 'rb') as f:
#     results = pickle.load(f)
#
# cutoff = .9
# points = list(results.keys())
# medians, quant_low, quant_high = [], [], []
#
# for npts in results.keys():
#     losses = results[npts]['loss']
#     inds = np.array(losses) <= np.quantile(losses, cutoff)
#     dists = np.array(results[npts]['dists'])[inds]
#
#     medians.append(np.median(dists))
#     quant_low.append(np.quantile(dists, .25))
#     quant_high.append(np.quantile(dists, .75))
#
# plt.figure(figsize=(4, 4))
# plt.fill_between(points, quant_low, quant_high, alpha=.5, color='gray')
# plt.plot(points, medians, lw=3, alpha=.8, color='k')
# plt.xlim(points[0], points[-1])
# # plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel('number of observations')
# plt.ylabel('average error')
# plt.tight_layout()
# plt.savefig('results/invariant_set/all_results.png')

# ================================================ repressilator =======================================================
# npts = 1500
# noise = 0
# time = 40
#
# base = {
#     'npts': npts, 'lin': False, 'nfreqs': 5, 'noise': noise,  'time': time, 'layers': 3,
# }
# npts = 1000
# om = 2
# freqs = 5
# exps = [
#     [{'path': f'results/repressilator_dim=2/its=2000_npts={npts}_freqs={freqs}_layers=3_multitraj_omega={om:.1f}/'}, 'Repressilator 2D'],
#     [{'path': f'results/repressilator_dim=3/its=2000_npts={npts}_freqs={freqs}_layers=3_proj=-1.0_multitraj_omega={om:.1f}/'}, '3D'],
#     [{'path': f'results/repressilator_dim=4/its=2000_npts={npts}_freqs={freqs}_layers=3_proj=-1.0_multitraj_omega={om:.1f}/'}, '4D'],
#     [{'path': f'results/repressilator_dim=5/its=2000_npts={npts}_freqs={freqs}_layers=3_proj=-1.0_multitraj_omega={om:.1f}/'}, '5D'],
#     [{'path': f'results/repressilator_dim=6/its=2000_npts={npts}_freqs={freqs}_layers=3_proj=-1.0_multitraj_omega={om:.1f}/'}, '6D'],
# ]
#
# plt.figure(figsize=(3.5*len(exps), 8))
# for i, exp in enumerate(exps):
#     plt.subplot(2, len(exps), i+1)
#     res_dict = make_dictionary(typ='repressilator', **exp[0], **base)
#     keys = list(res_dict['param_display'].keys())
#     params = res_dict['params']
#
#     preds = res_dict['preds']
#     probs = res_dict['probs']
#     ldets = np.array([res_dict['logdets'][i, res_dict['pred_archetypes'][i]] for i in range(len(probs))])
#     szprobs = np.array([probs[i, res_dict['pred_archetypes'][i]] for i in range(len(probs))])
#     # szprobs = 1/(np.abs(ldets)+1)
#
#     for j in range(len(res_dict['archetypes'])):
#         inds = np.array(res_dict['pred_archetypes']) == j
#         plt.scatter(params[inds, 0], params[inds, 1], 60, archetypes[j][1],
#                     label=archetypes[j][0], alpha=.6)
#     plot_boundary('repressilator')
#
#     lines_labels = [plt.gca().get_legend_handles_labels()]
#     lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#
#     plt.xlabel(res_dict['param_display'][keys[0]])
#     plt.xlim(res_dict['param_ranges'][keys[0]])
#     plt.ylabel(res_dict['param_display'][keys[1]])
#     plt.ylim(res_dict['param_ranges'][keys[1]])
#     acc = np.mean(1-np.abs(res_dict['preds'] - res_dict['true_labels'])/2)
#     plt.title( f'{exp[1]}\nacc={acc:.2f}')
#
#     plt.subplot(2, len(exps), len(exps)+i+1)
#     keys = list(res_dict['param_display'].keys())
#     params = res_dict['params']
#
#     pred_archetypes = np.argmin(res_dict['losses'], axis=1)
#     losses = np.sum(res_dict['probs']*res_dict['logdets'], axis=1)
#     # losses = res_dict['cycle_probs']
#
#     plot_boundary('repressilator')
#     plt.scatter(params[:, 0], params[:, 1], c=losses, cmap='coolwarm')
#     plt.colorbar()
#     plt.xlabel(res_dict['param_display'][keys[0]])
#     plt.xlim(res_dict['param_ranges'][keys[0]])
#     plt.ylabel(res_dict['param_display'][keys[1]])
#     plt.ylim(res_dict['param_ranges'][keys[1]])
#
# plt.gcf().legend(lines, labels, loc='center', ncol=4, fancybox=True, fontsize=16, bbox_to_anchor=(.5, .475))
#
# plt.tight_layout(h_pad=2, w_pad=-1)
# plt.savefig(f'graphics/repressilator.pdf')
# plt.savefig(f'graphics/repressilator.png')