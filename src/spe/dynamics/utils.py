import numpy as np
import torch
from scipy.optimize import minimize
from scipy import interpolate
from typing import Callable, Union
import ot


# ================================================= vv Flow Utils vv ===================================================
def cartesian_to_polar(x, y):
    """
    Transform cartesian coordinates to polar coordinates
    """
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return r, theta


def polar_derivative_to_cartesian_derivative(r, theta, rdot, thetadot):
    """
    Transform polar derivatives to cartesian derivatives
    """
    xdot = torch.cos(theta) * rdot - r * torch.sin(theta) * thetadot
    ydot = torch.sin(theta) * rdot + r * torch.cos(theta) * thetadot
    return xdot, ydot


def get_dist_from_bifur_curve(a, b, f, a_limits=None):
    """
    --------- copied from twa
    Computes distance of a,b points to the curve defined by (a,f(a)). Sign corresponds to the side of the curve.
    ab_s: (n,2) array of points corresponding to a,b parameters
    f: function of a
    """
    def obj(x):
        y = f(x)
        return np.sqrt((x - a)**2 + (y - b)**2)
    res = minimize(obj, x0=a, bounds=[a_limits])
    a_opt = res.x
    dist = obj(a_opt)
    return dist


def simulate_trajectory(grad, init: torch.Tensor, T: float, step: float=5e-2, euler: bool=False,
                        clamp: float=10., noise: Union[float, Callable]=None):
    """
    Discretization of a trajectory starting at some initial points using Runga-Kutta or the Euler method
    :param grad: the gradient function; inputs are [N, dim], and outputs are the same
    :param init: initial positions, a tensor with shape [N, dim]
    :param T: the amount of time to integrate the dynamics
    :param step: step size to use for the euler discretization
    :param euler: a boolean dictating whether an Euler (first-order) discretization should be used, other-wise a 4th
                  order method is used
    :param clamp: clamp the gradients to a particular value, to ensure numerical stability
    :param noise: if this nvariable is not None, then the Euler-Murayama method is used to simulate an SDE, with:
                  - noise is float: assumes the SDE has the same amount of noise uniformly everywhere and 'noise' is
                        the nvariance of the noise
                  - noise is a Callable: assumes that the input to the callable is a batch of positions 'x' ([N, dim])
                        and the  output is a torch tensor depicting the nvariance of the noise at 'x' ([N])
    :return: a torch tensor with shape [round(T/step), N, dim]
    """
    if noise is not None: euler = True   # only Euler-Murayama supported for SDE integration
    if isinstance(noise, int): noise = float(noise)
    if isinstance(noise, float):
        nvar = noise
        noise = lambda x: nvar*torch.ones_like(x)
    elif isinstance(noise, np.ndarray):
        nvar = torch.from_numpy(noise).float()
        noise = lambda x: nvar.to(x.device)
    elif isinstance(noise, torch.Tensor):
        nvar = noise
        noise = lambda x: nvar.to(x.device)

    flows = [init]
    pnts = init.clone()
    for t in np.arange(0, T, step):
        if euler:
            if noise is not None:
                eps = torch.sqrt(noise(pnts))*(step**0.5)*torch.randn_like(pnts)
            else:
                eps = 0
            pnts = pnts + step*torch.clamp(grad(pnts), -clamp, clamp) + eps
        else:
            k1 = torch.clamp(grad(pnts), -clamp, clamp)
            k2 = torch.clamp(grad(pnts+.5*step*k1), -clamp, clamp)
            k3 = torch.clamp(grad(pnts+.5*step*k2), -clamp, clamp)
            k4 = torch.clamp(grad(pnts+step*k3), -clamp, clamp)

            pnts = pnts + step*(k1 + 2*k2 + 2*k3 + k4)/6

        flows.append(pnts.clone())
    return torch.stack(flows)


def get_oscillator(a: float, omega: float, decay: float=1):
    """
    Creates a function that returns a simple oscillat's velocities. If the input dimensions are larger than 2, then
    all dimensions above the first two dimensions will decay to zero
    :param a: the dampening coefficient - smaller than 0 gives nodes, larger limit cycles
    :param omega: angular velocity - smaller than 0 gives counter clockwise behavior, larger gives clockwise behavior
    :param decay: how fast the "extra" dimensions decay to 0
    :return: a function that recieves as input a tensor with shape [N, dim] with dim >= 2 and outputs a tensor of the
             same dimensions, [N, dim]
    """
    def func(x: torch.Tensor):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        nonosc = x[..., 2:]  # non-oscillatory dimensions
        osc = x[..., :2]
        r = torch.sqrt(torch.sum(osc**2, dim=-1))
        theta = torch.atan2(osc[..., 1], osc[..., 0])

        rdot = r*(a-r*r)

        xdot = torch.cos(theta)*rdot - r*torch.sin(theta)*omega
        ydot = torch.sin(theta)*rdot + r*torch.cos(theta)*omega
        return torch.cat([xdot[..., None], ydot[..., None], -decay*nonosc], dim=-1).reshape(shape)
    return func


def interp_vectors(x: torch.Tensor, xdot: torch.Tensor, pos: torch.Tensor=None, smoothing: float=.5,
                   func: str='linear'):
    """
    Interpolate vector field
    """
    if pos is None: pos = x.clone()
    device = x.device
    x, xdot, pos = x.cpu().numpy(), xdot.cpu().numpy(), pos.cpu().numpy()
    dim = x.shape[-1]

    vectors = []
    for d in range(dim):
        f_src = xdot[..., d]

        interp = interpolate.Rbf(*x.T, f_src, function=func, smooth=smoothing)
        z = interp(*pos.T)

        vectors.append(z)

    vectors = np.stack(vectors, axis=-1)
    return torch.from_numpy(pos).float().to(device), torch.from_numpy(vectors).float().to(device)

# ================================================= ^^ Flow Utils ^^ ===================================================
# ================================================= vv Eval vv =========================================================


def dynamic_time_warp(x: torch.Tensor, y: torch.Tensor,
                      d: Callable=lambda x, y: torch.sum(torch.abs(x-y))) -> [float, torch.Tensor]:
    """
    Calculates the dynamic time warping (DTW) between x and y under the specified cost function
    :param x: the trajectory, a torch tensor with shape [N, dim]
    :param y: the predicted trajectory, a torch tensor with shape [M, dim]
    :param d: the cost function to use for the time warping, defaults to L1
    :return: - the DTW distance between x and y
             - y, warped to the best time-matching with x
    """
    dtw = torch.ones(x.shape[0], y.shape[0])*1e10
    dtw[0, 0] = 0
    for i in range(1, x.shape[0]):
        for j in range(1, y.shape[0]):
            cost = d(x[i], y[j]).item()
            dtw[i, j] = cost + min(dtw[i-1, j].item(), dtw[i, j-1].item(), dtw[i-1, j-1].item())
    return dtw[x.shape[0]-1, y.shape[0]-1]/x.shape[0], y[torch.argmin(dtw, dim=1)]


def MMD(x: torch.Tensor, y: torch.Tensor, scale: float=None):
    """
    Calculates the Maximum Mean Discrepancy (MMD) between the samples x and y under an RBF kernel
    :param x: an [n, d] tensor representing a d-dimensional point cloud with n points (one per row)
    :param y: an [m, d] tensor representing a d-dimensional point cloud with m points (one per row)
    :param scale: the scale of the RBF kernel - if None, the standard deviation of x is chosen as the scale
    :return: the MMD between x and y
    """
    if scale is None: scale = torch.std(x).item()
    scale = scale*scale

    Dxx = torch.sum(x*x, dim=-1)[:, None] - 2*x@x.T + torch.sum(x*x, dim=-1)[None]
    Dxy = torch.sum(x*x, dim=-1)[:, None] - 2*x@y.T + torch.sum(y*y, dim=-1)[None]
    Dyy = torch.sum(y*y, dim=-1)[:, None] - 2*y@y.T + torch.sum(y*y, dim=-1)[None]

    Dxx = torch.exp(-.5*Dxx/scale)
    Dxx.diagonal(dim1=0, dim2=1).zero_()
    Dxy = torch.exp(-.5*Dxy/scale)
    Dyy = torch.exp(-.5*Dyy/scale)
    Dyy.diagonal(dim1=0, dim2=1).zero_()

    return (torch.sum(Dxx)/(x.shape[0]*(x.shape[0]-1)) + torch.mean(Dxy)
            + torch.sum(Dyy)/(y.shape[0]*(y.shape[0]-1))).item()


def sinkhorn(x: torch.Tensor, y: torch.Tensor, p: float = 2, eps: float = 1e-1, max_iters: int = 10000,
             stop_thresh: float = 1e-8) -> [float, torch.Tensor, torch.Tensor]:
    """
    Compute the Entropy-Regularized p-Wasserstein Distance between two d-dimensional point clouds
    using the Sinkhorn scaling algorithm.
    Code adapted from: https://github.com/fwilliams/scalable-pytorch-sinkhorn/tree/main
    :param x: an [n, d] tensor representing a d-dimensional point cloud with n points (one per row)
    :param y: an [m, d] tensor representing a d-dimensional point cloud with m points (one per row)
    :param p: which norm to use. Must be an integer greater than 0.
    :param eps: the reciprocal of the sinkhorn entropy regularization parameter.
    :param max_iters: the maximum number of Sinkhorn iterations to perform.
    :param stop_thresh: stop if the maximum change in the parameters is below this amount
    :return: the approximate p-wasserstein distance between point clouds x and y
    """
    if p == 2:
        M = torch.sqrt(torch.clamp(torch.sum(x*x, dim=-1)[:, None] - 2*x@y.T + torch.sum(y*y, dim=-1)[None], 0, 1000))
    elif p == 1:
        M = torch.abs(x[:, None] - y[None]).sum(dim=-1)
    else:
        M = torch.sum((x[:, None] - y[None, :]).abs() ** p).pow(1 / p)

    return ot.sinkhorn2(a=np.ones(x.shape[0]) / x.shape[0], b=np.ones(y.shape[0]) / y.shape[0],
                        M=M.cpu().detach().numpy(), reg=eps, numItermax=max_iters, stopThr=stop_thresh)


def EMD(x: torch.Tensor, y: torch.Tensor, p: float=2):
    if p == 2:
        M = torch.sqrt(torch.clamp(torch.sum(x*x, dim=-1)[:, None] - 2*x@y.T + torch.sum(y*y, dim=-1)[None], 0, 1000))
    elif p==1:
        M = torch.abs(x[:, None] - y[None]).sum(dim=-1)
    else:
        M = torch.sum((x[:, None]-y[None, :]).abs()**p).pow(1/p)

    return ot.emd2(a=np.ones(x.shape[0])/x.shape[0], b=np.ones(y.shape[0])/y.shape[0], M=M.cpu().detach().numpy())


def duplicate_to_match_lengths(arr1, arr2):
    """
    Taken as is from https://github.com/ariel415el/GPDM
    Duplicates randomly selected entries from the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if arr1.shape[1] == arr2.shape[1]:
        return arr1, arr2
    elif arr1.shape[1] < arr2.shape[1]:
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[1] // arr2.shape[1]
    arr2 = torch.cat([arr2] * b, dim=1)
    if arr1.shape[1] > arr2.shape[1]:
        indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
        arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

    return arr1, arr2


def SWD(set1: torch.Tensor, set2: torch.Tensor, n_proj: int=100) -> float:
    """
    Measures the sliced Wasserstein distance (SWD) using random projections of the data
    :param set1: the training dataset, as a torch Tensor with shape [N_train, d1, d2, ...]
    :param set2: the testing dataset, as a torch Tensor with shape [N_test, d1, d2, ...]
    :param n_proj: number of projections to use for the mean SWD
    :return: a float depicting the average Wasserstein distance in different projections
    """
    set1, set2 = set1.reshape(set1.shape[0], -1), set2.reshape(set2.shape[0], -1)
    dim = set1.shape[-1]
    proj = torch.randn(dim, n_proj, device=set1.device)
    proj = proj / torch.norm(proj, dim=0, keepdim=True)

    proj1, proj2 = set1 @ proj, set2 @ proj

    proj1, proj2 = duplicate_to_match_lengths(proj1.T, proj2.T)
    return torch.abs(torch.sort(proj1, dim=0)[0] - torch.sort(proj2, dim=0)[0]).mean()


def invariant_distribution_error(field1: Union[Callable, torch.Tensor], field2: Union[Callable, torch.Tensor],
                                 inits: torch.Tensor=None, noise: Union[Callable, float]=None, T: float=10,
                                 standardize: bool=False,
                                 distance: str='swd', **distance_kwargs) -> float:
    with torch.no_grad():
        if isinstance(field1, Callable):
            pts = simulate_trajectory(field1, inits, T=T, step=1e-2, noise=noise)
            field1 = pts[-1]
        if isinstance(field2, Callable):
            pts = simulate_trajectory(field2, inits, T=T, step=1e-2, noise=noise)
            field2 = pts[-1]

    if standardize:
        m, std = torch.mean(field1, dim=-1, keepdim=True), torch.std(field1, dim=-1, keepdim=True)
        field1 = (field1-m)/std
        field2 = (field2-m)/std

    if distance.lower() == 'swd': return SWD(field1, field2, **distance_kwargs)
    elif distance.lower() == 'sinkhorn': return sinkhorn(field1, field2, **distance_kwargs)
    elif distance.lower() == 'emd': return EMD(field1, field2, **distance_kwargs)
    elif distance.lower() == 'mmd': return MMD(field1, field2, **distance_kwargs)
    else: raise NotImplementedError


# ================================================= ^^ Eval ^^ =========================================================
