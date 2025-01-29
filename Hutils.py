import numpy as np
import torch
from scipy.optimize import minimize
from scipy import interpolate
from NFDiffeo import Diffeo


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


def simulate_trajectory(grad, init: torch.Tensor, T: float, step: float=5e-2,
                        euler: bool=False):
    """
    Discretization of a trajectory starting at some initial points using Runga-Kutta
    :param grad: the gradient function; inputs are [N, dim], and outputs are the same
    :param init: initial positions, a tensor with shape [N, dim]
    :param T: the amount of time to integrate the dynamics
    :param step: step size to use for the euler discretization
    :param euler: a boolean dictating whether an Euler (first-order) discretization should be used, other-wise a 4th
                  order method is used
    :return: a torch tensor with shape [round(T/step), N, dim]
    """
    flows = [init]
    pnts = init.clone()
    for t in np.arange(0, T, step):
        if euler:
            pnts = pnts + step*grad(t, pnts)
        else:
            k1 = grad(t, pnts)
            k2 = grad(t+step/2, pnts+.5*step*k1)
            k3 = grad(t+step/2, pnts+.5*step*k2)
            k4 = grad(t+step, pnts+step*k3)

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


def project_onto_attractor(a: float, x: torch.Tensor) -> torch.Tensor:
    """
    Projects the observed points in x onto the attractor of a simple oscillator whose damping coefficient is a
    :param a: the damping coefficient of the simple oscillator prototype
    :param x: points to project onto the simple oscillator, a torch tensor of shape [N, dim]
    :return: the points, projected onto the invariant set of the oscillator defined by the damping coefficient a
    """
    rad = np.sqrt(a) if a > 0 else 0
    osc = rad*x[:, :2]/torch.norm(x[:, :2], dim=1, keepdim=True)
    nonosc = torch.zeros_like(x[:, 2:])
    return torch.cat([osc, nonosc], dim=1)


def cycle_error(H: Diffeo, x: torch.Tensor, a: float) -> float:
    """
    Calculates the cycle error of the observed points x
    :param H: the diffeomorphism to the simple oscillator prototype whose damping coefficient is the input "a"
    :param x: the trajectory, a torch tensor with shape [N, dim]
    :param a: the damping coefficient of the SO prototype
    :return: the cycle error of x onto the invariant set defined by the diffeomorphism H
    """
    y = H.reverse(project_onto_attractor(a, H(x)))
    err = (y-x)**2
    norm = torch.var(x)
    return torch.mean(err/norm).item()


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
