import numpy as np
import torch
from torch import nn
from .utils import simulate_trajectory, cartesian_to_polar, polar_derivative_to_cartesian_derivative


class Prototype(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_dim = 2
        self.optimize = True

    def forward(self, x: torch.Tensor): raise NotImplementedError

    @torch.no_grad()
    def trajectories(self, init: torch.Tensor, T: float=10., step: float=5e-2, euler: bool=False):
        return simulate_trajectory(self.forward, init, T=T, step=step, euler=euler)

    @torch.no_grad()
    def rand_on_traj(self, x: torch.Tensor, T: float, step: float=1e-2, min_time: float=0) -> torch.Tensor:
        strt = int(min_time/step)
        traj = self.trajectories(x, T=T, step=step)[strt:]
        pts = []
        for i in range(x.shape[0]):
            pts.append(traj[np.random.choice(traj.shape[0], 1)[0], i])
        return torch.stack(pts)

    def simulate(self, N: int, dim: int, T: float=5) -> torch.Tensor:
        """
        Simulates positions of N particles on the prototype
        :param N: number of particles to simulate
        :param dim: the dimension of the particles
        :param T: maximal time to iterate the simulation of the particles
        :return: a torch tensor with shape [N, dim] of the simulated particles
        """
        init = torch.randn(N, dim, device=self.device)*.01
        return self.rand_on_traj(init, T=T)

    def get_invariant(self, N: int, dim: int):
        """
        Get points from the invariant set of the prototype, useful for plotting
        :param N: number of points on the attractor to return
        :param dim: the dimension of the data
        :return: around N points on the attractor, a torch tensor with shape [~N, dim]
        """
        raise NotImplementedError

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the points x onto the invariant set of the prototype. In essence, this finds the closest point in the
        invariant set for each of the points x and allows for better evaluation
        :param x: the positions x, as a torch tensor with shape [N, dim]
        :return: the points after projecting them onto the invariant set, a torch tensor with shape [N, dim]
        """
        raise NotImplementedError

    def prior(self) -> torch.Tensor:
        """
        Calculate and return the negative log-prior on all model parameters
        :return: a torch tensor of the negative log-prior on all model parameters
        """
        if not self.optimize: return torch.tensor([0])
        else:
            prior = 0.
            for param in self.parameters():
                prior = prior + torch.sum(param * param)
            return .5 * prior

class SOPrototype(Prototype):

    def __init__(self, a: float=.25, omega: float=-.5, decay: float=.5, optimize: bool=False):
        super().__init__()
        if optimize:
            self.a = nn.Parameter(torch.tensor([a]))
            self.omega = nn.Parameter(torch.tensor([omega]))
            self.decay = nn.Parameter(torch.tensor([np.log(decay)]).float())
        else:
            self.register_buffer('a', torch.tensor([a]))
            self.register_buffer('omega', torch.tensor([omega]))
            self.register_buffer('decay', torch.tensor([np.log(decay)]).float())
        self.optimize = optimize
        self.proj_dim = 2

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        nonosc = x[..., 2:]  # non-oscillatory dimensions
        osc = x[..., :2]
        r = torch.sqrt(torch.sum(osc ** 2, dim=-1))
        theta = torch.atan2(osc[..., 1], osc[..., 0])

        rdot = r * (self.a - r * r)

        xdot = torch.cos(theta) * rdot - r * torch.sin(theta) * self.omega
        ydot = torch.sin(theta) * rdot + r * torch.cos(theta) * self.omega
        decay = torch.exp(self.decay)
        return torch.cat([xdot[..., None], ydot[..., None], - decay * nonosc], dim=-1).reshape(shape)

    def get_invariant(self, N: int, dim: int):
        if self.a > 0:
            nonosc = torch.zeros(N, dim-2, device=self.a.device)
            theta = torch.linspace(0, 2*np.pi, N, device=self.a.device)[:, None]
            R = torch.sqrt(self.a)
            return torch.cat([
                R*torch.cos(theta), R*torch.sin(theta), nonosc
            ], dim=-1)
        else:
            return torch.zeros(1, dim, device=self.a.device)

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        nonosc = x[..., 2:]  # non-oscillatory dimensions
        osc = x[..., :2]

        theta = torch.atan2(osc[..., 1], osc[..., 0])
        r = torch.ones_like(theta)*torch.sqrt(torch.clamp(self.a, 0))

        return torch.cat([
            (r * torch.cos(theta))[:, None], (r * torch.sin(theta))[:, None], nonosc*0
        ], dim=-1).reshape(shape)

    def prior(self) -> torch.Tensor:
        """
        Calculate and return the negative log-prior on all model parameters
        :return: a torch tensor of the negative log-prior on all model parameters
        """
        if not self.optimize: return torch.tensor([0])
        return .5*(
            torch.sum(self.a*self.a)/(.25**2) +  # gaussian prior with variance 0.25
            torch.sum(self.omega*self.omega) +   # gaussian prior around 0 with variance 1
            torch.sum(((self.decay - np.log(.25))/.25)**2) + .5*self.decay  # log-normal prior around 0.25
        )


class CyclePrototype(Prototype):

    def __init__(self, a: float=.25, omega: float=.0, decay: float=.5, optimize: bool=False):
        super().__init__()
        if optimize:
            self.a = nn.Parameter(torch.tensor([np.log(a)]).float())
            self.omega = nn.Parameter(torch.tensor([omega]).float())
            self.decay = nn.Parameter(torch.tensor([np.log(decay)]).float())
        else:
            self.register_buffer('a', torch.tensor([np.log(a)]).float())
            self.register_buffer('omega', torch.tensor([omega]).float())
            self.register_buffer('decay', torch.tensor([np.log(decay)]).float())
        self.optimize = optimize
        self.proj_dim = 2

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        nonosc = x[..., 2:]  # non-oscillatory dimensions
        osc = x[..., :2]
        r = torch.sqrt(torch.sum(osc ** 2, dim=-1))
        theta = torch.atan2(osc[..., 1], osc[..., 0])

        rdot = torch.exp(self.a) - r*r

        xdot = torch.cos(theta) * rdot - r * torch.sin(theta) * self.omega
        ydot = torch.sin(theta) * rdot + r * torch.cos(theta) * self.omega
        return torch.cat([xdot[..., None], ydot[..., None], - torch.exp(self.decay) * nonosc], dim=-1).reshape(shape)

    @torch.no_grad()
    def get_invariant(self, N: int, dim: int):
        nonosc = torch.zeros(N, dim-2, device=self.a.device)
        theta = torch.linspace(0, 2*np.pi, N, device=self.a.device)[:, None]
        R = torch.sqrt(torch.exp(self.a))
        return torch.cat([
            R*torch.cos(theta), R*torch.sin(theta), nonosc
        ], dim=-1)

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        nonosc = x[..., 2:]  # non-oscillatory dimensions
        osc = x[..., :2]

        theta = torch.atan2(osc[..., 1], osc[..., 0])
        r = torch.ones_like(theta)*torch.sqrt(torch.exp(self.a))

        return torch.cat([
            (r * torch.cos(theta))[:, None], (r * torch.sin(theta))[:, None], nonosc*0
        ], dim=-1).reshape(shape)

    def prior(self) -> torch.Tensor:
        omega_prior = .5*(self.omega*self.omega)  # assumes unit variance on omega
        a_prior = .5*((self.a - np.log(.25))/.5)**2
        return omega_prior + a_prior


class CylPrototype(Prototype):

    def __init__(self, a: float=.25, omega: float=-.5, cyl_dims: int=1, optimize: bool=True, decay: float=.5):
        super().__init__()
        if optimize:
            self.a = nn.Parameter(torch.tensor([a]))
            self.omega = nn.Parameter(torch.tensor([omega]))
        else:
            self.register_buffer('a', torch.tensor([a]))
            self.register_buffer('omega', torch.tensor([omega]))
        self.decay = decay
        self.cyl_dims = cyl_dims
        self.proj_dim = 2 + cyl_dims
        self.optimize = optimize

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        osc = x[..., :2]
        cyl = x[..., 2:2+self.cyl_dims]
        if x.shape[-1] > 2+self.cyl_dims:
            nonosc = x[..., 2+self.cyl_dims:]  # non-oscillatory dimensions
        else: nonosc = []

        r = torch.sqrt(torch.sum(osc ** 2, dim=-1))
        theta = torch.atan2(osc[..., 1], osc[..., 0])

        rdot = r * (self.a - r * r)

        xdot = torch.cos(theta) * rdot - r * torch.sin(theta) * self.omega
        ydot = torch.sin(theta) * rdot + r * torch.cos(theta) * self.omega
        return torch.cat([xdot[..., None], ydot[..., None], 0*cyl, -.5*nonosc], dim=-1).reshape(shape)

    def get_invariant(self, N: int, dim: int):
        # TODO actually get a cylinder, not just a cycle
        if self.a > 0:
            nonosc = torch.zeros(N, dim - 2, device=self.a.device)
            theta = torch.linspace(0, 2 * np.pi, N, device=self.a.device)[:, None]
            R = torch.sqrt(self.a)
            return torch.cat([
                R * torch.cos(theta), R * torch.sin(theta), nonosc
            ], dim=-1)
        else:
            return torch.zeros(1, dim, device=self.a.device)

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        osc = x[..., :2]
        cyl = x[..., 2:2+self.cyl_dims]
        if x.shape[-1] > 2+self.cyl_dims:
            nonosc = x[..., 2+self.cyl_dims:]*0  # non-oscillatory dimensions
        else: nonosc = []

        theta = torch.atan2(osc[..., 1], osc[..., 0])
        r = torch.ones_like(theta) * torch.sqrt(torch.clamp(self.a, 0))

        return torch.cat([
            (r * torch.cos(theta))[:, None], (r * torch.sin(theta))[:, None], cyl, nonosc
        ], dim=-1).reshape(shape)


class MultiStablePrototype(Prototype):

    def __init__(self, pos: torch.Tensor, sigma: float=.1, decay: float=.5, optimize: bool=False):
        """
        Initializes a prototype that is made up of multiple different attractors
        :param decay: the speed of the decay of the non-oscillating dimensions to zero
        :param optimize: whether or not to optimize the parameters of the model
        """
        super().__init__()
        self.decay = decay
        self. sigma = sigma
        if optimize:
            self.pos = nn.Parameter(pos)
        else:
            self.register_buffer('pos', pos)
        self.optimize = optimize

    def forward(self, x: torch.Tensor):
        ins = x[..., :self.pos.shape[1]]
        outs = x[..., self.pos.shape[1]:]

        d = torch.sum(ins * ins, dim=-1)[:, None] - 2 * ins @ self.pos.T + torch.sum(self.pos * self.pos, dim=-1)[None]
        p = -.5 * d / self.sigma
        p = torch.exp(p - torch.logsumexp(p, dim=-1, keepdim=True))

        node = self.pos[None] - ins[:, None]
        node = torch.sign(node)*torch.clamp(torch.abs(node), 0, 2)  # limit size of vectors

        return torch.cat([
            torch.sum(p[..., None] * node, dim=1),
            -self.decay * outs
        ], dim=1)

    def get_invariant(self, N: int, dim: int):
        return self.pos.detach()

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        ins = x[..., :self.pos.shape[1]]
        outs = x[..., self.pos.shape[1]:]

        d = torch.sum(ins * ins, dim=-1)[:, None] - 2 * ins @ self.pos.T + torch.sum(self.pos * self.pos, dim=-1)[None]
        inds = torch.argmin(d, dim=-1)

        return torch.cat([
            self.pos[inds],
            0*outs
        ], dim=1)
