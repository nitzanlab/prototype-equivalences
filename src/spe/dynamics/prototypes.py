import numpy as np
import torch
from torch import nn
from .utils import simulate_trajectory


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
        init = self.project_onto_invariant(torch.randn(N, dim))
        init = init + torch.randn_like(init)*.1
        return self.rand_on_traj(init, T=T)

    def get_invariant(self, N: int, dim: int, **kwargs):
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

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the potential component of the dynamics determined by the prototype
        :param x: the positions x, as a torch tensor with shape [N, dim]
        :return: the scalar potential of the prototype in the same positions, a torch tensor with shape [N,]
        """
        return torch.zeros_like(x[:, 0])  # default to a constant potential


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
    
    def potential(self, x):
        xx = x[..., 0]
        yy = x[..., 1]
        other = x[..., 2:]
        r_sq = xx*xx + yy*yy
        return .5*r_sq*(.5*r_sq-self.a) + .5*torch.exp(self.decay)*torch.sum(other**2, dim=-1)


class Attractor(Prototype):

    def __init__(self, decay: float=.5, optimize: bool=False):
        super().__init__()
        dec = np.log(decay+1e-6)
        if optimize:
            self.decay = nn.Parameter(torch.tensor([dec]).float())
        else:
            self.register_buffer('decay', torch.tensor([dec]).float())
        self.optimize = optimize

    def forward(self, x: torch.Tensor):
        decay = torch.exp(self.decay)
        return - decay*x

    def get_invariant(self, N: int, dim: int):
        return torch.zeros(1, dim, device=self.a.device)

    def project_onto_invariant(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def potential(self, x):
        return .5*torch.exp(self.decay)*torch.sum(x*x, dim=-1)


class HeteroclinicFlip(Prototype):

    def __init__(self, f: float=0., K: float=4., decay: float=.5, optimize: bool=False):
        super().__init__()
        if optimize:
            self.f = nn.Parameter(torch.tensor([f]))
            self.K = nn.Parameter(torch.tensor([K]))
            self.decay = nn.Parameter(torch.tensor([np.log(decay)]).float())
        else:
            self.register_buffer('f', torch.tensor([f]))
            self.register_buffer('K', torch.tensor([K]))
            self.register_buffer('decay', torch.tensor([np.log(decay)]).float())
        
        self.source = torch.tensor([0, 1])
        self.sinks = None
        self.optimize = optimize

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        nonosc = x[..., 2:]  # non-oscillatory dimensions
        fx = x[..., 0]
        fy = x[..., 1]
        
        xdot = 4*fx*fx*fx + 4*fx*fy + self.f
        ydot = 4*fy*fy*fy - 3*fy*fy + 2*fx*fx - 2*fy + self.K
        decay = torch.exp(self.decay)
        return torch.cat([- xdot[..., None], - ydot[..., None], - decay * nonosc], dim=-1).reshape(shape)

    @torch.no_grad()
    def get_invariant(self, N: int, dim: int):
        opt = self.optimize
        if self.sinks is None:
            opt = True
            self.sinks = -torch.ones(2, 2)
            self.sinks[1, 0] = 1
        if opt:
            self.sinks = self.trajectories(self.sinks, T=3)[-1]

        invariant = torch.stack([self.source, self.sinks[0], self.sinks[1]])
        return torch.cat([invariant, torch.zeros(3, dim-2)], dim=1)

    def project_onto_invariant(self, x: torch.Tensor, tau: float=.01) -> torch.Tensor:
        inv = self.get_invariant(N=3, dim=x.shape[1])  # (3, dim)
        D = torch.sum(x*x, dim=-1)[:, None] - 2*x@inv.T + torch.sum(inv*inv, dim=-1)[None]  # (N, 3)
        w = torch.softmax(-D / tau, dim=1)  # (N, 3)
        return w @ inv  # (N, dim)
    
    def simulate(self, N: int, dim: int, T: float=1.) -> torch.Tensor:
        init = self.project_onto_invariant(torch.randn(N, dim))
        init = init + torch.randn_like(init)*.1
        return self.rand_on_traj(init, T=T)
    
    def potential(self, x):
        xx = x[..., 0]
        yy = x[..., 1]
        other = x[..., 2:]
        return xx**4 + yy**4 - yy**3 + 2*yy*xx**2 - yy**2 + self.f*xx + self.K*yy  + .5*torch.exp(self.decay)*torch.sum(other**2, dim=-1)


class DualCusp(Prototype):

    def __init__(self, f: float=0., K: float=-.15, decay: float=.5, optimize: bool=False):
        super().__init__()
        if optimize:
            self.f = nn.Parameter(torch.tensor([f]))
            self.K = nn.Parameter(torch.tensor([K]))
            self.decay = nn.Parameter(torch.tensor([np.log(decay)]).float())
        else:
            self.register_buffer('f', torch.tensor([f]))
            self.register_buffer('K', torch.tensor([K]))
            self.register_buffer('decay', torch.tensor([np.log(decay)]).float())

        self.source = torch.tensor([0, 1])
        self.sinks = None
        self.optimize = optimize

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        nonosc = x[..., 2:]  # non-oscillatory dimensions
        fx = x[..., 0]
        fy = x[..., 1]
        
        xdot = 4*fx*fx*fx + 8*fx*fy + self.f
        ydot = 4*fy*fy*fy - 3*fy*fy + 4*fx*fx + 2*fy + self.K
        decay = torch.exp(self.decay)
        return torch.cat([- xdot[..., None], - ydot[..., None], - decay * nonosc], dim=-1).reshape(shape)

    @torch.no_grad()
    def get_invariant(self, N: int, dim: int):
        opt = self.optimize
        if self.sinks is None:
            opt = True
            self.sinks = -torch.ones(2, 2)
            self.sinks[1, 0] = 1
        if opt:
            self.sinks = self.trajectories(self.sinks, T=3)[-1]

        invariant = torch.stack([self.source, self.sinks[0], self.sinks[1]])
        return torch.cat([invariant, torch.zeros(3, dim-2)], dim=1)

    def project_onto_invariant(self, x: torch.Tensor, tau: float=.01) -> torch.Tensor:
        inv = self.get_invariant(N=3, dim=x.shape[1])  # (3, dim)
        D = torch.sum(x*x, dim=-1)[:, None] - 2*x@inv.T + torch.sum(inv*inv, dim=-1)[None]  # (N, 3)
        w = torch.softmax(-D / tau, dim=1)  # (N, 3)
        return w @ inv  # (N, dim)

    def simulate(self, N: int, dim: int, T: float=1.) -> torch.Tensor:
        init = self.project_onto_invariant(torch.randn(N, dim))
        init = init + torch.randn_like(init)*.1
        return self.rand_on_traj(init, T=T)

    def potential(self, x):
        xx = x[..., 0]
        yy = x[..., 1]
        other = x[..., 2:]
        return xx**4 + yy**4 - yy**3 + 4*yy*xx**2 + yy**2 + self.f*xx + self.K*yy + .5*torch.exp(self.decay)*torch.sum(other**2, dim=-1)


class Bifurcation(Prototype):

    def __init__(self, f: float=0., K: float=0.35, decay: float=.5, optimize: bool=False):
        super().__init__()
        if optimize:
            self.f = nn.Parameter(torch.tensor([f]))
            self.K = nn.Parameter(torch.tensor([K]))
            self.decay = nn.Parameter(torch.tensor([np.log(decay)]).float())
        else:
            self.register_buffer('f', torch.tensor([f]))
            self.register_buffer('K', torch.tensor([K]))
            self.register_buffer('decay', torch.tensor([np.log(decay)]).float())

        self.source = torch.tensor([0, 1])
        self.sinks = None
        self.optimize = optimize

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        nonosc = x[..., 2:]  # non-oscillatory dimensions
        fx = x[..., 0]
        fy = x[..., 1]
        
        xdot = 4*fx*fx*fx + 4*fx*fy + self.f
        ydot = 4*fy*fy*fy - 3*fy*fy + 2*fx*fx + 2*fy + self.K
        decay = torch.exp(self.decay)
        return torch.cat([- xdot[..., None], - ydot[..., None], - decay * nonosc], dim=-1).reshape(shape)

    @torch.no_grad()
    def get_invariant(self, N: int, dim: int):
        opt = self.optimize
        if self.sinks is None:
            opt = True
            self.sinks = -torch.ones(2, 2)
            self.sinks[1, 0] = 0
        if opt:
            self.sinks = self.trajectories(self.sinks, T=3)[-1]

        invariant = torch.stack([self.source, self.sinks[0], self.sinks[1]])
        return torch.cat([invariant, torch.zeros(3, dim-2)], dim=1)

    def project_onto_invariant(self, x: torch.Tensor, tau: float=.01) -> torch.Tensor:
        inv = self.get_invariant(N=3, dim=x.shape[1])  # (3, dim)
        D = torch.sum(x*x, dim=-1)[:, None] - 2*x@inv.T + torch.sum(inv*inv, dim=-1)[None]  # (N, 3)
        w = torch.softmax(-D / tau, dim=1)  # (N, 3)
        return w @ inv  # (N, dim)

    def simulate(self, N: int, dim: int, T: float=1.) -> torch.Tensor:
        init = self.project_onto_invariant(torch.randn(N, dim))
        init = init + torch.randn_like(init)*.1
        return self.rand_on_traj(init, T=T)

    def potential(self, x):
        xx = x[..., 0]
        yy = x[..., 1]
        other = x[..., 2:]
        return xx**4 + yy**4 - yy**3 + 2*yy*xx**2 + yy**2 + self.f*xx + self.K*yy + .5*torch.exp(self.decay)*torch.sum(other**2, dim=-1)


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


class MultiBasinPrototype(Prototype):

    def __init__(self, pos: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, decay: float=.5, optimize: bool=False):
        """
        Initializes a prototype that is made up of multiple different attractors
        :param decay: the speed of the decay of the non-oscillating dimensions to zero
        :param optimize: whether or not to optimize the parameters of the model
        """
        super().__init__()
        self.decay = decay
        if optimize:
            self.pos = nn.Parameter(pos)
            self.a = nn.Parameter(a)
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer('pos', pos)
            self.register_buffer('a', a)
            self.register_buffer('omega', omega)
        self.optimize = optimize

    def forward(self, x: torch.Tensor):
        ins = x[..., :2]
        outs = x[..., 2:]

        d = torch.sum(ins * ins, dim=-1)[:, None] - 2 * ins @ self.pos.T + torch.sum(self.pos * self.pos, dim=-1)[None]
        p = -.5 * d
        p = torch.exp(p - torch.logsumexp(p, dim=-1, keepdim=True))

        basin = self.pos[None] - ins[:, None]

        theta = torch.atan2(basin[..., 1], basin[..., 0])
        r = torch.sqrt(torch.sum(basin[..., -2:] ** 2, dim=-1))

        rdot = self.a[None] - r * r
        xdot = torch.cos(theta) * rdot - r * torch.sin(theta) * self.omega[None]
        ydot = torch.sin(theta) * rdot + r * torch.cos(theta) * self.omega[None]

        return torch.cat([
            torch.sum(p * xdot, dim=1)[:, None],
            torch.sum(p * ydot, dim=1)[:, None],
            -self.decay * outs
        ], dim=1)

    def get_invariant(self, N: int, dim: int, k: int=0):
        nonosc = torch.zeros(N, dim - 2, device=self.a.device)
        theta = torch.linspace(0, 2 * np.pi, N, device=self.a.device)[:, None]
        R = torch.sqrt(torch.clamp(self.a[k], .001))[None]
        return torch.cat([
            R * torch.cos(theta) + self.pos[k, 0],
            R * torch.sin(theta) + self.pos[k, 1],
            nonosc
        ], dim=-1).reshape(-1, dim)


class GoldbeterPrototype(Prototype):
    """
    Goldbeter model for cell cycle oscillations.

    Variables:
    C: cyclin concentration
    M: active (dephosphorylated) Cdk-cyclin complex
    X: inactive (phosphorylated) Cdk-cyclin complex
    """

    def __init__(self,
                 vi=0.025, vd=0.25, kd=0.02,
                 v1=3.0, v2=1.5, v3=1.0, v4=0.5,
                 k1=0.005, k2=0.005, k3=0.005, k4=0.005,
                 Kd=0.02, Kc=0.5, Km=0.1, n=4.0,
                 optimize=False):
        super().__init__()

        self.optimize = optimize
        self.proj_dim = 3  # C, M, X

        # Parameters as log-transformed where appropriate for positivity
        params = {
            'vi': np.log(vi), 'vd': np.log(vd), 'kd': np.log(kd),
            'v1': np.log(v1), 'v2': np.log(v2), 'v3': np.log(v3), 'v4': np.log(v4),
            'k1': np.log(k1), 'k2': np.log(k2), 'k3': np.log(k3), 'k4': np.log(k4),
            'Kd': np.log(Kd), 'Kc': np.log(Kc), 'Km': np.log(Km),
            'n': n  # Hill coefficient (no log transform)
        }

        if optimize:
            for name, val in params.items():
                setattr(self, name, nn.Parameter(torch.tensor([val]).float()))
        else:
            for name, val in params.items():
                self.register_buffer(name, torch.tensor([val]).float())

    def forward(self, x: torch.Tensor):
        """
        Compute dx/dt for the Goldbeter model.

        Args:
            x: tensor of shape [..., 3] or [..., >3] where first 3 dims are [C, M, X]

        Returns:
            dx/dt: tensor of same shape as x
        """
        shape = x.shape
        x = x.reshape(x.shape[0], -1)

        # Extract dimensions
        goldbeter_vars = x[..., :3]
        extra_dims = x[..., 3:] if x.shape[-1] > 3 else None

        C = goldbeter_vars[..., 0]
        M = goldbeter_vars[..., 1]
        X = goldbeter_vars[..., 2]

        # Exponentiate log-transformed parameters
        vi = torch.exp(self.vi)
        vd = torch.exp(self.vd)
        kd = torch.exp(self.kd)
        v1 = torch.exp(self.v1)
        v2 = torch.exp(self.v2)
        v3 = torch.exp(self.v3)
        v4 = torch.exp(self.v4)
        k1 = torch.exp(self.k1)
        k2 = torch.exp(self.k2)
        k3 = torch.exp(self.k3)
        k4 = torch.exp(self.k4)
        Kd = torch.exp(self.Kd)
        Km = torch.exp(self.Km)
        n = self.n

        # Michaelis-Menten kinetics
        inactive_cdk = C - M - X
        V1 = v1 * inactive_cdk / (k1 + inactive_cdk + 1e-8)
        V2 = v2 * M / (k2 + M + 1e-8)
        V3 = v3 * X * (M ** n) / ((k3 ** n + M ** n) * (Km + X) + 1e-8)
        V4 = v4 * M / (k4 + M + 1e-8)

        # ODEs
        dC_dt = vi - vd * C / (Kd + C + 1e-8) - kd * C
        dM_dt = V1 - V2 + V4 - V3
        dX_dt = V2 - V3 - V4

        goldbeter_dot = torch.stack([dC_dt, dM_dt, dX_dt], dim=-1)

        # Handle extra dimensions (if any)
        if extra_dims is not None:
            # Could add decay or other dynamics for extra dims
            extra_dot = torch.zeros_like(extra_dims)
            result = torch.cat([goldbeter_dot, extra_dot], dim=-1)
        else:
            result = goldbeter_dot

        return result.reshape(shape)