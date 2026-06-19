import numpy as np
import torch
from .utils import get_oscillator, simulate_trajectory,\
    cartesian_to_polar, polar_derivative_to_cartesian_derivative
from ..models.NFDiffeo import FFCoupling, NFCompose


class PhaseSpace:
    """
    Definition of a generic dynamical system class whose main purpose is to return points in phase space
    """

    def __init__(self, parameters: dict):
        for p in parameters:
            if parameters[p] is None:
                parameters[p] = \
                    (self.param_ranges[p][1]-self.param_ranges[p][0])*np.random.rand() + self.param_ranges[p][0]
        self.parameters = parameters
        self.parameters['bif'] = self.dist_from_bifur()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the velocities of the system at the positions given by z
        :param z: a torch tensor with shape [N, d] where N is the number of points and d the dimension
        :return: the velocities of the system, a torch tensor with shape [N, d]
        """
        raise NotImplementedError

    def __call__(self, z:torch.Tensor) -> torch.Tensor: return self.forward(z)

    def dist_from_bifur(self) -> float:
        """
        Checks how far the system is from the Hopf bifurcation
        :return: if smaller than 0, then the dynamics are a node attractor, otherwise they are cyclic
        """
        raise NotImplementedError

    @staticmethod
    def random_cycle_params() -> dict:
        """
        :return: a dictionary of parameters to the system for oscillatory behavior
        """
        raise NotImplementedError

    def trajectories(self, x: torch.Tensor, T: float, step: float=1e-2, euler: bool=False) -> torch.Tensor:
        # x = torch.clamp(x, self.position_lims[0], self.position_lims[1])
        return simulate_trajectory(self.forward, x, T=T, step=step, euler=euler)

    def rand_on_traj(self, x: torch.Tensor, T: float, step: float=1e-2, euler: bool=False,
                     min_time: float=0) -> torch.Tensor:
        strt = int(min_time/step)
        traj = self.trajectories(x, T=T, step=step, euler=euler)[strt:]
        pts = []
        for i in range(x.shape[0]):
            pts.append(traj[np.random.choice(traj.shape[0], 1)[0], i])
        return torch.stack(pts)

    def random_x(self, N: int, dim: int=2):
        """
        :param N: number of random positions to sample from the system's range
        :return: a torch tensor with shape [N, dim]
        """
        return torch.rand(N, dim)*(self.position_lims[1] - self.position_lims[0]) + self.position_lims[0]


# ================================================= ^^ Definitions ^^ =================================================
# ================================================= vv Systems vv =====================================================


class SO(PhaseSpace):
    """
    Simulates a multi-dimensional version of a oscillator
    """

    param_ranges = {
        'a': [-.5, .5],
        'omega': [-1, 1],
        'decay': [1/6, 1]
    }

    param_display = {
        'a': r'$a$',
        'omega': r'$\omega$'
    }

    position_lims = [-2, 2]

    def __init__(self, a: float=None, omega: float=None, decay: float=None):
        super().__init__({'a': a, 'omega': omega, 'decay': decay})
        self.osc = get_oscillator(a=self.parameters['a'],
                                  omega=self.parameters['omega'],
                                  decay=self.parameters['decay'])

    @staticmethod
    def random_cycle_params() -> dict:
        """
        :return: a dictionary of parameters to the system for oscillatory behavior
        """
        omega = np.random.rand()*(SO.param_ranges['omega'][1]-SO.param_ranges['omega'][0])+SO.param_ranges['omega'][0]
        a = np.random.rand()*SO.param_ranges['a'][1]
        return {'a': a, 'omega': omega}

    @torch.no_grad()
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        dx = self.osc(x)
        return dx.reshape(shape)

    def __call__(self, x): return self.forward(x)

    @torch.no_grad()
    def get_cycle(self):
        rad = np.sqrt(max(self.parameters['a'], 0))
        cycle_pts = np.stack([rad * np.cos(np.linspace(0, 2 * np.pi, 250)),
                              rad * np.sin(np.linspace(0, 2 * np.pi, 250))])
        return torch.from_numpy(cycle_pts).float().T

    def dist_from_bifur(self) -> float:
        """
        Checks how far the system is from the Hopf bifurcation
        :return: if smaller than 0, then the dynamics are a node attractor, otherwise they are cyclic
        """
        return self.parameters['a']

    def __str__(self): return 'SO'


class Repressilator(PhaseSpace):
    """
    Elowitz and Leibler repressilator:
        dm_i = -m_i + a/(1 + p_j^n) + a0
        dp_i = -b(p_i - m_i)
    where:
    - m_i (/p_i) denote mRNA(/protein) concentration of gene i
    - i = lacI, tetR, cI and j = cI, lacI, tetR
    - a0 - leaky mRNA expression
    - a - transcription rate without repression
    - b - ratio of protein and mRNA degradation rates
    - n - Hill coefficient

    All of the following code is copied and adapted from twa
    """
    param_ranges = {
        'alpha': [1e-4, 30],
        'beta': [1e-4, 10],
    }

    param_display = {
        'alpha': r'$\alpha$',
        'beta':  r'$\beta$',
    }

    position_lims = [1e-4, 20]

    def __init__(self, alpha: float=None, beta: float=None, n: float=2, alpha0: float=.2):
        """
        :param alpha: transcription rate without repression, 0 < alpha <= 30
        :param beta: ration of protein and mRNA degradation, 0 < beta <= 10
        :param n: Hill coefficient, greater than 1
        :param alpha0: leaky mRNA expression, greater than 0
        """
        self.n = n
        self.a0 = alpha0
        super().__init__({'alpha': alpha, 'beta': beta})

    def __str__(self): return 'Repressilator'

    @staticmethod
    def random_cycle_params() -> dict:
        """
        :return: a dictionary of parameters to the system for oscillatory behavior
        """
        beta_range = Repressilator.param_ranges['beta']
        alpha_range = Repressilator.param_ranges['alpha']
        beta = np.random.rand()*(beta_range[1] - beta_range[0]) + beta_range[0]
        alpha = None
        while alpha is None:
            a = np.random.rand()*(alpha_range[1] - alpha_range[0]) + alpha_range[0]
            if Repressilator._bifur_dist(alpha=a, beta=beta) > 0: alpha = a
        return {'alpha': alpha, 'beta': beta}

    @staticmethod
    def _bifur_dist(alpha, beta, a0=.2, n=2.):
        from scipy.optimize import fsolve
        a, b = alpha, beta

        def equation(x, a, a0, n):
            return a / (1 + x ** n) + a0 - x

        # Initial guess for x
        p0 = 1.0

        # Solve the equation using fsolve
        p = fsolve(equation, p0, args=(a, a0, n))

        xi = - (a * n * p ** (n - 1)) / (1 + p ** n) ** 2

        # check condition for stability
        return -(((b + 1) ** 2 / b) - 3 * xi ** 2 / (4 + 2 * xi))[0]

    def dist_from_bifur(self):
        """
        Checks how far the system is from the Hopf bifurcation
        :return: if smaller than 0, then the dynamics are a node attractor, otherwise they are cyclic
        """
        return self._bifur_dist(alpha=self.parameters['alpha'], beta=self.parameters['beta'], a0=self.a0, n=self.n)

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the velocities of the specific repressilator system at the positions given by z
        :param z: a torch tensor with shape [N, 6] where N is the number of points
        :return: the velocities of the repressilator, a torch tensor with shape [N, 6]
        """
        mLacI = z[..., 0]
        pLacI = z[..., 1]
        mTetR = z[..., 2]
        pTetR = z[..., 3]
        mcI = z[..., 4]
        pcI = z[..., 5]

        a, b = self.parameters['alpha'], self.parameters['beta']
        a0, n = self.a0, self.n

        mLacI_dot = -mLacI + a / (1 + pcI ** n) + a0
        pLacI_dot = -b * (pLacI - mLacI)

        mTetR_dot = -mTetR + a / (1 + pLacI ** n) + a0
        pTetR_dot = -b * (pTetR - mTetR)

        mcI_dot = -mcI + a / (1 + pTetR ** n) + a0
        pcI_dot = -b * (pcI - mcI)

        zdot = torch.cat([mLacI_dot.unsqueeze(-1),
                          pLacI_dot.unsqueeze(-1),
                          mTetR_dot.unsqueeze(-1),
                          pTetR_dot.unsqueeze(-1),
                          mcI_dot.unsqueeze(-1),
                          pcI_dot.unsqueeze(-1)], dim=-1)

        return zdot

    def trajectories(self, x: torch.Tensor, T: float, step: float=1e-2, euler: bool=False) -> torch.Tensor:
        dim = x.shape[-1]

        # for lower dimensions, concatenate random value for all unseen dimensions
        if dim == 2:
            y = torch.rand(x.shape[0], 4, device=x.device)*(self.position_lims[1] - self.position_lims[0]) + self.position_lims[0]
            x = torch.stack([y[:, 0], x[:, 0], y[:, 1], x[:, 1], y[:, 2], y[:, 3]]).T
        elif dim == 3:
            y = torch.rand(x.shape[0], 3, device=x.device) * (self.position_lims[1] - self.position_lims[0]) + self.position_lims[0]
            x = torch.stack([y[:, 0], x[:, 0], y[:, 1], x[:, 1], y[:, 2], x[:, 2]]).T
        elif dim == 4:
            y = torch.rand(x.shape[0], 2, device=x.device) * (self.position_lims[1] - self.position_lims[0]) + self.position_lims[0]
            x = torch.stack([x[:, 0], x[:, 1], y[:, 0], x[:, 2], y[:, 1], x[:, 3]]).T
        elif dim == 5:
            y = torch.rand(x.shape[0], 1, device=x.device) * (self.position_lims[1] - self.position_lims[0]) + self.position_lims[0]
            x = torch.stack([x[:, 0], x[:, 1], x[:, 2], x[:, 3], y[:, 0], x[:, 4]]).T

        x = torch.clamp(x, self.position_lims[0], self.position_lims[1])
        traj = simulate_trajectory(self.forward, x, T=T, step=step, euler=euler)
        if dim == 2: return traj[:, :, [1, 3]]
        elif dim == 3: return traj[:, :, [1, 3, 5]]
        elif dim == 4: return traj[:, :, [0, 1, 3, 5]]
        elif dim == 5: return traj[:, :, [0, 1, 2, 3, 5]]
        else: return traj


class BZreaction(PhaseSpace):
    """
    BZ reaction (undergoing Hopf bifurcation):
        xdot = a - x - 4*x*y / (1 + x^2)
        ydot = b * x * (1 - y / (1 + x^2))
    where:
        - 'a', 'b' depend on empirical rate constants and on concentrations of slow reactants
        - 'a' in [3, 19]
        - 'b' in [2, 6]

    Strogatz, p.256
    Copied, almost verbatim, from twa
    """
    param_ranges = {
        'a': [3, 19],
        'b': [2, 6],
        'decay': [1/6, 1],
    }

    param_display = {
        'a': r'$a$',
        'b': r'$b$',
    }

    position_lims = [0, 10]

    def __init__(self, a: float=None, b: float=None, decay: float=None):
        super().__init__({'a': a, 'b': b, 'decay': decay})

    @staticmethod
    def random_cycle_params() -> dict:
        """
        :return: a dictionary of parameters to the system for oscillatory behavior
        """
        a_range = BZreaction.param_ranges['a']
        b_range = BZreaction.param_ranges['b']
        a = np.random.rand() * (a_range[1] - a_range[0]) + a_range[0]
        max_b = 3*a/5 - 25/a
        b = np.random.rand() * (max_b - b_range[0]) + b_range[0]
        return {'a': a, 'b': b}

    def __str__(self): return 'BZReaction'

    def forward(self, z):
        a, b, decay = self.parameters['a'], self.parameters['b'], self.parameters['decay']
        x = z[..., 0] + 1e-3
        y = z[..., 1] + 1e-3
        nonosc = z[..., 2:]

        xdot = a - x - 4 * x * y / (1 + x ** 2)
        ydot = b * x * (1 - y / (1 + x ** 2))
        zdot = torch.cat([xdot[..., None], ydot[..., None], -decay*nonosc], dim=-1)
        return zdot

    def dist_from_bifur(self):
        a, b = self.parameters['a'], self.parameters['b']
        return 1 if b < 3*a/5 - 25/a else -1


class Selkov(PhaseSpace):
    """
    Selkov oscillator:
        xdot = -x + ay + x^2y
        ydot = b - ay - x^2y

    - 'a' in [.01, .11]
    - 'b' in [.02, 1.2]

    Strogatz, p. 209
    Copied, almost verbatim, from twa
    """
    param_ranges = {
        'a': [.01, .11],
        'b': [.02, 1.2],
        'decay': [1/6, 1]
    }

    param_display = {
        'a': r'$a$',
        'b': r'$b$',
    }

    position_lims = [0, 3]

    def __init__(self, a: float=None, b: float=None, decay: float=None):
        super().__init__({'a': a, 'b': b, 'decay': decay})

    @staticmethod
    def random_cycle_params() -> dict:
        """
        :return: a dictionary of parameters to the system for oscillatory behavior
        """
        a_range = Selkov.param_ranges['a']
        a = np.random.rand() * (a_range[1] - a_range[0]) + a_range[0]
        f_min = np.sqrt(1 / 2 * (1 - 2 * a - np.sqrt(1 - 8 * a)))
        f_max = np.sqrt(1 / 2 * (1 - 2 * a + np.sqrt(1 - 8 * a)))
        b = np.random.rand() * (f_max - f_min) + f_min
        return {'a': a, 'b': b}

    def __str__(self): return 'Selkov'

    def forward(self, z):
        a, b, decay = self.parameters['a'], self.parameters['b'], self.parameters['decay']

        x = z[..., 0]
        y = z[..., 1]
        nonosc = z[..., 2:]

        xdot = -x + a * y + x ** 2 * y
        ydot = b - a * y - x ** 2 * y
        zdot = torch.cat([xdot[..., None], ydot[..., None], -decay*nonosc], dim=-1)
        return zdot

    def dist_from_bifur(self):
        a, b, decay = self.parameters['a'], self.parameters['b'], self.parameters['decay']

        f_plus = np.sqrt(1 / 2 * (1 - 2 * a + np.sqrt(1 - 8 * a)))
        f_minus = np.sqrt(1 / 2 * (1 - 2 * a - np.sqrt(1 - 8 * a)))
        return 1 if f_minus <= b <= f_plus else -1


class SupercriticalHopf(PhaseSpace):
    """
    Supercritical Hopf bifurcation:
        rdot = mu * r - r^3
        thetadot = omega + b*r^2
    where:
    - mu controls stability of fixed point at the origin
    - omega controls frequency of oscillations
    - b controls dependence of frequency on amplitude

    Strogatz, p.250
    Copied from twa
    """

    param_ranges = {
        'mu': [-1, 1],
        'omega': [-1, 1],
        'b': [-1, 1],
        'decay': [1/6, 1]
    }

    param_display = {
        'mu': r'$\mu$',
        'omega': r'$\omega$',
        'b': r'$b$',
    }

    position_lims = [-1, 1]

    def __init__(self, mu: float=None, omega: float=None, b: float=None, decay: float=None):
        super().__init__({'mu': mu, 'omega': omega, 'b': b, 'decay': decay})

    def __str__(self): return 'SupercriticalHopf'

    def forward(self, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]
        mu = self.parameters['mu']
        omega = self.parameters['omega']
        b = self.parameters['b']
        decay = self.parameters['decay']
        r, theta = cartesian_to_polar(x, y)

        rdot = mu * r - r ** 3
        thetadot = omega + b * r ** 2

        xdot, ydot = polar_derivative_to_cartesian_derivative(r, theta, rdot, thetadot)

        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1), -decay*z[..., 2:]], dim=-1)
        return zdot

    def dist_from_bifur(self):
        return self.parameters['mu']


class VanDerPol(PhaseSpace):
    """
    Van der pol oscillator:
        xdot = y
        ydot = mu * (1-x^2) * y - x

    Strogatz, p. 198
    Copied from twa
    """

    param_ranges = {
        'mu': [-1, 1],
        'decay': [1/6, 1]
    }

    param_display = {
        'mu': r'$\mu$',
    }

    position_lims = [-3, 3]

    def __init__(self, mu: float=None, decay: float=None):
        super().__init__({'mu': mu, 'decay': decay})

    @staticmethod
    def random_cycle_params() -> dict:
        """
        :return: a dictionary of parameters to the system for oscillatory behavior
        """
        mu = VanDerPol.param_ranges['mu'][1]*np.random.rand()
        return {'mu': mu}

    def __str__(self): return 'VanDerPol'

    def forward(self, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]

        mu = self.parameters['mu']
        decay = self.parameters['decay']

        xdot = y
        ydot = mu * y - x - x ** 2 * y

        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1), -decay*z[..., 2:]], dim=-1)
        return zdot

    def dist_from_bifur(self):
        return self.parameters['mu']


class Lienard(PhaseSpace):
    """
    A general class of oscillators where:
        xdot = y
        ydot = -g(x) -f(x)*y

    And:
    - $f,g$ of polynomial basis (automatically then continuous and differentiable for all x)
    - $g$ is an odd function ($g(-x)=-g(x)$)
    - $g(x) > 0$ for $x>0$
    - cummulative function of f, $F(x)=\int_0^xf(u)du$, and is negative for $0<x<a, F(x)=0$, $x>a$ is positive and nondecreasing

    Copied from twa
    """

    def __init__(self, params, flip=False):
        super().__init__(params)

        self.flip = flip

    def forward(self, z):
        x = z[..., 0]
        y = z[..., 1]
        decay = self.parameters['decay']

        xdot = y
        ydot = -self.g(x) - self.f(x) * y
        if self.flip:
            xdot = -self.g(y) - self.f(y) * x
            ydot = x
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1), -decay*z[..., 2:]], dim=-1)

        return zdot

    def get_info(self):
        return super().get_info(exclude=super().exclude + ['f', 'g'])


class LienardPoly(Lienard):
    """
    Lienard oscillator with polynomial $f,g$ up to degree 3:
        f(x) = c + d x^2
        g(x) = a x + b x^3
    where c < 0 and a,b,d > 0 there is a limit cycle according to Lienard equations.
    Here we let c be positive to allow for a fixed point.
    Copied from twa
    """

    param_ranges = {
        'a': [0, 1],
        'c': [-1, 1],
        'b': [1, 1],
        'd': [1, 1],
        'decay': [1/6, 1]
    }

    param_display = {
        'a': r'$a$',
        'c': r'$c$',
    }

    position_lims = [-4.2, 4.2]

    def __init__(self, a: float=None, b: float=None, c: float=None, d: float=None, decay: float=None):
        super().__init__({'a': a, 'b': b, 'c': c, 'd': d, 'decay': decay})
        a, b, c, d = self.parameters['a'], self.parameters['b'], self.parameters['c'], self.parameters['d']
        self.g = lambda x: a * x + b * x ** 3
        self.f = lambda x: c + d * x ** 2

    @staticmethod
    def random_cycle_params() -> dict:
        """
        :return: a dictionary of parameters to the system for oscillatory behavior
        """
        c = LienardPoly.param_ranges['c'][0] * np.random.rand()
        return {'c': c}

    def __str__(self): return 'LienardPoly'

    def dist_from_bifur(self):
        return -self.parameters['c']


class LienardSigmoid(Lienard):
    """
    Lienard oscillator with polynomial $f$ and sigmoid $g$:
        f(x) = b + c x^2
        g(x) = 1 / (1 + e^(-ax)) - 0.5
    where b < 0 and a,c > 0 there is a limit cycle according to Lienard equations.
    Here we let b be positive to allow for a fixed point.
    Copied from twa
    """

    param_ranges = {
        'a': [1, 2],
        'b': [-1, 1],
        'c': [1, 1],
        'decay': [1/6, 1]
    }

    param_display = {
        'a': r'$a$',
        'b': r'$b$',
        'c': r'$c$',
    }

    position_lims = [-1.5, 1.5]

    def __init__(self, a: float=None, b: float=None, c: float=None, decay: float=None):
        super().__init__({'a': a, 'b': b, 'c': c, 'decay': decay})
        a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']

        self.g = lambda x: 1 / (1 + torch.exp(-a * x)) - 0.5
        self.f = lambda x: b + c * x ** 2

    @staticmethod
    def random_cycle_params() -> dict:
        """
        :return: a dictionary of parameters to the system for oscillatory behavior
        """
        b = LienardSigmoid.param_ranges['b'][0] * np.random.rand()
        return {'b': b}

    def __str__(self): return 'LienardSigmoid'

    def dist_from_bifur(self):
        return -self.parameters['b']


class MultiStable(PhaseSpace):

    position_lims = [-1, 1]

    def __init__(self, k: int=1, dim: int=2, a: torch.Tensor=None, pos: torch.Tensor=None, sigma: float=.1):
        super().__init__({})

        self.a = a
        if a is None: self.a = torch.ones(k)

        self.pos = pos
        if pos is None:
            self.pos = torch.rand(k, dim)*(self.position_lims[1]-self.position_lims[0]) + self.position_lims[0]
            self.pos = self.pos*.9

        self.sigma = sigma

    def __str__(self):
        return 'MultiStable'

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        d = torch.sum(z * z, dim=-1)[:, None] - 2 * z @ self.pos.T + torch.sum(self.pos * self.pos, dim=-1)[None]
        p = -.5 * d / self.sigma
        p = torch.exp(p - torch.logsumexp(p, dim=-1, keepdim=True))

        return torch.sum(p[..., None] * self.a[None, :, None] * (self.pos[None] - z[:, None]), dim=1)

    def dist_from_bifur(self) -> float: return 0


class MultiBasin(PhaseSpace):

    position_lims = [-3, 3]

    def __init__(self, n_basin: int, a: torch.Tensor=None, omega: torch.Tensor=None, logw: torch.Tensor=None,
                 centers: torch.Tensor=None, logs: torch.Tensor=None):
        super().__init__({})
        self.a = a if a is not None else torch.rand(n_basin)-.5
        self.om = omega if omega is not None else 2*torch.rand(n_basin)-1
        self.logw = logw if logw is not None else torch.zeros(n_basin)
        self.s = logs if logs is not None else torch.log(torch.tensor([.5]))
        self.cen = centers if centers is not None else torch.randn(n_basin, 2)

    def __str__(self):
        return f'MultiBasin_nodes={int(np.sum(self.a<0))}_cycles={int(np.sum(self.a>0))}'

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        m = x[:, None] - self.cen[None, :]
        r = torch.sqrt(torch.sum(m**2, dim=-1))
        theta = torch.atan2(m[..., 1], m[..., 0])

        rdot = r*(self.a[None]-r*r)

        xdot = torch.cat([
            (torch.cos(theta)*rdot - r*torch.sin(theta)*self.om[None])[..., None],
            (torch.sin(theta)*rdot + r*torch.cos(theta)*self.om[None])[..., None]
        ], dim=-1)
        w = torch.exp(self.logw)

        s = torch.exp(self.s)
        D = torch.sum(x*x, dim=-1)[:, None] - 2*x@self.cen.T + torch.sum(self.cen*self.cen, dim=-1)[None, :]
        D = torch.exp(-D/s - torch.logsumexp(-D/s, dim=1, keepdim=True))
        return torch.sum(xdot*D[..., None]*w[None, :, None], dim=1).reshape(shape)

    def dist_from_bifur(self) -> float: return 0


# ================================================= ^^ Systems ^^ =====================================================

# ================================================= vv Perturbations vv ================================================

class RBFPerturbation(PhaseSpace):
    """
    Perturbation of a given system using an RBF kernel
    """

    def __init__(self, system: PhaseSpace, n_control: int=1, dim: int=2, length_scale: float=None,
                 max_weight: float=None):
        self.system = system
        super().__init__(system.parameters)
        self.param_ranges = system.param_ranges
        self.param_display = system.param_display
        self.position_lims = system.position_lims

        if length_scale is None: length_scale = np.random.rand()*.9 + .1
        self.lscale = length_scale

        if max_weight is None: max_weight = np.random.rand()*.5
        self.max_weight = max_weight

        self.controls = self.random_x(n_control, dim)

    @staticmethod
    def random_cycle_params() -> dict: raise NotImplementedError

    def __str__(self): return f'perturbed_{str(self.system)}'

    def forward(self, z):
        D = torch.sum(z*z, dim=-1)[:, None] - 2*z@self.controls.T + torch.sum(self.controls*self.controls, dim=-1)[None]
        D = -.5*D/(self.lscale**2)
        weights = torch.exp(D)[:, :, None]*self.max_weight

        x = torch.sum((1-weights)*z[:, None] + weights*self.controls[None], dim=1)

        return self.system.forward(x)

    def dist_from_bifur(self):
        return self.system.dist_from_bifur()


class NFPerturbation(PhaseSpace):
    """
    Perturbation of a given system using a normalizing flow
    """

    def __init__(self, system: PhaseSpace, dim: int=2, n_layers: int=2, n_freqs: int=3, init: float=.1):
        self.system = system
        super().__init__(system.parameters)
        self.param_ranges = system.param_ranges
        self.param_display = system.param_display
        self.position_lims = system.position_lims

        layers = []
        for i in range(n_layers):
            layers += [FFCoupling(dim=dim, K=n_freqs, scale_free=True, R=15),
                       FFCoupling(dim=dim, K=n_freqs, scale_free=True, reverse=True, R=15)]
        self.NF = NFCompose(*layers)
        self.NF.requires_grad_(False)
        for name, param in self.NF.named_parameters():
            param.data = torch.randn_like(param)*init

    @staticmethod
    def random_cycle_params() -> dict: raise NotImplementedError

    def __str__(self): return f'perturbed_{str(self.system)}'

    def forward(self, z):
        x = self.NF(z)
        return self.system.forward(x)

    def dist_from_bifur(self):
        return self.system.dist_from_bifur()


class AffineLifting(PhaseSpace):
    """
    Lifting of a given system into a higher dimension through an affine transformation
    """

    def __init__(self, system: PhaseSpace, dim: int=3, decay: float=.5):
        self.system = system
        super().__init__(system.parameters)
        self.param_ranges = system.param_ranges
        self.param_display = system.param_display
        self.position_lims = system.position_lims

        self.trans = self.random_x(1, dim)[0]*0   # translation in space
        self.A = torch.randn(dim, 2)    # random hyperplane on which data is lifted
        self.proj = torch.linalg.solve(self.A.T@self.A, self.A.T)   # projection matrix onto hyperplane

        self.decay = decay

    @staticmethod
    def random_cycle_params() -> dict: raise NotImplementedError

    def __str__(self): return f'lifted_{str(self.system)}'

    def forward(self, z):
        x = (z-self.trans[None])@self.A
        xdot = self.system.forward(x)@self.proj

        proj_dot = self.decay*(z@self.proj.T@self.A.T - z)

        return xdot + proj_dot

    def dist_from_bifur(self):
        return self.system.dist_from_bifur()

# ================================================= ^^ Perturbations ^^ ================================================
