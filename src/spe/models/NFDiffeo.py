import torch
from torch import nn
import numpy as np
from typing import Tuple


class RFFCoupling(nn.Module):
    """
    An implementation of the invertible Fourier features transform, given by:
                    f(x_1, x_2) = (x_1, x_2 * exp(s(x_1)) + t(x_1))
    where the functions s(x_1) and t(x_1) are both defined according to the Fourier features:
                    t(x) = sum{ a_k sin(2*pi*k*x/R + b_k) } with k between 0 and K
    where a_k are the coefficients of the Fourier features and b_k are the phases. R defines the natural scale of the
    function and K is the number of Fourier components to be used.
    """
    def __init__(self, dim: int, K: int=32, reverse: bool=False):
        """
        Initializes the Fourier-features transform layer
        :param dim: number of dimensions expected for inputs
        :param K: number of Fourier coefficients to use
        :param reverse:
        """
        super().__init__()
        self.reord = reverse
        x1, x2 = self._split(torch.zeros(1, dim))
        self.W_s = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1]))
        self.W_t = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1]))
        self.g_s = nn.Parameter(1e-3*torch.randn(x2.shape[-1], K))
        self.g_t = nn.Parameter(1e-3*torch.randn(x2.shape[-1], K))
        self.b_s = nn.Parameter(torch.rand(K)*2*np.pi)
        self.b_t = nn.Parameter(torch.rand(K)*2*np.pi)

    @staticmethod
    def _ff(x: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: the output layer, a torch tensor with shape [dim_out, K]
        :param W: the hidden weights matrix, a torch tensor with shape [K, dim_in]
        :param phi: the bias, a torch tensor with shape [K]
        :return: the transformed input x, a tensor with shape [N, dim_out]
        """
        return torch.sin(x@W.T+phi[None])@gamma.T

    @staticmethod
    def _df(x: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: the output layer, a torch tensor with shape [dim_out, K]
        :param W: the hidden weights matrix, a torch tensor with shape [K, dim_in]
        :param phi: the bias, a torch tensor with shape [K]
        :return: the Jacobian of the FF function, with shape [N, dim_in, dim_out]
        """
        cos = torch.cos(x@W.T+phi[None])  # [N, k]
        interm = W.T[None]*cos[:, None, :]  # [N, dim_in, k]
        return interm@gamma.T  # [N, dim_in, dim_out]

    def _s(self, x1: torch.Tensor, rev: bool=False) -> torch.Tensor:
        return torch.exp(self._ff(x1, self.g_s, self.W_s, self.b_s)*(-1 if rev else 1))

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        return self._ff(x1, self.g_t, self.W_t, self.b_t)

    def _split(self, x: torch.Tensor) -> Tuple:
        if self.reord:
            x2, x1 = torch.chunk(x, 2, dim=1)
        else:
            x1, x2 = torch.chunk(x, 2, dim=1)
        return x1, x2

    def _cat(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.reord: return torch.cat([x2, x1], dim=1)
        else: return torch.cat([x1, x2], dim=1)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        self.W_s.requires_grad = not freeze
        self.b_s.requires_grad = not freeze
        self.g_s.requires_grad = not freeze

        self.W_t.requires_grad = not freeze
        self.b_t.requires_grad = not freeze
        self.g_t.requires_grad = not freeze

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        return self._ff(x1, self.g_s, self.W_s, self.b_s).sum(dim=1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian of the transform at points x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim, dim] of the Jacobians evaluated at each point x
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        x1, x2 = self._split(x)
        return self._cat(x1, self._s(x1)*x2 + self._t(x1))

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates both the forward pass, as well as a Jacobian-vector-product (JVP) and the log-abs-determinant
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the JVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (y, Jf, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed xs
                    - Jf: a torch tensor with shape [N, dim], which are the JVP with f
                    - logdet: a torch tensor with shape [N,] which are the log-abs-determinants evaluated at the
                              points x
        """
        x1, x2 = self._split(x)
        s, t = self._s(x1), self._t(x1)
        y = self._cat(x1, s*x2 + t)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        ds = (x2*s)[..., None, :]*self._df(x1, self.g_s, self.W_s, self.b_s)  # [N, dim_in, dim_out]
        dt = self._df(x1, self.g_t, self.W_t, self.b_t)  # [N, dim_in, dim_out]
        Jf2 = ((ds + dt).transpose(-2, -1)@f1[..., None])[..., 0] + f2*s
        Jf = self._cat(f1, Jf2)

        logdet = self._ff(x1, self.g_s, self.W_s, self.b_s).sum(dim=1)

        return y, Jf, logdet

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        y1, y2 = self._split(y)
        return self._cat(y1, self._s(y1, rev=True)*(y2 - self._t(y1)))

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        y1, y2 = self._split(y)
        s, t = self._s(y1, rev=True), self._t(y1)
        x2 = s * (y2 - self._t(y1))
        x = self._cat(y1, x2)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        ds = x2[:, None] * self._df(y1, self.g_s, self.W_s, self.b_s)  # [N, dim_in, dim_out]
        dt = s[:, None] * self._df(y1, self.g_t, self.W_t, self.b_t)  # [N, dim_in, dim_out]
        Jf2 = -((ds + dt).transpose(-2, -1) @ f1[..., None])[..., 0] + f2 * s
        Jf = self._cat(f1, Jf2)

        return x, Jf


class MLPCoupling(nn.Module):

    def __init__(self, dim: int, K: int=32, reverse: bool=False, depth: int=2, activation=nn.SiLU):
        """
        Initializes an AffineCoupling layer with MLPs with single hidden layers
        :param dim: number of dimensions expected for inputs
        :param K: width of MLP
        :param reverse: whether to reverse the order of x1 and x2
        :param depth: the depth of the MLP
        :param activation: the activation to use in the MLP
        """
        super().__init__()
        self.reord = reverse
        x1, x2 = self._split(torch.ones(1, dim))

        layers = [nn.Linear(x1.shape[-1], K), activation()]
        for i in range(depth-1):
            layers += [nn.Linear(K, K), activation()]
        layers.append(nn.Linear(K, x2.shape[-1]))
        self.s = nn.Sequential(*layers)
        for param in self.s.parameters(): param.data = 1e-5*torch.randn_like(param.data)

        layers = [nn.Linear(x1.shape[-1], K), activation()]
        for i in range(depth - 1):
            layers += [nn.Linear(K, K), activation()]
        layers.append(nn.Linear(K, x2.shape[-1]))
        self.t = nn.Sequential(*layers)
        for param in self.t.parameters(): param.data = 1e-3 * torch.randn_like(param.data)

    def _s(self, x1: torch.Tensor, rev: bool=False) -> torch.Tensor:
        return torch.exp(self.s(x1)*(-1 if rev else 1))

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        return self.t(x1)

    def _split(self, x: torch.Tensor) -> Tuple:
        if self.reord:
            x2, x1 = x.split(x.shape[-1]//2, dim=1)
        else:
            x1, x2 = x.split(x.shape[-1]//2, dim=1)
        return x1, x2

    def _cat(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.reord: return torch.cat([x2, x1], dim=1)
        else: return torch.cat([x1, x2], dim=1)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        for param in self.s.parameters():
            param.requires_grad = freeze

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        return self.s(x1).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        x1, x2 = self._split(x)
        return self._cat(x1, self._s(x1)*x2 + self._t(x1))

    def _Jf(self, x1: torch.Tensor, x2: torch.Tensor, f: torch.Tensor):
        f1, f2 = self._split(f.clone())
        xd = torch.clone(x1).requires_grad_(True)
        jf1 = torch.func.jvp(lambda x: self.s(x)*x2+self.t(x), (xd, ), (f1, ))[1]
        return self._cat(f1, jf1 + self.s(xd)*f2)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates both the forward pass, as well as a Jacobian-vector-product (JVP) and the log-abs-determinant
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the JVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (y, Jf, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed xs
                    - Jf: a torch tensor with shape [N, dim], which are the JVP with f
                    - logdet: a torch tensor with shape [N,] which are the log-abs-determinants evaluated at the
                              points x
        """
        x1, x2 = self._split(x)
        s, t = self._s(x1), self._t(x1)
        y = self._cat(x1, s*x2 + t)

        Jf = self._Jf(x1, x2, f)

        logdet = self.s(x1).sum(dim=1)

        return y, Jf, logdet

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        y1, y2 = self._split(y)
        return self._cat(y1, self._s(y1, rev=True)*(y2 - self._t(y1)))

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        raise NotImplementedError


class FFCoupling(nn.Module):
    """
    An implementation of the invertible Fourier features transform, given by:
                    f(x_1, x_2) = (x_1, x_2 * exp(s(x_1)) + t(x_1))
    where the functions s(x_1) and t(x_1) are both defined according to the Fourier features:
                    t(x) = sum{ a_k cos(2*pi*k*x/R + b_k) } with k between 0 and K
    where a_k are the coefficients of the Fourier features and b_k are the phases. R defines the natural scale of the
    function and K is the number of Fourier components to be used.
    """
    def __init__(self, dim: int, K: int=32, R: float=10, reverse: bool=False, scale_free: bool=False, split_dims=None):
        """
        Initializes the Fourier-features transform layer
        :param dim: number of dimensions expected for inputs
        :param K: number of Fourier coefficients to use
        :param R: the maximum interval assumed in the data
        :param scale_free: return a scaling-free version of the FFCoupling
        :param reverse:
        """
        super().__init__()
        self.reord = reverse
        self.split_dims = split_dims
        x1, x2 = self._split(torch.zeros(1, dim))
        self.a_t = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1], x2.shape[-1]))
        self.b_t = nn.Parameter(torch.rand(K, x1.shape[-1])*2*np.pi*1e-2)
        self.scale_free = scale_free
        self.R = R

        if self.scale_free:
            self.register_buffer('a_s', torch.zeros(K, x1.shape[-1], x2.shape[-1]))
            self.register_buffer('b_s', torch.zeros(K, x1.shape[-1]))
        else:
            self.a_s = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1], x2.shape[-1]))
            self.b_s = nn.Parameter(torch.rand(K, x1.shape[-1])*2*np.pi*1e-2)

    @staticmethod
    def _ff(x: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor, R: float) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: a torch tensor with shape [K, dim_in, dim_out] of the Fourier coefficients
        :param phi: a torch tensor with shape [K, dim_in] the phases of the transformation
        :param R: a float depicting the range
        :return: the transformed input x, a tensor with shape [N, dim_out]
        """
        freqs = torch.arange(0, gamma.shape[0], device=x.device)*2*np.pi/R
        # return torch.sum(torch.cos(freqs[None, :, None]*x[:, None] + phi[None])[..., None] * gamma[None], dim=(1, -2))
        return torch.sum(torch.sin(freqs[None, :, None]*x[:, None] + phi[None])[..., None] * gamma[None], dim=(1, -2))

    @staticmethod
    def _df(x: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor, R: float) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: a torch tensor with shape [K, dim_in, dim_out] of the Fourier coefficients
        :param phi: a torch tensor with shape [K, dim_in] the phases of the transformation
        :param R: a float depicting the range
        :return: the Jacobian of the FF function, with shape [N, dim_in, dim_out]
        """
        freqs = torch.arange(0, gamma.shape[0], device=x.device)*2*np.pi/R
        # sins = - freqs[None, :, None]*torch.sin(freqs[None, :, None]*x[:, None] + phi[None])  # [N, K, dim_in]
        sins = freqs[None, :, None]*torch.cos(freqs[None, :, None]*x[:, None] + phi[None])  # [N, K, dim_in]
        return torch.sum(sins[..., None]*gamma[None], dim=1)  # [N, dim_in, dim_out]

    def _s(self, x1: torch.Tensor, rev: bool=False) -> torch.Tensor:
        return torch.exp(self._ff(x1, self.a_s, self.b_s, self.R)*(-1 if rev else 1))

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        return self._ff(x1, self.a_t, self.b_t, self.R)

    def _split(self, x: torch.Tensor) -> Tuple:
        if self.split_dims is None:
            if self.reord:
                x2, x1 = torch.chunk(x, 2, dim=1)
            else:
                x1, x2 = torch.chunk(x, 2, dim=1)
        else:
            outinds = [i for i in range(x.shape[-1]) if i not in self.split_dims]
            x1 = x[:, self.split_dims]
            x2 = x[:, outinds]
        return x1, x2

    def _cat(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.split_dims is None:
            if self.reord: return torch.cat([x2, x1], dim=1)
            else: return torch.cat([x1, x2], dim=1)
        else:
            x = torch.zeros_like(torch.cat([x2, x1], dim=1))
            outinds = [i for i in range(x.shape[-1]) if i not in self.split_dims]
            x[:, self.split_dims] = x1
            x[:, outinds] = x2
            return x

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        self.a_s.requires_grad = not freeze
        self.b_s.requires_grad = not freeze
        self.a_t.requires_grad = not freeze
        self.b_t.requires_grad = not freeze

        if self.scale_free:
            self.a_s.requires_grad = False
            self.b_s.requires_grad = False

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        return self._ff(x1, self.a_s, self.b_s, self.R).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        x1, x2 = self._split(x)
        return self._cat(x1, self._s(x1)*x2 + self._t(x1))

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates both the forward pass, as well as a Jacobian-vector-product (JVP) and the log-abs-determinant
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the JVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (y, Jf, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed xs
                    - Jf: a torch tensor with shape [N, dim], which are the JVP with f
                    - logdet: a torch tensor with shape [N,] which are the log-abs-determinants evaluated at the
                              points x
        """
        x1, x2 = self._split(x)
        s, t = self._s(x1), self._t(x1)
        y = self._cat(x1, s*x2 + t)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        ds = (x2*s)[..., None, :]*self._df(x1, self.a_s, self.b_s, self.R)  # [N, dim_in, dim_out]
        dt = self._df(x1, self.a_t, self.b_t, self.R)  # [N, dim_in, dim_out]
        Jf2 = ((ds + dt).transpose(-2, -1)@f1[..., None])[..., 0] + f2*s
        Jf = self._cat(f1, Jf2)

        logdet = self._ff(x1, self.a_s, self.b_s, self.R).sum(dim=1)

        return y, Jf, logdet

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        y1, y2 = self._split(y)
        return self._cat(y1, self._s(y1, rev=True)*(y2 - self._t(y1)))

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        y1, y2 = self._split(y)
        s, t = self._s(y1, rev=True), self._t(y1)
        x2 = s*(y2 - self._t(y1))
        x = self._cat(y1, x2)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        ds = x2[:, None]*self._df(y1, self.a_s, self.b_s, self.R)  # [N, dim_in, dim_out]
        dt = s[:, None]*self._df(y1, self.a_t, self.b_t, self.R)  # [N, dim_in, dim_out]
        Jf2 = -((ds+dt).transpose(-2, -1) @ f1[..., None])[..., 0] + f2 * s
        Jf = self._cat(f1, Jf2)

        return x, Jf


class Affine(nn.Module):
    """
    An affine transformation module of a normalizing flow. To ensure invertibility, this transformation is defined as:
                                        y = WW^Tx + exp[phi] x + mu
    where the learnable parameters are W (with shape [dim, rank]), phi (with shape [dim]), and mu (with shape [dim])
    """
    def __init__(self, dim: int, rank: int=None, data_init: bool=False, mu: torch.Tensor=None):
        super().__init__()
        if rank is None: rank = dim
        self.mu = nn.Parameter(torch.randn(dim)*1e-3) if mu is None else nn.Parameter(mu.clone())
        self.W = nn.Parameter(torch.randn(dim, rank)*1e-3)
        self.phi = nn.Parameter(torch.randn(dim)*1e-3)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        return torch.linalg.slogdet(self.W@self.W.T + torch.diag(phi))[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        return phi[None]*x + (x@self.W)@self.W.T + self.mu[None]

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.forward(x)
        phi = torch.exp(self.phi)
        J = phi[None]*f + (f@self.W)@self.W.T
        return y, J, self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        M = self.W@self.W.T + torch.diag_embed(phi)
        m = y-self.mu[None]
        return torch.linalg.solve(M, m.T).T

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        phi = torch.exp(self.phi)
        M = self.W @ self.W.T + torch.diag_embed(phi)
        m = y - self.mu[None]
        return torch.linalg.solve(M, m.T).T, torch.linalg.solve(M, f.T).T


class ActNorm(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('mu', torch.zeros(dim))
        self.register_buffer('s', torch.zeros(dim))
        self.register_buffer('newinit', torch.ones(1))

    def _newinit(self, x: torch.Tensor):
        self.mu.data = torch.mean(x, dim=0).data
        self.s.data = torch.log(torch.std(x, dim=0).data + 1e-3)
        self.newinit.data[:] = 0

    def freeze_scale(self, freeze: bool = True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(-self.s)[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.newinit[0] == 1: self._newinit(x)
        return torch.exp(-self.s)[None]*(x-self.mu[None])

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(x), torch.exp(-self.s)[None]*f, self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.newinit[0] == 1: self._newinit(y)
        return torch.exp(self.s)[None]*y + self.mu[None]

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        return torch.exp(self.s)[None]*y + self.mu[None], torch.exp(self.s)[None]*f


class HouseholderTransf(nn.Module):
    """
    An implementation of Householder flows: https://arxiv.org/pdf/1611.09630
    """

    def __init__(self, dim: int):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dim) * 1e-3)

    def __transform(self, u: torch.Tensor):
        return u - 2*self.v[None]*torch.sum(self.v[None]*u, dim=1)[:, None]/torch.sum(self.v*self.v)

    def freeze_scale(self, freeze: bool = True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return 0

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eye(x.shape[-1]) - 2*self.v[:, None]@self.v[None, :]/torch.sum(self.v*self.v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__transform(x)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.__transform(x), self.__transform(f), self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.__transform(y)

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        return self.__transform(y), self.__transform(f)


class OrthoSylvester(nn.Module):
    """
    Implementation of orthogonal Sylvester flow (https://arxiv.org/pdf/1803.05649)
    """

    def __init__(self, dim: int, n_steps: int=30, prec: float=1e-5):
        """
        :param dim: dimensionality of the data
        :param n_steps: number of steps taken in order to project the transformation into an orthonormal one
        :param prec: precision parameter to stop orthogonalization iterations
        """
        super().__init__()
        self.Q = nn.Parameter(torch.randn(dim, dim)*1e-3)
        self.n_steps = n_steps
        self.dim = dim
        self.prec = prec
        self.register_buffer('Q_center', torch.eye(dim))

    def _Q(self):
        Q = self.Q + self.Q_center  # Q is the perturbation from the center
        Q = Q + 1e-6*torch.eye(self.dim, device=Q.device)  # numerical stability
        if torch.isnan(Q).any(): raise AssertionError('Affine transformation is nan')
        I = torch.eye(self.dim, device=Q.device)
        norm = torch.norm(Q.T@Q - I)
        if norm >= 1:
            with torch.no_grad():
                normalization = max(torch.max(Q).item()*Q.shape[0],
                                    torch.max(torch.sum(torch.abs(Q), dim=0)).item()*np.sqrt(Q.shape[0]))
            Q = Q/normalization
            # raise AssertionError('Distance of affine transformation from orthogonal too big')
        if torch.isnan(Q).any(): raise AssertionError('Affine transformation is nan')
        if norm > self.prec:
            for i in range(self.n_steps):
                P = I - Q.T@Q
                with torch.no_grad():
                    if torch.norm(P) <= self.prec: break
                Q = Q@(I + .5*P)
        if not self.training: self.Q.data = Q - self.Q_center  # Q is the perturbation from the centering
        if torch.isnan(Q).any(): raise AssertionError('Affine transformation is nan')
        return Q

    def freeze_scale(self, freeze: bool = True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self._Q()
        return x@Q

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = self._Q()
        return x@Q, f@Q, self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        Q = self._Q()
        return y@Q.T

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        Q = self._Q()
        return y@Q.T, f@Q.T


class ScaleTransf(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(dim)*1e-4)
        self.s = nn.Parameter(torch.randn(dim)*1e-4)

    def freeze_scale(self, freeze: bool = True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(-self.s)[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.s)[None]*(x-self.mu[None])

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(x), torch.exp(-self.s)[None]*f, self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.s)[None]*y + self.mu[None]

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        return torch.exp(self.s)[None]*y + self.mu[None], torch.exp(self.s)[None]*f


class NFCompose(nn.Module):
    """
    Composes a number of normalizing-flow layers into one, similar to nn.Sequential
    """
    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.transfs = nn.ModuleList(modules)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        for mod in self.transfs: mod.freeze_scale(freeze)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        full_jac = torch.eye(x.shape[-1], device=x.device)[None]
        for mod in self.transfs:
            jac = mod.jacobian(x)
            full_jac = jac@full_jac
            x = mod.forward(x)
        return full_jac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mod in self.transfs:
            x = mod.forward(x)
        return x

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ldets = 0
        for mod in self.transfs:
            x, f, ldet = mod.jvp_forward(x, f)
            ldets = ldets + ldet
        return x, f, ldets

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        for mod in self.transfs[::-1]:
            y = mod.reverse(y)
        return y

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for mod in self.transfs[::-1]:
            y, f = mod.jvp_reverse(y, f)
        return y, f


class Diffeo(nn.Module):

    def __init__(self, dim: int, rank: int=2, n_layers: int=4, K: int=15,
                 actnorm: bool=True, RFF: bool=False, MLP: bool=False,
                 n_householder: int=0, sylvester: bool=True, full_affine: bool=False,
                 R: float=10, autoregressive: bool=False):
        """
        Initializes a diffeomorphism, which is a normalizing flow with interleaved Affine transformations and
        AffineCoupling layers
        :param dim: the dimension of the data
        :param rank: the rank used in the Affine transformations (see the Affine object above)
        :param n_layers: number of Affine-Coupling-ReverseCoupling layers to use
        :param K: number of hidden units to use, either for the FFCoupling, MLP or RFF transformations
        :param MLP: if True, an AffineCoupling layer with an MLP will be used instead of the FFCoupling layer
        :param actnorm: whether the first layer is an invertible standardization of the data (a sort of preprocessing)
        :param RFF: if True, an AffineCoupling layer with an RFF will be used instead of the FFCoupling layer
        :param sylvester: whether to use a orthonormal Sylvester transformation (if dimension is greater than 2) at the beginning of the flow
        :param full_affine: whether to use a full Affine transformation instead of the simpler scale transformation
        :param n_householder: number of Householder transformations to use at the beginning of the flow
        :param R: range of fourier coefficients
        :param latent_dim: if not None, the first layer is a dimensionality reduction layer
        """
        super().__init__()

        self.actnorm = actnorm
        self.sylvester = sylvester

        layers = []

        if actnorm: layers.append(ActNorm(dim))   # an invertible z-scoring of the data

        if sylvester: layers.append(OrthoSylvester(dim=dim))   # an orthonormal transformation of the data

        if full_affine:
            layers.append(Affine(dim=dim, rank=dim))
        else:
            layers.append(ScaleTransf(dim=dim))

        # add househoulder transformations which can rotate space
        if n_householder > 0:
            for i in range(n_householder): layers.append(HouseholderTransf(dim))

        # build rest of network
        for i in range(n_layers):
            if full_affine: layers.append(Affine(dim=dim, rank=rank))
            else: layers.append(ScaleTransf(dim=dim))

            layer_type = MLPCoupling if MLP else RFFCoupling if RFF else FFCoupling
            if not autoregressive:
                layers.append(layer_type(dim=dim, K=K, R=R))
                layers.append(layer_type(dim=dim, K=K, R=R, reverse=True))
            else:   # if the transformation is autoregressive, couple each dimension to those that came before
                for d in range(dim):
                    layers.append(layer_type(dim=dim, K=K, R=R,
                                                split_dims=[m for m in range(dim) if m!=d]))
                

        self.transf = NFCompose(*layers)

    def initialize(self, x: torch.Tensor, xdot: torch.Tensor=None, zdot: torch.Tensor=None):
        with torch.no_grad():
            self.transf.freeze_scale(False)
            self.transf.forward(x)

            if self.sylvester and xdot is not None:
                if self.actnorm:
                    _, ydot, _ = self.transf.transfs[0].jvp_forward(x, xdot)
                else: ydot = xdot

                Ux, _, _ = torch.linalg.svd((ydot-ydot.mean(dim=0)).T, full_matrices=False)
                if zdot is not None: Uz, _, _ = torch.linalg.svd((zdot-zdot.mean(dim=0)).T, full_matrices=False)
                else: Uz = torch.eye(Ux.shape[0], device=Ux.device)
                self.transf.transfs[1].Q_center.data = Uz.T@Ux

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        self.transf.freeze_scale(freeze)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return self.transf.jacobian(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the forward transformation of the normalizing flow
        :param x: inputs as a torch tensor with shape [N, dim]
        :return: the transformed inputs, a tensor with shape [N, dim]
        """
        return self.transf.forward(x)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the forward transformation of the normalizing flow, plus a JVP and the log-determinant
        :param x: inputs as a torch tensor with shape [N, dim]
        :param f: vectors whose JVP should be calculated, a torch tensor with shape [N, dim]
        :return: - the transformed inputs, a tensor with shape [N, dim]
                 - the JVPs of the normalizing flow on the vectors in f, a tensor with shape [N, dim]
                 - the log-determinants of the normalizing flows on x, a tensor with shape [N]
        """
        return self.transf.jvp_forward(x, f)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation of the normalizing flow
        :param y: inputs as a torch tensor with shape [N, dim]
        :return: the transformed inputs, a tensor with shape [N, dim]
        """
        return self.transf.reverse(y)

    def jvp_reverse(self, y: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the reverse pass as well as the inverse Jacobian-vector-product (iJVP)
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector on which to evaluate the iJVP, a torch tensor with shape [N, dim] where N is the
                  number of points
        :return: the tuple (x, inv(J)f, logdet) where:
                    - y: a torch tensor with shape [N, dim], which are the transformed ys
                    - inv(J)f: a torch tensor with shape [N, dim], which is the iJVP with f
        """
        return self.transf.jvp_reverse(y, f)