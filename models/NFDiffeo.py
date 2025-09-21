import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from typing import Tuple


class NFModule(nn.Module):
    """
    The base class for a normalizing flow module.
    """
    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        raise NotImplementedError

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f
        """
        raise NotImplementedError

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        raise NotImplementedError

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        raise NotImplementedError

    def prior(self, base_var: float) -> torch.Tensor:
        """
        Calculates the prior probability on all of the module's weights. The default prior is a Gaussian prior with
        zero mean
        :param base_var: the base variance used for the Gaussian prior on the weights
        :return: the prior negative log-probability on the weights of the module
        """
        prior = torch.tensor(0.)
        for param in self.parameters():
            prior = prior + torch.sum(param*param)
        return .5*prior/base_var


class DimReduction(NFModule):

    def __init__(self, dim: int, latent: int):
        super().__init__()
        self.dim = dim
        self.latent = latent
        # self.register_buffer('A', torch.rand(dim, latent)/dim)
        # self.register_buffer('m', torch.zeros(dim))
        self.A = nn.Parameter(torch.rand(dim, latent)/dim)
        self.m = nn.Parameter(torch.rand(dim)/dim)
        self.register_buffer('inited', torch.tensor([0.]))

    def init_(self, x: torch.Tensor):
        """
        Fit the dimension reduction as a PCA. If xdot is also supplied, then the MLE transformation assuming a node
        attractor is fit
        :param x: the training data, a tensor with shape [N, dim]
        """
        self.m.data = x.mean(dim=0)
        m = x-self.m[None]
        u, _, _ = torch.linalg.svd(m.T)  # calculate SVD
        self.A.data = u[:, :self.latent]
        self.inited.data[:] = 1

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projected the samples onto the hyperplane
        :param x: the training data, a tensor with shape [N, dim]
        :return: the projected data, a tensor with shape [N, dim]
        """
        return x@self.A@self.A.T

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logdet(self.A.T@self.A)[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inited[0] != 1: self.init_(x)
        return (x-self.m[None])@self.A

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return f@self.A

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lstsq(self.A.T, y.T)[0].T + self.m[None]

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lstsq(self.A.T, f.T)[0].T



class CosineCoupling(NFModule):
    """
    Affine coupling layer with Cosine activations
    """
    def __init__(self, dim: int, K: int=32, reverse: bool=False):
        """
        Initializes the Fourier-features transform layer
        :param dim: number of dimensions expected for inputs
        :param K: number of Fourier coefficients to use
        :param R: the maximum interval assumed in the data
        :param reverse:
        """
        super().__init__()
        self.reord = reverse
        x1, x2 = self._split(torch.zeros(1, dim))
        # weights for scaling
        self.W_s = nn.Parameter(1e-5*torch.randn(K, x1.shape[-1]))
        self.phi_s = nn.Parameter(torch.rand(K) * 2 * np.pi * 1e-4)
        self.A_s = nn.Parameter(1e-5 * torch.randn(x2.shape[-1], K))
        self.b_s = nn.Parameter(1e-5*torch.randn(x2.shape[-1]))

        # weights for bias
        self.W_t = nn.Parameter(1e-5*torch.randn(K, x1.shape[-1]))
        self.phi_t = nn.Parameter(torch.rand(K) * 2 * np.pi * 1e-4)
        self.A_t = nn.Parameter(1e-5 * torch.randn(x2.shape[-1], K))
        self.b_t = nn.Parameter(1e-5 * torch.randn(x2.shape[-1]))

    @staticmethod
    def _ff(x: torch.Tensor, A: torch.Tensor, W: torch.Tensor, phi: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param A: the output layer, a torch tensor with shape [dim_out, K]
        :param W: the hidden weights matrix, a torch tensor with shape [K, dim_in]
        :param phi: the bias in the cosine, a torch tensor with shape [K]
        :param bias: the bias after the cosine, a torch tensor with shape [dim_out]
        :return: the transformed input x, a tensor with shape [N, dim_out]
        """
        return torch.cos(x@W.T+phi[None])@A.T + bias[None]

    @staticmethod
    def _df(x: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor, phi: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: the output layer, a torch tensor with shape [dim_out, K]
        :param W: the hidden weights matrix, a torch tensor with shape [K, dim_in]
        :param phi: the bias, a torch tensor with shape [K]
        :param bias: the bias after the cosine, a torch tensor with shape [dim_out]
        :return: the Jacobian of the FF function, with shape [N, dim_in, dim_out]
        """
        sin = torch.sin(x@W.T+phi[None])  # [N, k]
        interm = W.T[None]*sin[:, None, :]  # [N, dim_in, k]
        return - interm@gamma.T  # [N, dim_in, dim_out]

    def _s(self, x1: torch.Tensor, rev: bool=False) -> torch.Tensor:
        return torch.exp(self._ff(x1, self.A_s, self.W_s, self.phi_s, self.b_s) * (-1 if rev else 1))

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        return self._ff(x1, self.A_t, self.W_t, self.phi_t, self.b_t)

    def _split(self, x: torch.Tensor) -> Tuple:
        if self.reord:
            x2, x1 = torch.chunk(x, 2, dim=1)
        else:
            x1, x2 = torch.chunk(x, 2, dim=1)
        return x1, x2

    def _cat(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.reord: return torch.cat([x2, x1], dim=1)
        else: return torch.cat([x1, x2], dim=1)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        return self._ff(x1, self.A_s, self.W_s, self.phi_s, self.b_s).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        x1, x2 = self._split(x)
        return self._cat(x1, self._s(x1)*x2 + self._t(x1))

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f
        """
        x1, x2 = self._split(x)
        s, t = self._s(x1), self._t(x1)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        ds = (x2 * s)[..., None, :] * self._df(x1, self.A_s, self.W_s, self.phi_s, self.b_s)  # [N, dim_in, dim_out]
        dt = self._df(x1, self.A_t, self.W_t, self.phi_t, self.b_t)  # [N, dim_in, dim_out]
        Jf2 = ((ds + dt).transpose(-2, -1) @ f1[..., None])[..., 0] + f2 * s
        return self._cat(f1, Jf2)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        y1, y2 = self._split(y)
        return self._cat(y1, self._s(y1, rev=True)*(y2 - self._t(y1)))

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        y1, y2 = self._split(y)
        s, t = self._s(y1, rev=True), self._t(y1)
        x2 = s * (y2 - self._t(y1))

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        ds = x2[:, None] * self._df(y1, self.A_s, self.W_s, self.phi_s, self.b_s)  # [N, dim_in, dim_out]
        dt = s[:, None] * self._df(y1, self.A_t, self.W_t, self.phi_t, self.b_t)  # [N, dim_in, dim_out]
        Jf2 = -((ds + dt).transpose(-2, -1) @ f1[..., None])[..., 0] + f2 * s
        return self._cat(f1, Jf2)


class FFCoupling(NFModule):
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

        self.a_t = nn.Parameter(torch.randn(K, x1.shape[-1], x2.shape[-1])*1e-3)
        self.b_t = nn.Parameter(torch.rand(K, x1.shape[-1])*2*np.pi*1e-2)
        self.scale_free = scale_free
        self.R = R

        self.register_buffer('freqs_t', torch.arange(0, K) * 2 * np.pi / R)
        self.register_buffer('freqs_s', torch.arange(0, K) * 2 * np.pi / R)

        if self.scale_free:
            self.register_buffer('a_s', torch.zeros(K, x1.shape[-1], x2.shape[-1]))
            self.register_buffer('b_s', torch.zeros(K, x1.shape[-1]))
        else:
            self.a_s = nn.Parameter(torch.randn(K, x1.shape[-1], x2.shape[-1])*1e-3)
            self.b_s = nn.Parameter(torch.rand(K, x1.shape[-1])*2*np.pi*1e-2)

    @staticmethod
    def _ff(x: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: a torch tensor with shape [K, dim_in, dim_out] of the Fourier coefficients
        :param phi: a torch tensor with shape [K, dim_in] the phases of the transformation
        :param R: a float depicting the range
        :return: the transformed input x, a tensor with shape [N, dim_out]
        """
        # freqs = torch.arange(0, gamma.shape[0], device=x.device)*2*np.pi/R
        return torch.sum(torch.cos(freqs[None, :, None]*x[:, None] + phi[None])[..., None] * gamma[None], dim=(1, -2))

    @staticmethod
    def _df(x: torch.Tensor, gamma: torch.Tensor, phi: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: a torch tensor with shape [K, dim_in, dim_out] of the Fourier coefficients
        :param phi: a torch tensor with shape [K, dim_in] the phases of the transformation
        :param R: a float depicting the range
        :return: the Jacobian of the FF function, with shape [N, dim_in, dim_out]
        """
        # freqs = torch.arange(0, gamma.shape[0], device=x.device)*2*np.pi/R
        sins = - freqs[None, :, None]*torch.sin(freqs[None, :, None]*x[:, None] + phi[None])  # [N, K, dim_in]
        return torch.sum(sins[..., None]*gamma[None], dim=1)  # [N, dim_in, dim_out]

    def _s(self, x1: torch.Tensor, rev: bool=False) -> torch.Tensor:
        # return torch.exp(self._ff(x1, self.a_s, self.b_s, self.R)*(-1 if rev else 1))
        return torch.exp(self._ff(x1, self.a_s, self.b_s, self.freqs_s)*(-1 if rev else 1))

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        # return self._ff(x1, self.a_t, self.b_t, self.R)
        return self._ff(x1, self.a_t, self.b_t, self.freqs_t)

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

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        # return self._ff(x1, self.a_s, self.b_s, self.R).sum(dim=1)
        return self._ff(x1, self.a_s, self.b_s, self.freqs_s).sum(dim=1)

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f
        """
        x1, x2 = self._split(x)
        s, t = self._s(x1), self._t(x1)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        # ds = (x2 * s)[..., None, :] * self._df(x1, self.a_s, self.b_s, self.R)  # [N, dim_in, dim_out]
        ds = (x2 * s)[..., None, :] * self._df(x1, self.a_s, self.b_s, self.freqs_s)  # [N, dim_in, dim_out]
        # dt = self._df(x1, self.a_t, self.b_t, self.R)  # [N, dim_in, dim_out]
        dt = self._df(x1, self.a_t, self.b_t, self.freqs_t)  # [N, dim_in, dim_out]
        Jf2 = ((ds + dt).transpose(-2, -1) @ f1[..., None])[..., 0] + f2 * s
        return self._cat(f1, Jf2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        x1, x2 = self._split(x)
        return self._cat(x1, self._s(x1)*x2 + self._t(x1))

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        y1, y2 = self._split(y)
        return self._cat(y1, self._s(y1, rev=True)*(y2 - self._t(y1)))

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        y1, y2 = self._split(y)
        s, t = self._s(y1, rev=True), self._t(y1)
        x2 = s * (y2 - self._t(y1))

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        # ds = x2[:, None] * self._df(y1, self.a_s, self.b_s, self.R)  # [N, dim_in, dim_out]
        ds = x2[:, None] * self._df(y1, self.a_s, self.b_s, self.freqs_s)  # [N, dim_in, dim_out]
        # dt = s[:, None] * self._df(y1, self.a_t, self.b_t, self.R)  # [N, dim_in, dim_out]
        dt = s[:, None] * self._df(y1, self.a_t, self.b_t, self.freqs_t)  # [N, dim_in, dim_out]
        Jf2 = -((ds + dt).transpose(-2, -1) @ f1[..., None])[..., 0] + f2 * s
        return self._cat(f1, Jf2)


class Affine(NFModule):
    """
    An affine transformation module of a normalizing flow. To ensure invertibility, this transformation is defined as:
                                        y = WW^Tx + exp[phi] x + mu
    where the learnable parameters are W (with shape [dim, rank]), phi (with shape [dim]), and mu (with shape [dim])
    """
    def __init__(self, dim: int, rank: int=None):
        super().__init__()
        if rank is None: rank = dim
        self.mu = nn.Parameter(torch.randn(dim)*1e-3)
        self.W = nn.Parameter(torch.randn(dim, rank)*1e-3)
        self.phi = nn.Parameter(torch.randn(dim)*1e-3)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        return torch.logdet(self.W@self.W.T + torch.diag(phi)).repeat(x.shape[0])

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        return (self.W@self.W.T + torch.diag(phi))[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        return phi[None]*x + (x@self.W)@self.W.T + self.mu[None]

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f
        """
        phi = torch.exp(self.phi)
        return phi[None]*f + (f@self.W)@self.W.T

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        M = self.W@self.W.T + torch.diag_embed(phi)
        m = y-self.mu[None]
        return torch.linalg.solve(M, m.T).T

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        phi = torch.exp(self.phi)
        M = self.W @ self.W.T + torch.diag_embed(phi)
        return torch.linalg.solve(M, f.T).T


class LogTransf(NFModule):

    def __init__(self, precision: float=1e-4):
        super().__init__()
        self.prec = precision

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(-torch.log(x+self.prec), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x+self.prec)

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f
        """
        return f/(x+self.prec)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(y)-self.prec

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        return torch.exp(y)*f


class Log1PTransf(NFModule):

    def __init__(self):
        super().__init__()

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(-torch.log(x+1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x+1)

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f
        """
        return f/(x+1)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(y)-1

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        return torch.exp(y)*f


class ActNorm(NFModule):

    def __init__(self, dim: int, learnable: bool=True, scale_init: bool=True):
        super().__init__()
        if learnable:
            self.mu = nn.Parameter(torch.zeros(dim))
            self.s = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer('mu', nn.Parameter(torch.zeros(dim)))
            self.register_buffer('s', nn.Parameter(torch.zeros(dim)))
        self.register_buffer('newinit', torch.ones(1))
        self.scale_init = scale_init

    def _newinit(self, x: torch.Tensor):
        self.mu.data = torch.mean(x, dim=0).data
        if self.scale_init: self.s.data = torch.log(torch.std(x, dim=0).data + 1e-3)
        self.newinit.data[:] = 0

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(-self.s)[None]

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(torch.exp(-self.s))[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.newinit[0] == 1: self._newinit(x)
        return torch.exp(-self.s)[None]*(x-self.mu[None])

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f
        """
        return torch.exp(-self.s)[None]*f

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.newinit[0] == 1: self._newinit(y)
        return torch.exp(self.s)[None]*y + self.mu[None]

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        return torch.exp(self.s)[None]*f


class HouseholderTransf(NFModule):
    """
    An implementation of Householder flows: https://arxiv.org/pdf/1611.09630
    """

    def __init__(self, dim: int):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dim) * 1e-3)

    def __transform(self, u: torch.Tensor):
        return u - 2*self.v[None]*torch.sum(self.v[None]*u, dim=1)[:, None]/torch.sum(self.v*self.v)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:, 0])

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eye(x.shape[-1]) - 2*self.v[:, None]@self.v[None, :]/torch.sum(self.v*self.v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__transform(x)

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f
        """
        return self.__transform(f)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.__transform(y)

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        return self.__transform(f)


class NFCompose(NFModule):
    """
    Composes a number of normalizing-flow layers into one, similar to nn.Sequential
    """
    def __init__(self, *modules: NFModule):
        super().__init__()
        self.transfs = nn.ModuleList(modules)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        ldet = 0
        for mod in self.transfs:
            d = mod.logdet(x)
            ldet = ldet + d
            x = mod.forward(x)
        return ldet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mod in self.transfs:
            x = mod.forward(x)
        return x

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f
        """
        for mod in self.transfs:
            f = mod.jvp(x, f)
            x = mod.forward(x)
        return f

    def forward_jvp_logdet(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ldet = 0
        for mod in self.transfs:
            f = mod.jvp(x, f)
            d = mod.logdet(x)
            ldet = ldet + d
            x = mod.forward(x)
        return x, f, ldet

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        for mod in self.transfs[::-1]:
            y = mod.reverse(y)
        return y

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        for mod in self.transfs[::-1]:
            f = mod.inv_jvp(y, f)
            y = mod.reverse(y)
        return f


class Diffeo(NFModule):

    def __init__(self, dim: int, rank: int=2, n_layers: int=3, K: int=5, actnorm: bool=True,
                 n_householder: int=0, logtransf: bool=False, R: float=10, autoregressive: bool=False,
                 latent_dim: int=None):
        """
        Initializes a diffeomorphism, which is a normalizing flow with interleaved Affine transformations and
        AffineCoupling layers
        :param dim: the dimension of the data
        :param rank: the rank used in the Affine transformations (see the Affine object above)
        :param n_layers: number of Affine-Coupling-ReverseCoupling layers to use
        :param K: number of hidden units to use, either for the FFCoupling, MLP or RFF transformations
        :param actnorm: whether to add an invertible standardization of the data (a sort of preprocessing)
        :param n_householder: number of Householder transformations to use at the beginning of the flow
        :param logtransf: a boolean indicating if to log-transform in the first layer or not
        :param R: the range of the Fourier coupling layers
        :param autoregressive: whether to use an autoregressive scheme instead of the standard splitting in the Fourier
                               coupling; using autoregressive coupling is much slower than the standard, but
                               theoretically much more expressive (also prone to vanishing gradients)
        """
        super().__init__()

        layers = []

        if logtransf: layers.append(LogTransf())  # add a transformation to log space

        if latent_dim is not None:   # add a dimensionality reduction
            layers.append(DimReduction(dim=dim, latent=latent_dim))
            dim = latent_dim

        if actnorm: layers.append(ActNorm(dim, learnable=False))  # add an initial actnorm layer

        layers.append(Affine(dim=dim, rank=dim))  # first affine trasnformation

        if n_householder > 0:  # add householder transformations which can rotate space
            for i in range(n_householder): layers.append(HouseholderTransf(dim))

        # create all blocks
        for i in range(n_layers):
            layers.append(Affine(dim=dim, rank=rank))

            if not autoregressive:
                layers.append(FFCoupling(dim=dim, K=K, R=R))
                layers.append(FFCoupling(dim=dim, K=K, R=R, reverse=True))
            else:
                for d in range(dim):
                    layers.append(FFCoupling(dim=dim, K=K, R=R, split_dims=[m for m in range(dim) if m!=d]))

        self.transf = NFCompose(*layers)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return self.transf.logdet(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the forward transformation of the normalizing flow
        :param x: inputs as a torch tensor with shape [N, dim]
        :return: the transformed inputs, a tensor with shape [N, dim]
        """
        return self.transf.forward(x)

    def jvp(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian vector product (jvp) of the forward process at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f, a tensor with shape [N, dim]
        """
        return self.transf.jvp(x, f)

    def forward_jvp_logdet(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the forward operation, the jvp and the logdet of the transformation at point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose jvp needs to be calculated, a torch tensor with shape [N, dim]
        :return: a tuple:
                    - the transformed input, y=H(x), a tensor with shape [N, dim]
                    - the Jacobian at point x multiplied by the vector f: dH(x)/dx @ f, a tensor with shape [N, dim]
                    - the log-determinant of the Jacobian at point x, a tensor with shape [N,]
        """
        return self.transf.forward_jvp_logdet(x, f)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation of the normalizing flow
        :param y: inputs as a torch tensor with shape [N, dim]
        :return: the transformed inputs, a tensor with shape [N, dim]
        """
        return self.transf.reverse(y)

    def inv_jvp(self, y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse-Jacobian vector product (i-jvp) of the reverse process at the point y
        :param y: a torch tensor with shape [N, dim] where N is the number of points
        :param f: the vector whose i-jvp  needs to be calculated, a torch tensor with shape [N, dim]
        :return: the inverse-Jacobian at point y multiplied by the vector f: dH^-1(y)/dy @ f
        """
        return self.transf.inv_jvp(y, f)
