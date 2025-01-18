import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from typing import Tuple


class RFFCoupling(nn.Module):
    """
    An implementation of the invertible Fourier features transform, given by:
                    f(x_1, x_2) = (x_1, x_2 * exp(s(x_1)) + t(x_1))
    where the functions s(x_1) and t(x_1) are both defined according to the Fourier features:
                    t(x) = sum{ a_k cos(2*pi*k*x/R + b_k) } with k between 0 and K
    where a_k are the coefficients of the Fourier features and b_k are the phases. R defines the natural scale of the
    function and K is the number of Fourier components to be used.
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
        self.W_s = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1]))
        self.W_t = nn.Parameter(1e-3*torch.randn(K, x1.shape[-1]))
        self.g_s = nn.Parameter(1e-3*torch.randn(x2.shape[-1], K))
        self.g_t = nn.Parameter(1e-3*torch.randn(x2.shape[-1], K))
        self.b_s = nn.Parameter(torch.rand(K)*2*np.pi*1e-2)
        self.b_t = nn.Parameter(torch.rand(K)*2*np.pi*1e-2)

    @staticmethod
    def _ff(x: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: the output layer, a torch tensor with shape [dim_out, K]
        :param W: the hidden weights matrix, a torch tensor with shape [K, dim_in]
        :param phi: the bias, a torch tensor with shape [K]
        :return: the transformed input x, a tensor with shape [N, dim_out]
        """
        return torch.cos(x@W.T+phi[None])@gamma.T

    @staticmethod
    def _df(x: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim_in] the input to the transformation
        :param gamma: the output layer, a torch tensor with shape [dim_out, K]
        :param W: the hidden weights matrix, a torch tensor with shape [K, dim_in]
        :param phi: the bias, a torch tensor with shape [K]
        :return: the Jacobian of the FF function, with shape [N, dim_in, dim_out]
        """
        sin = torch.sin(x@W.T+phi[None])  # [N, k]
        interm = W.T[None]*sin[:, None, :]  # [N, dim_in, k]
        return - interm@gamma.T  # [N, dim_in, dim_out]

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
        self.W_s.requires_grad = freeze
        self.b_s.requires_grad = freeze
        self.g_s.requires_grad = freeze

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.W_t.data = torch.randn_like(self.W_t)*amnt
        self.b_t.data = torch.randn_like(self.b_t)*amnt
        self.g_t.data = torch.randn_like(self.g_t)*amnt

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


class FFCoupling(nn.Module):
    """
    An implementation of the invertible Fourier features transform, given by:
                    f(x_1, x_2) = (x_1, x_2 * exp(s(x_1)) + t(x_1))
    where the functions s(x_1) and t(x_1) are both defined according to the Fourier features:
                    t(x) = sum{ a_k cos(2*pi*k*x/R + b_k) } with k between 0 and K
    where a_k are the coefficients of the Fourier features and b_k are the phases. R defines the natural scale of the
    function and K is the number of Fourier components to be used.
    """
    def __init__(self, dim: int, K: int=32, R: int=10, reverse: bool=False):
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
        self.a_s = nn.Parameter(1e-2*torch.randn(K, x1.shape[-1], x2.shape[-1]))
        self.a_t = nn.Parameter(1e-2*torch.randn(K, x1.shape[-1], x2.shape[-1]))
        self.b_s = nn.Parameter(torch.rand(K, x1.shape[-1])*2*np.pi*1e-2)
        self.b_t = nn.Parameter(torch.rand(K, x1.shape[-1])*2*np.pi*1e-2)
        self.R = R

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
        return torch.sum(torch.cos(freqs[None, :, None]*x[:, None] + phi[None])[..., None] * gamma[None], dim=(1, -2))

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
        sins = - freqs[None, :, None]*torch.sin(freqs[None, :, None]*x[:, None] + phi[None])  # [N, K, dim_in]
        return torch.sum(sins[..., None]*gamma[None], dim=1)  # [N, dim_in, dim_out]

    def _s(self, x1: torch.Tensor, rev: bool=False) -> torch.Tensor:
        return torch.exp(self._ff(x1, self.a_s, self.b_s, self.R)*(-1 if rev else 1))

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        return self._ff(x1, self.a_t, self.b_t, self.R)

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
        self.a_s.requires_grad = freeze
        self.b_s.requires_grad = freeze
        self.a_t.requires_grad = freeze
        self.b_t.requires_grad = freeze

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.b_s.data = np.pi*torch.rand_like(self.b_s)*amnt
        self.b_t.data = np.pi*torch.rand_like(self.b_t)*amnt
        self.a_t.data = torch.randn_like(self.a_t)*amnt/np.sum(self.a_t.shape)
        self.a_s.data = torch.randn_like(self.a_t)*amnt*1e-2/np.sum(self.a_s.shape)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        return self._ff(x1, self.a_s, self.b_s, self.R).sum(dim=1)

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


class FFCoupling_scalefree(nn.Module):
    """
    An implementation of the scale-free invertible Fourier features transform, given by:
                    f(x_1, x_2) = (x_1, x_2 + t(x_1))
    where the functions s(x_1) and t(x_1) are both defined according to the Fourier features:
                    t(x) = sum{ a_k cos(2*pi*k*x/R + b_k) } with k between 0 and K
    where a_k are the coefficients of the Fourier features and b_k are the phases. R defines the natural scale of the
    function and K is the number of Fourier components to be used.
    """
    def __init__(self, dim: int, K: int=5, R: int=10, reverse: bool=False):
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
        self.a_t = nn.Parameter(1e-2*torch.randn(K, x1.shape[-1], x2.shape[-1]))
        self.b_t = nn.Parameter(torch.rand(K, x1.shape[-1])*2*np.pi*1e-2)
        self.R = R

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
        return torch.sum(torch.cos(freqs[None, :, None]*x[:, None] + phi[None])[..., None] * gamma[None], dim=(1, -2))

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
        sins = - freqs[None, :, None]*torch.sin(freqs[None, :, None]*x[:, None] + phi[None])  # [N, K, dim_in]
        return torch.sum(sins[..., None]*gamma[None], dim=1)  # [N, dim_in, dim_out]

    def _t(self, x1: torch.Tensor) -> torch.Tensor:
        return self._ff(x1, self.a_t, self.b_t, self.R)

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
        self.a_t.requires_grad = freeze
        self.b_t.requires_grad = freeze

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.b_t.data = np.pi*torch.rand_like(self.b_t)*amnt
        self.a_t.data = torch.randn_like(self.a_t)*amnt/np.sum(self.a_t.shape)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        return torch.zeros(x.shape[0], device=x.device)

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
        return self._cat(x1, x2 + self._t(x1))

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
        t = self._t(x1)
        y = self._cat(x1, x2 + t)

        f1, f2 = self._split(f)  # f1 shape: [N, dim_in], f2 shape: [N, dim_out]
        dt = self._df(x1, self.a_t, self.b_t, self.R)  # [N, dim_in, dim_out]
        Jf2 = (dt.transpose(-2, -1)@f1[..., None])[..., 0] + f2
        Jf = self._cat(f1, Jf2)

        logdet = torch.zeros(x.shape[0], device=x.device)

        return y, Jf, logdet

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        y1, y2 = self._split(y)
        return self._cat(y1, y2 - self._t(y1))


class AffineCoupling(nn.Module):

    def __init__(self, dim: int, K: int=32, reverse: bool=False):
        """
        Initializes an AffineCoupling layer with MLPs with single hidden layers
        :param dim: number of dimensions expected for inputs
        :param K: width of MLP
        :param reverse:
        """
        super().__init__()
        self.reord = reverse
        x1, x2 = self._split(torch.ones(1, dim))

        self.s = nn.Sequential(nn.Linear(x1.shape[-1], K), nn.GELU(), nn.Linear(K, K),
                               nn.GELU(), nn.Linear(K, x2.shape[-1]))
        for param in self.s.parameters(): param.data = 1e-3*torch.randn_like(param.data)

        self.t = nn.Sequential(nn.Linear(x1.shape[-1], K), nn.GELU(), nn.Linear(K, K),
                               nn.GELU(), nn.Linear(K, x2.shape[-1]))
        for param in self.t.parameters(): param.data = 1e-3*torch.randn_like(param.data)

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

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        for param in self.t.parameters(): param.data = torch.randn_like(param)*amnt

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        x1, _ = self._split(x)
        return self.s(x1).sum(dim=1)

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

    def _Jf(self, x1: torch.Tensor, x2: torch.Tensor, f: torch.Tensor):
        f1, f2 = self._split(f.clone())
        x = torch.clone(x1).requires_grad_(True)
        jf1 = torch.func.jvp(lambda x: self.s(x)*x2+self.t(x), (x, ), (f1, ))[1]
        return self._cat(f1, jf1 + self.s(x)*f2)

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


class PPT(nn.Module):
    """
    An implementation of the invertible positive-power transform, given by:
                    f(x) = sign(x) * |x| ^ beta
    where beta > 0. When beta < 1, this transform expands everything near the origin and "squishes" everything else, and
    does the opposite when beta > 1. The parameter beta is defined per coordinate and is parameterized as
    beta = exp(alpha) in order for beta to always be positive.
    """
    def __init__(self, dim: int):
        """
        Initializes the positive-power transform layer
        :param dim: number of dimensions expected for inputs
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(dim)*1e-2)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log-abs-determinant of the Jacobian evaluated at the point x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N,] of the log-abs-determinants of the Jacobian for each point x
        """
        beta = torch.exp(self.alpha)
        return torch.sum((beta[None]-1)*(self.alpha[None] + torch.log(torch.abs(x))), dim=-1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Jacobian of the transform at points x
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim, dim] of the Jacobians evaluated at each point x
        """
        beta = torch.exp(self.alpha)
        return torch.diag_embed(beta[None]*torch.pow(torch.abs(x), beta[None]-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch tensor with shape [N, dim] where N is the number of points
        :return: a torch tensor with shape [N, dim] of the transformed points
        """
        beta = torch.exp(self.alpha)
        return torch.sign(x)*torch.pow(torch.abs(x), beta[None, :])

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
        beta = torch.exp(self.alpha)
        y = torch.sign(x)*torch.pow(torch.abs(x), beta[None, :])
        J = torch.pow(torch.abs(x), beta[None, :]-1)*beta[None, :]
        return y, J*f, self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reverse transformation
        :param y: a torch tensor with shape [N, dim]
        :return: a torch tensor with shape [N, dim] which are the points after applying the reverse transformation
        """
        beta = torch.exp(-self.alpha)
        return torch.sign(y)*torch.pow(torch.abs(y), beta[None, :])


class Affine(nn.Module):

    def __init__(self, dim: int, rank: int=None, data_init: bool=False):
        super().__init__()
        if rank is None: rank = dim
        self.mu = nn.Parameter(torch.randn(dim)*1e-3)
        self.W = nn.Parameter(torch.randn(dim, rank)*1e-3)
        self.phi = nn.Parameter(torch.randn(dim)*1e-3)
        self.register_buffer('data_init', torch.ones(1) if data_init else torch.zeros(1))

    def _data_init(self, x: torch.Tensor):
        q = self.W.shape[-1]
        mean = torch.mean(x, dim=0)
        self.mu.data = mean
        y = x-mean[None]
        U, s, _ = torch.linalg.svd(y.T, full_matrices=False)
        self.W.data = U[:, :q]
        self.phi.data[:] = -3
        self.data_init.data[:] = 0

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.mu.data = torch.randn_like(self.mu)*amnt
        self.phi.data = torch.randn_like(self.phi)*amnt
        W = torch.randn_like(self.W)
        W = W@W.T
        W, _, _ = torch.svd(W)
        self.W.data = W[:, :self.W.shape[-1]]

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        return torch.linalg.slogdet(self.W@self.W.T + torch.diag(phi))[1]

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.exp(self.phi)
        return (self.W@self.W.T + torch.diag(phi))[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_init[0] == 1: self._data_init(x)
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


class LogTransf(nn.Module):

    def __init__(self, precision: float=1e-5):
        super().__init__()
        self.prec = precision

    def freeze_scale(self, freeze: bool = True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        pass

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        pass

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(-torch.log(x+self.prec), dim=-1)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(self._s())[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x+self.prec)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.log(x+self.prec), f/(x+self.prec), self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(y)-self.prec


class ActNorm(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('mu', nn.Parameter(torch.zeros(dim)))
        self.register_buffer('s', nn.Parameter(torch.zeros(dim)))
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

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.newinit.data[:] = 0
        self.s.data = 2*(torch.rand_like(self.s) - 1)*amnt
        self.mu.data = torch.randn_like(self.mu)*amnt

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(-self.s)[None]

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diag_embed(torch.exp(-self.s))[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.newinit[0] == 1: self._newinit(x)
        return torch.exp(-self.s)[None]*(x-self.mu[None])

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(x), torch.exp(-self.s)[None]*f, self.logdet(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.s)[None]*y + self.mu[None]


class NFCompose(nn.Module):

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.transfs = nn.ModuleList(modules)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        for mod in self.transfs: mod.freeze_scale(freeze)

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        for mod in self.transfs: mod.rand_init(amnt)

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


class Diffeo(nn.Module):

    def __init__(self, dim: int, rank: int=None, n_layers: int=4, K: int=15, add_log: bool=False,
                 MLP: bool=False, actnorm: bool=True, RFF: bool=False, scale_free: bool=False,
                 affine_init: bool=False):
        super().__init__()
        if rank is None: rank = dim

        layers = []
        if add_log: layers.append(LogTransf())
        if actnorm: layers.append(ActNorm(dim))
        layers.append(Affine(dim=dim, rank=dim, data_init=affine_init))
        for i in range(n_layers):
            layers.append(Affine(dim=dim, rank=rank))
            if MLP:
                layers.append(AffineCoupling(dim=dim, K=K))
                layers.append(AffineCoupling(dim=dim, K=K, reverse=True))
            elif RFF:
                layers.append(RFFCoupling(dim=dim, K=K))
                layers.append(RFFCoupling(dim=dim, K=K, reverse=True))
            elif scale_free:
                layers.append(FFCoupling_scalefree(dim=dim, K=K, R=10))
                layers.append(FFCoupling_scalefree(dim=dim, K=K, R=10, reverse=True))
            else:
                layers.append(FFCoupling(dim=dim, K=K, R=10))
                layers.append(FFCoupling(dim=dim, K=K, R=10, reverse=True))

        self.transf = NFCompose(*layers)

    def freeze_scale(self, freeze: bool=True):
        """
        Freeze all parameters that impact log-determinant
        :param freeze: a boolean indicating whether to freeze (True) or unfreeze (False)
        """
        self.transf.freeze_scale(freeze)

    def rand_init(self, amnt: float):
        """
        Random initialization of the layer
        :param amnt: a float with the strength of the initialization
        """
        self.transf.rand_init(amnt)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return self.transf.jacobian(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transf.forward(x)

    def jvp_forward(self, x: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.transf.jvp_forward(x, f)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.transf.reverse(y)

    def fit(self, x: torch.Tensor, y: torch.Tensor, lamb: float=0, lr: float=5e-3, its: int=100):
        for param in self.parameters(): param.requires_grad = True
        optim = Adam(self.parameters(), lr=lr)
        for i in range(its):
            optim.zero_grad()
            pred, _, det = self.jvp_forward(x, x)
            loss = torch.mean((pred-y)**2) + lamb*torch.abs(det).mean()
            loss.backward()
            optim.step()
        for param in self.parameters(): param.requires_grad = False
        return self
