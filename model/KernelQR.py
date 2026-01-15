import numpy as np
from cvxopt import solvers
solvers.options['show_progress'] = False  
from cvxopt.solvers import qp
from cvxopt import matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels  import Matern,ExpSineSquared, PairwiseKernel,Product
from sklearn.metrics.pairwise import chi2_kernel,cosine_similarity,laplacian_kernel,linear_kernel,polynomial_kernel,rbf_kernel,sigmoid_kernel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted





class KernelQR(RegressorMixin, BaseEstimator):
    """ Code implementing kernel quantile regression.

    Parameters
    ----------
    quantile : float or array-like
        quantile(s) under consideration

    kernel_type : str, default='gaussian_rbf'
        kind of kernel function    

    gamma : float, default=1
        bandwith parameter of rbf gaussian, laplacian, sigmoid, chi_squared, matern, periodic kernels

    sigma, omega, c, d, nu, p, gammas, var : see original docstring
    """

    def __init__(self, quantiles=0.5, kernel_type="gaussian_rbf", C=1, gamma=1, sigma=None, omega=None, c=None, d=None, nu=None, p=None, gammas=None, var=100.0):
        self.C = C
        self.quantiles = quantiles
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.sigma = sigma
        self.omega = omega
        self.c = c
        self.d = d
        self.nu = nu
        self.p = p
        self.gammas = gammas
        self.var = var

    def kernel(self, X, Y):
        if self.kernel_type == "gaussian_rbf":
            gaussian_rbf = Matern(length_scale=self.gamma, nu=np.inf)
            return self.var * gaussian_rbf(X, Y)
        elif self.kernel_type == "laplacian":
            laplacian_kernel_matern = self.var * Matern(length_scale=self.gamma, nu=0.5)
            return laplacian_kernel_matern(X, Y)
        elif self.kernel_type == "a_laplacian":
            return self.var * laplacian_kernel(X, Y, gamma=1 / self.gamma)
        elif self.kernel_type == "gaussian_rbf_":
            return self.var * rbf_kernel(X, Y, gamma=1 / self.gamma)
        elif self.kernel_type == "linear":
            return linear_kernel(X, Y)
        elif self.kernel_type == "cosine":
            return cosine_similarity(X, Y)
        elif self.kernel_type == "polynomial":
            return polynomial_kernel(X, Y, coef0=self.c, degree=self.d, gamma=1 / self.gamma)
        elif self.kernel_type == "sigmoid":
            return self.var * sigmoid_kernel(X, Y, coef0=self.c, gamma=1 / self.gamma)
        elif self.kernel_type == "matern_0.5":
            matern_kernel = self.var * Matern(length_scale=self.gamma, nu=0.5)
            return matern_kernel(X, Y)
        elif self.kernel_type == "matern_1.5":
            matern_kernel = self.var * Matern(length_scale=self.gamma, nu=1.5)
            return matern_kernel(X, Y)
        elif self.kernel_type == "matern_2.5":
            matern_kernel = self.var * Matern(length_scale=self.gamma, nu=2.5)
            return matern_kernel(X, Y)
        elif self.kernel_type == "chi_squared":
            return self.var * chi2_kernel(X, Y, gamma=1 / self.gamma)
        elif self.kernel_type == "periodic":
            periodic = self.var * ExpSineSquared(length_scale=self.gamma, periodicity=self.p)
            return periodic(X, Y)
        elif self.kernel_type == "gaussian_rbf_x_laplacian":
            return rbf_kernel(X, Y, gamma=1 / self.gamma) * laplacian_kernel(X, Y, gamma=1 / self.sigma)
        elif self.kernel_type == "se_ard":
            se_ard = 1
            for i in range(X.shape[1] - 1):
                se_ard *= laplacian_kernel(X[:, (i + 1)].reshape(-1, 1), Y[:, (i + 1)].reshape(-1, 1), gamma=1 / self.gammas[i])
            return se_ard
        elif self.kernel_type == "laplacian_x_periodic":
            kernel = self.var * Product(ExpSineSquared(length_scale=self.gamma, periodicity=24), PairwiseKernel(metric='laplacian', gamma=1 / self.sigma))
            return kernel(X, Y)
        elif self.kernel_type == "prod_1":
            matern_kernel = self.var * Matern(length_scale=self.gamma, nu=2.5)
            return matern_kernel(X[:, 0].reshape(-1, 1), Y[:, 0].reshape(-1, 1)) * laplacian_kernel(X[:, 0].reshape(-1, 1), Y[:, 0].reshape(-1, 1), gamma=1 / self.sigma)
        elif self.kernel_type == "prod_2":
            prod_2 = self.var * Product(PairwiseKernel(metric='rbf', gamma=1 / self.gamma), PairwiseKernel(metric='laplacian', gamma=1 / self.sigma))
            return prod_2(X, Y)
        else:
            raise NotImplementedError('No implementation for selected kernel')

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.detach().cpu().numpy() if hasattr(X, 'detach') else np.array(X)
        X = X.astype(float)
        if not isinstance(y, np.ndarray):
            y = y.detach().cpu().numpy() if hasattr(y, 'detach') else np.array(y)
        y = y.ravel().astype(float)
        
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if np.isscalar(self.quantiles):
            quantiles = np.array([self.quantiles])
        else:
            quantiles = np.array(self.quantiles).flatten()

        K = self.kernel(self.X_, self.X_)
        Kmat = matrix(K)
        r = matrix(y) * 1.0
        A = matrix(np.ones(y.size)).T
        b = matrix(0.0)
        n = y.size

        self.a_list = []
        self.b_list = []
        for quantile in quantiles:
            G1 = matrix(np.eye(n))
            h1 = matrix(self.C * quantile * np.ones(n))
            G2 = matrix(-np.eye(n))
            h2 = matrix(self.C * (1 - quantile) * np.ones(n))
            G = matrix([G1, G2])
            h = matrix([h1, h2])
            sol = qp(P=Kmat, q=-r, G=G, h=h, A=A, b=b)
            a = np.array(sol["x"]).flatten()
            squared_diff = (a - (self.C * quantile)) ** 2 + (a - (self.C * (quantile - 1))) ** 2
            offshift = int(np.argmin(squared_diff))
            bias = y[offshift] - a.T @ Kmat[:, offshift]
            self.a_list.append(a)
            self.b_list.append(bias)
        self.quantiles_ = quantiles
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_', 'a_list', 'b_list', 'quantiles_'])
        X = check_array(X)
        K = self.kernel(self.X_, X)
        preds = []
        for a, b in zip(self.a_list, self.b_list):
            preds.append(a.T @ K + b)
        return np.vstack(preds).T



