import numpy as np
import scipy as sp
from .omp import matrix_omp


class KSVD:
    """
    Implementation of the KSVD algorithm discribed in [1]. The truncated SVD
    operation is approximated using the algorithm in [2].
    
    References :
    ------------
    [1] : 'Sparse Representation for Color Image Restoration. J. Mairal, 
        M. Elad, G. Sapiro'.
    [2] : 'Efficient Implementation of the K-SVD Algorithm using Batch 
        Orthogonal Matching Pursuit. R. Rubinstein, M. Zibulevsky and M. Elad'.

    """

    def __init__(self, k, maxiter, omp_tol=None, omp_nnz=None, param_a=0.):
        """
        Parameters
        ----------
        k : int 
            Number of dictionary atoms
        maxiter : int
            Maximum number of iterations
        omp_tol : float
            Target precision in OMP
        omp_nnz : int
            Target number of non-zero coefficients in OMP
        param_a : float
           Correction parameter a (where gamma = 2a+a^2) for the modified scalar
           product.
        """
        self.k = k
        self.maxiter = maxiter
        self.omp_tol = omp_tol
        self.omp_nnz = omp_nnz
        self.dictionary = None
        self.param_a = param_a
        self.modified_metric_matrix = None
        
        if omp_tol is None and omp_nnz is None:
            raise ValueError("Either omp_tol or omp_nnz must be specified.")

    def _update_dict(self, Y, A, alpha):
        for j in range(self.k):
            I = alpha[:, j] > 0
            if np.sum(I) == 0:
                continue

            A[j, :] = 0
            g = alpha[I, j].T
            E = Y[I, :] - alpha[I, :] @ A
            d = E.T @ g
            d /= np.linalg.norm(d)
            g = E @ d
            A[j, :] = d
            alpha[I, j] = g.T
        return A, alpha

    def _initialize(self, Y):
        if min(Y.shape) < self.k:
            A = np.random.randn(self.k, Y.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(Y, k=self.k)
            A = np.diag(s) @ vt
        A /= np.linalg.norm(A, axis=1)[:, np.newaxis]

        # Metric modification matrix
        n = A.shape[1]
        I = np.eye(n)
        J = sp.ones((int(n/3),int(n/3)))
        K = sp.linalg.block_diag(J, J, J)
        self.modified_metric_matrix = I + (self.param_a/n) * K

        return A

    def _denoise(self, A, Y):
        
        A =  A @ self.modified_metric_matrix
        Y = Y @ self.modified_metric_matrix

        return matrix_omp(A.T, Y.T, tol=self.omp_tol, nnz=self.omp_nnz).T

    def learn_dictionary(self, Y):
        """
        Fit a dictionary for a given input signal.

        Parameters
        ----------
        Y : np.ndarray with 2 dims
            Input signal.
        """
        print("KSVD algorithm")
        print("--------------")
        A = self._initialize(Y)
        for i in range(self.maxiter):
            print(f"Iteration {i+1}/{self.maxiter}")
            print("  Sparse coding (OMP) step ...")
            alpha = self._denoise(A, Y)
            print("  Dictionary update step ...")
            A, alpha = self._update_dict(Y, A, alpha)

        self.dictionary = A

    def denoise(self, Y):
        """
        Find the sparse representation of the input signal.

        Parameters
        ----------
        Y : np.ndarray with 2 dims
            Input signal.
        """

        if self.dictionary is None:
            raise ValueError("Dictionary not learned yet, consider `learn_dictionary(Y)` first.")

        print("Denoising signal ...")
        return self._denoise(self.dictionary, Y)