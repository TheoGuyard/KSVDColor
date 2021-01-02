import numpy as np

def matrix_omp(A, Y, tol=None, nnz=None):
    """
    Orthogonal Matching Pursuit algorithm when the signal is a matrix.

    Parameters
    ----------
    A : np.ndarray with 2 dims
        Input dictionary. Columns are assumed to have unit norm.
    Y : np.array with 2 dims
        Input targets
    nnz : int
        Targeted number of non-zero elements.
    tol : float
        Targeted precision.
    """

    X = np.zeros((A.shape[1], Y.shape[1]))

    for k in range(Y.shape[1]):
        X_k, idx_k = omp(A, Y[:,k], tol, nnz)
        X[idx_k,k] = X_k

    return X

def omp(A, y, tol=None, nnz=None):  
    """
    Orthogonal Matching Pursuit algorithm.

    Parameters
    ----------
    A : np.ndarray with 2 dims
        Input dictionary. Columns are assumed to have unit norm.
    y : np.array with 1 dim
        Input targets
    nnz : int
        Targeted number of non-zero elements.
    tol : float
        Targeted precision.
    """

    m, n = A.shape

    if nnz is None and tol is None:
        raise ValueError("Either nnz or tol must be specified.")
    if nnz is not None and tol is not None:
        tol = None
    if nnz is None:
        nnz = n
    elif nnz > n:
        raise ValueError("Parameter nnz exceed A.shape[1].")
    if tol is None:
        tol = 0

    idx = []
    n_active = 0
    Aty = A.T @ y
    res = y

    while True:
        k = np.argmax(np.abs(A.T @ res))

        if k in idx:
            print("warning : OMP stopped because the same atom was selected twice.")
            break
        
        n_active += 1
        idx.append(k)
        x_idx = np.linalg.lstsq(A[:,idx], y, rcond=None)[0]
        res = y - A[:,idx] @ x_idx

        if np.linalg.norm(res) < tol or n_active >= nnz:
            break

    return x_idx, idx