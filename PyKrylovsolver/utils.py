from scipy.linalg import eigh
import numpy as np


def illinois_algorithm(f, a, b, y, margin=.00_001):
    ''' Bracketed approach of Root-finding with illinois method
    Parameters
    ----------
    f: callable, continuous function
    a: float, lower bound to be searched
    b: float, upper bound to be searched
    y: float, target value
    margin: float, margin of error in absolute term
    Returns
    -------
    A float c, where f(c) is within the margin of y
    '''

    lower = f(a)
    upper = f(b)

    assert lower <= y, f"y is smaller than the lower bound. {y} < {lower}"

    if upper <= y:
        return b

    stagnant = 0

    while 1:
        c = ((a * (upper - y)) - (b * (lower - y))) / (upper - lower)

        y_c = f(c)
        if abs((y_c) - y) < margin:
            # found!
            return c
        elif y < y_c:
            b, upper = c, y_c
            if stagnant == -1:
                # Lower bound is stagnant!
                lower += (y - lower) / 2
            stagnant = -1
        else:
            a, lower = c, y_c
            if stagnant == 1:
                # Upper bound is stagnant!
                upper -= (upper - y) / 2
            stagnant = 1


def optimizer(T, krylov_basis, tlist, tol, method="accuracy"):
    f = bound_function(T, krylov_basis=krylov_basis, t0=tlist[0], tf=tlist[-1])
    n = illinois_algorithm(f, a=tlist[0], b=tlist[-1], y=np.log10(tol), margin=0.1)
    return n


def bound_function(T, krylov_basis, t0, tf):
    eigenvalues1, eigenvectors1 = eigh(T[0:, 0:])
    U1 = np.matmul(krylov_basis[0:, 0:].T, eigenvectors1)
    e01 = eigenvectors1.conj().T[:, 0]

    if method == "accuracy":
        eigenvalues2, eigenvectors2 = eigh(T[0:-1, 0:T.shape[1] - 1])
        U2 = np.matmul(krylov_basis[0:-1, :].T, eigenvectors2)
        e02 = eigenvectors2.conj().T[:, 0]
    
        def f(t):
            delta_t = -1j * (t - t0)

            aux1 = np.multiply(np.exp(delta_t * eigenvalues1), e01)
            psi1 = np.matmul(U1, aux1)

            aux2 = np.multiply(np.exp(delta_t * eigenvalues2), e02)
            psi2 = np.matmul(U2, aux2)

            error = np.linalg.norm(psi1 - psi2)

            steps = 1 if t == t0 else max(1, tf // (t - t0))
            return np.log10(error) + np.log10(steps)
 
    if method == "loschmidt_echo":
        alphas = np.diagonal(T, offset=0).real
        betas = np.diagonal(T, offset=1).real
        alpha_mean = np.mean(alphas)
        beta_mean = np.mean(betas)
        
        _dim = T.shape[0]
        psi_extend = np.zeros(_dim + 1, dtype=complex)
        psi_extend[0] = 1 + 0*1j
        
        T_extend = np.zeros((_dim+1, _dim+1), dtype=complex)
        T_extend[:_dim, :_dim] = T
        T_extend[-2, -1] = beta_mean
        T_extend[-1, -2] = beta_mean
        T_extend[-1, -1] = alpha_mean
        
        eigenvalues2, eigenvectors2 = eigh(T_extend)
        U2 = np.exp(np.outer(-1j * eigenvalues, tlist))
        e02 = eigenvectors2.conj().T[:, 0]
        def f(t):
            delta_t = -1j * (t - t0)
    return f


def _make_partitions(tlist, n_timesteps):
    if n_timesteps == 1:
        partitions = [np.insert(tlist, 0, tlist[0])]
        return partitions
    n_timesteps += 1
    krylov_tlist = np.linspace(tlist[0], tlist[-1], n_timesteps)
    krylov_partitions = [np.array(krylov_tlist[i: i + 2]) for i in range(n_timesteps - 1)]
    partitions = []
    _tlist = np.copy(tlist)
    for krylov_partition in krylov_partitions:
        start = krylov_partition[0]
        end = krylov_partition[-1]
        condition = _tlist <= end
        partitions.append([start] + _tlist[condition].tolist() + [end])
        _tlist = _tlist[~condition]
    return partitions


def _a_posteriori_err_saad(_T_m, _psi_norm: float = 1):
    err = (_T_m[-1, -2] * np.abs(_psi_norm * expm(-1j * _T_m)[-1, 0])).real
    return err


def _happy_breakdown(T_m, v, beta, w, j):
    v = v[0:j + 1, :]
    v[j + 1, :] = w / beta

    T_m = T_m[0:j, 0:j]

    T_m[j + 2, j + 1] = beta

    return v, T_m
