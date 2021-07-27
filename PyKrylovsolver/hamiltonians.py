from qutip import qeye, sigmax, sigmay, sigmaz, tensor, Qobj
from scipy.stats import ortho_group
import numpy as np


def h_sho(dim):
    auts = np.linspace(0, dim, dim, dtype='complex')
    H = np.diag(auts + 1 / 2)
    O = ortho_group.rvs(dim)
    A = np.matmul(O.conj().T, np.matmul(H, O))
    A = A / np.linalg.norm(A, ord=2)
    A = Qobj(A)
    return A


def h_random(dim, distribution="normal"):
    if distribution == "uniform":
        H = np.random.random([dim, dim]) + 1j * np.random.random([dim, dim])
    if distribution == "normal":
        H = np.random.normal(size=[dim, dim]) + 1j * np.random.normal(size=[dim, dim])
    H = (H.conj().T + H) / 2
    H = H / np.linalg.norm(H, ord=2)
    H = Qobj(H)
    return (H)


def h_ising_transverse(N: int,
                       hx: float, hz: float,
                       Jx: float, Jy: float, Jz: float):
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += hz[n] * sz_list[n]

    for n in range(N):
        H += hx[n] * sx_list[n]

    # interaction terms
    for n in range(N - 1):
        H += -  Jx[n] * sx_list[n] * sx_list[n + 1]
        H += -  Jy[n] * sy_list[n] * sy_list[n + 1]
        H += - Jz[n] * sz_list[n] * sz_list[n + 1]
    return H