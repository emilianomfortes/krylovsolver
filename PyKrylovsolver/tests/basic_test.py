from PyKrylovsolver.krylovsolver import krylovsolve
from PyKrylovsolver.hamiltonians import h_sho, h_random, h_ising_transverse
from qutip.qobj import Qobj
from qutip import jmat, sesolve
import numpy as np


def basic_test(N=9):
    
    dim = 2 ** N
    psi0 = np.random.random(dim) + 1j * np.random.random(dim)
    psi0 = psi0 / np.linalg.norm(psi0)
    psi = Qobj(psi0)
    H = h_random(dim)
    H = Qobj(H)
    tlist = np.linspace(0, 10, 10)
    _ = krylovsolve(H, psi, tlist=tlist, tolerance=1e-2, krylov_dim=5, progress_bar=False, sparse=True)


def test_vs_sesolve(N=9, hamiltonian='random'):
    
    dim = 2 ** N
    if hamiltonian == 'random':
        H = h_random(dim)
        H = Qobj(H)
    elif hamiltonian == 'ising':
        hx = np.ones(N)
        hz = 0.5 * np.ones(N)
        Jx, Jy = 0 * np.ones(N), 0 * np.ones(N)
        Jz = np.ones(N)
        H = h_ising_transverse(N, hx, hz, Jx, Jy, Jz).full()
        H = H / np.linalg.norm(H, ord=2)
        H = Qobj(H)
    elif hamiltonian == 'sho':
        H = h_sho(dim)
        H = Qobj(H)
    elif hamiltonian == 'spin_y':
        H = 2 * np.pi * 0.5 * jmat(30, 'y')
        dim = H.shape[0]
    else:
        raise Exception

    psi0 = np.random.random(dim) + 1j * np.random.random(dim)
    psi0 = psi0 / np.linalg.norm(psi0)
    psi = Qobj(psi0)
    tlist = np.linspace(0, 10, 10)
    psi_krylov = krylovsolve(H, psi, tlist=tlist, tolerance=1e-7, krylov_dim=10, progress_bar=False, sparse=True).states
    psi_sesolve = sesolve(H, psi, tlist, progress_bar=None).states

    overlap = 1 - np.abs(np.vdot(psi_sesolve[-1], psi_krylov[-1])) ** 2
    return overlap


if __name__ == '__main__':
    basic_test(N=9)

    diff_random = test_vs_sesolve(N=9, hamiltonian='random')
    diff_ising = test_vs_sesolve(N=9, hamiltonian='ising')
    diff_sho = test_vs_sesolve(N=9, hamiltonian='sho')
    diff_spin_y = test_vs_sesolve(N=9, hamiltonian='spin_y')

    print(f'InFidelity between Krylov and sesolve in Random Hamiltonian: {diff_random:.2e}')
    print(f'InFidelity between Krylov and sesolve in Random Ising: {diff_ising:.2e}')
    print(f'InFidelity between Krylov and sesolve in Random Harmonic Oscillator: {diff_sho:.2e}')
    print(f'InFidelity between Krylov and sesolve in Spin y: {diff_spin_y:.2e}')
