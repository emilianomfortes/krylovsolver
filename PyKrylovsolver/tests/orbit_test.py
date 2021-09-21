from qutip import mcsolve, tensor, fock, qeye, destroy, Options
# from pylab import *
import numpy as np
from PyKrylovsolver.krylovsolve import krylovsolve
from matplotlib import pyplot as plt

if __name__ == '__main__':
    times = np.linspace(0.0, 10.0, 200)
    psi0 = tensor(fock(2, 0), fock(10, 5))
    a = tensor(qeye(2), destroy(10))
    sm = tensor(destroy(2), qeye(10))

    H = 2 * np.pi * a.dag() * a + 2 * np.pi * sm.dag() * sm + \
        2 * np.pi * 0.25 * (sm * a.dag() + sm.dag() * a)

    # run Monte Carlo solver
    psi_mcsolve = mcsolve(H, psi0, times,  # [np.sqrt(0.1) * a],
                          e_ops=[a.dag() * a, sm.dag() * sm], options=Options(store_states=True))

    psi_mcsolve_states = mcsolve(H, psi0, times)

    psi_krylov = krylovsolve(H, psi0, tlist=times,
                             e_ops=[a.dag() * a, sm.dag() * sm],
                             tolerance=1e-3, krylov_dim=3,
                             progress_bar=False, sparse=True, store_states=True)

    plt.plot(times, psi_mcsolve.expect[0], times, psi_mcsolve.expect[1])
    plt.plot(times, psi_krylov.expect[0], times, psi_krylov.expect[1])

    overlap = 1 - np.abs(np.vdot(psi_mcsolve_states.states[-1], psi_krylov.states[-1])) ** 2
    print(overlap)

    plt.title('Monte Carlo time evolution')
    plt.xlabel('Time')
    plt.ylabel('Expectation values')
    plt.legend(("cavity photon number", "atom excitation probability"))
    plt.savefig('mcsolve_krylov_expectation.png')
    plt.show()
