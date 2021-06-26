from qutip import sigmax, basis, sesolve, sigmaz, jmat, Qobj
import numpy as np
from src_clean.krylovsolver import krylovsolve


H = 2*np.pi * 0.5 * jmat(30,'y')
psi0 = np.random.random(H.shape[0]) + 1j * np.random.random(H.shape[0])
psi0 = psi0/np.linalg.norm(psi0)
psi0 = Qobj(psi0)

times = np.linspace(0.0, 10.0, 10)

result = sesolve(H, psi0, times, [jmat(30,'z'), jmat(30, '+')])
result_k = krylovsolve(H, psi0=psi0, tlist=times, krylov_dim=10, e_ops=[jmat(30,'z'), jmat(30, '+')])

print(f'Difference between Krylov and Sesolve Z expectation value {result.expect[0][-1]-result_k.expect[0][-1]}')
print(f'Difference between Krylov and Sesolve + expectation value {result.expect[1][-1]-result_k.expect[1][-1]}')
