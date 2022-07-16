import numpy as np

class Luvilian:
    def __init__(self, H) -> None:
        self.hamiltonian = H
    def __call__(self, O) -> np.ndarray:
        assert O.shape == self.hamiltonian.shape, "Operator shape does not match Hamiltonian shape"
        res = np.matmul(self.hamiltonian, O) - np.matmul(O, self.hamiltonian)
        return res

def matrix_inner_product(a, b):
    res = np.einsum('ji,ji->', a.conj(), b)
    # res = (a.T.conj()*b).sum()
    # res = np.trace(np.matmul(a, b))
    # res = np.trace(np.matmul(a.T.conj(), b))
    return res / a.shape[0]

def matrix_norm(a):
    return np.sqrt(matrix_inner_product(a,a))

class Lanczos:
    
    def __init__(self, inner_product, normalization, ortonormalization_order):
        self.inner_product = inner_product
        self.normalization = normalization
        self.ortonormalization_order = ortonormalization_order
        self.gram_schmidt = GramSchmidt(inner_product, normalization)
    
    def partial_lanczos(self, action_h, v0, itterations, base, betas, alphas):
        w0 = v0 / self.normalization(v0)
        base.append(w0)
        
        for _ in range(itterations):
            next_vector, next_beta, next_alpha = self._get_next_element(action_h, base)
            if np.abs(next_beta) < 1e-5:
                break
            base.append(next_vector)
            betas.append(next_beta)
            alphas.append(next_alpha)
        return base, betas, alphas

    def __call__(self, action_h, v0, itterations):
        base = list()
        betas = list()
        alphas = list()
        base, betas, alphas = self.partial_lanczos(action_h, v0, itterations, base, betas, alphas)
        return base, betas, alphas 
    
    def _get_next_element(self, action_h, base):
        w = action_h(base[-1])
        an = self.inner_product(w, base[-1])
        
        for _ in range(self.ortonormalization_order):
            w = self.gram_schmidt(w, base, normalize=False)
        bn = self.normalization(w)
        return w / bn, bn, an

class GramSchmidt:
    
    def __init__(self, inner_product, normalization):
        self.inner_product = inner_product
        self.normalization = normalization
    
    def __call__(self, v: np.array, w: list, normalize=True):
        """
        Gram-Schmidt orthonormalization
        """
        # for u in w:
        #     v < v - u <u,v>
        
        
        z = np.array(list(map(lambda x: x * self.inner_product(x, v)/self.normalization(x)**2, w)))
        # v - u <u,v> / <u,u>
        u = v - np.sum(z,axis=0)
        if normalize:
            u = u / self.normalization(u)
        return u


def test_gd():
    gs = GramSchmidt(np.dot, np.linalg.norm)
    w = ( np.array([1,0,0]), np.array([0,1,0]) )
    v = (1,2,3)
    res = gs(v, w)
    print(res, np.dot(res,w[0]), np.dot(res,w[1]))

import numpy as np
from qutip import sigmax, sigmay, sigmaz, tensor, qeye


def h_ising_transverse(N: int, hx: float, hz: float, jx: float, jy: float,
                       jz: float, to_numpy=False):
    params = N, *(x * np.ones(N) for x in [hx, hz, jx, jy, jz])
    h = _h_ising_transverse(*params)
    h = h.unit()
    if to_numpy:
        h = h.full()
    return h


def _h_ising_transverse(N: int, hx: np.ndarray, hz: np.ndarray, jx: np.ndarray, jy: np.ndarray, jz: list):
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
        H += -jx[n] * sx_list[n] * sx_list[n + 1]
        H += -jy[n] * sy_list[n] * sy_list[n + 1]
        H += -jz[n] * sz_list[n] * sz_list[n + 1]

    return H

from matplotlib import pyplot as plt

if __name__ == '__main__':
    ising_params = {
        'N': 4,
        'hx': 1,
        'hz': 1,
        'jx': 0,
        'jy': 0,
        'jz': 1,
    }
    h = h_ising_transverse(**ising_params)
    luvilian = Luvilian(h.full())
    
    op0 = tensor([sigmaz()] * ising_params['N']).full()
    base , beta, _ = Lanczos(matrix_inner_product,matrix_norm,2)(luvilian, op0, 100)
    
    # elemento ij
    # base[i] > operador

    # L_{ij} = matrix_inner_product(base[j], luvilian(base[i])) 
    
    L = np.zeros([len(base), len(base)])
    for i in range(len(base)):
        for j in range(len(base)):
            L[i,j] = matrix_inner_product(base[j], luvilian(base[i])) 

    L[L<1e-16] = 0

    print(L[-1,-2] - beta[-1])
    # beta = np.array(beta)

    # np.save('./betas_julian.npy', beta)

    # plt.plot(beta)
    # plt.show()