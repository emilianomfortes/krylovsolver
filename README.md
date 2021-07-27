# krylov
# krylov codes to add to qutip once properly tested

[E. M. Fortes](https://github.com/emilianomfortes),
[J. M. Ruffinelli](https://github.com/ruffa),
[M. Larocca](https://scholar.google.com/citations?user=mpQ0hgwAAAAJ&hl=es)
and [D. A. Wisniacki](https://scholar.google.com/citations?user=1tZ7BqoAAAAJ&hl=es).


The Krylov approximation method provides an efficient approach to perform time-evolutions of quantum states for systems with large-dimensional Hilbert spaces. 

The approximation can be described in terms of the Hamiltonian $H$, the initial state $\psi(t_0)$ and the final time $t_f$.



Support
-------

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](https://unitary.fund)

We are proud to have received a grant to develop this open-source code from [Unitary Fund](https://unitary.fund). We have also received funding from the Faculty of Exact and Natural Sciences at the University of Buenos Aires.

Installation
-------------

Dependencies:
```text
qutip
numpy
scipy
```

PyKrylovSolver is not currently in PyPi. You can install directly from GitHub by doing
```
pip install git+https://github.com/emilianomfortes/krylovsolver/
```





Documentation
-------------

The documentation website is coming up soon.

```python
from PyKrylovsolver.krylovsolver import krylovsolve
from qutip.qobj import Qobj
from qutip import jmat
import numpy as np
from PyKrylovsolver.hamiltonians import h_sho, h_random, h_ising_transverse
import numpy as np
from qutip import sesolve


dim = 2 ** N
psi0 = np.random.random(dim) + 1j * np.random.random(dim)
psi0 = psi0 / np.linalg.norm(psi0)
psi = Qobj(psi0)
H = h_random(dim)
H = Qobj(H)
tlist = np.linspace(0, 1, 100)
psi_evolved = krylovsolve(H, psi, tlist=tlist, tolerance=1e-2, krylov_dim=5, progress_bar=False, sparse=True)
```

Contribute
----------

You are most welcome to contribute in the development of the algorithm by forking this repository and sending pull requests, or filing bug reports at the [issues page](https://github.com/emilianomfortes/krylovsolver/issues).
Any code contributions will be acknowledged in the upcoming contributors section in the documentation.


Citing
------------

If you use our error bound approach for the Krylov approximation in your research, please cite the original paper available [here](https://arxiv.org/abs/2107.09805).
