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

 Time evolution of state vectors for time independent Hamiltonians.

 Evolve the state vector ("psi0") finding an approximation for the time 
 evolution operator of Hamiltonian ("H") by obtaining the projection of 
 the time evolution operator on a set of small dimensional Krylov 
 subspaces (m<<dim(H)).

 The output is either the state vector or the expectation values of supplied 
 operators ("e_ops") at arbitrary points in a time range built from inputs 
 "t0", "tf" and "dt". 
 Optionally, a custom ("tlist") without an even time 
 stepping between times can be provided, but the algorithm will become 
 slower. 
 **Additional options**

 Additional options to krylovsolve can be set with the following:

 "store_states": stores states even though expectation values are requested
 via the "e_ops" argument.
 
 "store_final_state": store final state even though expectation values are 
 requested via the "e_ops" argument.


```text
Parameters
   -------------
 H : :class:`qutip.Qobj`
    System Hamiltonian.
 psi0 : :class: `qutip.Qobj`
     initial state vector (ket).

 tlist : None / *list* / *array*
    List of times on which to evolve the initial state. If provided, it overrides
    t0, tf and dt parameters.

 krylov_dim: int
     Dimension of Krylov approximation subspaces used for the time evolution
     approximation.

 e_ops : None / list of :class:`qutip.Qobj`
     Single operator or list of operators for which to evaluate
     expectation values.
     
 if store_states : bool (default False)
     If e_ops is provided, store each state vector corresponding to each time
     in tlist.
     
 store_final_state : bool (default False)
     If e_ops is provided, store the final state vector of the evolution.

 progress_bar : None / BaseProgressBar
     Optional instance of BaseProgressBar, or a subclass thereof, for
     showing the progress of the simulation.          

 sparse : bool (default False)
     Use np.array to represent system Hamiltonians. If True, scipy sparse
     arrays are used instead.

 tolerance : :float: (default 1e-7)
     Minimum bound value for the final state infidelity with respect to 
     the exact state.       

Returns
---------
 result: :class:`qutip.Result`

    An instance of the class :class:`qutip.Result`, which contains
    either an *array* `result.expect` of expectation values for the times
    specified by range('t0', 'tf', 'dt') or `tlist`, or an *array* `result.states` 
    of state vectors corresponding to the times in range('t0', 'tf', 'dt') or
    `tlist` [if `e_ops` is an empty list].        
```

The documentation website is coming up soon.

```python
from PyKrylovsolver.krylovsolver import krylovsolve
from PyKrylovsolver.hamiltonians import  h_ising_transverse
from qutip.qobj import Qobj
import numpy as np

N = 8
dim = 2 ** N
psi = np.random.random(dim) + 1j * np.random.random(dim)
psi = psi / np.linalg.norm(psi)
psi = Qobj(psi)

hx, hz = np.ones(N), 0.5 * np.ones(N)
Jx, Jy, Jz = 0 * np.ones(N), 0 * np.ones(N), np.ones(N)
H = h_ising_transverse(N, hx, hz, Jx, Jy, Jz)

tlist = np.linspace(0, 1, 100)

psi_evolved = krylovsolve(H, psi, tlist=tlist, tolerance=1e-2, krylov_dim=5, progress_bar=False, sparse=True)
```

Krylov Aproximation
-------------------


Contribute
----------

You are most welcome to contribute in the development of the algorithm by forking this repository and sending pull requests, or filing bug reports at the [issues page](https://github.com/emilianomfortes/krylovsolver/issues).
Any code contributions will be acknowledged in the upcoming contributors section in the documentation.


Citing
------------

If you use our error bound approach for the Krylov approximation in your research, please cite the original paper available [here](https://arxiv.org/abs/2107.09805).
