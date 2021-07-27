from setuptools import setup, find_packages

setup(
    name="PyKrylovsolver",
    url="https://github.com/emilianomfortes/krylovsolver",
    author="Julian M. Ruffinelli, Emiliano M. Fortes, "
    + "Martin Larocca, Diego A. Wisniacki",
    author_email="julianm.ruffinelli@gmail.com, emilianomfortes@gmail.com",
    packages=["PyKrylovsolver"], #find_packages("src"),
    # package_dir={"":"src"},
    install_requires=["numpy", "scipy", "qutip"],
    version="1.0",
    description="This package provides an efficient computation of time evolution for quantum states using the Krylov subspace approximation of the time evolution operator.",
    long_description=open("README.md").read(),
)
