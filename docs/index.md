# PyTao

**PyTao** is wrapper on top of the particle accelerator design and simulation tool [_Tao_](https://www.classe.cornell.edu/bmad/tao.html).
It allows users to access _Tao_ via its shared library `libtao.so` using [ctypes](https://docs.python.org/3/library/ctypes.html).

## Tao and Bmad
**Tao** is an open source general purpose program for charged particle and X-ray
simulations in particle accelerators and storage rings. It is built on top of the _Bmad_
toolkit (software library) which provides the needed computational routines
needed to do simulations. Essentially you can think of _Tao_ as a car and _Bmad_
as the engine that powers the car. In fact _Bmad_ powers a number of other
simulation programs but that is getting outside of the scope of this manual.
It is sometimes convenient to be able to run _Tao_ via Python. For example, in an
online control system environment.

Documentation for _Bmad_ and _Tao_, as well as information for downloading the
code if needed is given on the [Bmad web site](https://www.classe.cornell.edu/bmad)

## Additional CLI Tools
In addition to the python library itself, `pytao` ships with some CLI tools useful for interacting with and maintaining Bmad
lattices using Tao.

- [`pytao`](usage.md#pytao-on-the-command-line): An IPython entrypoint dropping you into an interactive shell with your lattice loaded.
- [`pytao-constraints`](constraints/index.md): A constraint checker and regression testing tool for Bmad lattices built on top of `pytao`.