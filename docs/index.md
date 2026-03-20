# PyTao

**Tao** is an open source general purpose program for charged particle and X-ray
simulations in accelerators and storage rings. It is built on top of the _Bmad_
toolkit (software library) which provides the needed computational routines
needed to do simulations. Essentially you can think of _Tao_ as a car and _Bmad_
as the engine that powers the car. In fact _Bmad_ powers a number of other
simulation programs but that is getting outside of the scope of this manual.
It is sometimes convenient to be able to run _Tao_ via Python. For example, in an
online control system environment.

**PyTao** is wrapper on top of _Tao_ and allow users to access the _Tao_ library
via its shared library `libtao.so` via
[ctypes](https://docs.python.org/3/library/ctypes.html).

Documentation for _Bmad_ and _Tao_, as well as information for downloading the
code if needed is given on the [Bmad web site](https://www.classe.cornell.edu/bmad)
