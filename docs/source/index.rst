PyTao
=====

*Tao* is an open source general purpose program for charged particle and X-ray
simulations in accelerators and storage rings. It is built on top of the *Bmad*
toolkit (software library) which provides the needed computational routines
needed to do simulations. Essentially you can think of *Tao* as a car and *Bmad*
as the engine that powers the car. In fact *Bmad* powers a number of other
simulation programs but that is getting outside of the scope of this manual.

It is sometimes convenient to be able to run *Tao* via Python. For example,in an
online control system environment.

*PyTao* is wrapper on top of *Tao* and allow users to access the *Tao* library
via **ctypes** or **pexpect**.

Documentation for *Bmad* and *Tao*, as well as information for downloading the
code if needed is given on the `Bmad web site <https://www.classe.cornell.edu/bmad>`_.

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user_docs/index
   user_docs/interfaces
   user_docs/gui/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic
   examples/advanced
   examples/lattice
   examples/bunch
   examples/data
   examples/fodo

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   dev_docs/api/index
   dev_docs/documentation
   dev_docs/development

.. toctree::
   :hidden:
   :caption: External Links

   PyTao GitHub <https://github.com/bmad-sim/pytao>
   Bmad website <https://www.classe.cornell.edu/bmad>
   Tao website <https://www.classe.cornell.edu/bmad/tao.html>
