Scripting Interfaces
====================

*PyTao* allow users to interface with *Tao* in one of two ways. One way is using
the ctypes module. The other way is using the pexpect module.

A Web search will point to documentation on ctypes and pexpect.

The advantage of ctypes is that it directly accesses *Tao* code which makes
communication between Python and *Tao* more robust. The disadvantage of ctypes
is that it needs a shared-object version of the *Tao* library.

The disadvantage of pexpect is that it is slower and it is possible for pexpect
to time out waiting for a response from *Tao*.

.. toctree::
   :maxdepth: 1

   tao
   ctypes
   pexpect