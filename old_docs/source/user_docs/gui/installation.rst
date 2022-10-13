GUI Installation
================

Obtaining the Source Code
-------------------------

Source code and documentation for *Bmad* and *Tao*, if needed, at the *Bmad*
web site at: `https://www.classe.cornell.edu/bmad <https://www.classe.cornell.edu/bmad>`_

Source code for *PyTao*, including the GUI can be obtained at:
`https://www.github.com/bmad-sim/pytao <https://www.github.com/bmad-sim/pytao>`_

Building Tao
------------

As a prerequisite, if not already available, *Tao* must be built before using
the GUI. Build instructions are available on the *Bmad* web site. There are two
ways for the GUI scripts (written in Python) to interact with *Tao*. One way is
to use the \vn{pexpect} module which is a communications layer that interfaces
to *Tao*'s command line interface. The other way is to use ctypes (an
interface between Python and C) to communicate directly with the *Tao* subroutine
library (the *Tao* program is built by linking to the *Tao* library).

The advantage of using ctypes is that it is faster. The drawback is that
ctypes requires a version of the *Tao* library that is shared object.
If you are using a *Bmad* ``Distribution`` (a package that is downloaded from
the Web containing *Bmad*,  *Tao*, associated libraries, etc.), the default is
**not** to build shared object libraries. This default can, of course, be
changed but if you do not have control of how things are built, you may have to
use pexpect. To check if there is a shared object library built, issue the
following command:

.. code-block:: bash

   ls $ACC_ROOT_DIR/production/lib/libtao.*


[This assumes that you are not setting ``ACC_LOCAL_ROOT`` as discussed in
**Environment Variables**]

In all cases you will see a file *libtao.a*. This is a static library which
is always built but not of use. A file with an extension ``.so`` (UNIX) or
``*.dylib`` (Mac) is a shared object library.

Python Requirements
-------------------

Minimum Python version for the GUI is Python 3.6.

The GUI depends upon a number of modules that may have to be downloaded:
- tkinter
- ttk (may be called pyttk)
- pexpect         # If using pexpect instead of ctypes.
- matplotlib
- cycler
- ateutil
- tkagg

.. note::

   The GUI uses the TkAgg backend for matplotlib. There may be a problem with Python finding the
   TkAgg backend. On the mac, using macports, the solution is to install matplotlib with the
   ``tkinter`` variant. Something like:

   .. code-block:: bash

      sudo port uninstall py36-matplotlib           # May not be needed.
      sudo port install  py36-matplotlib +tkinter   # This is when using Python version 3.6

   For more information see: `https://matplotlib.org/tutorials/introductory/usage.html#backends <https://matplotlib.org/tutorials/introductory/usage.html#backends>`_

If one of the modules is missing, python will generate an error message. For example:

.. code-block:: bash

   > python ../../gui/main.py
   Exception in Tkinter callback
   Traceback (most recent call last):
     File "/opt/local/Library/Frameworks/Python.framework/Versions/3.7/lib/
                               python3.7/tkinter/__init__.py", line 1705, in __call__
       return self.func(*args)
     File "../../gui/main.py", line 372, in param_load
       from tao_interface import tao_interface
     File "/Users/dcs16/Bmad/bmad_dist/tao/gui/tao_interface.py", line 4, in <module>
       from tao_pipe import tao_io
     File "/Users/dcs16/Bmad/bmad_dist/tao/python/tao_pexpect/tao_pipe.py", line 14, in <module>
       import pexpect
   ModuleNotFoundError: No module named 'pexpect'

Notice that the last line shows that the pexpect module is needed.

How to install missing modules on the mac: [Note: The exact installation commands will depend upon
which version of python is being used. Use the "python --version" command to see what version you
are using.

Using macports and python 3.6:

.. code-block:: bash

   sudo port install py36-tkinter
   sudo port install py36-pexpect


Using pip (or pip3):

.. code-block:: bash

   sudo pip install pytkk
   sudo pip install pexpect

.. warning::

   It can be dangerous to use pip to install/modify modules in your system
   Python. A much safer way to install the modules you need is to set up a
   python virtual environment.  On Linux, you may also be able to find versions
   of the required modules in your system package manager, which are tailored to
   your Linux distribution and will not break your system python.

Environmental Variables
-----------------------

To run the GUI (or even to run  *Tao* without the GUI), certain environmental varibles must be
set. This is the same initialization that is done when compiling *Bmad* and  *Tao*. See your local Guru or the
*Bmad* web site for more details. To see if the environmental variables have been set, run the
``accinfo`` command.

It may be desireable to specify a local build tree as the place for the python scripts to find the *Tao*
executable or  *Tao* shared object library. To accomplish this, set the environmental variable
``ACC_LOCAL_ROOT`` to the base directory of your local build tree.

.. code-block:: bash

   export ACC_LOCAL_ROOT=/home/dcs16/bmad_dist

The standard place for the GUI script files is at:
``"${ACC_ROOT_DIR}/tao/python/pytao/gui``.
When doing GUI development work, the default location can be changed by setting
``ACC_PYTHONPATH``. Example:

.. code-block:: bash

   export ACC_PYTHONPATH="$ACC_LOCAL_ROOT/tao/python/pytao/gui"

[This assumes that ``ACC_LOCAL_ROOT`` has been set.]
``ACC_PYTHONPATH`` must be set before *Bmad* is initialized. That is, if *Bmad* is initialized in the \vn{.bashrc} file, \vn{ACC_PYTHONPATH} must
be initialized in the \vn{.bashrc} file before the *Bmad* initialization.

To check that ``PYTHONPATH`` has the correct value use the command:

.. code-block:: bash

   printenv |grep PYTHONPATH


Installation Troubleshooting
----------------------------

**Got error:**

.. code-block:: python

  ImportError: cannot import name _tkagg


**Solution:**

Uninstall and then reinstall matplotlib. For example, if using pip:

.. code-block:: bash

  sudo pip uninstall matplotlib
  sudo pip install matplotlib
