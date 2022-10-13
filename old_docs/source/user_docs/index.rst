Installation
============

Note! The **Bmad Distribution** (which includes *Tao*) must be installed before installing PyTao.
Additionally, the Bmad Distribution must be compiled with the ACC_ENABLE_SHARED="Y" flag set
in the *bmad_dist/util/dist_prefs* file.

For instructions on how to install the *Bmad Distribution*, please refer to the instructions
available at the *Bmad* website.

Since PyTao is a python package, it can be installed in a couple of different
ways:

Using setuptools
----------------

.. code-block:: bash

    python setup.py install

Using pip
---------

.. code-block:: bash

    # From PyPI distribution
    pip install pytao

    # or from the source folder
    pip install .

Using conda
-----------

.. code-block:: bash

    conda install -c conda-forge pytao


