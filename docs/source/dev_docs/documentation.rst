Documentation
=============

PyTao uses `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to generate its documentation.

The documentation is available online at `PyTao's website <https://bmad-sim.github.io/pytao/>`_
and it is kept up-to-date with the codebase via automated routines (GitHub Actions) for deployment
of files when the code is changed.

In case you have the need to build the documentation locally please follow the procedure
below.

System Dependencies
-------------------

Here are the system dependencies required to properly build the documentation:

- `pandoc <https://pandoc.org/>`_
    Pandoc is an universal document converter.

- `Python >= 3.7 <https://www.python.org/>`_
    Python is an interpreted, high-level and general-purpose programming language.

Feel free to install them using the package manager or your choice or build from
source.

Sphinx & Python Dependencies
----------------------------

To obtain the Sphinx and Python dependencies we will use pip to install them all.
For that please run the following command from the root folder of this repository:

.. code-block:: bash

  pip install -r docs-requirements.txt

The command above will install all dependencies listed in the file `docs-requirements.txt`.

Building the Documentation
--------------------------

Now that we have all the dependencies ready the next step is to build the documentation.
For that follow the instructions below:

.. code-block:: bash

   cd docs
   make html

The command above will create the documentation in HTML format.
The generated documentation will be on the `build/html` folder.

To visualize the documentation that you just built you just need to open the
`build/html/index.html` file with your browser.
