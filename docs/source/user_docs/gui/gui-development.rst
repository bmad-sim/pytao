GUI Development
===============

To profile the GUI, first create a `gui.init` file with:

.. code-block:: yaml

   skip_setup:T
   do_mainloop:F

thenuse the python script:

.. code-block:: python

   from pytao import gui
   import cProfile
   cProfile.run('gui.tao_root_window()', 'profile.dat')
