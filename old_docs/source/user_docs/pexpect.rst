pexpect
=======

The `tao_pipe` module uses the **pexpect** module. The **pexpect** module is a
general purpose tool for interfacing Python with programs like *Tao*.
If **pexpect** is not present on your system, it can be downloaded from
`www.noah.org/wiki/pexpect <www.noah.org/wiki/pexpect>`.

Example
-------

.. code-block:: python

   import tao_pipe # import module

   p = tao_pipe.tao_io("../bin/tao -lat my_lat.bmad") # init session

   p.cmd_in("show global") # Command to Tao
   print(p.output) # print the output from Tao

   p.cmd("show global") # Like p.cmd_in() excepts prints the output too.

API
---

.. autoclass:: pytao.tao_pexpect.tao_pipe.tao_io
   :members:
   :undoc-members:
   :noindex:
