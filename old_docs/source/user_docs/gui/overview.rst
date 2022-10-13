Overview
--------

*Tao* is an open source general purpose program for charged particle and X-ray simulations in
accelerators and storage rings. It is built on top of the *Bmad* toolkit (software library) which
provides the needed computational routines needed to do simulations. Essentially you can think of
*Tao* as a car and *Bmad* as the engine that powers the car. In fact *Bmad* powers a number of other
simulation programs but that is getting outside of the scope of this manual.

Documentation for *Bmad* and *Tao*, as well as information for downloading the code if needed is given
on the *Bmad* web site `https://www.classe.cornell.edu/bmad <https://www.classe.cornell.edu/bmad>`_

*Tao* by itself is a command line program. To make it more readily accessible, a graphic user
interface (GUI) has been developed and this is the subject of this manual. The GUI is written in
Python. The Python tkinter package is used for windowing. Tkinter being an interface layer
for the Tk widget toolset. The Python based plotting package MatPlotLib can be used for graphics.
Alternatively, the PGPLOT/PLPLOT plotting that comes standard with Tao can be used.

It is assumed in this manual that the reader already has some familiarity with *Bmad* and *Tao*. If not,
there are manuals for *Bmad* and *Tao* posted on the *Bmad* web site along with a beginner's tutorial.

The coding the of GUI was a joint development project with John Mastroberti, Kevin Kowalski, and
David Sagan. Thanks must go to Thomas Gläße for helping with the inital development.
