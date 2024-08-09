# Usage

## PyTao with JupyterLab

PyTao has advanced JupyterLab integration for plotting and is generally the recommended method for using PyTao.

Start up JupyterLab as you would normally:

```bash
jupyter lab

```

And then use PyTao, enabling your preferred plotting backend:

```python
from pytao import Tao

# Best available (Bokeh first, Matplotlib second)
tao = Tao(init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init", plot=True)
# Matplotlib:
tao = Tao(init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init", plot="mpl")
# Bokeh
tao = Tao(init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init", plot="bokeh")

```

If you wish to use Tao's internal plotting mechanism, leave the `plot` keyword argument off or specify `plot="tao"`:

```python
from pytao import Tao

# Use Tao's internal plotting mechanism:
tao = Tao(init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init")

```

To disable plotting entirely, use:

```python
from pytao import Tao

# Use Tao's internal plotting mechanism:
tao = Tao(init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init", noplot=True)

```

The `Tao` object supports all Tao initialization arguments as Python keyword arguments.
That is, any of the following may be specified to `Tao`:

```python


Tao(
  beam_file="file_name",                # File containing the tao_beam_init namelist.
  beam_init_position_file="file_name",  # File containing initial particle positions.
  building_wall_file="file_name",       # Define the building tunnel wall
  command="command_string",             # Commands to run after startup file commands
  data_file="file_name",                # Define data for plotting and optimization
  debug=True,                           # Debug mode for Wizards
  disable_smooth_line_calc=True,        # Disable the smooth line calc used in plotting
  external_plotting=True,               # Tells Tao that plotting is done externally to Tao.
  geometry="<width>x<height>",          # Plot window geometry (pixels)
  hook_init_file="file_name",           # Init file for hook routines (Default = tao_hook.init)
  init_file="file_name",                # Tao init file
  lattice_file="file_name",             # Bmad lattice file
  log_startup=True,                     # Write startup debugging info
  no_stopping=True,                     # For debugging: Prevents Tao from exiting on errors
  noinit=True,                          # Do not use Tao init file.
  noplot=True,                          # Do not open a plotting window
  nostartup=True,                       # Do not open a startup command file
  no_rad_int=True,                      # Do not do any radiation integrals calculations.
  plot_file="file_name",                # Plotting initialization file
  prompt_color="color",                 # Set color of prompt string. Default is blue.
  reverse=True,                         # Reverse lattice element order?
  rf_on=True,                           # Use "--rf_on" to turn off RF (default is now RF on)
  quiet=True,                           # Suppress terminal output when running a command file?
  slice_lattice="ele_list",             # Discards elements from lattice that are not in the list
  start_branch_at="ele_name",           # Start lattice branch at element.
  startup_file="file_name",             # Commands to run after parsing Tao init file
  symbol_import=True,                   # Import symbols defined in lattice files(s)?
  var_file="file_name",                 # Define variables for plotting and optimization
)
```

## PyTao on the command-line

PyTao has a simple IPython entrypoint, giving you quick access to Tao in Python.

The following will start IPython with a `Tao` instance available as `tao`:

```bash
pytao -init_file "$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init"
```

To use PyTao's Matplotlib backend, do the following:

```bash
PYTAO_PLOT=mpl pytao -init_file "$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init"
```

```python
In [1]: tao.plot("beta")

In [2]: plt.show()
```

## PyTao (deprecated/experimental) GUI

Start the experimental (and mostly unsupported/deprecated) GUI by using the following:

```bash
pytao-gui -init_file "$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init"
```
