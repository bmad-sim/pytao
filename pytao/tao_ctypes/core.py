import os
import ctypes
import numpy as np
from pytao import tao_ctypes
from pytao.tao_ctypes import extra_commands
from pytao.tao_ctypes.util import error_in_lines
from pytao.util.parameters import tao_parameter_dict
from .tools import full_path
import tempfile
import shutil
import types


import logging
logger = logging.getLogger(__name__)


#--------------------------------------

class Tao:
    """
    Class to run and interact with Tao. Requires libtao shared object.

    Setup:

    import os
    import sys
    TAO_PYTHON_DIR=os.environ['ACC_ROOT_DIR'] + '/tao/python'
    sys.path.insert(0, TAO_PYTHON_DIR)

    import tao_ctypes
    tao = tao_ctypes.Tao()
    tao.init("command line args here...")
    """

    #---------------------------------------------

    def __init__(self, init='', so_lib=''):
        # TL/DR; Leave this import out of the global scope.
        #
        # Make it lazy import to avoid cyclical dependency.
        # at __init__.py there is an import for Tao which
        # would cause interface_commands to be imported always
        # once we import pytao.
        # If by any chance the interface_commands.py is broken and
        # we try to autogenerate it will complain about the broken
        # interface_commands file.
        
        from pytao import interface_commands 
        
        # Library needs to be set.
        self.so_lib_file = None
        if so_lib == '':
            # Search
            ACC_ROOT_DIR = os.getenv('ACC_ROOT_DIR', '')
            if ACC_ROOT_DIR:
                BASE_DIR = os.path.join(ACC_ROOT_DIR, 'production', 'lib')
                self.so_lib_file = find_libtao(BASE_DIR)
        else:
            self.so_lib_file = so_lib

        if self.so_lib_file:
            self.so_lib = ctypes.CDLL(self.so_lib_file)
        else:
            lib, lib_file = auto_discovery_libtao()

            if lib:
                self.so_lib = lib
                self.so_lib_file = lib_file
            else:
                raise ValueError(f'Shared object libtao library not found.')

        self.so_lib.tao_c_out_io_buffer_get_line.restype = ctypes.c_char_p
        self.so_lib.tao_c_out_io_buffer_reset.restype = None

        # Extra methods
        self._import_commands(interface_commands)
        self._import_commands(extra_commands)

        try:
            self.register_cell_magic()
        except:
            pass

        if init:
            self.init(init)
            
            
    def _import_commands(self, module):
        deny_list = getattr(module, '__deny_list', [])
        # Add in methods from `interface_commands`
        methods = [m for m in dir(module) if not m.startswith('__') and m not in deny_list]
        for m in methods:
            func = module.__dict__[m]
            setattr(self, m, types.MethodType(func, self))            

    #---------------------------------------------
    # Used by init and cmd routines

    def get_output(self, reset=True):
        """
        Returns a list of output strings.
        If reset, the internal Tao buffers will be reset.
        """
        n_lines = self.so_lib.tao_c_out_io_buffer_num_lines()
        lines = [self.so_lib.tao_c_out_io_buffer_get_line(i).decode('utf-8') for i in range(1, n_lines+1)]
        if reset:
            self.so_lib.tao_c_out_io_buffer_reset()
        return lines
    
    def reset_output(self):
        """
        Resets all output buffers
        """
        self.so_lib.tao_c_out_io_buffer_reset()

    #---------------------------------------------
    # Init Tao

    def init(self, cmd):

        if not tao_ctypes.initialized:
            logger.debug(f'Initializing Tao with: {cmd}')
            err = self.so_lib.tao_c_init_tao(cmd.encode('utf-8'))
            if err != 0:
                raise ValueError(f'Unable to init Tao with: {cmd}')
            tao_ctypes.initialized = True
            return self.get_output()
        else:
            # Reinit
            return self.cmd(f'reinit tao -clear {cmd}', raises=True)

    #---------------------------------------------
    # Send a command to Tao and return the output

    def cmd(self, cmd, raises=True):
        """
        Runs a command, and returns the text output
        
        cmd: command string
        raises: will raise an exception of [ERROR or [FATAL is detected in the output
        
        Returns a list of strings
        """
        
        logger.debug(f'Tao> {cmd}')

        self.so_lib.tao_c_command(cmd.encode('utf-8'))
        lines = self.get_output()
        
        # Error checking
        if not raises:
            return lines
        
        err = error_in_lines(lines)
        if err:
            raise RuntimeError(f'Command: {cmd} causes error: {err}')
        
        return lines
    
    def cmds(self, cmds, 
             suppress_lattice_calc=True, 
             suppress_plotting=True, 
             raises=True):
        """
        Runs a list of commands
    
        Args:
            cmds: list of commands
            
            suppress_lattice_calc: bool, optional
                If True, will suppress lattice calc when applying the commands
                Default: True
                
            suppress_plotting: bool, optional
                If True, will suppress plotting when applying commands
                Default: True
                
            raises: bool, optional
                If True will raise an exception of [ERROR or [FATAL is detected in the 
                output
                Default: True
            
        Returns:
            list of results corresponding to the commands
        
        """
        # Get globals to detect plotting
        g = self.tao_global()
        ploton, laton = g['plot_on'], g['lattice_calc_on']
        
        if suppress_plotting and ploton:
            self.cmd('set global plot_on = F')
        if suppress_lattice_calc and laton:
            self.cmd('set global lattice_calc_on = F')            
    
        # Actually apply commands
        results = []
        for cmd in cmds:
            res = self.cmd(cmd, raises=raises)
            results.append(res)
            
        if suppress_plotting and ploton:
            self.cmd('set global plot_on = T')
        if suppress_lattice_calc and laton:
            self.cmd('set global lattice_calc_on = T')               
            
        return results
            
        
    
    #---------------------------------------------
    # Get real array output.
    # Only python commands that load the real array buffer can be used with this method.

    def cmd_real (self, cmd, raises=True):
        logger.debug(f'Tao> {cmd}')
        
        self.so_lib.tao_c_command(cmd.encode('utf-8'))
        n = self.so_lib.tao_c_real_array_size()
        self.so_lib.tao_c_get_real_array.restype = ctypes.POINTER(ctypes.c_double * n)

        # Check the output for errors
        lines = self.get_output(reset=False)
        err = error_in_lines(lines)
        if err:
            self.reset_output()
            if raises:
                raise RuntimeError(err)
            else:
                return None
    
        # Extract array data
        # This is a pointer to the scratch space.
        array = np.ctypeslib.as_array(
            (ctypes.c_double * n).from_address(ctypes.addressof(self.so_lib.tao_c_get_real_array().contents)))
        
        array = array.copy()
        self.reset_output()
        
        return array  

    #----------
    # Get integer array output.
    # Only python commands that load the integer array buffer can be used with this method.

    def cmd_integer (self, cmd, raises=True):
        logger.debug(f'Tao> {cmd}')
        
        self.so_lib.tao_c_command(cmd.encode('utf-8'))
        n = self.so_lib.tao_c_integer_array_size()
        self.so_lib.tao_c_get_integer_array.restype = ctypes.POINTER(ctypes.c_int * n)

        # Check the output for errors
        lines = self.get_output(reset=False)
        err = error_in_lines(lines)
        if err:
            self.reset_output()
            if raises:
                raise RuntimeError(err)
            else:
                return None  
        
        # Extract array data
        # This is a pointer to the scratch space.
        array = np.ctypeslib.as_array(
            (ctypes.c_int * n).from_address(ctypes.addressof(self.so_lib.tao_c_get_integer_array().contents)))

        array = array.copy()
        self.reset_output()
        
        return array  
    
 

    #---------------------------------------------

    def register_cell_magic(self):
      """
      Registers a cell magic in Jupyter notebooks
      Invoke by
      %%tao
      sho lat
      """

      from IPython.core.magic import register_cell_magic
      @register_cell_magic
      def tao(line, cell):
          cell = cell.format(**globals())
          cmds=cell.split('\n')
          output = []
          for c in cmds:
              print('-------------------------')
              print('Tao> '+c)
              res = self.cmd(c)
              for l in res:
                   print(l)
      del tao


def find_libtao(base_dir):  
    """
    Searches base_for for an appropriate libtao shared library. 
    """
    for lib in ['libtao.so', 'libtao.dylib', 'libtao.dll']:
        so_lib_file = os.path.join(base_dir, lib)
        if os.path.exists(so_lib_file):
            return so_lib_file
    return None
    

def auto_discovery_libtao():
    """
    Use system loader to try and find libtao.
    """
    for lib in ['libtao.so', 'libtao.dylib', 'libtao.dll']:
        try:
            lib_handler = ctypes.CDLL(lib)
            return lib_handler, lib
        except OSError:
            continue
    return None, None


#----------------------------------------------------------------------

class TaoModel(Tao):
    """
    Base class for setting up a Tao model in a directory. Builds upon the Tao class.

    If use_tempdir==True, then the input_file and its directory will be copied to a temporary directory.
    If workdir is given, then this temporary directory will be placed in workdir.
    """

    def __init__(self,
          input_file='tao.init',
          ploton = True,
          use_tempdir=True,
          workdir=None,
          verbose=True,
          so_lib='',  # Passed onto Tao superclass
          auto_configure = True # Should be disables if inheriting.
          ):

        # NOTE: SUPER is being called from configure(...)

        # Save init

        self.original_input_file = input_file
        self.ploton = ploton
        self.use_tempdir = use_tempdir
        self.workdir = workdir
        if workdir: assert os.path.exists(workdir), 'workdir does not exist: '+workdir

        self.verbose=verbose
        self.so_lib=so_lib

        # Run control
        self.finished = False
        self.configured = False

        if os.path.exists(os.path.expandvars(input_file)):
            f = full_path(input_file)
            self.original_path, self.original_input_file = os.path.split(f) # Get original path, filename
            if auto_configure:
                self.configure()
        else:
            self.vprint('Warning: Input file does not exist. Cannot configure.')

    def configure(self):

        # Set paths
        if self.use_tempdir:
            # Need to attach this to the object. Otherwise it will go out of scope.
            self.tempdir = tempfile.TemporaryDirectory(dir=self.workdir)
            # Make yet another directory to overcome the limitations of shutil.copytree
            self.path = full_path(os.path.join(self.tempdir.name, 'tao/'))
            # Copy everything in original_path
            shutil.copytree(self.original_path, self.path, symlinks=True)
        else:
            # Work in place
            self.path = self.original_path

        self.input_file = os.path.join(self.path, self.original_input_file)

        self.vprint('Initialized Tao with '+self.input_file)


        # Set up Tao library
        super().__init__(init=self.init_line(), so_lib=self.so_lib)

        self.configured = True

    def init_line(self):
        line = '-init '+self.input_file
        if self.ploton:
            line += ' --noplot'
        else:
            line += ' -noplot'
        return line

    def reinit(self):
        line = 'reinit tao '+self.init_line()
        self.cmd(line)
        self.vprint('Re-initialized with '+line)

    def vprint(self, *args, **kwargs):
        # Verbose print
        if self.verbose:
            print(*args, **kwargs)
            
    #---------------------------------
    # Conveniences        

    @property
    def globals(self):
        """
        Returns dict of tao parameters.
        Note that the name of this function cannot be named 'global'
        """

        dat = self.cmd('python global')
        return tao_parameter_dict(dat)            

    #---------------------------------
    # [] for set command

    def __setitem__(self, key, item):
        """
        Issue a set command separated by :
        
        Example:
            TaoModel['global:track_type'] = 'beam'
        will issue command:
            set global track_type = beam
        """

        cmd = form_set_command(key, item,  delim=':')
        self.vprint(cmd)
        self.cmd(cmd)

    #---------------------------------
    def evaluate(self, expression):
        """
        Example: 
            .evaluate('lat::orbit.x[beginning:end]')
        Returns an np.array of floats
        """

        return tao_object_evaluate(self, expression)

    #---------------------------------
    def __str__(self):
        s = 'Tao Model initialized from: '+self.original_path
        s +='\n Working in path: '+self.path
        return s
        
#------------------------------------------------------------------------------- 
#------------------------------------------------------------------------------- 
# Helper functions        
     
def tao_object_evaluate(tao_object, expression):
    """
    Evaluates an expression and returns 
    
    Example expressions:
        beam::norm_emit.x[end]        # returns a single float
        lat::orbit.x[beginning:end]   # returns an np array of floats 
    """
    
    cmd = f'python evaluate {expression}'
    res = tao_object.cmd(cmd)

    # Cast to float
    vals = [x.split(';')[1] for x in res]
    
    try:
        fvals = np.asarray(vals, dtype=np.float)
    except:
        fvals = vals
    
    # Return single value, or array
    if len(fvals) == 1:
        return fvals[0]
    return fvals        
        
def form_set_command(s, value,  delim=':'):    
    """
    Forms a set command string that is separated by delim.
    
    Splits into three parts:
    command:what:attribute
    
    If 'what' had delim inside, the comma should preserve that.
    
    Example:
    >>>form_set_command('ele:BEG:END:a', 1.23)
    'set ele BEG:END a = 1.23'
    
    """
    x = s.split(delim)
    
    cmd0 = x[0]
    what = ':'.join(x[1:-1])
    att = x[-1]
    cmd = f'set {cmd0} {what} {att} = {value}'
    
    # cmd = 'set '+' '.join(x) + f' = {value}'
  
    return cmd
    
    
def apply_settings(tao_object, settings):
    """
    Applies multiple settings to a tao object.
    Checks for lattice_calc_on and plot_on, and temporarily disables these for speed.
    """
    
    cmds = []
    
    # Save these
    plot_on = tao_object.globals['plot_on'].value
    lattice_calc_on = tao_object.globals['lattice_calc_on'].value
    
    if plot_on:
        cmds.append('set global plot_on = F')
    if lattice_calc_on:
        cmds.append('set global lattice_calc_on = F')
    
    
    for k, v in settings.items():
        cmd = form_set_command(k, v)
        cmds.append(cmd)
        
    # Restore
    if lattice_calc_on:
        cmds.append('set global lattice_calc_on = T')

    if plot_on:
        cmds.append('set global plot_on = T')    
        
        
    for cmd in cmds:
        tao_object.vprint(cmd)
        tao_object.cmd(cmd, raises=True)
        
    return cmds    

#------------------------------------------------------------------------------- 
#------------------------------------------------------------------------------- 
# Helper functions        

def run_tao(settings=None,
                run_commands=['set global track_type=single'],
                input_file='tao.init',
                ploton=False,
                workdir=None,
                so_lib='',
                verbose=False):
    """
    Creates an LCLSTaoModel object, applies settings, and runs the beam.
    """
    
    assert os.path.exists(input_file), f'Tao input file does not exist: {input_file}'
    
    M = TaoModel(input_file=input_file,
                 ploton = ploton,
                 use_tempdir=True,
                 workdir=workdir,
                 verbose=verbose,
                 so_lib=so_lib,  # Passed onto Tao superclass
                 auto_configure = True) # Should be disables if inheriting.
                
    # Move to local dir, so call commands work 
    init_dir = os.getcwd()
    os.chdir(M.path)
    
    try:
        if settings:
            apply_settings(M, settings)
        
        for command in run_commands:
            if verbose:
                print('run command:', command)
            M.cmd(command, raises=True)
    
    finally:
        # Return to init_dir
        os.chdir(init_dir)    

    return M
