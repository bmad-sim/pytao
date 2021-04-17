"""
pytao specific utilities

"""

import numpy as np



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_bool(s):
    x = s.upper()[0]
    if x == 'T':
        return True
    elif x == 'F':
        return False
    else:
        raise ValueError ('Unknown bool: '+s) 


def parse_tao_lat_ele_list(lines):
    """
    returns mapping of names to index
    
    TODO: if elements are duplicated, this returns only the last one.
    
    Example: 
    ixlist = parse_tao_lat_ele_list(tao.cmd('python lat_ele_list 1@0'))
    """
    ix = {}
    for l in lines:
        index, name = l.split(';')
        ix[name] = int(index)
    return ix


def parse_pytype(type, val):
    """
    Parses the various types from tao_python_cmd

    
    """
    
    # Handle 
    if isinstance(val, list):
        if len(val) == 1:
            val = val[0]
    
    if type  in ['STR', 'ENUM', 'FILE', 'CRYSTAL', 'COMPONENT',
                 'DAT_TYPE', 'DAT_TYPE_Z', 'SPECIES', 'ELE_PARAM']:
        return val
    
    if type == 'LOGIC':
        return parse_bool(val)

    if type in ['INT', 'INUM']:
        return int(val)     
  
    if type == 'REAL':
        return float(val)

    if type == 'REAL_ARR':
        return np.array(val).astype(float)

    if type == 'COMPLEX':
        return complex(*val)
      
    if type == 'STRUCT':
        return {name:parse_pytype(t1, v1) for name, t1, v1 in chunks(val, 3)}
        
    # Not found
    raise ValueError ('Unknown type: '+type)


def parse_tao_python_data1(line, clean_key=True):
    """
    Parses most common data output from a Tao>python command
    <component_name>;<type>;<is_variable>;<component_value>
    
    and returns a dict
    Example: 
        eta_x;REAL;F;  9.0969865321048662E+00
    parses to:
        {'eta_x':9.0969865321048662E+00}
    
    If clean key, the key will be cleaned up by replacing '.' with '_' for use as class attributes.
    
    See: tao_python_cmd.f90
    """
    dat = {}

    sline = line.split(';')
    name, type, setable = sline[0:3]
    component_value = sline[3:]
    
    # Parse
    dat = parse_pytype(type, component_value)

    if clean_key:
        name = name.replace('.', '_')
        
    return {name:dat}

def parse_tao_python_data(lines, clean_key=True):
    """
    returns dict with data
    """
    dat = {}
    for l in lines:
        dat.update(parse_tao_python_data1(l, clean_key))
        
    return dat
    
    
    
def simple_lat_table(tao, ix_universe=1, ix_branch=0, which='model', who='twiss'):
    """
    Takes the tao object, and returns columns of parameters associated with lattice elements
     "which" is one of:
       model
       base
       design
     and "who" is one of:
       general         ! ele%xxx compnents where xxx is "simple" component (not a structure nor an array, nor allocatable, nor pointer).
       parameters      ! parameters in ele%value array
       multipole       ! nonzero multipole components.
       floor           ! floor coordinates.
       twiss           ! twiss parameters at exit end.
       orbit           ! orbit at exit end.
     Example:
    
    
    """
    # Form list of ele names
    cmd = 'python lat_ele_list '+str(ix_universe)+'@'+str(ix_branch)
    lines = tao.cmd(cmd)
    # initialize 
    ele_table = {}
    for x in lines:
        ix, name = x.split(';')
        # Single element information
        cmd = 'python lat_ele1 '+str(ix_universe)+'@'+str(ix_branch)+'>>'+str(ix)+'|'+which+' '+who
        lines2=tao.cmd(cmd)
        # Parse, setting types correctly
        ele = parse_tao_python_data(lines2)
        # Add name and index
        ele['name'] = name
        ele['ix_ele'] = int(ix)
        
        # Add data to columns 
        for key in ele:
            if key not in ele_table:
                ele_table[key] = [ele[key]]
            else:
                ele_table[key].append(ele[key])
        
        # Stop at the end ele
        if name == 'END': 
            break
    return ele_table
