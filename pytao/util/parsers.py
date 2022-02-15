import numpy as np





# Column names and types for parse_data_d_array
DATA_D_COLS =  ['ix_d1',
 'data_type',
 'merit_type',
 'ele_ref_name',
 'ele_start_name',
 'ele_name',
 'meas_value',
 'model_value',
 'design_value',
 'useit_opt',
 'useit_plot',
 'good_user',
 'weight',
 'exists']
DATA_D_TYPES = [int, str, str, str, str, str, float, float, float, bool, bool, bool, float, bool]

def parse_data_d_array(lines):
    """
    Parses the output of the 'python data_d_array' command into a list of dicts. 
    
    This can be easily be case into a table. For example:
    
    import pandas as pd
    ...
    lines = tao.data_d_array('orbit', 'x')
    dat = parse_data_d_array(lines)
    df = pd.DataFrame(dat)
    
    
    Parameters
    ----------
    lines : list of str
        The output of the 'python data_d_array' command to parse
    
    Returns
    -------
    datums: list of dicts
            Each dict has keys:
            'ix_d1', 'data_type', 'merit_type', 
            'ele_ref_name', 'ele_start_name', 'ele_name', 
            'meas_value', 'model_value', 'design_value', 
            'useit_opt', 'useit_plot', 'good_user', 
            'weight', 'exists'
    
    """ 
    result = []
    for line in lines:
        d = {}
        result.append(d)
        vals = line.split(';')
        for name, typ, val in zip(DATA_D_COLS, DATA_D_TYPES, vals):
            d[name] = typ(val)
        
    return result



def parse_derivative(lines):
    """
    Parses the output of tao python derivative
    
    Parameters
    ----------
    lines : list of str
        The output of the 'python derivative' command to parse
    
    Returns
    -------
    out : dict
        Dictionary with keys corresponding to universe indexes (int),
        with dModel_dVar as the value:
            np.ndarray with shape (n_data, n_var)    
    """
    universes = {}

    # Build up matrices
    for line in lines:
        x = line.split(';')
        if len(x) <= 1:
            continue
        iu = int(x[0])
          
        if iu not in universes:
            # new universe
            rows = universes[iu] = []
            rowdat = []
            row_id = int(x[1])
        
        if int(x[1]) == row_id:
            # accumulate more data
            rowdat += x[3:]     
        else:
            # Finish row
            rows.append(rowdat)   
            rowdat = x[3:]  
            row_id = int(x[1])
           
    # cast to float    
    out = {}
    for iu, vals in universes.items():
        out[iu] = np.array(vals).astype(float)
        
    return out



def parse_lat_ele_list(lines):
    """
    Parses the output of tao python lat_ele_list
    
    Parameters
    ----------
    lines : list of str
        The output of the 'python lat_ele_list' command to parse
    
    Returns
    -------
    list of str of element names
    
    """
    
    return [s.split(';')[1] for s in lines]


def parse_matrix(lines):
    """
    Parses the output of a tao python matix
    
    Parameters
    ----------
    lines : list of str
        The output of the 'python matrix' command to parse
    
    Returns
    -------
    dict with keys:
        'mat6' : np.array of shape (6,6)
        'vec6' : np.array of shape(6)
        
    
    """
    m7 = np.array([[float(x) for x in line.split(';')[1:]] for line in lines])
    return {'mat6':m7[:,0:6], 'vec0':m7[:,6]}



