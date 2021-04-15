import numpy as np


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
