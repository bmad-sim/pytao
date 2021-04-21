import numpy as np

# These methods will be added to the Tao class
# Skip these:
__deny_list = ['np']



def bunch_data(tao, ele_id, *, which='model', ix_bunch=1,  verbose=False):
    """    
    Returns bunch data in openPMD-beamphysics format/notation.
    
    Notes
    -----
    Note that Tao's 'write beam' will also write a proper h5 file in this format.
    
    Expected usage:
        data = bunch_data(tao, 'end')
        from pmd_beamphysics import ParticleGroup
        P = ParicleGroup(data=data)
        
        
    Returns
    -------
    data : dict of arrays, with keys:
        'x', 'px', 'y', 'py', 't', 'pz', 'status', 'weight', 'z', 'species'
        
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/csr_beam_tracking/tao.init
     args:
       ele_id: end
       which: model
       ix_bunch: 1  
    
    """
    
    # Get species
    stats = tao.bunch1(ele_id, which=which, ix_bunch=ix_bunch, verbose=verbose)
    species = stats['species']
    
    dat = {}
    for coordinate in ['x', 'px', 'y', 'py',  't', 'pz', 'p0c', 'charge', 'state']:
        dat[coordinate] = tao.bunch1(ele_id, coordinate=coordinate, which=which, ix_bunch=ix_bunch, verbose=verbose)
        
    # Remove normalizations
    p0c = dat.pop('p0c')
    
    dat['status'] = dat.pop('state')
    dat['weight'] = dat.pop('charge')
    
    # px from Bmad is px/p0c 
    # pz from Bmad is delta = p/p0c -1. 
    # pz = sqrt( (delta+1)**2 -px**2 -py**2)*p0c
    dat['pz'] = np.sqrt((dat['pz'] + 1)**2 - dat['px']**2 - dat['py']**2) * p0c
    dat['px'] = dat['px']*p0c
    dat['py'] = dat['py']*p0c

    # z = 0 by definition
    dat['z'] = np.full(len(dat['x']), 0)
        
    dat['species'] = species.lower()
    
    return dat