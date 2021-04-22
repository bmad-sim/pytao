
from pytao.tao_ctypes.util import parse_tao_python_data
from pytao.util.parameters import tao_parameter_dict
from pytao.util import parsers as __parsers


def __execute(tao, cmd, as_dict=True, method_name=None, cmd_type="string_list"):
    func_for_type = {
        "string_list": tao.cmd,
        "real_array": tao.cmd_real,
        "integer_array": tao.cmd_integer
    }
    func = func_for_type.get(cmd_type, tao.cmd)
    ret = func(cmd)
    special_parser = getattr(__parsers, f'parse_{method_name}', "")
    if special_parser:
        data = special_parser(ret)
        return data
    if "string" in cmd_type:
        try:
            if as_dict:
                data = parse_tao_python_data(ret)
            else:
                data = tao_parameter_dict(ret)
        except Exception as ex:
            print('Failed to parse string data. Returning raw value. Exception was: ', ex)
            return ret
            
        return data
        
    return ret


def beam(tao, *, ix_universe='1', verbose=False, as_dict=True):
    """
    
    Output beam parameters that are not in the beam_init structure.
    
    Parameters
    ----------
    ix_universe : default=1
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python beam {ix_universe}
    where
      {ix_universe} is a universe index.
    To set beam_init parameters use the "set beam" command
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/csr_beam_tracking/tao.init
     args:
       ix_universe: 1
    
    """
    cmd = f'python beam {ix_universe}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='beam', cmd_type='string_list')


def beam_init(tao, *, ix_universe='1', verbose=False, as_dict=True):
    """
    
    Output beam_init parameters.
    
    Parameters
    ----------
    ix_universe : default=1
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python beam_init {ix_universe}
    where
      {ix_universe} is a universe index.
    To set beam_init parameters use the "set beam_init" command
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/csr_beam_tracking/tao.init
     args:
       ix_universe: 1
    
    """
    cmd = f'python beam_init {ix_universe}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='beam_init', cmd_type='string_list')


def bmad_com(tao, *, verbose=False, as_dict=True):
    """
    
    Bmad_com structure components
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python bmad_com
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
    
    """
    cmd = f'python bmad_com'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='bmad_com', cmd_type='string_list')


def branch1(tao, *, ix_universe='1', ix_branch='0', verbose=False, as_dict=True):
    """
    
    Lattice element list.
    
    Parameters
    ----------
    ix_universe : default=1
    ix_branch : default=0
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python branch1 {ix_universe}@{ix_branch}
    where
      {ix_universe} is a universe index
      {ix_branch} is a lattice branch index
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_universe: 1
       ix_branch: 0
    
    """
    cmd = f'python branch1 {ix_universe}@{ix_branch}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='branch1', cmd_type='string_list')


def bunch1(tao, ele_id, *, which='model', ix_bunch='1', coordinate='', verbose=False, as_dict=True):
    """
    
    Bunch parameters at the exit end of a given lattice element.
    
    Parameters
    ----------
    ele_id
        Element name or index
    which : default=model
        One of: "model", "base" or "design"
    ix_bunch : default=1
    coordinate : optional
        If one of: x, px, y, py, z, pz, 's', 't', 'charge', 'p0c', 'state'
    
    Returns
    -------
    string_list
        if not coordinate
    real_array
        if coordinate and coordinate != 'state'
    integer_array
        if coordinate == 'state'
    
    Notes
    -----
    Command syntax:
    python bunch1 {ele_id}|{which} {ix_bunch} {coordinate}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/csr_beam_tracking/tao.init
     args:
       ele_id: end
       which: model
       ix_bunch: 1
       coordinate:
    
    Example: 2
     init: $ACC_ROOT_DIR/tao/examples/csr_beam_tracking/tao.init
     args:
       ele_id: end
       which: model
       ix_bunch: 1
       coordinate: x
    
    """
    cmd = f'python bunch1 {ele_id}|{which} {ix_bunch} {coordinate}'
    if verbose: print(cmd)
    if not coordinate:
        return __execute(tao, cmd, as_dict, method_name='bunch1', cmd_type='string_list')
    if coordinate and coordinate != 'state':
        return __execute(tao, cmd, as_dict, method_name='bunch1', cmd_type='real_array')
    if coordinate == 'state':
        return __execute(tao, cmd, as_dict, method_name='bunch1', cmd_type='integer_array')


def building_wall_list(tao, *, ix_section='', verbose=False, as_dict=True):
    """
    
    List of building wall sections or section points
    
    Parameters
    ----------
    ix_section : optional
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python building_wall_list {ix_section}
    If {ix_section} is not present then a list of building wall sections is given.
    If {ix_section} is present then a list of section points is given
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/examples/tutorial_bmad_tao/lattice_files/building_wall_optimization/tao.init
     args:
       ix_section:
    
    Example: 2
     init: $ACC_ROOT_DIR/examples/tutorial_bmad_tao/lattice_files/building_wall_optimization/tao.init
     args:
       ix_section: 1
    
    """
    cmd = f'python building_wall_list {ix_section}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='building_wall_list', cmd_type='string_list')


def building_wall_graph(tao, graph, *, verbose=False, as_dict=True):
    """
    
    (x, y) points for drawing the building wall for a particular graph.
    
    Parameters
    ----------
    graph
    
    Returns
    -------
    string_list
    
    Notes
    -----
    The graph defines the coordinate system for the (x, y) points.
    Command syntax:
      python building_wall_graph {graph}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/examples/tutorial_bmad_tao/lattice_files/building_wall_optimization/tao.init
     args:
       graph:
    
    """
    cmd = f'python building_wall_graph {graph}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='building_wall_graph', cmd_type='string_list')


def building_wall_point(tao, ix_section, ix_point, z, x, radius, z_center, x_center, *, verbose=False, as_dict=True):
    """
    
    add or delete a building wall point
    
    Parameters
    ----------
    ix_section
    ix_point
    z
    x
    radius
    z_center
    x_center
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python building_wall_point {ix_section}^^{ix_point}^^{z}^^{x}^^{radius}^^{z_center}^^{x_center}
    Where:
      {ix_section}    -- Section index.
      {ix_point}      -- Point index. Points of higher indexes will be moved up 
                           if adding a point and down if deleting.
      {z}, etc...     -- See tao_building_wall_point_struct components.
                      -- If {z} is set to "delete" then delete the point.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/examples/tutorial_bmad_tao/lattice_files/building_wall_optimization/tao.init
     args:
       ix_section: 1
       ix_point: 1
       z: 0
       x: 0
       radius: 0
       z_center: 0
       x_center: 0
    
    """
    cmd = f'python building_wall_point {ix_section}^^{ix_point}^^{z}^^{x}^^{radius}^^{z_center}^^{x_center}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='building_wall_point', cmd_type='string_list')


def building_wall_section(tao, ix_section, sec_name, sec_constraint, *, verbose=False, as_dict=True):
    """
    
    add or delete a building wall section
    
    Parameters
    ----------
    ix_section
    sec_name
    sec_constraint
    
    Returns
    -------
    None
    
    Notes
    -----
    Command syntax:
      python building_wall_section {ix_section}^^{sec_name}^^{sec_constraint}
    Where:
      {ix_section}      -- Section index. Sections with higher indexes will be
                             moved up if adding a section and down if deleting.
      {sec_name}        -- Section name.
      {sec_constraint}  -- Must be one of:
          delete     -- Delete section. Anything else will add the section.
          none
          left_side
          right_side
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_section: 1
       sec_name: test
       sec_constraint: none
    
    """
    cmd = f'python building_wall_section {ix_section}^^{sec_name}^^{sec_constraint}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='building_wall_section', cmd_type='None')


def constraints(tao, who, *, verbose=False, as_dict=True):
    """
    
    Optimization data and variables that contribute to the merit function.
    
    Parameters
    ----------
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python constraints {who}
    {who} is one of:
      data
      var
    Data constraints output is:
      data name
      constraint type
      evaluation element name
      start element name
      end/reference element name
      measured value
      ref value (only relavent if global%opt_with_ref = T)
      model value
      base value (only relavent if global%opt_with_base = T)
      weight
      merit value
      location where merit is evaluated (if there is a range)
    Var constraints output is:
      var name
      Associated varible attribute
      meas value
      ref value (only relavent if global%opt_with_ref = T)
      model value
      base value (only relavent if global%opt_with_base = T)
      weight
      merit value
      dmerit/dvar
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       who: data
    
    Example: 2
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       who:var
    
    """
    cmd = f'python constraints {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='constraints', cmd_type='string_list')


def da_aperture(tao, *, ix_uni='1', verbose=False, as_dict=True):
    """
    
    Dynamic aperture data
    
    Parameters
    ----------
    ix_uni : default=1
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python da_aperture {ix_uni}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_uni:1
    
    """
    cmd = f'python da_aperture {ix_uni}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='da_aperture', cmd_type='string_list')


def da_params(tao, *, ix_uni='1', verbose=False, as_dict=True):
    """
    
    Dynamic aperture input parameters
    
    Parameters
    ----------
    ix_uni : default=1
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python da_params {ix_uni}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_uni:1
    
    """
    cmd = f'python da_params {ix_uni}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='da_params', cmd_type='string_list')


def data(tao, d2_name, d1_datum, *, ix_universe='1', dat_index='1', verbose=False, as_dict=True):
    """
    
    Individual datum info.
    
    Parameters
    ----------
    d2_name
    d1_datum
    ix_universe : default=1
    dat_index : default=1
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python data {ix_universe}@{d2_name}.{d1_datum}[{dat_index}]
    Use the "python data-d1" command to get detailed info on a specific d1 array.
    Output syntax is parameter list form. See documentation at the beginning of this file.
    Example:
      python data 1@orbit.x[10]
    Note : By default dat_index is 1.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       ix_universe:
       d2_name: twiss
       d1_datum: end 
       dat_index: 
    
    Example: 2
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       ix_universe: 1
       d2_name: twiss
       d1_datum: end
       dat_index: 1
    
    """
    cmd = f'python data {ix_universe}@{d2_name}.{d1_datum}[{dat_index}]'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='data', cmd_type='string_list')


def data_d2_create(tao, d2_name, n_d1_data, d_data_arrays_name_min_max, *, ix_uni='1', verbose=False, as_dict=True):
    """
    
    Create a d2 data structure along with associated d1 and data arrays.
    
    Parameters
    ----------
    d2_name
    n_d1_data
    d_data_arrays_name_min_max
    ix_uni : default=1
    
    Returns
    -------
    None
    
    Notes
    -----
    Command syntax:
      python data_d2_create {d2_name}^^{n_d1_data}^^{d_data_arrays_name_min_max}
    {d2_name} should be of the form {ix_uni}@{d2_datum_name}
    {n_d1_data} is the number of associated d1 data structures.
    {d_data_arrays_name_min_max} has the form
      {name1}^^{lower_bound1}^^{upper_bound1}^^....^^{nameN}^^{lower_boundN}^^{upper_boundN}
    where {name} is the data array name and {lower_bound} and {upper_bound} are the bounds 
    of the array.
    
    Example:
      python data_d2_create 2@orbit^^2^^x^^0^^45^^y^^1^^47
    This example creates a d2 data structure called "orbit" with 
    two d1 structures called "x" and "y".
    
    The "x" d1 structure has an associated data array with indexes in the range [0, 45].
    The "y" d1 structure has an associated data arrray with indexes in the range [1, 47].
    
    Use the "set data" command to set a created datum parameters.
    Note: When setting multiple data parameters, 
          temporarily toggle s%global%lattice_calc_on to False
      ("set global lattice_calc_on = F") to prevent Tao trying to 
          evaluate the partially created datum and generating unwanted error messages.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       ix_uni: 1
       d2_name: orbit
       n_d1_data: 2 
       d_data_arrays_name_min_max: x^^0^^45^^y^^1^^47
    
    """
    cmd = f'python data_d2_create {d2_name}^^{n_d1_data}^^{d_data_arrays_name_min_max}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='data_d2_create', cmd_type='None')


def data_d2_destroy(tao, d2_datum, *, ix_uni='1', verbose=False, as_dict=True):
    """
    
    Destroy a d2 data structure along with associated d1 and data arrays.
    
    Parameters
    ----------
    d2_datum
    ix_uni : default=1
    
    Returns
    -------
    None
    
    Notes
    -----
    Command syntax:
      python data_d2_destroy {d2_datum}
    {d2_datum} should be of the form
      {ix_uni}@{d2_datum_name}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       ix_uni: 1
       d2_datum: orbit
    
    """
    cmd = f'python data_d2_destroy {d2_datum}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='data_d2_destroy', cmd_type='None')


def data_d2(tao, d2_datum, *, ix_uni='1', verbose=False, as_dict=True):
    """
    
    Information on a d2_datum.
    
    Parameters
    ----------
    d2_datum
    ix_uni : default=1
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python data_d2 {d2_datum}
    {d2_datum} should be of the form
      {ix_uni}@{d2_datum_name}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       ix_uni: 1
       d2_datum: twiss
    
    """
    cmd = f'python data_d2 {d2_datum}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='data_d2', cmd_type='string_list')


def data_d_array(tao, d1_datum, *, ix_uni='1', verbose=False, as_dict=True):
    """
    
    List of datums for a given data_d1.
    
    Parameters
    ----------
    d1_datum
    ix_uni : default=1
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python data_d_array {d1_datum}
    {d1_datum} should be for the form
      {ix_uni}@{d2_datum_name}.{d1_datum_name}
    Example:
      python data_d_array 1@orbit.x
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       ix_uni: 1 
       d1_datum: twiss.end
    
    """
    cmd = f'python data_d_array {d1_datum}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='data_d_array', cmd_type='string_list')


def data_d1_array(tao, d2_datum, *, ix_uni='1', verbose=False, as_dict=True):
    """
    
    List of d1 arrays for a given data_d2.
    
    Parameters
    ----------
    d2_datum
    ix_uni : default=1
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python data_d1_array {d2_datum}
    {d2_datum} should be of the form
      {ix_uni}@{d2_datum_name}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       ix_uni: 1 
       d2_datum: twiss
    
    """
    cmd = f'python data_d1_array {d2_datum}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='data_d1_array', cmd_type='string_list')


def data_parameter(tao, data_array, parameter, *, verbose=False, as_dict=True):
    """
    
    Given an array of datums, generate an array of values for a particular datum parameter.
    
    Parameters
    ----------
    data_array
    parameter
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python data_parameter {data_array} {parameter}
    {parameter} may be any tao_data_struct parameter.
    Example:
      python data_parameter orbit.x model_value
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       data_array: twiss.end 
       parameter: model_value
    
    """
    cmd = f'python data_parameter {data_array} {parameter}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='data_parameter', cmd_type='string_list')


def data_d2_array(tao, ix_universe, *, verbose=False, as_dict=True):
    """
    
    Data d2 info for a given universe.
    
    Parameters
    ----------
    ix_universe
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python data_d2_array {ix_universe}
    Example:
      python data_d2_array 1
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_universe : 1 
    
    """
    cmd = f'python data_d2_array {ix_universe}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='data_d2_array', cmd_type='string_list')


def data_set_design_value(tao, *, verbose=False, as_dict=True):
    """
    
    Set the design (and base & model) values for all datums.
    
    Returns
    -------
    None
    
    Notes
    -----
    Command syntax:
      python data_set_design_value
    Example:
      python data_set_design_value
    
    Note: Use the "data_d2_create" and "datum_create" first to create datums.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
    
    """
    cmd = f'python data_set_design_value'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='data_set_design_value', cmd_type='None')


def datum_create(tao, datum_name, data_type, *, ele_ref_name='', ele_start_name='', ele_name='', merit_type='', meas='0', good_meas='F', ref='0', good_ref='F', weight='0', good_user='T', data_source='lat', eval_point='END', s_offset='0', ix_bunch='0', invalid_value='0', spin_axis_n0_1='', spin_axis_n0_2='', spin_axis_n0_3='', spin_axis_l_1='', spin_axis_l_2='', spin_axis_l_3='', verbose=False, as_dict=True):
    """
    
    Create a datum.
    
    Parameters
    ----------
    datum_name          ! EG: orb.x[3]
    data_type           ! EG: orbit.x
    ele_ref_name : optional
    ele_start_name : optional
    ele_name : optional
    merit_type : optional
    meas : default=0
    good_meas : default=F
    ref : default=0
    good_ref : default=F
    weight : default=0
    good_user : default=T
    data_source : default=lat
    eval_point : default=END
    s_offset : default=0
    ix_bunch : default=0
    invalid_value : default=0
    spin_axis%n0(1) : optional
    spin_axis%n0(2) : optional
    spin_axis%n0(3) : optional
    spin_axis%l(1) : optional
    spin_axis%l(2) : optional
    spin_axis%l(3) : optional
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python datum_create {datum_name}^^{data_type}^^{ele_ref_name}^^{ele_start_name}^^
                          {ele_name}^^{merit_type}^^{meas}^^{good_meas}^^{ref}^^
                          {good_ref}^^{weight}^^{good_user}^^{data_source}^^
                          {eval_point}^^{s_offset}^^{ix_bunch}^^{invalid_value}^^
                          {spin_axis%n0(1)}^^{spin_axis%n0(2)}^^{spin_axis%n0(3)}^^
                          {spin_axis%l(1)}^^{spin_axis%l(2)}^^{spin_axis%l(3)}
    
    Note: The 3 values for spin_axis%n0, as a group, are optional. 
          Also the 3 values for spin_axis%l are, as a group, optional.
    Note: Use the "data_d2_create" first to create a d2 structure with associated d1 arrays.
    Note: After creating all your datums, use the "data_set_design_value" routine to 
          set the design (and model) values.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       datum_name: twiss.end[6]
       data_type: beta.y
       ele_ref_name:
       ele_start_name:
       ele_name: P1
       merit_type: target
       meas: 0
       good_meas: T
       ref: 0
       good_ref: T
       weight: 0.3
       good_user: T
       data_source: lat
       eval_point: END
       s_offset: 0
       ix_bunch: 1
       invalid_value: 0
    
    """
    cmd = f'python datum_create {datum_name}^^{data_type}^^{ele_ref_name}^^{ele_start_name}^^{ele_name}^^{merit_type}^^{meas}^^{good_meas}^^{ref}^^{good_ref}^^{weight}^^{good_user}^^{data_source}^^{eval_point}^^{s_offset}^^{ix_bunch}^^{invalid_value}^^{spin_axis_n0_1}^^{spin_axis_n0_2}^^{spin_axis_n0_3}^^{spin_axis_l_1}^^{spin_axis_l_2}^^{spin_axis_l_3}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='datum_create', cmd_type='string_list')


def datum_has_ele(tao, datum_type, *, verbose=False, as_dict=True):
    """
    
    Does datum type have an associated lattice element?
    
    Parameters
    ----------
    datum_type
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python datum_has_ele {datum_type}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
       datum_type: twiss.end 
    
    """
    cmd = f'python datum_has_ele {datum_type}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='datum_has_ele', cmd_type='string_list')


def derivative(tao, *, verbose=False, as_dict=True):
    """
    
    Optimization derivatives
    
    Returns
    -------
    out : dict
        Dictionary with keys corresponding to universe indexes (int),
        with dModel_dVar as the value:
            np.ndarray with shape (n_data, n_var)    
    
    Notes
    -----
    Command syntax:
      python derivative
    Note: To save time, this command will not recalculate derivatives. 
    Use the "derivative" command beforehand to recalcuate if needed.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/optics_matching/tao.init
     args:
    
    """
    cmd = f'python derivative'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='derivative', cmd_type='string_list')


def ele_head(tao, ele_id, *, which='model', verbose=False, as_dict=True):
    """
    
    "Head" Element attributes
    
    Parameters
    ----------
    ele_id
    which : default=model
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:head {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:head 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:head {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_head', cmd_type='string_list')


def ele_methods(tao, ele_id, *, which='model', verbose=False, as_dict=True):
    """
    
    Element methods
    
    Parameters
    ----------
    ele_id
    which : default=model
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:methods {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:methods 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:methods {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_methods', cmd_type='string_list')


def ele_gen_attribs(tao, ele_id, *, which='model', verbose=False, as_dict=True):
    """
    
    Element general attributes
    
    Parameters
    ----------
    ele_id
    which : default=model
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:gen_attribs {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:gen_attribs 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:gen_attribs {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_gen_attribs', cmd_type='string_list')


def ele_multipoles(tao, ele_id, *, which='model', verbose=False, as_dict=True):
    """
    
    Element multipoles
    
    Parameters
    ----------
    ele_id
    which : default=model
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:multipoles {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:multipoles 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:multipoles {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_multipoles', cmd_type='string_list')


def ele_ac_kicker(tao, ele_id, *, which='model', verbose=False, as_dict=True):
    """
    
    Element ac_kicker
    
    Parameters
    ----------
    ele_id
    which : default=model
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:ac_kicker {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:ac_kicker 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:ac_kicker {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_ac_kicker', cmd_type='string_list')


def ele_cartesian_map(tao, ele_id, index, who, *, which='model', verbose=False, as_dict=True):
    """
    
    Element cartesian_map
    
    Parameters
    ----------
    ele_id
    index
    who
    which : default=model
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:cartesian_map {ele_id}|{which} {index} {who}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    {index} is the index number in the ele%cartesian_map(:) array
    {who} is one of:
      base
      terms
    Example:
      python ele:cartesian_map 3@1>>7|model 2 base
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      index: 1
      who: base
    
    """
    cmd = f'python ele:cartesian_map {ele_id}|{which} {index} {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_cartesian_map', cmd_type='string_list')


def ele_chamber_wall(tao, ele_id, index, who, *, which='model', verbose=False, as_dict=True):
    """
    
    Element beam chamber wall
    
    Parameters
    ----------
    ele_id
    index
    who
    which : default=model
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:chamber_wall {ele_id}|{which} {index} {who}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    {index} is index of the wall.
    {who} is one of:
      x       ! Return min/max in horizontal plane
      y       ! Return min/max in vertical plane
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/examples/tutorial_bmad_tao/lattice_files/building_wall_optimization/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      index: 1
      who: x
    
    """
    cmd = f'python ele:chamber_wall {ele_id}|{which} {index} {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_chamber_wall', cmd_type='string_list')


def ele_cylindrical_map(tao, ele_id, which, index, who, *, verbose=False, as_dict=True):
    """
    
    Element cylindrical_map
    
    Parameters
    ----------
    ele_id
    which
    index
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:cylindrical_map {ele_id}|{which} {index} {who}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    {index} is the index number in the ele%cylindrical_map(:) array
    {who} is one of:
      base
      terms
    Example:
      python ele:cylindrical_map 3@1>>7|model 2 base
    This gives map #2 of element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      index: 1
      who: base
    
    """
    cmd = f'python ele:cylindrical_map {ele_id}|{which} {index} {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_cylindrical_map', cmd_type='string_list')


def ele_taylor(tao, ele_id, which, *, verbose=False, as_dict=True):
    """
    
    Element taylor
    
    Parameters
    ----------
    ele_id
    which
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:taylor {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:taylor 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:taylor {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_taylor', cmd_type='string_list')


def ele_spin_taylor(tao, ele_id, which, *, verbose=False, as_dict=True):
    """
    
    Element spin_taylor
    
    Parameters
    ----------
    ele_id
    which
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:spin_taylor {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:spin_taylor 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:spin_taylor {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_spin_taylor', cmd_type='string_list')


def ele_wake(tao, ele_id, which, who, *, verbose=False, as_dict=True):
    """
    
    Element wake
    
    Parameters
    ----------
    ele_id
    which
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:wake {ele_id}|{which} {who}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    {Who} is one of
      base
      sr_long     sr_long_table
      sr_trans    sr_trans_table
      lr_mode_table
    Example:
      python ele:wake 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      who: base
    
    """
    cmd = f'python ele:wake {ele_id}|{which} {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_wake', cmd_type='string_list')


def ele_wall3d(tao, ele_id, which, index, who, *, verbose=False, as_dict=True):
    """
    
    Element wall3d
    
    Parameters
    ----------
    ele_id
    which
    index
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:wall3d {ele_id}|{which} {index} {who}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    {index} is the index number in the ele%wall3d(:) array (size obtained from "ele:head").
    {who} is one of:
      base
      table
    Example:
      python ele:wall3d 3@1>>7|model 2 base
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      index: 1
      who: base
    
    """
    cmd = f'python ele:wall3d {ele_id}|{which} {index} {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_wall3d', cmd_type='string_list')


def ele_twiss(tao, ele_id, which, *, verbose=False, as_dict=True):
    """
    
    Element twiss
    
    Parameters
    ----------
    ele_id
    which
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:twiss {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:twiss 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:twiss {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_twiss', cmd_type='string_list')


def ele_control(tao, ele_id, which, *, verbose=False, as_dict=True):
    """
    
    Element control
    
    Parameters
    ----------
    ele_id
    which
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:control {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:control 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:control {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_control', cmd_type='string_list')


def ele_orbit(tao, ele_id, which, *, verbose=False, as_dict=True):
    """
    
    Element orbit
    
    Parameters
    ----------
    ele_id
    which
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:orbit {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:orbit 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:orbit {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_orbit', cmd_type='string_list')


def ele_mat6(tao, ele_id, which, who, *, verbose=False, as_dict=True):
    """
    
    Element mat6
    
    Parameters
    ----------
    ele_id
    which
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:mat6 {ele_id}|{which} {who}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    {who} is one of:
      mat6
      vec0
      err
    Example:
      python ele:mat6 3@1>>7|model mat6
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      who: mat6
    
    """
    cmd = f'python ele:mat6 {ele_id}|{which} {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_mat6', cmd_type='string_list')


def ele_taylor_field(tao, ele_id, which, index, who, *, verbose=False, as_dict=True):
    """
    
    Element taylor_field
    
    Parameters
    ----------
    ele_id
    which
    index
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:taylor_field {ele_id}|{which} {index} {who}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    {index} is the index number in the ele%taylor_field(:) array
    {who} is one of:
      base
      terms
    Example:
      python ele:taylor_field 3@1>>7|model 2 base
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      index: 1
      who: base
    
    """
    cmd = f'python ele:taylor_field {ele_id}|{which} {index} {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_taylor_field', cmd_type='string_list')


def ele_grid_field(tao, ele_id, which, index, who, *, verbose=False, as_dict=True):
    """
    
    Element grid_field
    
    Parameters
    ----------
    ele_id
    which
    index
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:grid_field {ele_id}|{which} {index} {who}
    where {ele_id} is an element name or index and {which} is one of
      model, base, design
    {index} is the index number in the ele%grid_field(:) array.
    {who} is one of:
      base, points
    Example:
      python ele:grid_field 3@1>>7|model 2 base
    This gives grid #2 of element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      index: 1
      who: base 
    
    """
    cmd = f'python ele:grid_field {ele_id}|{which} {index} {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_grid_field', cmd_type='string_list')


def ele_floor(tao, ele_id, which, *, where='end', verbose=False, as_dict=True):
    """
    
    Element floor coordinates. The output gives two lines. "Reference" is
    without element misalignments and "Actual" is with misalignments.
    
    Parameters
    ----------
    ele_id
    which
    where : default=end
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:floor {ele_id}|{which} {where}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    {where} is an optional argument which, if present, is one of
      beginning  ! Upstream end
      center     ! Middle of element
      end        ! Downstream end (default)
    Note: {where} ignored for photonic elements crystal, mirror, and multilayer_mirror.
    Example:
      python ele:floor 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      where 
    
    Example: 2
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      where: center
    
    """
    cmd = f'python ele:floor {ele_id}|{which} {where}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_floor', cmd_type='string_list')


def ele_photon(tao, ele_id, which, who, *, verbose=False, as_dict=True):
    """
    
    Element photon
    
    Parameters
    ----------
    ele_id
    which
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:photon {ele_id}|{which} {who}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    {who} is one of:
      base
      material
      surface
    Example:
      python ele:photon 3@1>>7|model base
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
      who: base
    
    """
    cmd = f'python ele:photon {ele_id}|{which} {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_photon', cmd_type='string_list')


def ele_lord_slave(tao, ele_id, which, *, verbose=False, as_dict=True):
    """
    
    Lists the lord/slave tree of an element.
    
    Parameters
    ----------
    ele_id
    which
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:lord_slave {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:lord_slave 3@1>>7|model
    This gives lord and slave info on element number 7 in branch 1 of universe 3.
    Note: The lord/slave info is independent of the setting of {which}.
    
    The output is a number of lines, each line giving information on an element (element index, etc.).
    Some lines begin with the word "Element". 
    After each "Element" line, there are a number of lines (possibly zero) that begin with the word "Slave or "Lord".
    These "Slave" and "Lord" lines are the slaves and lords of the "Element" element.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:lord_slave {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_lord_slave', cmd_type='string_list')


def ele_elec_multipoles(tao, ele_id, which, *, verbose=False, as_dict=True):
    """
    
    Element electric multipoles
    
    Parameters
    ----------
    ele_id
    which
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python ele:elec_multipoles {ele_id}|{which}
    where {ele_id} is an element name or index and {which} is one of
      model
      base
      design
    Example:
      python ele:elec_multipoles 3@1>>7|model
    This gives element number 7 in branch 1 of universe 3.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
      ele_id: 1@0>>1
      which: model
    
    """
    cmd = f'python ele:elec_multipoles {ele_id}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='ele_elec_multipoles', cmd_type='string_list')


def evaluate(tao, expression, *, flags='-array_out', verbose=False, as_dict=True):
    """
    
    Evaluate an expression. The result may be a vector.
    
    Parameters
    ----------
    expression
    flags : default=-array_out
        If -array_out, the output will be available in the tao_c_interface_com%c_real.!
    
    Returns
    -------
    string_list
        if '-array_out' not in flags
    real_array
        if '-array_out' in flags
    
    Notes
    -----
    Command syntax:
      python evaluate {flags} {expression}
    
    Example:
      python evaluate data::cbar.11[1:10]|model
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       expression: data::cbar.11[1:10]|model
    
    """
    cmd = f'python evaluate {flags} {expression}'
    if verbose: print(cmd)
    if '-array_out' not in flags:
        return __execute(tao, cmd, as_dict, method_name='evaluate', cmd_type='string_list')
    if '-array_out' in flags:
        return __execute(tao, cmd, as_dict, method_name='evaluate', cmd_type='real_array')


def em_field(tao, ele_id, which, x, y, z, t_or_z, *, verbose=False, as_dict=True):
    """
    
    EM field at a given point generated by a given element.
    
    Parameters
    ----------
    ele_id
    which
    x
    y
    z
    t_or_z
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python em_field {ele_id}|{which} {x}, {y}, {z}, {t_or_z}
    where {which} is one of:
      model
      base
      design
    Where:
      {x}, {y}  -- Transverse coords.
      {z}       -- Longitudainal coord with respect to entrance end of element.
      {t_or_z}  -- time or phase space z depending if lattice is setup for absolute time tracking.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ele_id: 1@0>>1
       which: model
       x: 0
       y: 0
       z: 0
       t_or_z: z
    
    """
    cmd = f'python em_field {ele_id}|{which} {x}, {y}, {z}, {t_or_z}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='em_field', cmd_type='string_list')


def enum(tao, enum_name, *, verbose=False, as_dict=True):
    """
    
    List of possible values for enumerated numbers.
    
    Parameters
    ----------
    enum_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python enum {enum_name}
    Example:
      python enum tracking_method
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       enum_name: tracking_method
    
    """
    cmd = f'python enum {enum_name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='enum', cmd_type='string_list')


def floor_plan(tao, graph, *, verbose=False, as_dict=True):
    """
    
    Floor plan elements
    
    Parameters
    ----------
    graph
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python floor_plan {graph}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       graph:
    
    """
    cmd = f'python floor_plan {graph}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='floor_plan', cmd_type='string_list')


def floor_orbit(tao, graph, *, verbose=False, as_dict=True):
    """
    
    (x, y) coordinates for drawing the particle orbit on a floor plan.
    
    Parameters
    ----------
    graph
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python floor_orbit {graph}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       graph: 
    
    """
    cmd = f'python floor_orbit {graph}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='floor_orbit', cmd_type='string_list')


def tao_global(tao, *, verbose=False, as_dict=True):
    """
    
    Global parameters
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python global
    Output syntax is parameter list form. See documentation at the beginning of this file.
    
    Note: The follow is intentionally left out:
      optimizer_allow_user_abort
      quiet
      single_step
      prompt_color
      prompt_string
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
    
    """
    cmd = f'python global'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='tao_global', cmd_type='string_list')


def help(tao, *, verbose=False, as_dict=True):
    """
    
    returns list of "help xxx" topics
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python help
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
    
    """
    cmd = f'python help'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='help', cmd_type='string_list')


def inum(tao, who, *, verbose=False, as_dict=True):
    """
    
    INUM
    
    Parameters
    ----------
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python inum {who}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       who: ix_universe
    
    """
    cmd = f'python inum {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='inum', cmd_type='string_list')


def lat_calc_done(tao, branch_name, *, verbose=False, as_dict=True):
    """
    
    Check if a lattice recalculation has been proformed since the last time
      "python lat_calc_done" was called.
    
    Parameters
    ----------
    branch_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python lat_calc_done
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       branch_name: 1@0
    
    """
    cmd = f'python lat_calc_done'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='lat_calc_done', cmd_type='string_list')


def lat_ele_list(tao, branch_name, *, verbose=False, as_dict=True):
    """
    
    Lattice element list.
    
    Parameters
    ----------
    branch_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python lat_ele {branch_name}
    {branch_name} should have the form:
      {ix_uni}@{ix_branch}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       branch_name: 1@0
    
    """
    cmd = f'python lat_ele {branch_name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='lat_ele_list', cmd_type='string_list')


def lat_general(tao, *, ix_universe='1', verbose=False, as_dict=True):
    """
    
    Lattice general
    
    Parameters
    ----------
    ix_universe : default=1
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python lat_general {ix_universe}
    
    Output syntax:
      branch_index;branch_name;n_ele_track;n_ele_max
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_universe: 1
    
    """
    cmd = f'python lat_general {ix_universe}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='lat_general', cmd_type='string_list')


def lat_list(tao, elements, who, *, ix_uni='1', ix_branch='0', which='model', flags='-array_out -track_only', verbose=False, as_dict=True):
    """
    
    List of parameters at ends of lattice elements
    
    Parameters
    ----------
    elements
    who
    ix_uni : default=1
    ix_branch : default=0
    which : default=model
    flags : optional, default=-array_out -track_only
    
    Returns
    -------
    string_list
        if ('-array_out' not in flags) or (who in ['ele.name'])
    real_array
        if ('-array_out' in flags or 'real:' in who) and (who not in ['orbit.state'])
    integer_array
        if '-array_out' in flags and who in ['orbit.state']
    
    Notes
    -----
    Command syntax:
      python lat_list {flags} {ix_uni}@{ix_branch}>>{elements}|{which} {who}
    where:
     Optional {flags} are:
      -no_slaves : If present, multipass_slave and super_slave elements will not be matched to.
      -track_only : If present, lord elements will not be matched to.
      -index_order : If present, order elements by element index instead of the standard s-position.
      -array_out : If present, the output will be available in the tao_c_interface_com%c_real or
        tao_c_interface_com%c_integer arrays. See the code below for when %c_real vs %c_integer is used.
        Note: Only a single {who} item permitted when -array_out is present.
    
      {which} is one of:
        model
        base
        design
    
      {who} is a comma deliminated list of:
        orbit.floor.x, orbit.floor.y, orbit.floor.z    ! Floor coords at particle orbit.
        orbit.spin.1, orbit.spin.2, orbit.spin.3,
        orbit.vec.1, orbit.vec.2, orbit.vec.3, orbit.vec.4, orbit.vec.5, orbit.vec.6,
        orbit.t, orbit.beta,
        orbit.state,     ! Note: state is an integer. alive$ = 1, anything else is lost.
        orbit.energy, orbit.pc,
        ele.name, ele.ix_ele, ele.ix_branch
        ele.a.beta, ele.a.alpha, ele.a.eta, ele.a.etap, ele.a.gamma, ele.a.phi,
        ele.b.beta, ele.b.alpha, ele.b.eta, ele.b.etap, ele.b.gamma, ele.b.phi,
        ele.x.eta, ele.x.etap,
        ele.y.eta, ele.y.etap,
        ele.s, ele.l
        ele.e_tot, ele.p0c
        ele.mat6, ele.vec0
    
      {elements} is a string to match element names to.
        Use "*" to match to all elements.
    
    Examples:
      python lat_list -track 3@0>>Q*|base ele.s,orbit.vec.2
      python lat_list 3@0>>Q*|base real:ele.s    
    
    Note: vector layout of mat6(6,6) is: [mat6(1,:), mat6(2,:), ...mat6(6,:)]
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_uni: 1  
       ix_branch: 0 
       elements: Q* 
       which: model
       who: orbit.floor.x
    
    """
    cmd = f'python lat_list {flags} {ix_uni}@{ix_branch}>>{elements}|{which} {who}'
    if verbose: print(cmd)
    if ('-array_out' not in flags) or (who in ['ele.name']):
        return __execute(tao, cmd, as_dict, method_name='lat_list', cmd_type='string_list')
    if ('-array_out' in flags or 'real:' in who) and (who not in ['orbit.state']):
        return __execute(tao, cmd, as_dict, method_name='lat_list', cmd_type='real_array')
    if '-array_out' in flags and who in ['orbit.state']:
        return __execute(tao, cmd, as_dict, method_name='lat_list', cmd_type='integer_array')


def lat_param_units(tao, param_name, *, verbose=False, as_dict=True):
    """
    
    Units of a parameter associated with a lattice or lattice element.
    
    Parameters
    ----------
    param_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python lat_param_units {param_name}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       param_name: L   
    
    """
    cmd = f'python lat_param_units {param_name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='lat_param_units', cmd_type='string_list')


def matrix(tao, ele1_id, ele2_id, *, verbose=False, as_dict=True):
    """
    
    Matrix value from the exit end of one element to the exit end of the other.
    
    Parameters
    ----------
    ele1_id
    ele2_id
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python matrix {ele1_id} {ele2_id}
    where:
      {ele1_id} is the start element.
      {ele2_id} is the end element.
    If {ele2_id} = {ele1_id}, the 1-turn transfer map is computed.
    Note: {ele2_id} should just be an element name or index without universe, branch, or model/base/design specification.
    
    Example:
      python matrix 2@1>>q01w|design q02w
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ele1_id: 2@1>>q01w|design
       ele2_id: q02w
    
    """
    cmd = f'python matrix {ele1_id} {ele2_id}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='matrix', cmd_type='string_list')


def merit(tao, *, verbose=False, as_dict=True):
    """
    
    Merit value.
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python merit
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
    
    """
    cmd = f'python merit'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='merit', cmd_type='string_list')


def orbit_at_s(tao, s, *, ix_uni='1', ix_branch='0', which='model', verbose=False, as_dict=True):
    """
    
    Twiss at given s position.
    
    Parameters
    ----------
    s
    ix_uni : default=1
    ix_branch : default=0
    which : default=model
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python orbit_at_s {ix_uni}@{ix_branch}>>{s}|{which}
    where:
      {which} is one of:
        model
        base
        design
      {s} is the longitudinal s-position.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_uni: 1
       ix_branch: 0
       s: 0.001
       which: model
    
    """
    cmd = f'python orbit_at_s {ix_uni}@{ix_branch}>>{s}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='orbit_at_s', cmd_type='string_list')


def place_buffer(tao, *, verbose=False, as_dict=True):
    """
    
    Output place command buffer and reset the buffer.
    The contents of the buffer are the place commands that the user has issued.
    
    Returns
    -------
    None
    
    Notes
    -----
    Command syntax:
      python place_buffer
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
    
    """
    cmd = f'python place_buffer'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='place_buffer', cmd_type='None')


def plot_curve(tao, curve_name, *, verbose=False, as_dict=True):
    """
    
    Curve information for a plot
    
    Parameters
    ----------
    curve_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python plot_curve {curve_name}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       curve_name: top.x.c1
    
    """
    cmd = f'python plot_curve {curve_name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_curve', cmd_type='string_list')


def plot_lat_layout(tao, ix_universe: 1, ix_branch: 0, *, verbose=False, as_dict=True):
    """
    
    Plot Lat_layout info
    
    Parameters
    ----------
    ix_universe: 1
    ix_branch: 0
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python plot_lat_layout {ix_universe}@{ix_branch}
    Note: The returned list of element positions is not ordered in increasing
          longitudinal position.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_universe: 1
       ix_branch: 0 
    
    """
    cmd = f'python plot_lat_layout {ix_universe}@{ix_branch}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_lat_layout', cmd_type='string_list')


def plot_list(tao, r_or_g, *, verbose=False, as_dict=True):
    """
    
    List of plot templates or plot regions.
    
    Parameters
    ----------
    r_or_g
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python plot_list {r_or_g}
    where "{r/g}" is:
      "r"      ! list regions
      "t"      ! list template plots
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       r_or_g: r
    
    """
    cmd = f'python plot_list {r_or_g}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_list', cmd_type='string_list')


def plot_graph(tao, graph_name, *, verbose=False, as_dict=True):
    """
    
    Graph
    
    Parameters
    ----------
    graph_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python plot_graph {graph_name}
    {graph_name} is in the form:
      {p_name}.{g_name}
    where
      {p_name} is the plot region name if from a region or the plot name if a template plot.
      This name is obtained from the python plot_list command.
      {g_name} is the graph name obtained from the python plot1 command.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       graph_name: r13.g
    
    """
    cmd = f'python plot_graph {graph_name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_graph', cmd_type='string_list')


def plot_histogram(tao, curve_name, *, verbose=False, as_dict=True):
    """
    
    Plot Histogram
    
    Parameters
    ----------
    curve_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python plot_histograph {curve_name}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       curve_name: c1
    
    """
    cmd = f'python plot_histograph {curve_name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_histogram', cmd_type='string_list')


def plot_plot_manage(tao, plot_location, plot_name, n_graph, graph1_name, graph2_name, graphN_name, *, verbose=False, as_dict=True):
    """
    
    Template plot creation or destruction.
    
    Parameters
    ----------
    plot_location
    plot_name
    n_graph
    graph1_name
    graph2_name
    graphN_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      pyton plot_plot_manage {plot_location}^^{plot_name}^^
                             {n_graph}^^{graph1_name}^^{graph2_name}...{graphN_name}
    Use "@Tnnn" sytax for {plot_location} to place a plot. A plot may be placed in a 
    spot where there is already a template.
    If {n_graph} is set to -1 then just delete the plot.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       plot_location: r11
       plot_name: beta
       n_graph: 1
       graph1_name: g
       graph2_name:
       graphN_name:
    
    """
    cmd = f'pyton plot_plot_manage {plot_location}^^{plot_name}^^{n_graph}^^{graph1_name}^^{graph2_name}...{graphN_name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_plot_manage', cmd_type='string_list')


def plot_curve_manage(tao, graph_name, curve_index, curve_name, *, verbose=False, as_dict=True):
    """
    
    Template plot curve creation/destruction
    
    Parameters
    ----------
    graph_name
    curve_index
    curve_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      pyton plot_curve_manage {graph_name}^^{curve_index}^^{curve_name}
    If {curve_index} corresponds to an existing curve then this curve is deleted.
    In this case the {curve_name} is ignored and does not have to be present.
    If {curve_index} does not not correspond to an existing curve, {curve_index}
    must be one greater than the number of curves.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       graph_name: g
       curve_index: 1
       curve_name: c1
    
    """
    cmd = f'pyton plot_curve_manage {graph_name}^^{curve_index}^^{curve_name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_curve_manage', cmd_type='string_list')


def plot_graph_manage(tao, plot_name, graph_index, graph_name, *, verbose=False, as_dict=True):
    """
    
    Template plot graph creation/destruction
    
    Parameters
    ----------
    plot_name
    graph_index
    graph_name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      pyton plot_graph_manage {plot_name}^^{graph_index}^^{graph_name}
    If {graph_index} corresponds to an existing graph then this graph is deleted.
    In this case the {graph_name} is ignored and does not have to be present.
    If {graph_index} does not not correspond to an existing graph, {graph_index}
    must be one greater than the number of graphs.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       plot_name: beta
       graph_index: 1
       graph_name: g
    
    """
    cmd = f'pyton plot_graph_manage {plot_name}^^{graph_index}^^{graph_name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_graph_manage', cmd_type='string_list')


def plot_line(tao, region_name, graph_name, curve_name, x_or_y, *, verbose=False, as_dict=True):
    """
    
    Points used to construct a smooth line for a plot curve.
    
    Parameters
    ----------
    region_name
    graph_name
    curve_name
    x_or_y
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python plot_line {region_name}.{graph_name}.{curve_name} {x_or_y}
    Optional {x-or-y} may be set to "x" or "y" to get the smooth line points x or y 
    component put into the real array buffer.
    Note: The plot must come from a region, and not a template, since no template plots 
          have associated line data.
    Examples:
      python plot_line r13.g.a   ! String array output.
      python plot_line r13.g.a x ! x-component of line points loaded into the real array buffer.
      python plot_line r13.g.a y ! y-component of line points loaded into the real array buffer.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       region_name: r11
       graph_name: g
       curve_name: a
       x_or_y: x
    
    """
    cmd = f'python plot_line {region_name}.{graph_name}.{curve_name} {x_or_y}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_line', cmd_type='string_list')


def plot_symbol(tao, region_name, graph_name, curve_name, x_or_y, *, verbose=False, as_dict=True):
    """
    
    Locations to draw symbols for a plot curve.
    
    Parameters
    ----------
    region_name
    graph_name
    curve_name
    x_or_y
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python plot_symbol {region_name}.{graph_name}.{curve_name} {x_or_y}
    Optional {x_or_y} may be set to "x" or "y" to get the symbol x or y 
    positions put into the real array buffer.
    Note: The plot must come from a region, and not a template, 
          since no template plots have associated symbol data.
    Examples:
      python plot_symbol r13.g.a       ! String array output.
      python plot_symbol r13.g.a x     ! x-component of the symbol positions 
                                         loaded into the real array buffer.
      python plot_symbol r13.g.a y     ! y-component of the symbol positions 
                                         loaded into the real array buffer.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       region_name: r11
       graph_name: g
       curve_name: a
       x_or_y: x 
    
    """
    cmd = f'python plot_symbol {region_name}.{graph_name}.{curve_name} {x_or_y}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_symbol', cmd_type='string_list')


def plot_transfer(tao, from_plot, to_plot, *, verbose=False, as_dict=True):
    """
    
    Transfer plot parameters from the "from plot" to the "to plot" (or plots).
    
    Parameters
    ----------
    from_plot
    to_plot
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python plot_transfer {from_plot} {to_plot}
    To avoid confusion, use "@Tnnn" and "@Rnnn" syntax for {from_plot}.
    If {to_plot} is not present and {from_plot} is a template plot, the "to plots" 
     are the equivalent region plots with the same name. And vice versa 
     if {from_plot} is a region plot.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       from_plot: r13
       to_plot: 
    
    """
    cmd = f'python plot_transfer {from_plot} {to_plot}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot_transfer', cmd_type='string_list')


def plot1(tao, name, *, verbose=False, as_dict=True):
    """
    
    Info on a given plot.
    
    Parameters
    ----------
    name
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python plot1 {name}
    {name} should be the region name if the plot is associated with a region.
    Output syntax is parameter list form. See documentation at the beginning of this file.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       name: r11
    
    """
    cmd = f'python plot1 {name}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='plot1', cmd_type='string_list')


def shape_list(tao, who, *, verbose=False, as_dict=True):
    """
    
    lat_layout and floor_plan shapes list
    
    Parameters
    ----------
    who
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python shape_list {who}
    {who} is one of:
      lat_layout
      floor_plan
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       who: floor_plan  
    
    """
    cmd = f'python shape_list {who}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='shape_list', cmd_type='string_list')


def shape_manage(tao, who, index, add_or_delete, *, verbose=False, as_dict=True):
    """
    
    element shape creation or destruction
    
    Parameters
    ----------
    who
    index
    add_or_delete
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python shape_manage {who} {index} {add_or_delete}
    
    {who} is one of:
      lat_layout
      floor_plan
    {add_or_delete} is one of:
      add     -- Add a shape at {index}. 
                 Shapes with higher index get moved up one to make room.
      delete  -- Delete shape at {index}. 
                 Shapes with higher index get moved down one to fill the gap.
    
    Example:
      python shape_manage floor_plan 2 add
    Note: After adding a shape use "python shape_set" to set shape parameters.
    This is important since an added shape is in a ill-defined state.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       who: floor_plan
       index: 1
       add_or_delete: add
    
    """
    cmd = f'python shape_manage {who} {index} {add_or_delete}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='shape_manage', cmd_type='string_list')


def shape_pattern_list(tao, *, ix_pattern='', verbose=False, as_dict=True):
    """
    
    List of shape patterns
    
    Parameters
    ----------
    ix_pattern : optional
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python shape_pattern_list {ix_pattern}
    If optional {ix_pattern} index is omitted then list all the patterns
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_pattern: 
    
    Example: 2
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_pattern: 1 
    
    """
    cmd = f'python shape_pattern_list {ix_pattern}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='shape_pattern_list', cmd_type='string_list')


def shape_pattern_manage(tao, ix_pattern, pat_name, pat_line_width, *, verbose=False, as_dict=True):
    """
    
    Add or remove shape pattern
    
    Parameters
    ----------
    ix_pattern
    pat_name
    pat_line_width
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python shape_pattern_manage {ix_pattern}^^{pat_name}^^{pat_line_width}
    where:
      {ix_pattern}      -- Pattern index. Patterns with higher indexes will be moved up 
                                          if adding a pattern and down if deleting.
      {pat_name}        -- Pattern name.
      {pat_line_width}  -- Line width. Integer. If set to "delete" then section 
                                                will be deleted.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_pattern : 5
       pat_name : new_pat
       pat_line_width : 1
    
    """
    cmd = f'python shape_pattern_manage {ix_pattern}^^{pat_name}^^{pat_line_width}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='shape_pattern_manage', cmd_type='string_list')


def shape_pattern_point_manage(tao, ix_pattern, ix_point, s, x, *, verbose=False, as_dict=True):
    """
    
    Add or remove shape pattern point
    
    Parameters
    ----------
    ix_pattern
    ix_point
    s
    x
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python shape_pattern_point_manage {ix_pattern}^^{ix_point}^^{s}^^{x}
    where:
      {ix_pattern}      -- Pattern index.
      {ix_point}        -- Point index. Points of higher indexes will be moved up
                                        if adding a point and down if deleting.
      {s}, {x}          -- Point location. If {s} is "delete" then delete the point.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       ix_pattern: 1
       ix_point: 1
       s: 0
       x: 0
    
    """
    cmd = f'python shape_pattern_point_manage {ix_pattern}^^{ix_point}^^{s}^^{x}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='shape_pattern_point_manage', cmd_type='string_list')


def shape_set(tao, who, shape_index, ele_name, shape, color, shape_size, type_label, shape_draw, multi_shape, line_width, *, verbose=False, as_dict=True):
    """
    
    lat_layout or floor_plan shape set
    
    Parameters
    ----------
    who
    shape_index
    ele_name
    shape
    color
    shape_size
    type_label
    shape_draw
    multi_shape
    line_width
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python shape_set {who}^^{shape_index}^^{ele_name}^^{shape}^^{color}^^
                       {shape_size}^^{type_label}^^{shape_draw}^^
                       {multi_shape}^^{line_width}
    {who} is one of:
      lat_layout
      floor_plan
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       who: floor_plan
       shape_index: 1
       ele_name: Q1
       shape:
       color:
       shape_size:
       type_label:
       shape_draw:
       multi_shape:
       line_width:
    
    """
    cmd = f'python shape_set {who}^^{shape_index}^^{ele_name}^^{shape}^^{color}^^{shape_size}^^{type_label}^^{shape_draw}^^{multi_shape}^^{line_width}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='shape_set', cmd_type='string_list')


def show(tao, line, *, verbose=False, as_dict=True):
    """
    
    Show command pass through
    
    Parameters
    ----------
    line
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python show {line}
    {line} is the string to pass through to the show command.
    Example:
      python show lattice -python
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       line: -python
    
    """
    cmd = f'python show {line}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='show', cmd_type='string_list')


def species_to_int(tao, species_str, *, verbose=False, as_dict=True):
    """
    
    Convert species name to corresponding integer
    
    Parameters
    ----------
    species_str
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python species_to_int {species_str}
    Example:
      python species_to_int CO2++
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       species_str: CO2++
    
    """
    cmd = f'python species_to_int {species_str}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='species_to_int', cmd_type='string_list')


def species_to_str(tao, species_int, *, verbose=False, as_dict=True):
    """
    
    Convert species integer id to corresponding
    
    Parameters
    ----------
    species_int
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python species_to_str {species_int}
    Example:
      python species_to_str -1     ! Returns 'Electron'
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       species_int: -1
    
    """
    cmd = f'python species_to_str {species_int}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='species_to_str', cmd_type='string_list')


def spin_polarization(tao, *, ix_uni='1', ix_branch='0', which='model', verbose=False, as_dict=True):
    """
    
    Spin information
    
    Parameters
    ----------
    ix_uni : default=1
    ix_branch : default=0
    which : default=model
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python spin {ix_uni}@{ix_branch}|{which}
    where {which} is one of:
      model
      base
      design
    Example:
      python spin 1@0|model
    
    Note: This command is under development. If you want to use please contact David Sagan.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args: 
       ix_uni: 1
       ix_branch: 0
       which: model
    
    """
    cmd = f'python spin {ix_uni}@{ix_branch}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='spin_polarization', cmd_type='string_list')


def super_universe(tao, *, verbose=False, as_dict=True):
    """
    
    Super_Universe information
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python super_universe
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args: 
    
    """
    cmd = f'python super_universe'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='super_universe', cmd_type='string_list')


def twiss_at_s(tao, ix_uni, ix_branch, s, which, *, verbose=False, as_dict=True):
    """
    
    Twiss at given s position
    
    Parameters
    ----------
    ix_uni
    ix_branch
    s
    which
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python twiss_at_s {ix_uni}@{ix_branch}>>{s}|{which}
    where {which} is one of:
      model
      base
      design
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args: 
       ix_uni: 1
       ix_branch: 0
       s: 0
       which: model 
    
    """
    cmd = f'python twiss_at_s {ix_uni}@{ix_branch}>>{s}|{which}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='twiss_at_s', cmd_type='string_list')


def universe(tao, ix_universe, *, verbose=False, as_dict=True):
    """
    
    Universe info
    
    Parameters
    ----------
    ix_universe
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python universe {ix_universe}
    Use "python global" to get the number of universes.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args: 
       ix_universe: 1
    
    """
    cmd = f'python universe {ix_universe}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='universe', cmd_type='string_list')


def var(tao, var, *, verbose=False, as_dict=True):
    """
    
    Info on an individual variable
    
    Parameters
    ----------
    var
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python var {var}        or
      python var {var} slaves
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args: 
       var: quad_k1[1]
    
    """
    cmd = f'python var {var}        or'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='var', cmd_type='string_list')


def var_create(tao, var_name, ele_name, attribute, universes, weight, step, low_lim, high_lim, merit_type, good_user, key_bound, key_delta, *, verbose=False, as_dict=True):
    """
    
    Create a single variable
    
    Parameters
    ----------
    var_name
    ele_name
    attribute
    universes
    weight
    step
    low_lim
    high_lim
    merit_type
    good_user
    key_bound
    key_delta
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python var_create {var_name}^^{ele_name}^^{attribute}^^{universes}^^
                        {weight}^^{step}^^{low_lim}^^{high_lim}^^{merit_type}^^
                        {good_user}^^{key_bound}^^{key_delta}
    {var_name} is something like "kick[5]".
    Before using var_create, setup the appropriate v1_var array using 
    the "python var_v1_create" command.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       var_name: kick[5]
       ele_name: Q01W
       attribute: L
       universes: 1
       weight: 0.001
       step: 0.001
       low_lim: -10
       high_lim: 10
       merit_type: 
       good_user: 1
       key_bound: T
       key_delta: 0.01 
    
    """
    cmd = f'python var_create {var_name}^^{ele_name}^^{attribute}^^{universes}^^{weight}^^{step}^^{low_lim}^^{high_lim}^^{merit_type}^^{good_user}^^{key_bound}^^{key_delta}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='var_create', cmd_type='string_list')


def var_general(tao, *, verbose=False, as_dict=True):
    """
    
    List of all variable v1 arrays
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python var_general
    Output syntax:
      {v1_var name};{v1_var%v lower bound};{v1_var%v upper bound}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
    
    """
    cmd = f'python var_general'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='var_general', cmd_type='string_list')


def var_v_array(tao, v1_var, *, verbose=False, as_dict=True):
    """
    
    List of variables for a given data_v1.
    
    Parameters
    ----------
    v1_var
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python var_v_array {v1_var}
    Example:
      python var_v_array quad_k1
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       v1_var: quad_k1
    
    """
    cmd = f'python var_v_array {v1_var}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='var_v_array', cmd_type='string_list')


def var_v1_array(tao, v1_var, *, verbose=False, as_dict=True):
    """
    
    List of variables in a given variable v1 array
    
    Parameters
    ----------
    v1_var
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python var_v1_array {v1_var}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       v1_var: quad_k1 
    
    """
    cmd = f'python var_v1_array {v1_var}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='var_v1_array', cmd_type='string_list')


def var_v1_create(tao, v1_name, n_var_min, n_var_max, *, verbose=False, as_dict=True):
    """
    
    Create a v1 variable structure along with associated var array.
    
    Parameters
    ----------
    v1_name
    n_var_min
    n_var_max
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python var_v1_create {v1_name} {n_var_min} {n_var_max}
    {n_var_min} and {n_var_max} are the lower and upper bounds of the var
    Example:
      python var_v1_create quad_k1 0 45
    This example creates a v1 var structure called "quad_k1" with an associated
    variable array that has the range [0, 45].
    
    Use the "set variable" command to set a created variable parameters.
    In particular, to slave a lattice parameter to a variable use the command:
      set {v1_name}|ele_name = {lat_param}
    where {lat_param} is of the form {ix_uni}@{ele_name_or_location}{param_name}]
    Examples:
      set quad_k1[2]|ele_name = 2@q01w[k1]
      set quad_k1[2]|ele_name = 2@0>>10[k1]
    Note: When setting multiple variable parameters, 
          temporarily toggle s%global%lattice_calc_on to False
      ("set global lattice_calc_on = F") to prevent Tao trying to evaluate the 
    partially created variable and generating unwanted error messages.
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       v1_name: quad_k1 
       n_var_min: 0 
       n_var_max: 45 
    
    """
    cmd = f'python var_v1_create {v1_name} {n_var_min} {n_var_max}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='var_v1_create', cmd_type='string_list')


def var_v1_destroy(tao, v1_datum, *, verbose=False, as_dict=True):
    """
    
    Destroy a v1 var structure along with associated var sub-array.
    
    Parameters
    ----------
    v1_datum
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python var_v1_destroy {v1_datum}
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       v1_datum: quad_k1
    
    """
    cmd = f'python var_v1_destroy {v1_datum}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='var_v1_destroy', cmd_type='string_list')


def wave(tao, what, *, verbose=False, as_dict=True):
    """
    
    Wave analysis info.
    
    Parameters
    ----------
    what
    
    Returns
    -------
    string_list
    
    Notes
    -----
    Command syntax:
      python wave {what}
    Where {what} is one of:
      params
      loc_header
      locations
      plot1, plot2, plot3
    
    Examples
    --------
    Example: 1
     init: $ACC_ROOT_DIR/tao/examples/cesr/tao.init
     args:
       what: params
    
    """
    cmd = f'python wave {what}'
    if verbose: print(cmd)
    return __execute(tao, cmd, as_dict, method_name='wave', cmd_type='string_list')

