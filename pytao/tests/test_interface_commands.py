
import os
from pytao import Tao
from pytao import interface_commands


def test_beam_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/csr_beam_tracking/tao.init -noplot'))
    ret = interface_commands.beam(tao, ix_uni='1')
            
        
def test_beam_init_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/csr_beam_tracking/tao.init -noplot'))
    ret = interface_commands.beam_init(tao, ix_uni='1')
            
        
def test_bmad_com_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.bmad_com(tao)
            
        
def test_branch1_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.branch1(tao, ix_uni='1', ix_branch='0')
            
        
def test_bunch_comb_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/csr_beam_tracking/tao.init -noplot'))
    ret = interface_commands.bunch_comb(tao, who='x.beta')
            
        
def test_bunch_params_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/csr_beam_tracking/tao.init -noplot'))
    ret = interface_commands.bunch_params(tao, ele_id='end', which='model')
            
        
def test_bunch1_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/csr_beam_tracking/tao.init -noplot'))
    ret = interface_commands.bunch1(tao, ele_id='end', coordinate='x', which='model', ix_bunch='1')
            
        
def test_building_wall_list_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall -noplot'))
    ret = interface_commands.building_wall_list(tao, ix_section='')
            
        
def test_building_wall_list_2():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall -noplot'))
    ret = interface_commands.building_wall_list(tao, ix_section='1')
            
        
def test_building_wall_graph_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall -noplot'))
    ret = interface_commands.building_wall_graph(tao, graph='floor_plan.g')
            
        
def test_building_wall_point_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall -noplot'))
    ret = interface_commands.building_wall_point(tao, ix_section='1', ix_point='1', z='0', x='0', radius='0', z_center='0', x_center='0')
            
        
def test_building_wall_section_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.building_wall_section(tao, ix_section='1', sec_name='test', sec_constraint='none')
            
        
def test_constraints_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.constraints(tao, who='data')
            
        
def test_constraints_2():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.constraints(tao, who='var')
            
        
def test_data_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.data(tao, ix_uni='', d2_name='twiss', d1_name='end', dat_index='1')
            
        
def test_data_2():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.data(tao, ix_uni='1', d2_name='twiss', d1_name='end', dat_index='1')
            
        
def test_data_d_array_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.data_d_array(tao, ix_uni='1', d2_name='twiss', d1_name='end')
            
        
def test_data_d1_array_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.data_d1_array(tao, ix_uni='1', d2_datum='twiss')
            
        
def test_data_d2_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.data_d2(tao, ix_uni='1', d2_name='twiss')
            
        
def test_data_d2_array_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.data_d2_array(tao, ix_uni='1')
            
        
def test_data_d2_create_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.data_d2_create(tao, ix_uni='1', d2_name='orbit', n_d1_data='2', d_data_arrays_name_min_max='x^^0^^45^^y^^1^^47')
            
        
def test_data_d2_destroy_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.data_d2_destroy(tao, d2_name='orbit')
            
        
def test_data_parameter_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.data_parameter(tao, data_array='twiss.end', parameter='model_value')
            
        
def test_data_set_design_value_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.data_set_design_value(tao)
            
        
def test_datum_create_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.datum_create(tao, datum_name='twiss.end[6]', data_type='beta.y', ele_ref_name='', ele_start_name='', ele_name='P1', merit_type='target', meas='0', good_meas='T', ref='0', good_ref='T', weight='0.3', good_user='T', data_source='lat', eval_point='END', s_offset='0', ix_bunch='1', invalid_value='0')
            
        
def test_datum_has_ele_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.datum_has_ele(tao, datum_type='twiss.end')
            
        
def test_derivative_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.derivative(tao)
            
        
def test_ele_ac_kicker_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_ac_kicker(tao, ele_id='1@0>>1', which='model')
            
        
def test_ele_cartesian_map_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_em_field -noplot'))
    ret = interface_commands.ele_cartesian_map(tao, ele_id='1@0>>1', which='model', index='1', who='base')
            
        
def test_ele_chamber_wall_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall3d -noplot'))
    ret = interface_commands.ele_chamber_wall(tao, ele_id='1@0>>1', which='model', index='1', who='x')
            
        
def test_ele_control_var_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_control_var(tao, ele_id='1@0>>873', which='model')
            
        
def test_ele_cylindrical_map_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_em_field -noplot'))
    ret = interface_commands.ele_cylindrical_map(tao, ele_id='1@0>>5', which='model', index='1', who='base')
            
        
def test_ele_elec_multipoles_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_elec_multipoles(tao, ele_id='1@0>>1', which='model')
            
        
def test_ele_floor_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_floor(tao, ele_id='1@0>>1', which='model', where='')
            
        
def test_ele_floor_2():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_floor(tao, ele_id='1@0>>1', which='model', where='center')
            
        
def test_ele_grid_field_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_grid -noplot'))
    ret = interface_commands.ele_grid_field(tao, ele_id='1@0>>1', which='model', index='1', who='base')
            
        
def test_ele_gen_attribs_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_gen_attribs(tao, ele_id='1@0>>1', which='model')
            
        
def test_ele_head_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_head(tao, ele_id='1@0>>1', which='model')
            
        
def test_ele_lord_slave_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_lord_slave(tao, ele_id='1@0>>1', which='model')
            
        
def test_ele_mat6_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_mat6(tao, ele_id='1@0>>1', which='model', who='mat6')
            
        
def test_ele_methods_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_methods(tao, ele_id='1@0>>1', which='model')
            
        
def test_ele_multipoles_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_multipoles(tao, ele_id='1@0>>1', which='model')
            
        
def test_ele_orbit_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_orbit(tao, ele_id='1@0>>1', which='model')
            
        
def test_ele_param_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_photon -noplot'))
    ret = interface_commands.ele_param(tao, ele_id='1@0>>1', which='model', who='orbit.vec.1')
            
        
def test_ele_photon_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_photon -noplot'))
    ret = interface_commands.ele_photon(tao, ele_id='1@0>>1', which='model', who='base')
            
        
def test_ele_spin_taylor_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_spin -noplot'))
    ret = interface_commands.ele_spin_taylor(tao, ele_id='1@0>>2', which='model')
            
        
def test_ele_taylor_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_taylor -noplot'))
    ret = interface_commands.ele_taylor(tao, ele_id='1@0>>34', which='model')
            
        
def test_ele_twiss_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ele_twiss(tao, ele_id='1@0>>1', which='model')
            
        
def test_ele_wake_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_wake -noplot'))
    ret = interface_commands.ele_wake(tao, ele_id='1@0>>1', which='model', who='sr_long')
            
        
def test_ele_wall3d_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall3d -noplot'))
    ret = interface_commands.ele_wall3d(tao, ele_id='1@0>>1', which='model', index='1', who='table')
            
        
def test_evaluate_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.evaluate(tao, expression='data::cbar.11[1:10]|model')
            
        
def test_em_field_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.em_field(tao, ele_id='1@0>>22', which='model', x='0', y='0', z='0', t_or_z='0')
            
        
def test_enum_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.enum(tao, enum_name='tracking_method')
            
        
def test_floor_plan_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.floor_plan(tao, graph='r13.g')
            
        
def test_floor_orbit_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_floor_orbit -noplot'))
    ret = interface_commands.floor_orbit(tao, graph='r33.g')
            
        
def test_tao_global_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.tao_global(tao)
            
        
def test_help_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.help(tao)
            
        
def test_inum_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.inum(tao, who='ix_universe')
            
        
def test_lat_calc_done_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.lat_calc_done(tao, branch_name='1@0')
            
        
def test_lat_ele_list_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.lat_ele_list(tao, branch_name='1@0')
            
        
def test_lat_branch_list_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.lat_branch_list(tao, ix_uni='1')
            
        
def test_lat_list_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.lat_list(tao, ix_uni='1', ix_branch='0', elements='Q*', which='model', who='orbit.floor.x')
            
        
def test_lat_list_2():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.lat_list(tao, ix_uni='1', ix_branch='0', elements='Q*', which='design', who='ele.ix_ele')
            
        
def test_lat_param_units_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.lat_param_units(tao, param_name='L')
            
        
def test_matrix_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.matrix(tao, ele1_id='1@0>>q01w|design', ele2_id='q02w')
            
        
def test_merit_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.merit(tao)
            
        
def test_orbit_at_s_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.orbit_at_s(tao, ix_uni='1', ele='10', s_offset='0.7', which='model')
            
        
def test_place_buffer_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.place_buffer(tao)
            
        
def test_plot_curve_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.plot_curve(tao, curve_name='r13.g.a')
            
        
def test_plot_lat_layout_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.plot_lat_layout(tao, ix_uni='1', ix_branch='0')
            
        
def test_plot_list_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.plot_list(tao, r_or_g='r')
            
        
def test_plot_graph_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.plot_graph(tao, graph_name='beta.g')
            
        
def test_plot_histogram_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.plot_histogram(tao, curve_name='r33.g.x')
            
        
def test_plot_template_manage_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.plot_template_manage(tao, template_location='@T1', template_name='beta', n_graph='2', graph_names='g1^^g2')
            
        
def test_plot_curve_manage_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.plot_curve_manage(tao, graph_name='beta.g', curve_index='1', curve_name='r13.g.a')
            
        
def test_plot_graph_manage_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.plot_graph_manage(tao, plot_name='beta', graph_index='1', graph_name='beta.g')
            
        
def test_plot_line_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_plot_line -external_plotting -noplot'))
    ret = interface_commands.plot_line(tao, region_name='beta', graph_name='g', curve_name='a', x_or_y='')
            
        
def test_plot_line_2():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_plot_line -external_plotting -noplot'))
    ret = interface_commands.plot_line(tao, region_name='beta', graph_name='g', curve_name='a', x_or_y='y')
            
        
def test_plot_symbol_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_plot_line -external_plotting -noplot'))
    ret = interface_commands.plot_symbol(tao, region_name='r13', graph_name='g', curve_name='a', x_or_y='')
            
        
def test_plot_symbol_2():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_plot_line -external_plotting -noplot'))
    ret = interface_commands.plot_symbol(tao, region_name='r13', graph_name='g', curve_name='a', x_or_y='y')
            
        
def test_plot_transfer_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.plot_transfer(tao, from_plot='r13', to_plot='r23')
            
        
def test_plot1_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.plot1(tao, name='beta')
            
        
def test_ptc_com_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ptc_com(tao)
            
        
def test_ring_general_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.ring_general(tao, ix_uni='1', ix_branch='0', which='model')
            
        
def test_shape_list_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.shape_list(tao, who='floor_plan')
            
        
def test_shape_manage_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.shape_manage(tao, who='floor_plan', index='1', add_or_delete='add')
            
        
def test_shape_pattern_list_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_shape -noplot'))
    ret = interface_commands.shape_pattern_list(tao, ix_pattern='')
            
        
def test_shape_pattern_manage_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_shape -noplot'))
    ret = interface_commands.shape_pattern_manage(tao, ix_pattern='1', pat_name='new_pat', pat_line_width='1')
            
        
def test_shape_pattern_point_manage_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_shape -noplot'))
    ret = interface_commands.shape_pattern_point_manage(tao, ix_pattern='1', ix_point='1', s='0', x='0')
            
        
def test_shape_set_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.shape_set(tao, who='floor_plan', shape_index='1', ele_name='Q1', shape='circle', color='', shape_size='', type_label='', shape_draw='', multi_shape='', line_width='')
            
        
def test_show_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.show(tao, line='-python')
            
        
def test_species_to_int_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.species_to_int(tao, species_str='electron')
            
        
def test_species_to_str_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.species_to_str(tao, species_int='-1')
            
        
def test_spin_invariant_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.spin_invariant(tao, who='l0', ix_uni='1', ix_branch='0', which='model')
            
        
def test_spin_polarization_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.spin_polarization(tao, ix_uni='1', ix_branch='0', which='model')
            
        
def test_spin_resonance_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.spin_resonance(tao, ix_uni='1', ix_branch='0', which='model')
            
        
def test_super_universe_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.super_universe(tao)
            
        
def test_twiss_at_s_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.twiss_at_s(tao, ix_uni='1', ele='10', s_offset='0.7', which='model')
            
        
def test_universe_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.universe(tao, ix_uni='1')
            
        
def test_var_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.var(tao, var='quad[1]', slaves='')
            
        
def test_var_2():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.var(tao, var='quad[1]', slaves='slaves')
            
        
def test_var_create_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_optics_matching -noplot'))
    ret = interface_commands.var_create(tao, var_name='quad[1]', ele_name='Q1', attribute='L', universes='1', weight='0.001', step='0.001', low_lim='-10', high_lim='10', merit_type='', good_user='T', key_bound='T', key_delta='0.01')
            
        
def test_var_general_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.var_general(tao)
            
        
def test_var_v_array_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.var_v_array(tao, v1_var='quad_k1')
            
        
def test_var_v1_array_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.var_v1_array(tao, v1_var='quad_k1')
            
        
def test_var_v1_create_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.var_v1_create(tao, v1_name='quad_k1', n_var_min='0', n_var_max='45')
            
        
def test_var_v1_destroy_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.var_v1_destroy(tao, v1_datum='quad_k1')
            
        
def test_wave_1():
    
    tao = Tao(os.path.expandvars('-init $ACC_ROOT_DIR/regression_tests/python_test/cesr/tao.init -noplot'))
    ret = interface_commands.wave(tao, who='params')
            
        