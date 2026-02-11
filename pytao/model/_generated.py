# vi: syntax=python sw=4 ts=4 sts=4
"""
This file is auto-generated; do not hand-edit it.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    ClassVar,
    Sequence,
)

from pydantic import Field

from .base import (
    TaoModel,
    TaoSettableModel,
)
from .types import (
    FloatSequence,
    IntSequence,
)

logger = logging.getLogger(__name__)


class Beam(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe beam`, for example.

    Attributes
    ----------
    always_reinit : bool
    comb_ds_save : float
        Master parameter for %bunch_params_comb(:)%ds_save
    ds_save : float
        Min distance between points.
    dump_at : str
    dump_file : str
    saved_at : str
    track_beam_in_universe : bool
        Beam tracking enabled in this universe?
    track_end : str
    track_start : str
        Tracking start element.
    """

    _tao_command_attr_: ClassVar[str] = "beam"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {"ix_branch": 0, "ix_uni": 1}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    always_reinit: bool = False
    comb_ds_save: float = Field(
        default=-1.0, description="Master parameter for %bunch_params_comb(:)%ds_save"
    )
    ds_save: float = Field(
        default=-1.0, description="Min distance between points.", frozen=True
    )
    dump_at: str = ""
    dump_file: str = ""
    saved_at: str = ""
    track_beam_in_universe: bool = Field(
        default=False, description="Beam tracking enabled in this universe?", frozen=True
    )
    track_end: str = ""
    track_start: str = Field(default="", description="Tracking start element.")


class BeamInit(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe beam_init`, for example.

    Attributes
    ----------
    a_emit : float
        a-mode emittance
    a_norm_emit : float
        a-mode normalized emittance (emit * beta * gamma)
    b_emit : float
        b-mode emittance
    b_norm_emit : float
        b-mode normalized emittance (emit * beta * gamma)
    bunch_charge : float
        charge (Coul) in a bunch.
    center : sequence of floats
        Bench phase space center offset relative to reference.
    center_jitter : sequence of floats
        Bunch center rms jitter
    distribution_type : Sequence[str]
        distribution type (in x-px, y-py, and z-pz planes) "ELLIPSE", "KV",
    "GRID", "FILE", "RAN_GAUSS" or "" = "RAN_GAUSS"
    dpz_dz : float
        Correlation of Pz with long position.
    dt_bunch : float
        Time between bunches.
    ellipse_1_n_ellipse : int or None
    ellipse_1_part_per_ellipse : int or None
    ellipse_1_sigma_cutoff : float or None
    ellipse_2_n_ellipse : int or None
    ellipse_2_part_per_ellipse : int or None
    ellipse_2_sigma_cutoff : float or None
    ellipse_3_n_ellipse : int or None
    ellipse_3_part_per_ellipse : int or None
    ellipse_3_sigma_cutoff : float or None
    emit_jitter : sequence of floats
        a and b bunch emittance rms jitter normalized to emittance
    full_6d_coupling_calc : bool
        Use V from 6x6 1-turn mat to match distribution? Else use 4x4 1-turn
    mat used.
    grid_1_n_px : int or None
    grid_1_n_x : int or None
    grid_1_px_max : float or None
    grid_1_px_min : float or None
    grid_1_x_max : float or None
    grid_1_x_min : float or None
    grid_2_n_px : int or None
    grid_2_n_x : int or None
    grid_2_px_max : float or None
    grid_2_px_min : float or None
    grid_2_x_max : float or None
    grid_2_x_min : float or None
    grid_3_n_px : int or None
    grid_3_n_x : int or None
    grid_3_px_max : float or None
    grid_3_px_min : float or None
    grid_3_x_max : float or None
    grid_3_x_min : float or None
    ix_turn : int
        Turn index used to adjust particles time if needed.
    kv_a : float
    kv_n_i2 : int
    kv_part_per_phi : sequence of integers
    n_bunch : int
        Number of bunches.
    n_particle : int
        Number of particles per bunch.
    position_file : str
        File with particle positions.
    random_engine : str
        Or 'quasi'. Random number engine to use.
    random_gauss_converter : str
        Or 'quick'. Uniform to gauss conversion method.
    random_sigma_cutoff : float
        Cut-off in sigmas.
    renorm_center : bool
        Renormalize centroid?
    renorm_sigma : bool
        Renormalize sigma?
    sig_pz : float
        pz sigma
    sig_pz_jitter : float
        RMS pz spread jitter
    sig_z : float
        Z sigma in m.
    sig_z_jitter : float
        bunch length RMS jitter
    species : str
        "positron", etc. "" => use referece particle.
    spin : sequence of floats
        Spin (x, y, z)
    t_offset : float
        Time center offset
    use_particle_start : bool
        Use lat%particle_start instead of beam_init%center, %t_offset, and
    %spin?
    use_t_coords : bool
        If true, the distributions will be taken as in t-coordinates
    use_z_as_t : bool
        Only used if  use_t_coords = .true. If true,  z describes the t
    distribution If false, z describes the s distribution
    """

    _tao_command_attr_: ClassVar[str] = "beam_init"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {"ix_branch": 0, "ix_uni": 1}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = (
        "a_emit",
        "b_emit",
        "a_norm_emit",
        "b_norm_emit",
    )
    a_emit: float = Field(default=0.0, description="a-mode emittance")
    a_norm_emit: float = Field(
        default=0.0, description="a-mode normalized emittance (emit * beta * gamma)"
    )
    b_emit: float = Field(default=0.0, description="b-mode emittance")
    b_norm_emit: float = Field(
        default=0.0, description="b-mode normalized emittance (emit * beta * gamma)"
    )
    bunch_charge: float = Field(default=0.0, description="charge (Coul) in a bunch.")
    center: FloatSequence = Field(
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        max_length=6,
        description="Bench phase space center offset relative to reference.",
    )
    center_jitter: FloatSequence = Field(
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        max_length=6,
        description="Bunch center rms jitter",
    )
    distribution_type: Sequence[str] = Field(
        default=["RAN_GAUSS", "RAN_GAUSS", "RAN_GAUSS"],
        max_length=3,
        description=(
            "distribution type (in x-px, y-py, and z-pz planes) 'ELLIPSE', 'KV', "
            "'GRID', 'FILE', 'RAN_GAUSS' or '' = 'RAN_GAUSS'"
        ),
    )
    dpz_dz: float = Field(default=0.0, description="Correlation of Pz with long position.")
    dt_bunch: float = Field(default=0.0, description="Time between bunches.")
    ellipse_1_n_ellipse: int | None = Field(default=None, alias="ellipse(1)%n_ellipse")
    ellipse_1_part_per_ellipse: int | None = Field(
        default=None, alias="ellipse(1)%part_per_ellipse"
    )
    ellipse_1_sigma_cutoff: float | None = Field(default=None, alias="ellipse(1)%sigma_cutoff")
    ellipse_2_n_ellipse: int | None = Field(default=None, alias="ellipse(2)%n_ellipse")
    ellipse_2_part_per_ellipse: int | None = Field(
        default=None, alias="ellipse(2)%part_per_ellipse"
    )
    ellipse_2_sigma_cutoff: float | None = Field(default=None, alias="ellipse(2)%sigma_cutoff")
    ellipse_3_n_ellipse: int | None = Field(default=None, alias="ellipse(3)%n_ellipse")
    ellipse_3_part_per_ellipse: int | None = Field(
        default=None, alias="ellipse(3)%part_per_ellipse"
    )
    ellipse_3_sigma_cutoff: float | None = Field(default=None, alias="ellipse(3)%sigma_cutoff")
    emit_jitter: FloatSequence = Field(
        default=[0.0, 0.0],
        max_length=2,
        description="a and b bunch emittance rms jitter normalized to emittance",
    )
    full_6d_coupling_calc: bool = Field(
        default=False,
        description="Use V from 6x6 1-turn mat to match distribution? Else use 4x4 1-turn mat used.",
    )
    grid_1_n_px: int | None = Field(default=None, alias="grid(1)%n_px")
    grid_1_n_x: int | None = Field(default=None, alias="grid(1)%n_x")
    grid_1_px_max: float | None = Field(default=None, alias="grid(1)%px_max")
    grid_1_px_min: float | None = Field(default=None, alias="grid(1)%px_min")
    grid_1_x_max: float | None = Field(default=None, alias="grid(1)%x_max")
    grid_1_x_min: float | None = Field(default=None, alias="grid(1)%x_min")
    grid_2_n_px: int | None = Field(default=None, alias="grid(2)%n_px")
    grid_2_n_x: int | None = Field(default=None, alias="grid(2)%n_x")
    grid_2_px_max: float | None = Field(default=None, alias="grid(2)%px_max")
    grid_2_px_min: float | None = Field(default=None, alias="grid(2)%px_min")
    grid_2_x_max: float | None = Field(default=None, alias="grid(2)%x_max")
    grid_2_x_min: float | None = Field(default=None, alias="grid(2)%x_min")
    grid_3_n_px: int | None = Field(default=None, alias="grid(3)%n_px")
    grid_3_n_x: int | None = Field(default=None, alias="grid(3)%n_x")
    grid_3_px_max: float | None = Field(default=None, alias="grid(3)%px_max")
    grid_3_px_min: float | None = Field(default=None, alias="grid(3)%px_min")
    grid_3_x_max: float | None = Field(default=None, alias="grid(3)%x_max")
    grid_3_x_min: float | None = Field(default=None, alias="grid(3)%x_min")
    ix_turn: int = Field(
        default=0, description="Turn index used to adjust particles time if needed."
    )
    kv_a: float = Field(default=0.0, alias="kv%A")
    kv_n_i2: int = Field(default=0, alias="kv%n_I2")
    kv_part_per_phi: IntSequence = Field(
        default_factory=list, max_length=2, alias="kv%part_per_phi"
    )
    n_bunch: int = Field(default=0, description="Number of bunches.")
    n_particle: int = Field(default=0, description="Number of particles per bunch.")
    position_file: str = Field(default="", description="File with particle positions.")
    random_engine: str = Field(
        default="pseudo", description="Or 'quasi'. Random number engine to use."
    )
    random_gauss_converter: str = Field(
        default="exact", description="Or 'quick'. Uniform to gauss conversion method."
    )
    random_sigma_cutoff: float = Field(default=-1.0, description="Cut-off in sigmas.")
    renorm_center: bool = Field(default=True, description="Renormalize centroid?")
    renorm_sigma: bool = Field(default=True, description="Renormalize sigma?")
    sig_pz: float = Field(default=0.0, description="pz sigma")
    sig_pz_jitter: float = Field(default=0.0, description="RMS pz spread jitter")
    sig_z: float = Field(default=0.0, description="Z sigma in m.")
    sig_z_jitter: float = Field(default=0.0, description="bunch length RMS jitter")
    species: str = Field(
        default="", description="'positron', etc. '' => use referece particle."
    )
    spin: FloatSequence = Field(
        default=[0.0, 0.0, 0.0], max_length=3, description="Spin (x, y, z)"
    )
    t_offset: float = Field(default=0.0, description="Time center offset")
    use_particle_start: bool = Field(
        default=False,
        description="Use lat%particle_start instead of beam_init%center, %t_offset, and %spin?",
    )
    use_t_coords: bool = Field(
        default=False,
        description="If true, the distributions will be taken as in t-coordinates",
    )
    use_z_as_t: bool = Field(
        default=False,
        description=(
            "Only used if  use_t_coords = .true. If true,  z describes the t "
            "distribution If false, z describes the s distribution"
        ),
    )


class BmadCom(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe bmad_com`, for example.

    Attributes
    ----------
    abs_tol_adaptive_tracking : float
        Runge-Kutta tracking absolute tolerance.
    abs_tol_tracking : float
        Closed orbit absolute tolerance.
    absolute_time_ref_shift : bool
        Apply reference time shift when using absolute time tracking?
    absolute_time_tracking : bool
        Absolute or relative time tracking?
    aperture_limit_on : bool
        Use apertures in tracking?
    auto_bookkeeper : bool
        Deprecated and no longer used.
    autoscale_amp_abs_tol : float
        Autoscale absolute amplitude tolerance (eV).
    autoscale_amp_rel_tol : float
        Autoscale relative amplitude tolerance
    autoscale_phase_tol : float
        Autoscale phase tolerance.
    conserve_taylor_maps : bool
        Enable bookkeeper to set ele%taylor_map_includes_offsets = F?
    convert_to_kinetic_momentum : bool
        Cancel kicks due to finite vector potential when doing symplectic
    tracking? Set to True to test symp_lie_bmad against runge_kutta.
    csr_and_space_charge_on : bool
        Space charge switch.
    d_orb : sequence of floats
        Orbit deltas for the mat6 via tracking calc.
    debug : bool
        Used for code debugging.
    default_ds_step : float
        Default integration step for eles without an explicit step calc.
    default_integ_order : int
        PTC integration order.
    electric_dipole_moment : float
        Particle's EDM. Call set_ptc to transfer value to PTC.
    fatal_ds_adaptive_tracking : float
        If actual step size is below this particle is lost.
    init_ds_adaptive_tracking : float
        Initial step size
    lr_wakes_on : bool
        Long range wakefields
    max_aperture_limit : float
        Max Aperture.
    max_num_runge_kutta_step : int
        Maximum number of RK steps before particle is considered lost.
    min_ds_adaptive_tracking : float
        Min step size to take.
    normalize_twiss : bool
        Normalize matrix when computing Twiss for off-energy ref?
    radiation_damping_on : bool
        Radiation damping toggle.
    radiation_fluctuations_on : bool
        Radiation fluctuations toggle.
    radiation_zero_average : bool
        Shift damping to be zero on the zero orbit to get rid of sawtooth?
    rel_tol_adaptive_tracking : float
        Runge-Kutta tracking relative tolerance.
    rel_tol_tracking : float
        Closed orbit relative tolerance.
    rf_phase_below_transition_ref : bool
        Autoscale uses below transition stable point for RFCavities?
    runge_kutta_order : int
        Runge Kutta order.
    sad_amp_max : float
        Used in sad_mult step length calc.
    sad_eps_scale : float
        Used in sad_mult step length calc.
    sad_n_div_max : int
        Used in sad_mult step length calc.
    significant_length : float
        meter
    spin_n0_direction_user_set : bool
        User sets direction of n0 for closed geometry branches?
    spin_sokolov_ternov_flipping_on : bool
        Spin flipping during synchrotron radiation emission?
    spin_tracking_on : bool
        spin tracking?
    sr_wakes_on : bool
        Short range wakefields?
    synch_rad_scale : float
        Synch radiation kick scale. 1 => normal, 0 => no kicks.
    taylor_order : int
        Taylor order to use. 0 -> default = ptc_private%taylor_order_saved.
    """

    _tao_command_attr_: ClassVar[str] = "bmad_com"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    abs_tol_adaptive_tracking: float = Field(
        default=1e-10, description="Runge-Kutta tracking absolute tolerance."
    )
    abs_tol_tracking: float = Field(
        default=1e-12, description="Closed orbit absolute tolerance."
    )
    absolute_time_ref_shift: bool = Field(
        default=True,
        description="Apply reference time shift when using absolute time tracking?",
    )
    absolute_time_tracking: bool = Field(
        default=False, description="Absolute or relative time tracking?"
    )
    aperture_limit_on: bool = Field(default=True, description="Use apertures in tracking?")
    auto_bookkeeper: bool = Field(default=True, description="Deprecated and no longer used.")
    autoscale_amp_abs_tol: float = Field(
        default=0.1, description="Autoscale absolute amplitude tolerance (eV)."
    )
    autoscale_amp_rel_tol: float = Field(
        default=1e-06, description="Autoscale relative amplitude tolerance"
    )
    autoscale_phase_tol: float = Field(default=1e-05, description="Autoscale phase tolerance.")
    conserve_taylor_maps: bool = Field(
        default=True,
        description="Enable bookkeeper to set ele%taylor_map_includes_offsets = F?",
    )
    convert_to_kinetic_momentum: bool = Field(
        default=False,
        description=(
            "Cancel kicks due to finite vector potential when doing symplectic "
            "tracking? Set to True to test symp_lie_bmad against runge_kutta."
        ),
    )
    csr_and_space_charge_on: bool = Field(default=False, description="Space charge switch.")
    d_orb: FloatSequence = Field(
        default=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        max_length=6,
        description="Orbit deltas for the mat6 via tracking calc.",
    )
    debug: bool = Field(default=False, description="Used for code debugging.")
    default_ds_step: float = Field(
        default=0.2,
        description="Default integration step for eles without an explicit step calc.",
    )
    default_integ_order: int = Field(default=2, description="PTC integration order.")
    electric_dipole_moment: float = Field(
        default=0.0, description="Particle's EDM. Call set_ptc to transfer value to PTC."
    )
    fatal_ds_adaptive_tracking: float = Field(
        default=1e-08, description="If actual step size is below this particle is lost."
    )
    init_ds_adaptive_tracking: float = Field(default=0.001, description="Initial step size")
    lr_wakes_on: bool = Field(default=True, description="Long range wakefields")
    max_aperture_limit: float = Field(default=1000.0, description="Max Aperture.")
    max_num_runge_kutta_step: int = Field(
        default=10000,
        description="Maximum number of RK steps before particle is considered lost.",
    )
    min_ds_adaptive_tracking: float = Field(default=0.0, description="Min step size to take.")
    normalize_twiss: bool = Field(
        default=False, description="Normalize matrix when computing Twiss for off-energy ref?"
    )
    radiation_damping_on: bool = Field(default=False, description="Radiation damping toggle.")
    radiation_fluctuations_on: bool = Field(
        default=False, description="Radiation fluctuations toggle."
    )
    radiation_zero_average: bool = Field(
        default=False,
        description="Shift damping to be zero on the zero orbit to get rid of sawtooth?",
    )
    rel_tol_adaptive_tracking: float = Field(
        default=1e-08, description="Runge-Kutta tracking relative tolerance."
    )
    rel_tol_tracking: float = Field(
        default=1e-09, description="Closed orbit relative tolerance."
    )
    rf_phase_below_transition_ref: bool = Field(
        default=False,
        description="Autoscale uses below transition stable point for RFCavities?",
    )
    runge_kutta_order: int = Field(default=4, description="Runge Kutta order.")
    sad_amp_max: float = Field(default=0.05, description="Used in sad_mult step length calc.")
    sad_eps_scale: float = Field(
        default=0.005, description="Used in sad_mult step length calc."
    )
    sad_n_div_max: int = Field(default=1000, description="Used in sad_mult step length calc.")
    significant_length: float = Field(default=1e-10, description="meter")
    spin_n0_direction_user_set: bool = Field(
        default=False, description="User sets direction of n0 for closed geometry branches?"
    )
    spin_sokolov_ternov_flipping_on: bool = Field(
        default=False, description="Spin flipping during synchrotron radiation emission?"
    )
    spin_tracking_on: bool = Field(default=False, description="spin tracking?")
    sr_wakes_on: bool = Field(default=True, description="Short range wakefields?")
    synch_rad_scale: float = Field(
        default=1.0, description="Synch radiation kick scale. 1 => normal, 0 => no kicks."
    )
    taylor_order: int = Field(
        default=0,
        description="Taylor order to use. 0 -> default = ptc_private%taylor_order_saved.",
    )


class ElementBunchParams(TaoModel):
    """
    Structure which corresponds to Tao `pipe bunch_params 1`, for example.

    Attributes
    ----------
    beam_saved : bool
    centroid_beta : float
    centroid_p0c : float
    centroid_t : float
    centroid_vec_1 : float
    centroid_vec_2 : float
    centroid_vec_3 : float
    centroid_vec_4 : float
    centroid_vec_5 : float
    centroid_vec_6 : float
    charge_live : float
        Charge of all non-lost particle
    direction : int
    ix_ele : int
        Lattice element where params evaluated at.
    location : str
        Location in element: upstream_end$, inside$, or downstream_end$
    n_particle_live : int
        Number of non-lost particles
    n_particle_lost_in_ele : int
        Number lost in element (not calculated by Bmad)
    n_particle_tot : int
        Total number of particles
    rel_max_1 : float
    rel_max_2 : float
    rel_max_3 : float
    rel_max_4 : float
    rel_max_5 : float
    rel_max_6 : float
    rel_min_1 : float
    rel_min_2 : float
    rel_min_3 : float
    rel_min_4 : float
    rel_min_5 : float
    rel_min_6 : float
    s : float
        Longitudinal position.
    sigma_11 : float
    sigma_12 : float
    sigma_13 : float
    sigma_14 : float
    sigma_15 : float
    sigma_16 : float
    sigma_21 : float
    sigma_22 : float
    sigma_23 : float
    sigma_24 : float
    sigma_25 : float
    sigma_26 : float
    sigma_31 : float
    sigma_32 : float
    sigma_33 : float
    sigma_34 : float
    sigma_35 : float
    sigma_36 : float
    sigma_41 : float
    sigma_42 : float
    sigma_43 : float
    sigma_44 : float
    sigma_45 : float
    sigma_46 : float
    sigma_51 : float
    sigma_52 : float
    sigma_53 : float
    sigma_54 : float
    sigma_55 : float
    sigma_56 : float
    sigma_61 : float
    sigma_62 : float
    sigma_63 : float
    sigma_64 : float
    sigma_65 : float
    sigma_66 : float
    sigma_t : float
        RMS of time spread.
    species : str
    t : float
        Time.
    twiss_alpha_a : float
    twiss_alpha_b : float
    twiss_alpha_c : float
    twiss_alpha_x : float
    twiss_alpha_y : float
    twiss_alpha_z : float
    twiss_beta_a : float
    twiss_beta_b : float
    twiss_beta_c : float
    twiss_beta_x : float
    twiss_beta_y : float
    twiss_beta_z : float
    twiss_dalpha_dpz_a : float
    twiss_dalpha_dpz_b : float
    twiss_dalpha_dpz_c : float
    twiss_dalpha_dpz_x : float
    twiss_dalpha_dpz_y : float
    twiss_dalpha_dpz_z : float
    twiss_dbeta_dpz_a : float
    twiss_dbeta_dpz_b : float
    twiss_dbeta_dpz_c : float
    twiss_dbeta_dpz_x : float
    twiss_dbeta_dpz_y : float
    twiss_dbeta_dpz_z : float
    twiss_deta_dpz_a : float
    twiss_deta_dpz_b : float
    twiss_deta_dpz_c : float
    twiss_deta_dpz_x : float
    twiss_deta_dpz_y : float
    twiss_deta_dpz_z : float
    twiss_deta_ds_a : float
    twiss_deta_ds_b : float
    twiss_deta_ds_c : float
    twiss_deta_ds_x : float
    twiss_deta_ds_y : float
    twiss_deta_ds_z : float
    twiss_detap_dpz_a : float
    twiss_detap_dpz_b : float
    twiss_detap_dpz_c : float
    twiss_detap_dpz_x : float
    twiss_detap_dpz_y : float
    twiss_detap_dpz_z : float
    twiss_emit_a : float
    twiss_emit_b : float
    twiss_emit_c : float
    twiss_emit_x : float
    twiss_emit_y : float
    twiss_emit_z : float
    twiss_eta_a : float
    twiss_eta_b : float
    twiss_eta_c : float
    twiss_eta_x : float
    twiss_eta_y : float
    twiss_eta_z : float
    twiss_etap_a : float
    twiss_etap_b : float
    twiss_etap_c : float
    twiss_etap_x : float
    twiss_etap_y : float
    twiss_etap_z : float
    twiss_gamma_a : float
    twiss_gamma_b : float
    twiss_gamma_c : float
    twiss_gamma_x : float
    twiss_gamma_y : float
    twiss_gamma_z : float
    twiss_norm_emit_a : float
    twiss_norm_emit_b : float
    twiss_norm_emit_c : float
    twiss_norm_emit_x : float
    twiss_norm_emit_y : float
    twiss_norm_emit_z : float
    twiss_phi_a : float
    twiss_phi_b : float
    twiss_phi_c : float
    twiss_phi_x : float
    twiss_phi_y : float
    twiss_phi_z : float
    twiss_sigma_a : float
    twiss_sigma_b : float
    twiss_sigma_c : float
    twiss_sigma_p_a : float
    twiss_sigma_p_b : float
    twiss_sigma_p_c : float
    twiss_sigma_p_x : float
    twiss_sigma_p_y : float
    twiss_sigma_p_z : float
    twiss_sigma_x : float
    twiss_sigma_y : float
    twiss_sigma_z : float
    """

    _tao_command_attr_: ClassVar[str] = "bunch_params"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    beam_saved: bool = False
    centroid_beta: float = Field(default=0.0, frozen=True)
    centroid_p0c: float = Field(default=0.0, frozen=True)
    centroid_t: float = Field(default=0.0, frozen=True)
    centroid_vec_1: float = Field(default=0.0, frozen=True)
    centroid_vec_2: float = Field(default=0.0, frozen=True)
    centroid_vec_3: float = Field(default=0.0, frozen=True)
    centroid_vec_4: float = Field(default=0.0, frozen=True)
    centroid_vec_5: float = Field(default=0.0, frozen=True)
    centroid_vec_6: float = Field(default=0.0, frozen=True)
    charge_live: float = Field(
        default=0.0, description="Charge of all non-lost particle", frozen=True
    )
    direction: int = Field(default=0, frozen=True)
    ix_ele: int = Field(
        default=-1, description="Lattice element where params evaluated at.", frozen=True
    )
    location: str = Field(
        default="",
        description="Location in element: upstream_end$, inside$, or downstream_end$",
        frozen=True,
    )
    n_particle_live: int = Field(
        default=0, description="Number of non-lost particles", frozen=True
    )
    n_particle_lost_in_ele: int = Field(
        default=0, description="Number lost in element (not calculated by Bmad)", frozen=True
    )
    n_particle_tot: int = Field(
        default=0, description="Total number of particles", frozen=True
    )
    rel_max_1: float = Field(default=0.0, frozen=True)
    rel_max_2: float = Field(default=0.0, frozen=True)
    rel_max_3: float = Field(default=0.0, frozen=True)
    rel_max_4: float = Field(default=0.0, frozen=True)
    rel_max_5: float = Field(default=0.0, frozen=True)
    rel_max_6: float = Field(default=0.0, frozen=True)
    rel_min_1: float = Field(default=0.0, frozen=True)
    rel_min_2: float = Field(default=0.0, frozen=True)
    rel_min_3: float = Field(default=0.0, frozen=True)
    rel_min_4: float = Field(default=0.0, frozen=True)
    rel_min_5: float = Field(default=0.0, frozen=True)
    rel_min_6: float = Field(default=0.0, frozen=True)
    s: float = Field(default=-1.0, description="Longitudinal position.", frozen=True)
    sigma_11: float = Field(default=0.0, frozen=True)
    sigma_12: float = Field(default=0.0, frozen=True)
    sigma_13: float = Field(default=0.0, frozen=True)
    sigma_14: float = Field(default=0.0, frozen=True)
    sigma_15: float = Field(default=0.0, frozen=True)
    sigma_16: float = Field(default=0.0, frozen=True)
    sigma_21: float = Field(default=0.0, frozen=True)
    sigma_22: float = Field(default=0.0, frozen=True)
    sigma_23: float = Field(default=0.0, frozen=True)
    sigma_24: float = Field(default=0.0, frozen=True)
    sigma_25: float = Field(default=0.0, frozen=True)
    sigma_26: float = Field(default=0.0, frozen=True)
    sigma_31: float = Field(default=0.0, frozen=True)
    sigma_32: float = Field(default=0.0, frozen=True)
    sigma_33: float = Field(default=0.0, frozen=True)
    sigma_34: float = Field(default=0.0, frozen=True)
    sigma_35: float = Field(default=0.0, frozen=True)
    sigma_36: float = Field(default=0.0, frozen=True)
    sigma_41: float = Field(default=0.0, frozen=True)
    sigma_42: float = Field(default=0.0, frozen=True)
    sigma_43: float = Field(default=0.0, frozen=True)
    sigma_44: float = Field(default=0.0, frozen=True)
    sigma_45: float = Field(default=0.0, frozen=True)
    sigma_46: float = Field(default=0.0, frozen=True)
    sigma_51: float = Field(default=0.0, frozen=True)
    sigma_52: float = Field(default=0.0, frozen=True)
    sigma_53: float = Field(default=0.0, frozen=True)
    sigma_54: float = Field(default=0.0, frozen=True)
    sigma_55: float = Field(default=0.0, frozen=True)
    sigma_56: float = Field(default=0.0, frozen=True)
    sigma_61: float = Field(default=0.0, frozen=True)
    sigma_62: float = Field(default=0.0, frozen=True)
    sigma_63: float = Field(default=0.0, frozen=True)
    sigma_64: float = Field(default=0.0, frozen=True)
    sigma_65: float = Field(default=0.0, frozen=True)
    sigma_66: float = Field(default=0.0, frozen=True)
    sigma_t: float = Field(default=0.0, description="RMS of time spread.", frozen=True)
    species: str = Field(default="", frozen=True)
    t: float = Field(default=-1.0, description="Time.", frozen=True)
    twiss_alpha_a: float = Field(default=0.0, frozen=True)
    twiss_alpha_b: float = Field(default=0.0, frozen=True)
    twiss_alpha_c: float = Field(default=0.0, frozen=True)
    twiss_alpha_x: float = Field(default=0.0, frozen=True)
    twiss_alpha_y: float = Field(default=0.0, frozen=True)
    twiss_alpha_z: float = Field(default=0.0, frozen=True)
    twiss_beta_a: float = Field(default=0.0, frozen=True)
    twiss_beta_b: float = Field(default=0.0, frozen=True)
    twiss_beta_c: float = Field(default=0.0, frozen=True)
    twiss_beta_x: float = Field(default=0.0, frozen=True)
    twiss_beta_y: float = Field(default=0.0, frozen=True)
    twiss_beta_z: float = Field(default=0.0, frozen=True)
    twiss_dalpha_dpz_a: float = Field(default=0.0, frozen=True)
    twiss_dalpha_dpz_b: float = Field(default=0.0, frozen=True)
    twiss_dalpha_dpz_c: float = Field(default=0.0, frozen=True)
    twiss_dalpha_dpz_x: float = Field(default=0.0, frozen=True)
    twiss_dalpha_dpz_y: float = Field(default=0.0, frozen=True)
    twiss_dalpha_dpz_z: float = Field(default=0.0, frozen=True)
    twiss_dbeta_dpz_a: float = Field(default=0.0, frozen=True)
    twiss_dbeta_dpz_b: float = Field(default=0.0, frozen=True)
    twiss_dbeta_dpz_c: float = Field(default=0.0, frozen=True)
    twiss_dbeta_dpz_x: float = Field(default=0.0, frozen=True)
    twiss_dbeta_dpz_y: float = Field(default=0.0, frozen=True)
    twiss_dbeta_dpz_z: float = Field(default=0.0, frozen=True)
    twiss_deta_dpz_a: float = Field(default=0.0, frozen=True)
    twiss_deta_dpz_b: float = Field(default=0.0, frozen=True)
    twiss_deta_dpz_c: float = Field(default=0.0, frozen=True)
    twiss_deta_dpz_x: float = Field(default=0.0, frozen=True)
    twiss_deta_dpz_y: float = Field(default=0.0, frozen=True)
    twiss_deta_dpz_z: float = Field(default=0.0, frozen=True)
    twiss_deta_ds_a: float = Field(default=0.0, frozen=True)
    twiss_deta_ds_b: float = Field(default=0.0, frozen=True)
    twiss_deta_ds_c: float = Field(default=0.0, frozen=True)
    twiss_deta_ds_x: float = Field(default=0.0, frozen=True)
    twiss_deta_ds_y: float = Field(default=0.0, frozen=True)
    twiss_deta_ds_z: float = Field(default=0.0, frozen=True)
    twiss_detap_dpz_a: float = Field(default=0.0, frozen=True)
    twiss_detap_dpz_b: float = Field(default=0.0, frozen=True)
    twiss_detap_dpz_c: float = Field(default=0.0, frozen=True)
    twiss_detap_dpz_x: float = Field(default=0.0, frozen=True)
    twiss_detap_dpz_y: float = Field(default=0.0, frozen=True)
    twiss_detap_dpz_z: float = Field(default=0.0, frozen=True)
    twiss_emit_a: float = Field(default=0.0, frozen=True)
    twiss_emit_b: float = Field(default=0.0, frozen=True)
    twiss_emit_c: float = Field(default=0.0, frozen=True)
    twiss_emit_x: float = Field(default=0.0, frozen=True)
    twiss_emit_y: float = Field(default=0.0, frozen=True)
    twiss_emit_z: float = Field(default=0.0, frozen=True)
    twiss_eta_a: float = Field(default=0.0, frozen=True)
    twiss_eta_b: float = Field(default=0.0, frozen=True)
    twiss_eta_c: float = Field(default=0.0, frozen=True)
    twiss_eta_x: float = Field(default=0.0, frozen=True)
    twiss_eta_y: float = Field(default=0.0, frozen=True)
    twiss_eta_z: float = Field(default=0.0, frozen=True)
    twiss_etap_a: float = Field(default=0.0, frozen=True)
    twiss_etap_b: float = Field(default=0.0, frozen=True)
    twiss_etap_c: float = Field(default=0.0, frozen=True)
    twiss_etap_x: float = Field(default=0.0, frozen=True)
    twiss_etap_y: float = Field(default=0.0, frozen=True)
    twiss_etap_z: float = Field(default=0.0, frozen=True)
    twiss_gamma_a: float = Field(default=0.0, frozen=True)
    twiss_gamma_b: float = Field(default=0.0, frozen=True)
    twiss_gamma_c: float = Field(default=0.0, frozen=True)
    twiss_gamma_x: float = Field(default=0.0, frozen=True)
    twiss_gamma_y: float = Field(default=0.0, frozen=True)
    twiss_gamma_z: float = Field(default=0.0, frozen=True)
    twiss_norm_emit_a: float = Field(default=0.0, frozen=True)
    twiss_norm_emit_b: float = Field(default=0.0, frozen=True)
    twiss_norm_emit_c: float = Field(default=0.0, frozen=True)
    twiss_norm_emit_x: float = Field(default=0.0, frozen=True)
    twiss_norm_emit_y: float = Field(default=0.0, frozen=True)
    twiss_norm_emit_z: float = Field(default=0.0, frozen=True)
    twiss_phi_a: float = Field(default=0.0, frozen=True)
    twiss_phi_b: float = Field(default=0.0, frozen=True)
    twiss_phi_c: float = Field(default=0.0, frozen=True)
    twiss_phi_x: float = Field(default=0.0, frozen=True)
    twiss_phi_y: float = Field(default=0.0, frozen=True)
    twiss_phi_z: float = Field(default=0.0, frozen=True)
    twiss_sigma_a: float = Field(default=0.0, frozen=True)
    twiss_sigma_b: float = Field(default=0.0, frozen=True)
    twiss_sigma_c: float = Field(default=0.0, frozen=True)
    twiss_sigma_p_a: float = Field(default=0.0, frozen=True)
    twiss_sigma_p_b: float = Field(default=0.0, frozen=True)
    twiss_sigma_p_c: float = Field(default=0.0, frozen=True)
    twiss_sigma_p_x: float = Field(default=0.0, frozen=True)
    twiss_sigma_p_y: float = Field(default=0.0, frozen=True)
    twiss_sigma_p_z: float = Field(default=0.0, frozen=True)
    twiss_sigma_x: float = Field(default=0.0, frozen=True)
    twiss_sigma_y: float = Field(default=0.0, frozen=True)
    twiss_sigma_z: float = Field(default=0.0, frozen=True)


class ElementChamberWall(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:chamber_wall 1 1 x`, for example.

    Attributes
    ----------
    longitudinal_position : float
    section : int
    z1 : float
    z2_neg : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_chamber_wall"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    longitudinal_position: float = 0.0
    section: int = 0
    z1: float = 0.0
    z2_neg: float = Field(default=0.0, alias="-z2")


class ElementGridField(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:grid_field G1 1 base`, for example.

    Attributes
    ----------
    curved_ref_frame : bool
    dr : sequence of floats
        Grid spacing.
    ele_anchor_pt : str
        anchor_beginning$, anchor_center$, or anchor_end$
    field_scale : float
        Factor to scale the fields by
    field_type : str
        or magnetic$ or electric$
    file : str
    grid_field_geometry : str
    harmonic : int
        Harmonic of fundamental for AC fields.
    interpolation_order : int
        Possibilities are 1 or 3.
    master_parameter : str
        Master parameter in ele%value(:) array to use for scaling the field.
    phi0_fieldmap : float
        Mode oscillates as: twopi * (f * t + phi0_fieldmap)
    r0 : sequence of floats
        Field origin relative to ele_anchor_pt.
    """

    _tao_command_attr_: ClassVar[str] = "ele_grid_field"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    curved_ref_frame: bool = False
    dr: FloatSequence = Field(
        default=[0.0, 0.0, 0.0], max_length=3, description="Grid spacing."
    )
    ele_anchor_pt: str = Field(
        default="", description="anchor_beginning$, anchor_center$, or anchor_end$"
    )
    field_scale: float = Field(default=1.0, description="Factor to scale the fields by")
    field_type: str = Field(default="", description="or magnetic$ or electric$")
    file: str = ""
    grid_field_geometry: str = Field(default="", alias="grid_field^geometry")
    harmonic: int = Field(default=0, description="Harmonic of fundamental for AC fields.")
    interpolation_order: int = Field(default=1, description="Possibilities are 1 or 3.")
    master_parameter: str = Field(
        default=0,
        description="Master parameter in ele%value(:) array to use for scaling the field.",
    )
    phi0_fieldmap: float = Field(
        default=0.0, description="Mode oscillates as: twopi * (f * t + phi0_fieldmap)"
    )
    r0: FloatSequence = Field(
        default=[0.0, 0.0, 0.0],
        max_length=3,
        description="Field origin relative to ele_anchor_pt.",
    )


class ElementGridFieldPoints(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:grid_field G1 1 points`, for example.

    Attributes
    ----------
    data : FloatSequence
    i : int
    j : int
    k : int
    """

    _tao_command_attr_: ClassVar[str] = "ele_grid_field"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    data: FloatSequence = list()
    i: int = 0
    j: int = 0
    k: int = 0


class ElementHead(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:head 1`, for example.

    Attributes
    ----------
    alias : str
        Another name.
    descrip : str
        Description string.
    has_ab_multipoles : bool
    has_ac_kick : bool
    has_control : bool
    has_floor : bool
    has_kt_multipoles : bool
    has_lord_slave : bool
    has_mat6 : bool
    has_methods : bool
    has_multipoles_elec : bool
    has_photon : bool
    has_spin_taylor : bool
    has_taylor : bool
    has_twiss : bool
    has_wake : bool
    has_wall3d : int
    is_on : bool
        For turning element on/off.
    ix_branch : int
        Index in lat%branch(:) array. Note: lat%ele => lat%branch(0).
    ix_ele : int
        Index in branch ele(0:) array. Set to ix_slice_slave$ = -2 for
    slice_slave$ elements.
    key : str
        Element class (quadrupole, etc.).
    name : str
        name of element.
    num_cartesian_map : int
    num_cylindrical_map : int
    num_gen_grad_map : int
    num_grid_field : int
    ref_time : float
        Time ref particle passes exit end.
    s : float
        longitudinal ref position at the exit end.
    s_start : float
        longitudinal ref position at entrance_end
    type : str
        type name.
    universe : int
    """

    _tao_command_attr_: ClassVar[str] = "ele_head"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    alias: str = Field(default="", description="Another name.")
    descrip: str = Field(default="", description="Description string.")
    has_ab_multipoles: bool = Field(default=False, alias="has#ab_multipoles", frozen=True)
    has_ac_kick: bool = Field(default=False, alias="has#ac_kick", frozen=True)
    has_control: bool = Field(default=False, alias="has#control", frozen=True)
    has_floor: bool = Field(default=False, alias="has#floor", frozen=True)
    has_kt_multipoles: bool = Field(default=False, alias="has#kt_multipoles", frozen=True)
    has_lord_slave: bool = Field(default=False, alias="has#lord_slave", frozen=True)
    has_mat6: bool = Field(default=False, alias="has#mat6", frozen=True)
    has_methods: bool = Field(default=False, alias="has#methods", frozen=True)
    has_multipoles_elec: bool = Field(default=False, alias="has#multipoles_elec", frozen=True)
    has_photon: bool = Field(default=False, alias="has#photon", frozen=True)
    has_spin_taylor: bool = Field(default=False, alias="has#spin_taylor", frozen=True)
    has_taylor: bool = Field(default=False, alias="has#taylor", frozen=True)
    has_twiss: bool = Field(default=False, alias="has#twiss", frozen=True)
    has_wake: bool = Field(default=False, alias="has#wake", frozen=True)
    has_wall3d: int = Field(default=0, alias="has#wall3d", frozen=True)
    is_on: bool = Field(default=True, description="For turning element on/off.")
    ix_branch: int = Field(
        default=0,
        description="Index in lat%branch(:) array. Note: lat%ele => lat%branch(0).",
        alias="1^ix_branch",
    )
    ix_ele: int = Field(
        default=-1,
        description=(
            "Index in branch ele(0:) array. Set to ix_slice_slave$ = -2 for "
            "slice_slave$ elements."
        ),
        frozen=True,
    )
    key: str = Field(default=0, description="Element class (quadrupole, etc.).", frozen=True)
    name: str = Field(default="<Initialized>", description="name of element.", frozen=True)
    num_cartesian_map: int = Field(default=0, alias="num#cartesian_map", frozen=True)
    num_cylindrical_map: int = Field(default=0, alias="num#cylindrical_map", frozen=True)
    num_gen_grad_map: int = Field(default=0, alias="num#gen_grad_map", frozen=True)
    num_grid_field: int = Field(default=0, alias="num#grid_field", frozen=True)
    ref_time: float = Field(
        default=0.0, description="Time ref particle passes exit end.", frozen=True
    )
    s: float = Field(
        default=0.0, description="longitudinal ref position at the exit end.", frozen=True
    )
    s_start: float = Field(
        default=0.0, description="longitudinal ref position at entrance_end", frozen=True
    )
    type: str = Field(default="", description="type name.")
    universe: int = Field(default=0, frozen=True)


class ElementLordSlave(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:lord_slave 1 1 x`, for example.

    Attributes
    ----------
    key : str
    location_name : str
    name : str
    status : str
    type : str
    """

    _tao_command_attr_: ClassVar[str] = "ele_lord_slave"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    key: str = ""
    location_name: str = ""
    name: str = ""
    status: str = ""
    type: str = ""


class ElementMat6(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:mat6 1 mat6`, for example.

    Attributes
    ----------
    data_1 : sequence of floats
    data_2 : sequence of floats
    data_3 : sequence of floats
    data_4 : sequence of floats
    data_5 : sequence of floats
    data_6 : sequence of floats
    """

    _tao_command_attr_: ClassVar[str] = "ele_mat6"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    data_1: FloatSequence = Field(default_factory=list, max_length=6, alias="1", frozen=True)
    data_2: FloatSequence = Field(default_factory=list, max_length=6, alias="2", frozen=True)
    data_3: FloatSequence = Field(default_factory=list, max_length=6, alias="3", frozen=True)
    data_4: FloatSequence = Field(default_factory=list, max_length=6, alias="4", frozen=True)
    data_5: FloatSequence = Field(default_factory=list, max_length=6, alias="5", frozen=True)
    data_6: FloatSequence = Field(default_factory=list, max_length=6, alias="6", frozen=True)


class ElementMat6Error(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:mat6 1 err`, for example.

    Attributes
    ----------
    symplectic_error : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_mat6"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    symplectic_error: float = Field(default=0.0, frozen=True)


class ElementMat6Vec0(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:mat6 1 vec0`, for example.

    Attributes
    ----------
    vec0 : sequence of floats
    """

    _tao_command_attr_: ClassVar[str] = "ele_mat6"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    vec0: FloatSequence = Field(default_factory=list, max_length=6, frozen=True)


class ElementMultipoles_Data(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 13`, for example.

    Attributes
    ----------
    an_equiv : float
    bn_equiv : float
    index : int
    knl : float
    knl_w_tilt : float
    tn : float
    tn_w_tilt : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    an_equiv: float = Field(default=0.0, alias="An (equiv)")
    bn_equiv: float = Field(default=0.0, alias="Bn (equiv)")
    index: int = 0
    knl: float = Field(default=0.0, alias="KnL")
    knl_w_tilt: float = Field(default=0.0, alias="KnL (w/Tilt)")
    tn: float = Field(default=0.0, alias="Tn")
    tn_w_tilt: float = Field(default=0.0, alias="Tn (w/Tilt)")


class ElementMultipoles(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 13`, for example.

    Attributes
    ----------
    data : ElementMultipoles_Data
        Structure which corresponds to Tao `pipe ele:multipoles 13`, for
    example.
    multipoles_on : bool
        For turning multipoles on/off
    scale_multipoles : bool or None
        Are ab_multipoles within other elements (EG: quads, etc.) scaled by
    the strength of the element?
    """

    _tao_command_attr_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    data: Sequence[ElementMultipoles_Data] = Field(
        default_factory=list,
        description="Structure which corresponds to Tao `pipe ele:multipoles 13`, for example.",
    )
    multipoles_on: bool = Field(default=True, description="For turning multipoles on/off")
    scale_multipoles: bool | None = Field(
        default=None,
        description=(
            "Are ab_multipoles within other elements (EG: quads, etc.) scaled by the "
            "strength of the element?"
        ),
    )


class ElementMultipolesAB_Data(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 4`, for example.

    Attributes
    ----------
    an : float
    an_w_tilt : float
    bn : float
    bn_w_tilt : float
    index : int
    knl_equiv : float
    tn_equiv : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    an: float = Field(default=0.0, alias="An")
    an_w_tilt: float = Field(default=0.0, alias="An (w/Tilt)")
    bn: float = Field(default=0.0, alias="Bn")
    bn_w_tilt: float = Field(default=0.0, alias="Bn (w/Tilt)")
    index: int = 0
    knl_equiv: float = Field(default=0.0, alias="KnL (equiv)")
    tn_equiv: float = Field(default=0.0, alias="Tn (equiv)")


class ElementMultipolesAB(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 4`, for example.

    Attributes
    ----------
    data : ElementMultipolesAB_Data
        Structure which corresponds to Tao `pipe ele:multipoles 4`, for
    example.
    multipoles_on : bool
        For turning multipoles on/off
    """

    _tao_command_attr_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    data: Sequence[ElementMultipolesAB_Data] = Field(
        default_factory=list,
        description="Structure which corresponds to Tao `pipe ele:multipoles 4`, for example.",
    )
    multipoles_on: bool = Field(default=True, description="For turning multipoles on/off")


class ElementMultipolesScaled_Data(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 16`, for example.

    Attributes
    ----------
    an : float
    an_scaled : float
    an_w_tilt : float
    bn : float
    bn_scaled : float
    bn_w_tilt : float
    index : int
    knl_equiv : float
    tn_equiv : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    an: float = Field(default=0.0, alias="An")
    an_scaled: float = Field(default=0.0, alias="An (Scaled)")
    an_w_tilt: float = Field(default=0.0, alias="An (w/Tilt)")
    bn: float = Field(default=0.0, alias="Bn")
    bn_scaled: float = Field(default=0.0, alias="Bn (Scaled)")
    bn_w_tilt: float = Field(default=0.0, alias="Bn (w/Tilt)")
    index: int = 0
    knl_equiv: float = Field(default=0.0, alias="KnL (equiv)")
    tn_equiv: float = Field(default=0.0, alias="Tn (equiv)")


class ElementMultipolesScaled(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:multipoles 16`, for example.

    Attributes
    ----------
    data : ElementMultipolesScaled_Data
        Structure which corresponds to Tao `pipe ele:multipoles 16`, for
    example.
    multipoles_on : bool
        For turning multipoles on/off
    scale_multipoles : bool
        Are ab_multipoles within other elements (EG: quads, etc.) scaled by
    the strength of the element?
    """

    _tao_command_attr_: ClassVar[str] = "ele_multipoles"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    data: Sequence[ElementMultipolesScaled_Data] = Field(
        default_factory=list,
        description="Structure which corresponds to Tao `pipe ele:multipoles 16`, for example.",
    )
    multipoles_on: bool = Field(default=True, description="For turning multipoles on/off")
    scale_multipoles: bool = Field(
        default=True,
        description=(
            "Are ab_multipoles within other elements (EG: quads, etc.) scaled by the "
            "strength of the element?"
        ),
    )


class ElementOrbit(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:orbit 1`, for example.

    Attributes
    ----------
    beta : float
        Velocity / c_light.
    charge : float
        Macroparticle weight (which is different from particle species
    charge). For some space charge calcs the weight is in Coulombs.
    direction : int
        +1 or -1. Sign of longitudinal direction of motion (ds/dt). This is
    independent of the element orientation.
    dt_ref : float
        Used in: * time tracking for computing z. * by coherent photons =
    path_length/c_light.
    field : sequence of floats
        Photon E-field intensity (x,y).
    ix_ele : int
        Index of the lattice element the particle is in. May be -1 if element
    is not associated with a lattice.
    location : str
        upstream_end$, inside$, or downstream_end$
    p0c : float
        For non-photons: Reference momentum. For photons: Photon momentum (not
    reference).
    phase : sequence of floats
        Photon E-field phase (x,y). For charged particles, phase(1) is RF
    phase.
    px : float
    py : float
    pz : float
    s : float
        Longitudinal position
    species : str
        positron$, proton$, etc.
    spin : sequence of floats
        Spin.
    state : str
        alive$, lost$, lost_neg_x_aperture$, lost_pz$, etc.
    t : float
        Absolute time (not relative to reference). Note: Quad precision!
    x : float
    y : float
    z : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_orbit"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    beta: float = Field(default=-1.0, description="Velocity / c_light.", frozen=True)
    charge: float = Field(
        default=0.0,
        description=(
            "Macroparticle weight (which is different from particle species charge). "
            "For some space charge calcs the weight is in Coulombs."
        ),
        frozen=True,
    )
    direction: int = Field(
        default=1,
        description=(
            "+1 or -1. Sign of longitudinal direction of motion (ds/dt). This is "
            "independent of the element orientation."
        ),
        frozen=True,
    )
    dt_ref: float = Field(
        default=0.0,
        description=(
            "Used in: * time tracking for computing z. * by coherent photons = "
            "path_length/c_light."
        ),
        frozen=True,
    )
    field: FloatSequence = Field(
        default=[0.0, 0.0],
        max_length=2,
        description="Photon E-field intensity (x,y).",
        frozen=True,
    )
    ix_ele: int = Field(
        default=-1,
        description=(
            "Index of the lattice element the particle is in. May be -1 if element is "
            "not associated with a lattice."
        ),
        frozen=True,
    )
    location: str = Field(
        default="", description="upstream_end$, inside$, or downstream_end$", frozen=True
    )
    p0c: float = Field(
        default=0.0,
        description=(
            "For non-photons: Reference momentum. For photons: Photon momentum (not "
            "reference)."
        ),
        frozen=True,
    )
    phase: FloatSequence = Field(
        default=[0.0, 0.0],
        max_length=2,
        description="Photon E-field phase (x,y). For charged particles, phase(1) is RF phase.",
        frozen=True,
    )
    px: float = Field(default=0.0, frozen=True)
    py: float = Field(default=0.0, frozen=True)
    pz: float = Field(default=0.0, frozen=True)
    s: float = Field(default=0.0, description="Longitudinal position", frozen=True)
    species: str = Field(default="", description="positron$, proton$, etc.", frozen=True)
    spin: FloatSequence = Field(
        default=[0.0, 0.0, 0.0], max_length=3, description="Spin.", frozen=True
    )
    state: str = Field(
        default="",
        description="alive$, lost$, lost_neg_x_aperture$, lost_pz$, etc.",
        frozen=True,
    )
    t: float = Field(
        default=0.0,
        description="Absolute time (not relative to reference). Note: Quad precision!",
        frozen=True,
    )
    x: float = Field(default=0.0, frozen=True)
    y: float = Field(default=0.0, frozen=True)
    z: float = Field(default=0.0, frozen=True)


class ElementPhotonBase(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:photon 1 base`, for example.

    Attributes
    ----------
    has_material : bool
    has_pixel : bool
    """

    _tao_command_attr_: ClassVar[str] = "ele_photon"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    has_material: bool = Field(default=False, alias="has#material", frozen=True)
    has_pixel: bool = Field(default=False, alias="has#pixel", frozen=True)


class ElementPhotonCurvature(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:photon 1 curvature`, for example.

    Attributes
    ----------
    elliptical_curvature : sequence of floats
    spherical_curvature : float
    xy_0 : sequence of floats
    xy_1 : sequence of floats
    xy_2 : sequence of floats
    xy_3 : sequence of floats
    xy_4 : sequence of floats
    xy_5 : sequence of floats
    xy_6 : sequence of floats
    """

    _tao_command_attr_: ClassVar[str] = "ele_photon"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    elliptical_curvature: FloatSequence = Field(default_factory=list, max_length=3)
    spherical_curvature: float = 0.0
    xy_0: FloatSequence = Field(default_factory=list, max_length=7, alias="xy(0,:)")
    xy_1: FloatSequence = Field(default_factory=list, max_length=7, alias="xy(1,:)")
    xy_2: FloatSequence = Field(default_factory=list, max_length=7, alias="xy(2,:)")
    xy_3: FloatSequence = Field(default_factory=list, max_length=7, alias="xy(3,:)")
    xy_4: FloatSequence = Field(default_factory=list, max_length=7, alias="xy(4,:)")
    xy_5: FloatSequence = Field(default_factory=list, max_length=7, alias="xy(5,:)")
    xy_6: FloatSequence = Field(default_factory=list, max_length=7, alias="xy(6,:)")


class ElementPhotonMaterial(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:photon 2 material`, for example.

    Attributes
    ----------
    f0_m1 : complex or None
    f0_m2 : complex
    f_h : complex
    f_hbar : complex
    sqrt_f_h_f_hbar : complex
    """

    _tao_command_attr_: ClassVar[str] = "ele_photon"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    f0_m1: complex | None = Field(default=None, alias="F0_m1")
    f0_m2: complex = Field(default=0j, alias="F0_m2")
    f_h: complex = Field(default=0j, alias="F_H")
    f_hbar: complex = Field(default=0j, alias="F_Hbar")
    sqrt_f_h_f_hbar: complex = Field(default=0j, alias="Sqrt(F_H*F_Hbar)")


class ElementTwiss(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:twiss 1`, for example.

    Attributes
    ----------
    alpha_a : float
    alpha_b : float
    beta_a : float
    beta_b : float
    dalpha_dpz_a : float
    dalpha_dpz_b : float
    dbeta_dpz_a : float
    dbeta_dpz_b : float
    deta_dpz_a : float
    deta_dpz_b : float
    deta_dpz_x : float
    deta_dpz_y : float
    deta_ds_a : float
    deta_ds_b : float
    deta_dsx : float
    deta_dsy : float
    detap_dpz_a : float
    detap_dpz_b : float
    detap_dpz_x : float
    detap_dpz_y : float
    eta_a : float
    eta_b : float
    eta_x : float
    eta_y : float
    etap_a : float
    etap_b : float
    etap_x : float
    etap_y : float
    gamma_a : float
    gamma_b : float
    mode_flip : bool
    phi_a : float
    phi_b : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_twiss"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    alpha_a: float = Field(default=0.0, frozen=True)
    alpha_b: float = Field(default=0.0, frozen=True)
    beta_a: float = Field(default=0.0, frozen=True)
    beta_b: float = Field(default=0.0, frozen=True)
    dalpha_dpz_a: float = Field(default=0.0, frozen=True)
    dalpha_dpz_b: float = Field(default=0.0, frozen=True)
    dbeta_dpz_a: float = Field(default=0.0, frozen=True)
    dbeta_dpz_b: float = Field(default=0.0, frozen=True)
    deta_dpz_a: float = Field(default=0.0, frozen=True)
    deta_dpz_b: float = Field(default=0.0, frozen=True)
    deta_dpz_x: float = Field(default=0.0, frozen=True)
    deta_dpz_y: float = Field(default=0.0, frozen=True)
    deta_ds_a: float = Field(default=0.0, frozen=True)
    deta_ds_b: float = Field(default=0.0, frozen=True)
    deta_dsx: float = Field(default=0.0, frozen=True)
    deta_dsy: float = Field(default=0.0, frozen=True)
    detap_dpz_a: float = Field(default=0.0, frozen=True)
    detap_dpz_b: float = Field(default=0.0, frozen=True)
    detap_dpz_x: float = Field(default=0.0, frozen=True)
    detap_dpz_y: float = Field(default=0.0, frozen=True)
    eta_a: float = Field(default=0.0, frozen=True)
    eta_b: float = Field(default=0.0, frozen=True)
    eta_x: float = Field(default=0.0, frozen=True)
    eta_y: float = Field(default=0.0, frozen=True)
    etap_a: float = Field(default=0.0, frozen=True)
    etap_b: float = Field(default=0.0, frozen=True)
    etap_x: float = Field(default=0.0, frozen=True)
    etap_y: float = Field(default=0.0, frozen=True)
    gamma_a: float = Field(default=0.0, frozen=True)
    gamma_b: float = Field(default=0.0, frozen=True)
    mode_flip: bool = Field(default=False, frozen=True)
    phi_a: float = Field(default=0.0, frozen=True)
    phi_b: float = Field(default=0.0, frozen=True)


class ElementWakeBase(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wake P3 base`, for example.

    Attributes
    ----------
    has_lr_mode : bool
    has_sr_long : bool
    has_sr_trans : bool
    lr_amp_scale : float
        Wake amplitude scale factor.
    lr_freq_spread : float
        Random frequency spread of long range modes.
    lr_self_wake_on : bool
        Long range self-wake used in tracking?
    lr_time_scale : float
        time scale factor.
    sr_amp_scale : float
        Wake amplitude scale factor.
    sr_scale_with_length : bool
        Scale wake with element length?
    sr_z_max : float
        Max allowable z value. 0-> ignore
    sr_z_scale : float
        z-distance scale factor.
    """

    _tao_command_attr_: ClassVar[str] = "ele_wake"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    has_lr_mode: bool = Field(default=False, alias="has#lr_mode", frozen=True)
    has_sr_long: bool = Field(default=False, alias="has#sr_long", frozen=True)
    has_sr_trans: bool = Field(default=False, alias="has#sr_trans", frozen=True)
    lr_amp_scale: float = Field(
        default=1.0, description="Wake amplitude scale factor.", alias="lr%amp_scale"
    )
    lr_freq_spread: float = Field(
        default=0.0,
        description="Random frequency spread of long range modes.",
        alias="lr%freq_spread",
    )
    lr_self_wake_on: bool = Field(
        default=True,
        description="Long range self-wake used in tracking?",
        alias="lr%self_wake_on",
    )
    lr_time_scale: float = Field(
        default=1.0, description="time scale factor.", alias="lr%time_scale"
    )
    sr_amp_scale: float = Field(
        default=1.0, description="Wake amplitude scale factor.", alias="sr%amp_scale"
    )
    sr_scale_with_length: bool = Field(
        default=True,
        description="Scale wake with element length?",
        alias="sr%scale_with_length",
    )
    sr_z_max: float = Field(
        default=0.0, description="Max allowable z value. 0-> ignore", alias="sr%z_max"
    )
    sr_z_scale: float = Field(
        default=1.0, description="z-distance scale factor.", alias="sr%z_scale"
    )


class ElementWakeSrLong(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wake P3 sr_long`, for example.

    Attributes
    ----------
    z_ref : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_wake"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    z_ref: float = 0.0


class ElementWakeSrTrans(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wake P3 sr_long`, for example.

    Attributes
    ----------
    z_ref : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_wake"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    z_ref: float = 0.0


class ElementWall3DBase(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wall3d 1 1 base`, for example.

    Attributes
    ----------
    clear_material : str or None
    ele_anchor_pt : str
        anchor_beginning$, anchor_center$, or anchor_end$
    name : str
        Identifying name
    opaque_material : str or None
    superimpose : bool or None
        Can overlap another wall
    thickness : float or None
        Material thickness.
    """

    _tao_command_attr_: ClassVar[str] = "ele_wall3d"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    clear_material: str | None = None
    ele_anchor_pt: str = Field(
        default="", description="anchor_beginning$, anchor_center$, or anchor_end$"
    )
    name: str = Field(default="", description="Identifying name")
    opaque_material: str | None = None
    superimpose: bool | None = Field(default=None, description="Can overlap another wall")
    thickness: float | None = Field(default=None, description="Material thickness.")


class ElementWall3DTable_Data(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wall3d 1 1 table`, for example.

    Attributes
    ----------
    j : int
    radius_x : float
    radius_y : float
    tilt : float
    x : float
    y : float
    """

    _tao_command_attr_: ClassVar[str] = "ele_wall3d"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    j: int = 0
    radius_x: float = 0.0
    radius_y: float = 0.0
    tilt: float = 0.0
    x: float = 0.0
    y: float = 0.0


class ElementWall3DTable(TaoModel):
    """
    Structure which corresponds to Tao `pipe ele:wall3d 1 1 table`, for example.

    Attributes
    ----------
    data : ElementWall3DTable_Data
        Structure which corresponds to Tao `pipe ele:wall3d 1 1 table`, for
    example.
    r0 : sequence of floats
        Center of section Section-to-section spline interpolation of the
    center of the section
    s : float
        Longitudinal position
    section : int
    vertex : int
    wall3d_section_type : str
    """

    _tao_command_attr_: ClassVar[str] = "ele_wall3d"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    data: Sequence[ElementWall3DTable_Data] = Field(
        default_factory=list,
        description="Structure which corresponds to Tao `pipe ele:wall3d 1 1 table`, for example.",
    )
    r0: FloatSequence = Field(
        default=[0.0, 0.0],
        max_length=2,
        description=(
            "Center of section Section-to-section spline interpolation of the center of "
            "the section"
        ),
    )
    s: float = Field(default=0.0, description="Longitudinal position")
    section: int = Field(default=0, frozen=True)
    vertex: int = Field(default=0, frozen=True)
    wall3d_section_type: str = Field(default="", alias="wall3d_section^type")


class SpaceChargeCom(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe space_charge_com`, for example.

    Attributes
    ----------
    abs_tol_tracking : float
        Absolute tolerance for tracking.
    beam_chamber_height : float
        Used in shielding calculation.
    cathode_strength_cutoff : float
        Cutoff for the cathode field calc.
    csr3d_mesh_size : sequence of integers
        Gird size for CSR.
    diagnostic_output_file : str
        If non-blank write a diagnostic (EG wake) file
    ds_track_step : float
        CSR tracking step size
    dt_track_step : float
        Time Runge kutta initial step.
    lsc_kick_transverse_dependence : bool
    lsc_sigma_cutoff : float
        Cutoff for the 1-dim longitudinal SC calc. If a bin sigma is < cutoff
    * sigma_ave then ignore.
    n_bin : int
        Number of bins used
    n_shield_images : int
        Chamber wall shielding. 0 = no shielding.
    particle_bin_span : int
        Longitudinal particle length / dz_bin
    particle_sigma_cutoff : float
        3D SC calc cutoff for particles with (x,y,z) position far from the
    center. Negative or zero means ignore.
    rel_tol_tracking : float
        Relative tolerance for tracking.
    sc_min_in_bin : int
        Minimum number of particles in a bin for sigmas to be valid.
    space_charge_mesh_size : sequence of integers
        Gird size for fft_3d space charge calc.
    """

    _tao_command_attr_: ClassVar[str] = "space_charge_com"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    abs_tol_tracking: float = Field(
        default=1e-10, description="Absolute tolerance for tracking."
    )
    beam_chamber_height: float = Field(
        default=0.0, description="Used in shielding calculation."
    )
    cathode_strength_cutoff: float = Field(
        default=0.01, description="Cutoff for the cathode field calc."
    )
    csr3d_mesh_size: IntSequence = Field(
        default=[32, 32, 64], max_length=3, description="Gird size for CSR."
    )
    diagnostic_output_file: str = Field(
        default="", description="If non-blank write a diagnostic (EG wake) file"
    )
    ds_track_step: float = Field(default=0.0, description="CSR tracking step size")
    dt_track_step: float = Field(default=1e-12, description="Time Runge kutta initial step.")
    lsc_kick_transverse_dependence: bool = False
    lsc_sigma_cutoff: float = Field(
        default=0.1,
        description=(
            "Cutoff for the 1-dim longitudinal SC calc. If a bin sigma is < cutoff * "
            "sigma_ave then ignore."
        ),
    )
    n_bin: int = Field(default=0, description="Number of bins used")
    n_shield_images: int = Field(
        default=0, description="Chamber wall shielding. 0 = no shielding."
    )
    particle_bin_span: int = Field(
        default=2, description="Longitudinal particle length / dz_bin"
    )
    particle_sigma_cutoff: float = Field(
        default=-1.0,
        description=(
            "3D SC calc cutoff for particles with (x,y,z) position far from the center. "
            "Negative or zero means ignore."
        ),
    )
    rel_tol_tracking: float = Field(
        default=1e-08, description="Relative tolerance for tracking."
    )
    sc_min_in_bin: int = Field(
        default=10, description="Minimum number of particles in a bin for sigmas to be valid."
    )
    space_charge_mesh_size: IntSequence = Field(
        default=[32, 32, 64],
        max_length=3,
        description="Gird size for fft_3d space charge calc.",
    )


class TaoGlobal(TaoSettableModel):
    """
    Structure which corresponds to Tao `pipe global`, for example.

    Attributes
    ----------
    beam_timer_on : bool
        For timing the beam tracking calculation.
    box_plots : bool
        For debugging plot layout issues.
    bunch_to_plot : int
        Which bunch to plot
    concatenate_maps : bool
        False => tracking using DA.
    de_lm_step_ratio : float
        Scaling for step sizes between DE and LM optimizers.
    de_var_to_population_factor : float
        DE population = max(n_var*factor, 20)
    debug_on : bool
        For debugging.
    delta_e_chrom : float
        Delta E used from chrom calc.
    derivative_recalc : bool
        Recalc before each optimizer run?
    derivative_uses_design : bool
        Derivative calc uses design lattice instead of model?
    disable_smooth_line_calc : bool
        Global disable of the smooth line calculation.
    dmerit_stop_value : float
        Fractional Merit change below which an optimizer will stop.
    draw_curve_off_scale_warn : bool
        Display warning on graphs?
    external_plotting : bool
        Used with matplotlib and gui.
    label_keys : bool
        For lat_layout plots
    label_lattice_elements : bool
        For lat_layout plots
    lattice_calc_on : bool
        Turn on/off beam and single particle calculations.
    lm_opt_deriv_reinit : float
        Reinit derivative matrix cutoff
    lmdif_eps : float
        Tollerance for lmdif optimizer.
    lmdif_negligible_merit : float
    merit_stop_value : float
        Merit value below which an optimizer will stop.
    n_opti_cycles : int
        Number of optimization cycles
    n_opti_loops : int
        Number of optimization loops
    n_threads : int
        Number of OpenMP threads for parallel calculations.
    n_top10_merit : int
        Number of top merit constraints to print.
    only_limit_opt_vars : bool
        Only apply limits to variables used in optimization.
    opt_match_auto_recalc : bool
        Set recalc = True for match elements before each cycle?
    opt_with_base : bool
        Use base data in optimization?
    opt_with_ref : bool
        Use reference data in optimization?
    opti_write_var_file : bool
        "run" command writes var_out_file
    optimizer : str
        optimizer to use.
    optimizer_allow_user_abort : bool
        See Tao manual for more details.
    optimizer_var_limit_warn : bool
        Warn when vars reach a limit with optimization.
    phase_units : str
        Phase units on output.
    plot_on : bool
        Do plotting?
    print_command : str
    random_engine : str
        Non-beam random number engine
    random_gauss_converter : str
        Non-beam
    random_seed : int
        Use system clock by default
    random_sigma_cutoff : float
        Cut-off in sigmas.
    rf_on : bool
        RFcavities on or off? Does not affect lcavities.
    srdt_gen_n_slices : int
        Number times to slice elements for summation RDT calculation
    srdt_sxt_n_slices : int
        Number times to slice sextupoles for summation RDT calculation
    srdt_use_cache : bool
        Create cache for SRDT calculations.  Can use lots of memory if
    srdt_*_n_slices large.
    stop_on_error : bool
        For debugging: False prevents tao from exiting on an error.
    svd_cutoff : float
        SVD singular value cutoff.
    svd_retreat_on_merit_increase : bool
    symbol_import : bool
        Import symbols from lattice file(s)? Internal stuff
    track_type : str
        or 'beam'
    unstable_penalty : float
        Used in unstable_ring datum merit calculation.
    var_limits_on : bool
        Respect the variable limits?
    var_out_file : str
    wait_for_cr_in_single_mode : bool
        For use with a python GUI.
    """

    _tao_command_attr_: ClassVar[str] = "tao_global"
    _tao_set_name_: ClassVar[str] = "global"
    _tao_command_default_args_: ClassVar[dict[str, Any]] = {}
    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()
    beam_timer_on: bool = Field(
        default=False, description="For timing the beam tracking calculation."
    )
    box_plots: bool = Field(default=False, description="For debugging plot layout issues.")
    bunch_to_plot: int = Field(default=1, description="Which bunch to plot")
    concatenate_maps: bool = Field(default=False, description="False => tracking using DA.")
    de_lm_step_ratio: float = Field(
        default=1.0, description="Scaling for step sizes between DE and LM optimizers."
    )
    de_var_to_population_factor: float = Field(
        default=5.0, description="DE population = max(n_var*factor, 20)"
    )
    debug_on: bool = Field(default=False, description="For debugging.")
    delta_e_chrom: float = Field(default=0.0, description="Delta E used from chrom calc.")
    derivative_recalc: bool = Field(
        default=True, description="Recalc before each optimizer run?"
    )
    derivative_uses_design: bool = Field(
        default=False, description="Derivative calc uses design lattice instead of model?"
    )
    disable_smooth_line_calc: bool = Field(
        default=False, description="Global disable of the smooth line calculation."
    )
    dmerit_stop_value: float = Field(
        default=0.0, description="Fractional Merit change below which an optimizer will stop."
    )
    draw_curve_off_scale_warn: bool = Field(
        default=True, description="Display warning on graphs?"
    )
    external_plotting: bool = Field(
        default=False, description="Used with matplotlib and gui.", frozen=True
    )
    label_keys: bool = Field(default=True, description="For lat_layout plots")
    label_lattice_elements: bool = Field(default=True, description="For lat_layout plots")
    lattice_calc_on: bool = Field(
        default=True, description="Turn on/off beam and single particle calculations."
    )
    lm_opt_deriv_reinit: float = Field(
        default=-1.0, description="Reinit derivative matrix cutoff"
    )
    lmdif_eps: float = Field(default=1e-12, description="Tollerance for lmdif optimizer.")
    lmdif_negligible_merit: float = 1e-30
    merit_stop_value: float = Field(
        default=0.0, description="Merit value below which an optimizer will stop."
    )
    n_opti_cycles: int = Field(default=20, description="Number of optimization cycles")
    n_opti_loops: int = Field(default=1, description="Number of optimization loops")
    n_threads: int = Field(
        default=1, description="Number of OpenMP threads for parallel calculations."
    )
    n_top10_merit: int = Field(
        default=10, description="Number of top merit constraints to print."
    )
    only_limit_opt_vars: bool = Field(
        default=False, description="Only apply limits to variables used in optimization."
    )
    opt_match_auto_recalc: bool = Field(
        default=False, description="Set recalc = True for match elements before each cycle?"
    )
    opt_with_base: bool = Field(default=False, description="Use base data in optimization?")
    opt_with_ref: bool = Field(
        default=False, description="Use reference data in optimization?"
    )
    opti_write_var_file: bool = Field(
        default=True, description="'run' command writes var_out_file"
    )
    optimizer: str = Field(default="lm", description="optimizer to use.")
    optimizer_allow_user_abort: bool = Field(
        default=True, description="See Tao manual for more details."
    )
    optimizer_var_limit_warn: bool = Field(
        default=True, description="Warn when vars reach a limit with optimization."
    )
    phase_units: str = Field(default="", description="Phase units on output.")
    plot_on: bool = Field(default=True, description="Do plotting?")
    print_command: str = "lpr"
    random_engine: str = Field(default="", description="Non-beam random number engine")
    random_gauss_converter: str = Field(default="", description="Non-beam")
    random_seed: int = Field(default=-1, description="Use system clock by default")
    random_sigma_cutoff: float = Field(default=-1.0, description="Cut-off in sigmas.")
    rf_on: bool = Field(
        default=True, description="RFcavities on or off? Does not affect lcavities."
    )
    srdt_gen_n_slices: int = Field(
        default=10, description="Number times to slice elements for summation RDT calculation"
    )
    srdt_sxt_n_slices: int = Field(
        default=20,
        description="Number times to slice sextupoles for summation RDT calculation",
    )
    srdt_use_cache: bool = Field(
        default=True,
        description=(
            "Create cache for SRDT calculations.  Can use lots of memory if "
            "srdt_*_n_slices large."
        ),
    )
    stop_on_error: bool = Field(
        default=True, description="For debugging: False prevents tao from exiting on an error."
    )
    svd_cutoff: float = Field(default=1e-05, description="SVD singular value cutoff.")
    svd_retreat_on_merit_increase: bool = True
    symbol_import: bool = Field(
        default=False, description="Import symbols from lattice file(s)? Internal stuff"
    )
    track_type: str = Field(default="single", description="or 'beam'")
    unstable_penalty: float = Field(
        default=0.001, description="Used in unstable_ring datum merit calculation."
    )
    var_limits_on: bool = Field(default=True, description="Respect the variable limits?")
    var_out_file: str = "var#.out"
    wait_for_cr_in_single_mode: bool = Field(
        default=False, description="For use with a python GUI."
    )
