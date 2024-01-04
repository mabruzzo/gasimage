import numpy as np
import unyt

cimport numpy as np
cimport cython

from libc.math cimport exp as exp_f64
from libc.math cimport sqrt as sqrt_f64
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from *:
    """
    #define INV_SQRT_PI 0.5641895835477563
    #define QUARTER_DIV_PI 0.07957747154594767
    // define some Macros equal to some yt-constants
    #define C_CGS 29979245800.0            /* yt.units.c_cgs */
    #define HPLANCK_CGS 6.62606957e-27     /* yt.units.h_cgs */
    #define MH_CGS 1.6737352238051868e-24  /* yt.units.mh_cgs */
    #define KBOLTZ_CGS 1.3806488e-16       /* yt.units.kboltz_cgs */
    """
    double INV_SQRT_PI
    double QUARTER_DIV_PI
    double C_CGS
    double HPLANCK_CGS
    double MH_CGS
    double KBOLTZ_CGS


cdef extern from "cpp/stuff.hpp":
    struct LinInterpPartitionFn:
        const double* log10_T
        const double* log10_partition
        long len

    double eval_partition_fn(const LinInterpPartitionFn& pack,
                             double T_val)
        

def _nosubtype_isinstance(obj, classinfo):
    return isinstance(obj, classinfo) and obj.__class__ == np.ndarray

@cython.auto_pickle(True)
cdef class PyLinInterpPartitionFunc:
    cdef object log10_T_arr
    cdef object log10_partition_arr

    def __init__(self, log10_T, log10_partition):
        assert _nosubtype_isinstance(log10_T, np.ndarray)
        assert _nosubtype_isinstance(log10_partition, np.ndarray)
        assert log10_T.ndim == 1
        assert log10_T.size > 1
        assert log10_T.shape == log10_partition.shape

        assert np.isfinite(log10_T).all() and np.isfinite(log10_partition).all()
        assert (log10_T[1:] > log10_T[:-1]).all()

        _kw = dict(order = 'C', dtype = 'f8', casting = 'safe', copy = True)
        self.log10_T_arr = log10_T.astype(**_kw)
        self.log10_partition_arr = log10_partition.astype(**_kw)

    @property
    def log10_T_vals(self): # this is just for testing purposes
        return self.log10_T_arr.copy()

    @property
    def log10_partition_arr(self): # this is just for testing purposes
        return self.log10_partition_arr.copy()

    cdef LinInterpPartitionFn get_partition_fn_struct(self):
        # WARNING: make sure the returned object does not outlive self
        cdef const double[::1] log10_T = self.log10_T_arr
        cdef const double[::1] log10_partition = self.log10_partition_arr

        cdef LinInterpPartitionFn out
        out.log10_T = &log10_T[0]
        out.log10_partition = &log10_partition[0]
        out.len = self.log10_T_arr.size
        return out

    def __call__(self, T_vals):
        # assumes that T_vals is a 1D regular numpy array with units of Kelvin
        # (it's NOT a unyt.unyt_array)

        cdef LinInterpPartitionFn pack = self.get_partition_fn_struct()

        if np.ndim(T_vals) == 0:
            return eval_partition_fn(pack,float(T_vals))

        T_vals = np.asanyarray(T_vals)

        cdef const double[:] T_view = T_vals
        out = np.empty(shape = T_vals.shape, dtype = T_vals.dtype)
        cdef double[:] out_view = out
        cdef Py_ssize_t num_vals = T_view.shape[0]
        cdef Py_ssize_t i

        for i in range(num_vals):
            out_view[i] = eval_partition_fn(pack,T_view[i])
        return out

# When considering a transition we construct a new LineProfileStruct for each
# gas element.
# -> An instance is configured based on the doppler_parameter_b value and los
#    component of that element. It also depends on the transition's rest-frame
#    frequency
# -> An instance holds some precalcuated quantities for evaluating the
#    normalized gaussian:
#        norm * exp( neg_half_div_sigma2 *
#                    (obs_freq*emit_freq_factor - rest_freq)**2 )
cdef struct LineProfileStruct:
    # norm is the normalization factor of the line profile (in frequency
    # space). Multiplying this quantity by the exponential term will normalize
    # the gaussian. (This has units of Hz**-1)
    double norm

    # half_div_sigma2 is used in the exponential. (This has units of Hz**-1)
    double neg_half_div_sigma2

    # emit_freq_factor is ``1.0 / (1 + bulk_vlos / c)``. Multiplying the
    # observed frequency by this factor gives the emission-frequency (i.e. the
    # frequency in the reference frame where the los-component of the bulk
    # velocity is zero)
    double emit_freq_factor

@cython.cdivision(True)
cdef inline LineProfileStruct prep_LineProfHelper(double rest_freq,
                                                  double doppler_parameter_b,
                                                  double velocity_offset) nogil:
    # compute the INVERSE of what Rybicki and Lightman call the "Doppler width"
    # -> NOTE: the input doppler_parameter_b argument specifies a value that is
    #    the standard deviation of the los VELOCITY profile times sqrt(2)
    # -> the quantity Rybicki and Lightman call the "Doppler width" is the
    #    standard-deviation of the FREQUENCY profile times sqrt(2)
    cdef double temp = C_CGS / (rest_freq * doppler_parameter_b)

    # now fill in the LineProfileStruct
    cdef LineProfileStruct out
    out.norm = INV_SQRT_PI * temp
    out.neg_half_div_sigma2 = -1.0 * temp * temp
    out.emit_freq_factor = 1.0 / (1.0 + velocity_offset/C_CGS)
    return out

cdef inline double eval_line_profile(double obs_freq, double rest_freq,
                                     LineProfileStruct prof) nogil:
    # convert from the observed frequency, obs_freq, to emit_freq, the frequency
    # emitted/absorbed (in the frame where the bulk gas velocity is zero)
    #
    # obs_freq = emit_freq * (1 + bulk_vlos / c)
    # emit_freq = obs_freq / (1 + bulk_vlos / c)
    cdef double emit_freq = obs_freq * prof.emit_freq_factor

    # compute the exponential term:
    #     exp(-(emit_freq - rest_freq)**2 / (2 * sigma**2))
    cdef double delta_freq = (emit_freq - rest_freq)
    cdef double exp_term = exp_f64(
        delta_freq * delta_freq * prof.neg_half_div_sigma2
    )

    # finally, multiply the exponential term by the normalization
    return prof.norm * exp_term


@cython.boundscheck(False)
@cython.wraparound(False)
def full_line_profile_evaluation(obs_freq, doppler_parameter_b,
                                 rest_freq, velocity_offset = None):
    """
    This exists purely for testing purposes. It lets the line_profile code be
    called outside of cython.

    Evaluates the line profile, assuming that Doppler broadening is the only
    source of broadening, for each specified observed frequencies and for each
    set of specified gas-properties.

    Parameters
    ----------
    rest_freq: `unyt.unyt_quantity`
        Specifies the rest-frame frequency of the transition.
    obs_freq: `unyt.unyt_array`, (n,)
        An array of frequencies to evaluate the line profile at.
    doppler_parameter_b: `unyt.unyt_array`, shape(m,)
        Array of values for the Doppler parameter (aka Doppler broadening
        parameter) for each of the gas cells that the line_profile is
        evaluated for. This quantity is commonly represented by the variable
        ``b`` and has units consistent with velocity. ``b/sqrt(2)`` specifies
        the standard deviation of the line-of-sight velocity component. When
        this quantity is multiplied by ``rest_freq/unyt.c_cgs``, you get what
        Rybicki and Lightman call the "Doppler width". This alternative
        quantity divided by ``sqrt(2)`` gives the standard deviation of the
        frequency profile.
    velocity_offset: `unyt.unyt_array`, shape(m,), optional
        Array of line-of-sight velocities for each of the gas cells that the
        line_profile is evaluated for. When omitted, this is assumed to have a
        value of zero.

    Returns
    -------
    out: ndarray, shape(n,m)
       Holds the results of the evaluated line profile.
    """
    assert np.ndim(rest_freq) == 0 and np.size(rest_freq) == 1
    assert np.ndim(obs_freq) == 1 and np.size(obs_freq) > 0
    assert (np.ndim(doppler_parameter_b) == 1 and
            np.size(doppler_parameter_b) > 0)

    out = np.empty((doppler_parameter_b.size,obs_freq.size), dtype = '=f8')

    doppler_parameter_b = doppler_parameter_b.to('cm/s').v
    obs_freq = obs_freq.to('Hz').v

    cdef double _rest_freq = rest_freq.to('Hz').v

    cdef double[:,::1] _out = out
    cdef double[::1] _doppler_parameter_b = doppler_parameter_b
    cdef double[::1] _obs_freq = obs_freq
    cdef double[::1] _velocity_offset
    #import datetime
    #t1 = datetime.datetime.now()

    cdef LineProfileStruct prof
    cdef Py_ssize_t num_gas_bins = doppler_parameter_b.size
    cdef Py_ssize_t num_obs_freq = obs_freq.size
    cdef Py_ssize_t i, j

    if velocity_offset is None:
        for i in range(num_gas_bins):
            prof = prep_LineProfHelper(_rest_freq, doppler_parameter_b[i], 0.0)
            for j in range(num_obs_freq):
                _out[i,j] = eval_line_profile(_obs_freq[j], _rest_freq, prof)
        
    else:
        assert velocity_offset.shape == doppler_parameter_b.shape
        velocity_offset = velocity_offset.to('cm/s').v
        _velocity_offset = velocity_offset
        for i in range(num_gas_bins):
            prof = prep_LineProfHelper(_rest_freq, _doppler_parameter_b[i],
                                       _velocity_offset[i])
            for j in range(num_obs_freq):
                _out[i,j] = eval_line_profile(_obs_freq[j], _rest_freq, prof)
    #t2 = datetime.datetime.now()
    #print(t2-t1)
    return out.T / unyt.Hz

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _generate_ray_spectrum_cy(const double[::1] obs_freq,
                                const double[:] velocities,
                                const double[:] ndens_HI_n1state,
                                const double[:] doppler_parameter_b,
                                const double[:] dz,
                                double rest_freq, double A10_Hz,
                                double[::1] out,
                                const double[:] kinetic_T = None):
    """
    Compute specific intensity (aka monochromatic intensity) from data measured
    along the path of a single ray.

    While we only consider the path of a given ray, specific intensity
    actually describes a continuum of rays with infintesimal differences.
    Specific intensity is the amount of energy carried by light in frequency
    range $d\nu$, over time $dt$ along the collection of all rays that pass
    through a patch of area $dA$ (oriented normal to the given-ray) whose
    directions subtend a solid angle of $d\Omega$ around the given-ray.

    For computational convenience, a single call to this function will compute
    the specific intensity at multiple different frequencies.

    The resulting array should be understood to have units of
    erg/(cm**2 * Hz * s * steradian).

    Parameters
    ----------
    obs_freq : ndarray, shape(nfreq,)
        Array of frequencies to perform radiative transfer at (in Hz).
    velocities : ndarray, shape(ngas,)
        Velocities of the gas in cells along the ray (in cm/s).
    ndens_HI_n1state : ndarray, shape(ngas,)
        The number density of neutral hydrogen in cells along the ray (in 
        cm**-3)
    doppler_parameter_b : ndarray, shape(ngas,)
        The Doppler parameter (aka Doppler broadening parameter) of the gas in
        cells along the ray, often abbreviated with the variable ``b``. This
        has units of cm/s. ``b/sqrt(2)`` specifies the standard deviation of
        the line-of-sight velocity component. When this quantity is multiplied
        by ``rest_freq/unyt.c_cgs.v``, you get what Rybicki and Lightman call
        the "Doppler width". This alternative quantity is a factor of
        ``sqrt(2)`` larger than the standard deviation of the frequency profile.
    dz : ndarray, shape(ngas,)
        The distance travelled by the ray through each cell (in cm).
    rest_freq : float
        The rest-frame frequency of the transition (in Hz)
    A10_Hz : float
        The einstein coefficient for spontaneous emission (in Hz)
    out : ndarray, shape(nfreq,)
        Array to hold the outputs
    """
    # this ASSUMES that ``exp(-h_cgs*rest_freq / (kboltz_cgs * T_spin))`` is
    # approximately 1 (this is a fairly decent assumption)

    cdef Py_ssize_t num_cells = dz.size
    cdef Py_ssize_t num_obs_freq = obs_freq.size
    cdef Py_ssize_t i, j

    for j in range(num_obs_freq):
        out[j] = 0

    cdef double n1, cur_dz, profile_val, cur_jnu
    cdef LineProfileStruct prof

    for i in range(num_cells): # iterate over gas-elements

        # compute ndens in state 1, by combining info from following points:
        # - since level_pops_from_stat_weights is True: n1/n0 == g1/g0
        # - for 21cm transition, g1 = 3 and g0 = 1 
        # - since n0 and n1 are the only states, n0 + n1 == ndens
        # -> Putting these together: n1/3 + n1 == ndens OR 4*n1/3 == ndens
        n1 = 0.75*ndens_HI_n1state[i]

        # fetch the length of the path through the current gas-element
        cur_dz = dz[i]

        # construct the profile for the current gas-element
        prof = prep_LineProfHelper(rest_freq, doppler_parameter_b[i],
                                   velocities[i])

        for j in range(num_obs_freq): # iterate over oberved frequencies
            profile_val = eval_line_profile(obs_freq[j], rest_freq, prof)
            cur_jnu = (
                HPLANCK_CGS * rest_freq * n1* A10_Hz * profile_val *
                QUARTER_DIV_PI
            )
            out[j] += cur_jnu * cur_dz

    return out

from .ray_traversal import traverse_grid, max_num_intersections
from .rt_config import default_spin_flip_props
from .utils.misc import check_consistent_arg_dims, _has_consistent_dims

# DEFINE DIFFERENT APPROACHES FOR COMPUTING THE DOPPLER PARAMETER, B
#
# BACKGROUND: The Doppler parameter had units consistent with velocity and is
# often represented by the variable ``b``. ``b/sqrt(2)`` specifies the standard
# deviation of the line-of-sight velocity component. For a given line
# transition with a rest-frame frequency ``rest_freq``,
# ``b * rest_freq/unyt.c_cgs`` specifies what Rybicki and Lightman call the
# "Doppler width". The "Doppler width" divided by ``sqrt(2)`` is the standard
# deviation of the line-profile for the given transition.
#
# We choose to represent the different strategies as distinct ExtensionTypes so
# we can use cython's ability to dispatch based on the desired strategy for
# computing the doppler-broadening
# -> a potential optimization my be to treat the extension types as structs
#    (and avoid attaching methods to them)

# NOTE: I would like to refactor things so that temperature_arr is not stored
# as an attribute... Instead, I want to pass a single value temperature as an
# argument

cdef class PrecomputedDopplerParameter:
    cdef double[:,:,:] doppler_parameter_b_arr

    def __init__(self, doppler_parameter_b_arr):
        self.doppler_parameter_b_arr = doppler_parameter_b_arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef get_vals_cm_per_s(self, long[:,:] idx3Darr,
                           double [::1] out):
        cdef Py_ssize_t i
        for i in range(idx3Darr.shape[1]):
            out[i] = self.doppler_parameter_b_arr[idx3Darr[0,i], idx3Darr[1,i],
                                                  idx3Darr[2,i]]
        

def build_precomputed_doppler_parameter(grid, fixed_val = None):
    if fixed_val is not None:
        assert fixed_val.ndim == 0
        assert fixed_val.to('cm/s').v > 0
        b_arr = np.broadcast_to(fixed_val.to('cm/s').v,
                                shape = grid['gas', 'velocity_x'].shape)
        return PrecomputedDopplerParameter(b_arr)

    # this is the "incorrect" legacy approach!
    T_field = ('gas','temperature')
    mmw_field = ('gas','mean_molecular_weight')
    T_vals, mmw_vals = grid[T_field], grid[mmw_field]

    # previously, a bug in the units caused a really hard to debug error
    # (when running the program in parallel). So now we manually check!
    if not _has_consistent_dims(T_vals, unyt.dimensions.temperature):
        raise RuntimeError(f"{T_field!r}'s units are wrong")
    elif not _has_consistent_dims(mmw_vals, unyt.dimensions.dimensionless):
        raise RuntimeError(f"{mmw_field!r}'s units are wrong")

    to_Kelvin_factor = float(T_vals.uq.to('K').v)
    b_arr = np.sqrt(2.0 * KBOLTZ_CGS * to_Kelvin_factor * T_vals.ndview /
                    (mmw_vals.ndview * MH_CGS))

    return PrecomputedDopplerParameter(b_arr)

cdef double doppler_parameter_b_from_temperature(double kinetic_T,
                                                 double inv_particle_mass_cgs):
    # this assumes that the temperature is already in units of kelvin
    return sqrt_f64(2.0 * KBOLTZ_CGS * kinetic_T * inv_particle_mass_cgs)


cdef class DopplerParameterFromTemperature:
    cdef double[:,:,::1] temperature_arr
    cdef double inv_particle_mass_cgs
    cdef double internal_to_Kelvin_factor

    def __init__(self, grid, particle_mass_in_grams):
        T_field = ('gas','temperature')
        T_vals = grid[T_field]

        # previously, a bug in the units caused a really hard to debug error
        # (when running the program in parallel). So now we manually check!
        if not _has_consistent_dims(T_vals, unyt.dimensions.temperature):
            raise RuntimeError(f"{T_field!r}'s units are wrong")

        self.temperature_arr = T_vals.ndview # strip units
        self.inv_particle_mass_cgs = 1.0 / particle_mass_in_grams

        self.internal_to_Kelvin_factor = float(T_vals.uq.to('K').v)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef get_vals_cm_per_s(self, long[:,:] idx3Darr,
                           double [::1] out):
        cdef Py_ssize_t i
        for i in range(idx3Darr.shape[1]):
            out[i] = doppler_parameter_b_from_temperature(
                self.temperature_arr[idx3Darr[0,i],idx3Darr[1,i],idx3Darr[2,i]],
                self.inv_particle_mass_cgs)

ctypedef fused FusedDopplerParameterType:
    PrecomputedDopplerParameter
    DopplerParameterFromTemperature

def _construct_Doppler_Parameter_Approach(grid, approach,
                                          particle_mass_in_grams = None):
    if (approach is None):
        raise RuntimeError("The default approach for computing the Doppler "
                           "Parameter is changing!")
    elif isinstance(approach,str):
        if approach == 'legacy':
            return build_precomputed_doppler_parameter(grid, None)
        elif approach == 'normal':
            return DopplerParameterFromTemperature(grid, particle_mass_in_grams)
        else:
            raise ValueError(
                "when doppler_parameter_b approach is a str, it must be "
                f"'legacy' or 'normal', not {approach!r}")
    else:
        return build_precomputed_doppler_parameter(grid, approach)

def _calc_doppler_parameter_b(grid, idx3Darr, approach = None,
                              particle_mass_in_grams = None):
    """
    Compute the Doppler parameter (aka Doppler broadening parameter) of the gas
    in cells specified by the inidices idx.

    Note: idx = (idx3Darr[0], idx3Darr[1], idx3Darr[2]) could be used to index
    a numpy array

    Note
    ----
    The Doppler parameter had units consistent with velocity and is often
    represented by the variable ``b``. ``b/sqrt(2)`` specifies the standard
    deviation of the line-of-sight velocity component.

    For a given line-transition with a rest-frame frequency ``rest_freq``,
    ``b * rest_freq/unyt.c_cgs`` specifies what Rybicki and Lightman call the
    "Doppler width". The "Doppler width" divided by ``sqrt(2)`` is the standard
    deviation of the line-profile for the given transition.

    (THIS IS DEFINED FOR BACKWARDS COMPATABILITY)
    """
    out = np.empty(shape = (idx3Darr.shape[1],), dtype = 'f8')
    tmp = _construct_Doppler_Parameter_Approach(
        grid, approach, particle_mass_in_grams = particle_mass_in_grams)
    if isinstance(tmp, PrecomputedDopplerParameter):
        (<PrecomputedDopplerParameter>tmp).get_vals_cm_per_s(
            idx3Darr = idx3Darr, out = out)
    elif isinstance(tmp, DopplerParameterFromTemperature):
        (<DopplerParameterFromTemperature>tmp).get_vals_cm_per_s(
            idx3Darr = idx3Darr, out = out)
    else:
        raise RuntimeError("SOMETHING IS WRONG")
    return out * (unyt.cm/unyt.s)

def generate_ray_spectrum(grid, spatial_grid_props,
                          full_ray_start, full_ray_uvec,
                          rest_freq, obs_freq,
                          doppler_parameter_b = None,
                          ndens_HI_n1state_field = ('gas',
                                                    'H_p0_number_density'),
                          out = None):
    # By default, ``ndens_HI_n1state_field`` is set to the yt-field specifying
    # the number density of all neutral Hydrogen. See the docstring of
    # optically_thin_ppv for further discussion about this approximation

    # do NOT use ``grid`` to access length-scale information. This will really
    # mess some things up related to rescale_length_factor

    check_consistent_arg_dims(obs_freq, unyt.dimensions.frequency, 'obs_freq')
    assert obs_freq.ndim == 1
    assert str(obs_freq.units) == 'Hz'
    cdef const double[::1] _obs_freq_Hz_view = obs_freq.ndview

    nrays = full_ray_uvec.shape[0]
    if out is not None:
        assert out.shape == (nrays, obs_freq.size)
    else:
        out = np.empty(shape = (nrays, obs_freq.size), dtype = np.float64)

    # Prefetch some quantities and do some work to figure out unit conversions
    # ahead of time:

    tmp_ndens_HI_n1state_grid = grid[ndens_HI_n1state_field]
    # get factor that must be multiplied by this ndens to convert to cm**-3
    ndens_to_cc_factor = float(tmp_ndens_HI_n1state_grid.uq.to('cm**-3').v)
    ndens_HI_n1state_grid = tmp_ndens_HI_n1state_grid.ndview

    # load in velocity information:

    tmp_vx = grid['gas', 'velocity_x']
    tmp_vy = grid['gas', 'velocity_y']
    tmp_vz = grid['gas', 'velocity_z']
    assert ((tmp_vx.units == tmp_vy.units) and (tmp_vx.units == tmp_vx.units))
    # compute the factor that must be multiplied by velocity to convert to cm/s
    vel_to_cm_per_s_factor = float(tmp_vx.uq.to('cm/s').v)
    # now, get versions of velocity components without units
    vx, vy, vz = tmp_vx.ndview, tmp_vy.ndview, tmp_vz.ndview

    # currently, this function can only be used for the 21 spin-flip transition
    # -> consequently, we ALWAY specify that the particle mass is that of a
    #    Hydrogen atom!
    doppler_parameter_b = _construct_Doppler_Parameter_Approach(
        grid, approach = doppler_parameter_b, particle_mass_in_grams = MH_CGS
    )

    return _generate_ray_spectrum(
        spatial_grid_props = spatial_grid_props,
        full_ray_start = full_ray_start, full_ray_uvec = full_ray_uvec,
        line_props = default_spin_flip_props(),
        obs_freq_Hz = _obs_freq_Hz_view,
        vx = vx, vy = vy, vz = vz,
        vel_to_cm_per_s_factor = vel_to_cm_per_s_factor,
        doppler_parameter_b = doppler_parameter_b,
        ndens_HI_n1state_grid = ndens_HI_n1state_grid,
        ndens_to_cc_factor = ndens_to_cc_factor,
        out = out)


def _generate_ray_spectrum(object spatial_grid_props,
                           object full_ray_start, object full_ray_uvec,
                           object line_props,
                           double[::1] obs_freq_Hz,
                           double[:,:,::1] vx, double[:,:,::1] vy,
                           double[:,:,::1] vz,
                           double vel_to_cm_per_s_factor,
                           FusedDopplerParameterType doppler_parameter_b,
                           double[:,:,::1] ndens_HI_n1state_grid,
                           double ndens_to_cc_factor,
                           double[:,::1] out):

    cdef bint legacy_optically_thin_spin_flip = True
    cdef list out_l = []

    cdef double _A_Hz = float(line_props.A_quantity.to('Hz').v)
    cdef double _rest_freq_Hz = float(line_props.freq_quantity.to('Hz').v)

    cdef Py_ssize_t nrays = full_ray_uvec.shape[0]
    cdef Py_ssize_t i, j

    # declaring types used by the nested loop:
    cdef long i0, i1, i2
    cdef const double[::1] cur_uvec_view
    cdef long[:,:] idx3D_view

    cdef long max_num = max_num_intersections(spatial_grid_props.grid_shape)
    _vlos_npy = np.empty(shape = (max_num,), dtype = 'f8')
    cdef double[::1] vlos = _vlos_npy
    _ndens_HI_n1state_npy = np.empty(shape = (max_num,), dtype = 'f8')
    cdef double[::1] ndens_HI_n1state = _ndens_HI_n1state_npy

    _cur_doppler_parameter_b_npy = np.empty(shape = (max_num,), dtype = 'f8')
    cdef double[::1] cur_doppler_parameter_b = _cur_doppler_parameter_b_npy

    cdef double cm_per_length_unit = spatial_grid_props.cm_per_length_unit

    # some additional optimizations are definitely still possible:
    # - we can redefine traverse_grid so that it is a cpdef-ed function (avoid
    #   python overhead). We could also define a variant that
    #     (i) specifies tmp_idx with a more optimal layout
    #     (ii) lets us preallocate tmp_idx and dz
    # - we can preallocate buffers for storing the data along rays. Then we can
    #   avoid using numpy's fancy-indexing and directly index ourselves

    try:
        # loop over the rays
        for i in range(nrays):
            ray_start = full_ray_start[i,:]
            ray_uvec = full_ray_uvec[i,:]

            tmp_idx, dz = traverse_grid(line_uvec = ray_uvec,
                                        line_start = ray_start,
                                        spatial_props = spatial_grid_props)
            idx3D_view = tmp_idx

            # convert dz to cm to avoid problems later
            dz *= cm_per_length_unit

            doppler_parameter_b.get_vals_cm_per_s(
                idx3Darr = idx3D_view, out = cur_doppler_parameter_b
            )

            cur_uvec_view = ray_uvec
            with cython.boundscheck(False), cython.wraparound(False):
                for j in range(dz.size):
                    i0 = idx3D_view[0,j]
                    i1 = idx3D_view[1,j]
                    i2 = idx3D_view[2,j]

                    # should probably confirm correctness of velocity sign
                    vlos[j] = (
                        cur_uvec_view[0] * vx[i0,i1,i2] +
                        cur_uvec_view[1] * vy[i0,i1,i2] +
                        cur_uvec_view[2] * vz[i0,i1,i2]
                    ) * vel_to_cm_per_s_factor

            if legacy_optically_thin_spin_flip:
                for j in range(dz.size):
                    i0 = idx3D_view[0,j]
                    i1 = idx3D_view[1,j]
                    i2 = idx3D_view[2,j]

                    ndens_HI_n1state[j] = (ndens_HI_n1state_grid[i0,i1,i2] *
                                           ndens_to_cc_factor)

                _generate_ray_spectrum_cy(
                    obs_freq = obs_freq_Hz,
                    velocities = vlos,
                    ndens_HI_n1state = ndens_HI_n1state,
                    doppler_parameter_b = cur_doppler_parameter_b,
                    rest_freq = _rest_freq_Hz,
                    dz = dz,
                    A10_Hz = _A_Hz,
                    out = out[i,:])
            else:
                # NOT TESTED YET!!!
                
                tmp = _generate_noscatter_spectrum_cy(
                    line_props = line_props,
                    obs_freq = obs_freq_Hz,
                    vLOS = vlos,
                    #ndens = ndens,
                    ndens = None,
                    kinetic_T = None,
                    doppler_parameter_b = cur_doppler_parameter_b,
                    dz = dz,
                    level_pops_arg = None)#partition_func)
                out_l.append({'integrated_source' : tmp[0],
                              'total_tau' : tmp[1]})
    except:
        print(
            "There was a problem!",
            f'line_uvec = {np.array2string(ray_uvec, floatmode = "unique")}',
            f'line_start = {np.array2string(ray_start, floatmode = "unique")}',
            f'spatial_grid_props = {spatial_grid_props!r}',
            sep = '\n  '
        )
        raise


    if legacy_optically_thin_spin_flip:
        return out
    else:
        return out_l


####### DOWN HERE WE DEFINE STUFF RELATED TO FULL (NO-SCATTER) RT

"""
The following struct is used to solve for the optical depth and the integrated
source_function diminished by absorption.

Notes
-----
Our definition of optical depth, differs from Rybicki and Lightman. They
would define the maximum optical depth at the observer's location. Our
choice of definition is a little more consistent with the choice used in
the context of stars.

We are essentially solving the following 2 equations:

.. math::

  \tau_\nu (s) = \int_{s}^{s_0} \alpha_\nu(s^\prime)\, ds^\prime

and

.. math::

  I_\nu (\tau_\nu=0) =  I_\nu(\tau_\nu)\, e^{-\tau_\nu} + f,

where :math:`f`, the integral term, is given by:

.. math::

  f = -\int_{\tau_\nu}^0  S_\nu(\tau_\nu^\prime)\, e^{-\tau_\nu^\prime}\, d\tau_\nu^\prime.

"""

# We construct a new IntegralStructNoScatterRT for every ray
# -> it essentially tracks the output variables and the scratch-space used
#    while performing the integral
# -> in the future, I would like to make this a c++ class
cdef struct IntegralStructNoScatterRT:
    # this specifies the length of all pointers held by this struct
    Py_ssize_t nfreq

    # the next 2 variables are accumulator variables that serve as outputs
    # -> they each accumulate values separately for each considered frequency
    # -> their lifetimes are externally managed
    double* total_tau
    double* integrated_source

    # this last variable is managed by the struct (we use it to cache
    # expensive exponential evaluations)
    double* segStart_expNegTau

cdef IntegralStructNoScatterRT prep_IntegralStructNoScatterRT(
    double[::1] total_tau, double[::1] integrated_source):
    # the nfreq field of the resulting struct is set to 0 if this function
    # encounters issues

    cdef IntegralStructNoScatterRT out
    if ((total_tau.shape[0] == 0) or
        (total_tau.shape[0] != integrated_source.shape[0])):
        out.nfreq = 0
        return out

    out.nfreq = total_tau.shape[0]
    out.total_tau = &total_tau[0]
    out.integrated_source = &integrated_source[0]
    out.segStart_expNegTau = <double*> PyMem_Malloc(sizeof(double) * out.nfreq)
    if out.segStart_expNegTau == NULL:
        out.nfreq = 0

    # finally, lets initialize total_tau & segStart_expNegTau so that they have
    # the correct values for the start of the integral
    # -> essentially, we need to initialize the value at the end of the ray
    #    closest to the observer
    # -> by convention, we define the tau at this location to have an optical
    #    depth of zeros at all frequencies (we can always increase the optical
    #    depth used here after we finish the integral)
    cdef Py_ssize_t freq_i
    for freq_i in range(out.nfreq):
        out.total_tau[freq_i] = 0.0
        out.segStart_expNegTau[freq_i] = 1.0 # = exp_f64(-out.total_tau[freq_i])
    return out


# this function effectively updates the tracked integral (at all frequencies,
# for a single segment of the ray)
#
# In general, we are effectively solving the following integral (dependence on
# frequency is dropped to simplify notation)
#     f = -∫ S(𝜏) * exp(-𝜏) d𝜏 integrated from 𝜏 to 0
# We are integrating the light as it moves from the far end of the ray
# towards the observer.
#
# We can reverse this integral so we are integrating along the ray from
# near to far
#     f = ∫ S(𝜏) * exp(-𝜏) d𝜏 integrated from 0 to 𝜏
#
# Now let's break this integral up into N segments
#
#    f = ∑_(i=0)^(N-1) ∫_i S(𝜏) * exp(-𝜏) d𝜏
# - each integral integrates between 𝜏_i and 𝜏_(i+1) (which correspond to
#   the tau values at the edges of each segment.
#
# Now if we assume that S(𝜏) has a constant value S_i we can pull S_i out
# of the integral and solve the integral analytically.
# ->  ∫ exp(-𝜏) d𝜏 from 𝜏_i to 𝜏_(i+1) is
#           -exp(-𝜏_(i+1)) - (-exp(-𝜏_i))
#     OR equivalently, it's
#           exp(-𝜏_i) - exp(-𝜏_(i+1))
#
# Putting this togeter, we find that:
#    f = ∑_(i=0)^(N-1) S_i * ( exp(-𝜏_i) - exp(-𝜏_(i+1)) )
#
# Coming back the following function:
# -> the function considers a single section of the above summation and
#    evaluates the integral over tau AND the integrated source-term
#
# This function should be repeatedly called moving progressively further from
# the observer
cdef void update_IntegralStructNoScatterRT(const IntegralStructNoScatterRT obj,
                                           const double* absorption_coef,
                                           double source_function, double dz):
    # NOTE: its ok to pass obj by value since the only thing being updated are
    #       pointers pointer held by obj
    # implicit assumption: absorption_coef has obj.nfreq entries

    cdef Py_ssize_t freq_i
    cdef double diff, cur_segEnd_expNegTau
    for freq_i in range(obj.nfreq):
        # part 0: precompute -exp(obj.total_tau[freq_i])
        # -> we need to know the exponential of negative optical depth at start
        #    of current segment
        #
        # this is done implicitly. The value is equivalent to
        # obj.segStart_expNegTau[freq_i]

        # part 1: update obj.total_tau[freq_i] so that it holds the
        #         optical-depth at the end of the current segment
        # -> this is equivalent to saying do the integral over tau in the
        #    current segment
        #
        # recall: we defined tau so that it is increasing as we move away from
        #         the observer
        obj.total_tau[freq_i] += (absorption_coef[freq_i] * dz)

        # part 2: perform the integral over the source term in current segment
        # first, compute the value of exp(-tau) at end of the current segment
        cur_segEnd_expNegTau = exp_f64(-obj.total_tau[freq_i])
        # next, update the integrated source term
        diff = obj.segStart_expNegTau[freq_i] - cur_segEnd_expNegTau
        obj.integrated_source[freq_i] += (source_function * diff)

        # part 3: prepare for next segment (the value of expNegTau at the end of
        # the current segment is the value at the start of the next segment)
        obj.segStart_expNegTau[freq_i] = cur_segEnd_expNegTau

cdef void clean_IntegralStructNoScatterRT(const IntegralStructNoScatterRT obj):
    if ((obj.nfreq > 0) and (obj.segStart_expNegTau != NULL)):
        PyMem_Free(obj.segStart_expNegTau)



from enum import Enum

class NdensStrategy(Enum):
    IonNDensGrid_LTERatio = 1
    # -> we have implemented the above case
    # -> the grid has a field for the total number density of the ion species
    # -> the user must provide a partition function

    #AbsorberNDensGrid_LTERatio = 2
    #AbsorberNDensGrid_ArbitraryRatio = 3
    # -> this case would be nice to have... This is the most generic case!

    SpecialSpinFlipCase = 4

def generate_noscatter_ray_spectrum(
        grid, spatial_grid_props,
        full_ray_start, full_ray_uvec,
        obs_freq, line_props, particle_mass_g, ndens_field,
        partition_func = None, ndens_ratio_field = None,
        ndens_strategy = NdensStrategy.IonNDensGrid_LTERatio):

    cdef list out_l = []

    check_consistent_arg_dims(obs_freq, unyt.dimensions.frequency, 'obs_freq')
    assert obs_freq.ndim == 1
    assert str(obs_freq.units) == 'Hz'
    cdef const double[::1] obs_freq_Hz = obs_freq.ndview

    cdef long max_num = max_num_intersections(spatial_grid_props.grid_shape)

    # load in velocity information:
    tmp_vx = grid['gas', 'velocity_x']
    tmp_vy = grid['gas', 'velocity_y']
    tmp_vz = grid['gas', 'velocity_z']
    assert ((tmp_vx.units == tmp_vy.units) and (tmp_vx.units == tmp_vx.units))
    # compute the factor that must be multiplied by velocity to convert to cm/s
    cdef double vel_to_cm_per_s_factor = float(tmp_vx.uq.to('cm/s').v)
    # now, get versions of velocity components without units
    cdef double[:,:,::1] vx = tmp_vx.ndview
    cdef double[:,:,::1] vy = tmp_vy.ndview
    cdef double[:,:,::1] vz = tmp_vz.ndview
    # prepare the buffer used to hold the LOS velocity data along the ray
    _vlos_npy = np.empty(shape = (max_num,), dtype = 'f8')
    cdef double[::1] vlos = _vlos_npy

    # load in the grid number-density data and prepare the buffer that will be
    # used to hold the number-density data along the ray
    _ndens_grid = grid[ndens_field]
    cdef const double[:,:,::1] ndens_grid = _ndens_grid.ndview
    cdef double ndens_to_invcc = float(_ndens_grid.uq.to('cm**-3').v)
    _ndens_npy = np.empty(shape = (max_num,), dtype = 'f8')
    cdef double[::1] ndens = _ndens_npy

    # load in the grid kinetic-T data and prepare the buffer that will be
    # used to hold the kinetic-T data along the ray
    _kinetic_T_grid = grid['gas','temperature']
    cdef const double[:,:,::1] kinetic_T_grid = _kinetic_T_grid.ndview
    cdef double kinetic_T_to_K = float(_kinetic_T_grid.uq.to('K').v)
    _kinetic_T_npy = np.empty(shape = (max_num,), dtype = 'f8')
    cdef double[::1] kinetic_T_view = _kinetic_T_npy

    # prepare the DopplerParameter-calculator
    cdef DopplerParameterFromTemperature doppler_parameter_b = \
        <DopplerParameterFromTemperature?> \
        _construct_Doppler_Parameter_Approach(
            grid, approach = 'normal',
            particle_mass_in_grams = particle_mass_g)
    # prepare the 1d buffer used to hold the doppler parameter data along the
    # ray
    _cur_doppler_parameter_b_npy = np.empty(shape = (max_num,), dtype = 'f8')
    cdef double[::1] cur_doppler_parameter_b = _cur_doppler_parameter_b_npy

    if ndens_strategy == NdensStrategy.IonNDensGrid_LTERatio:
        assert partition_func is not None
        assert ndens_ratio_field is None
    elif ndens_strategy == NdensStrategy.SpecialSpinFlipCase:
        assert partition_func is None
        assert ndens_ratio_field is not None
        raise NotImplementedError("this case must be implemented!")
    else:
        raise NotImplementedError("unimplemented ndens_strategy")

    cdef Py_ssize_t nrays = full_ray_uvec.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t num_cells

    # declaring types used by the nested loop:
    cdef long i0, i1, i2
    cdef const double[::1] cur_uvec_view
    cdef long[:,:] idx3D_view

    try:

        for i in range(nrays):
            ray_start = full_ray_start[i,:]
            ray_uvec = full_ray_uvec[i,:]

            tmp_idx, dz = traverse_grid(line_uvec = ray_uvec,
                                        line_start = ray_start,
                                        spatial_props = spatial_grid_props)

            idx3D_view = tmp_idx

            # convert dz to cm to avoid problems later (but DON'T convert
            # it into a unyt_array)
            dz = dz * spatial_grid_props.cm_per_length_unit

            doppler_parameter_b.get_vals_cm_per_s(
                idx3Darr = idx3D_view, out = cur_doppler_parameter_b
            )

            # copy grid data along the ray into 1d buffers!
            cur_uvec_view = ray_uvec
            num_cells = dz.size
            with cython.wraparound(False):
                for j in range(num_cells):
                    i0 = idx3D_view[0,j]
                    i1 = idx3D_view[1,j]
                    i2 = idx3D_view[2,j]

                    # should probably confirm correctness of velocity sign
                    vlos[j] = (
                        cur_uvec_view[0] * vx[i0,i1,i2] +
                        cur_uvec_view[1] * vy[i0,i1,i2] +
                        cur_uvec_view[2] * vz[i0,i1,i2]
                    ) * vel_to_cm_per_s_factor

                    ndens[j] = ndens_grid[i0,i1,i2] * ndens_to_invcc

                    kinetic_T_view[j] = (kinetic_T_grid[i0,i1,i2] *
                                         kinetic_T_to_K)

            tmp = _generate_noscatter_spectrum_cy(
                line_props = line_props, obs_freq = obs_freq_Hz,
                vLOS = vlos, ndens = ndens,
                # our choice to pass a numpy array to kinetic_T is deliberate
                kinetic_T = _kinetic_T_npy[:num_cells],
                doppler_parameter_b = cur_doppler_parameter_b,
                dz = dz,
                level_pops_arg = partition_func
            )
            out_l.append({'integrated_source' : tmp[0],
                          'total_tau' : tmp[1]})

    except:
        print(
            "There was a problem!",
            f'line_uvec = {np.array2string(ray_uvec, floatmode = "unique")}',
            f'line_start = {np.array2string(ray_start, floatmode = "unique")}',
            f'spatial_grid_props = {spatial_grid_props!r}',
            sep = '\n  '
        )
        raise
    return out_l

class NdensRatio:
    hi_div_lo: np.ndarray

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void ndens_and_ratio_from_partition(double[::1] ndens_1,
                                         double[::1] n2g1_div_n1g2,
                                         PyLinInterpPartitionFunc partition_fn,
                                         const double[::1] kinetic_T,
                                         const double[::1] ndens_ion_species,
                                         const double g_1,
                                         const double energy_1_erg,
                                         const double rest_freq_Hz,
                                         const Py_ssize_t num_segments):
    cdef double restframe_energy_photon_erg = rest_freq_Hz * HPLANCK_CGS
    cdef double beta_cgs

    cdef LinInterpPartitionFn partition_fn_pack \
        = partition_fn.get_partition_fn_struct()

    for pos_i in range(num_segments):
        # thermodynamic beta
        beta_cgs = 1.0 / (kinetic_T[pos_i] * KBOLTZ_CGS)

        ndens_1[pos_i] = (
            ndens_ion_species[pos_i] * g_1 * exp_f64(-energy_1_erg * beta_cgs)
            / eval_partition_fn(partition_fn_pack, kinetic_T[pos_i]) 
        )

        # n1/n2 = (g1/g2) * exp(restframe_energy_photon_cgs * beta_cgs)
        # n2/n1 = (g2/g1) * exp(-restframe_energy_photon_cgs * beta_cgs)
        # (n2*g1)/(n1*g2) = exp(-restframe_energy_photon_cgs * beta_cgs)
        n2g1_div_n1g2[pos_i] = exp_f64(-1 * restframe_energy_photon_erg *
                                       beta_cgs)

# further optimization ideas:
# 1. don't allow level_pops_arg to be an arbitrary python object
#    (i.e. implement the partition function interpolation in C/C++)
#    -> this would allow us to make use of fused types
#       -> this would further allow us to entirely avoid allocating 2 arrays
#          to hold precomputed values
#    -> we could also then declare kinetic_T as a memoryview
#
# 2. modify update_IntegralStructNoScatterRT so that it operates on individual
#    frequencies (either by making the method operate on individual frequencies
#    or by allocating separate IntegralStructNoScatterRT for each freq)
#
# 3. Modify the doppler-parameter calculation to happen on the fly (since we
#    are already passing in the kinetic temperature anyways)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _generate_noscatter_spectrum_cy(object line_props,
                                    const double[::1] obs_freq,
                                    const double[::1] vLOS,
                                    const double[::1] ndens,
                                    const double[::1] kinetic_T,
                                    const double[::1] doppler_parameter_b,
                                    const double[::1] dz,
                                    object level_pops_arg):
    """
    Compute the specific intensity (aka monochromatic intensity) and optical
    depth from data measured along the path of a single ray.

    While we only consider the path of a given ray, specific intensity 
    actually describes a continuum of rays with infintesimal differences.
    Specific intensity is the amount of energy carried by light in frequency 
    range $d\nu$, over time $dt$ along the collection of all rays that pass 
    through a patch of area $dA$ (oriented normal to the given-ray) whose 
    directions subtend a solid angle of $d\Omega$ around the given-ray.

    For computational convenience, a single call to this function will compute
    the specific intensity at multiple different frequencies.

    While the result is an ordinary numpy array, the units should be understood
    to be erg/(cm**2 * Hz * s * steradian).

    Parameters
    ----------
    obs_freq : unyt.unyt_array, shape(nfreq,)
        An array of frequencies to perform rt at.
    vLOS : np.ndarray, shape(ngas,)
        Velocities of the gas in cells along the ray (in cm/s).
    ndens : np.ndarray, shape(ngas,)
        The number density in cells along the ray (in cm**-3). The exact
        meaning of this argument depends on ``level_pops_arg``.
    kinetic_T : np.ndarray, shape(ngas,)
        Specifies the kinetic temperatures along the ray (in units of Kelvin)
    doppler_parameter_b : unyt.unyt_array, shape(ngas,)
        The Doppler parameter (aka Doppler broadening parameter) of the gas in
        cells along the ray, often abbreviated with the variable ``b``. This
        has units consistent with velocity. ``b/sqrt(2)`` specifies the
        standard deviation of the line-of-sight velocity component. When this
        quantity is multiplied by ``rest_freq/unyt.c_cgs``, you get what 
        Rybicki and Lightman call the "Doppler width". This alternative
        quantity is a factor of ``sqrt(2)`` larger than the standard deviation
        of the frequency profile.
    dz : np.ndarray, shape(ngas,)
        The distance travelled by the ray through each cell (in cm).
    level_pops_arg : NdensRatio or callable
        If this is an instance of NdensRatio, it specifies the ratio of number
        densities and the ``ndens`` argument is treated as the number density
        of particles in the lower state. Otherwise this argument must be a 
        callable (that accepts temperature as an argument) representing the 
        Partition Function. In this scenario, we assume LTE and treat the 
        ``ndens`` argument as the number density of all particles described by
        the partition function.

    Returns
    -------
    optical_depth: `numpy.ndarray`, shape(nfreq,ngas+1)
        Holds the integrated optical depth as a function of frequency computed
        at the edge of each ray-segment. We define optical depth such that it 
        is zero at the observer,
        and such that it increases as we move backwards along the ray (i.e. it
        increases with distance from the observer)
    integrated_source: ndarray, shape(nfreq,)
        Holds the integrated_source function as a function of frequency. This 
        is also the total intensity if there is no background intensity. If
        the background intensity is given by ``bkg_intensity`` (an array where
        theat varies with frequency), then the total intensity is just
        ``bkg_intensity*np.exp(-tau[:, -1]) + integrated_source``.
    """
    cdef Py_ssize_t num_segments = dz.size
    cdef Py_ssize_t pos_i, freq_i # indexing variables

    # load transition properties:
    cdef double rest_freq_Hz = line_props.freq_Hz

    # consider 2 states: states 1 and 2.
    # - State2 is the upper level and State1 is the lower level
    # - B12 multiplied by average intensity gives the rate of absorption
    # - A21 gives the rate of spontaneous emission
    cdef double B12_cgs = line_props.B_absorption_cgs
    cdef double g_1 = float(line_props.g_lo)
    cdef double g_2 = float(line_props.g_hi)

    # allocate & precompute both ndens_1 & n2g1_div_n1g2
    _ndens_1 = np.empty(shape = (num_segments,), dtype = 'f8')
    cdef double[::1] ndens_1 = _ndens_1
    _n2g1_div_n1g2 = np.empty(shape = (num_segments,), dtype = 'f8')
    cdef double[::1] n2g1_div_n1g2 = _n2g1_div_n1g2

    if isinstance(level_pops_arg, PyLinInterpPartitionFunc):
        # in this case:
        # -> treat level_pops_arg as partition function
        # -> treat ndens as number density of particles described by partition
        #    function
        # -> assume LTE

        ndens_and_ratio_from_partition(
            ndens_1 = ndens_1, n2g1_div_n1g2 = n2g1_div_n1g2,
            partition_fn = <PyLinInterpPartitionFunc>level_pops_arg,
            kinetic_T = kinetic_T, ndens_ion_species = ndens,
            g_1 = g_1, energy_1_erg = line_props.energy_lo_erg,
            rest_freq_Hz = rest_freq_Hz, num_segments = num_segments)
    elif isinstance(level_pops_arg, NdensRatio):
        raise RuntimeError("UNTESTED")
        _ndens_1[:] = ndens # in this case, treat ndens as lower-state ndens
        _n2g1_div_n1g2[:] = level_pop_arg.hi_div_lo * g_1 / g_2
    else:
        raise ValueError("unallowed level_pops_arg")

    # Down below we actually perform the integral!

    cdef Py_ssize_t nfreq = obs_freq.shape[0]

    # the following are output variables (they will be updated by functions
    # operating upon accumulator)
    tau = np.empty(shape = (nfreq,), dtype = 'f8')
    integrated_source = np.zeros(shape=(nfreq,), dtype = 'f8')

    cdef IntegralStructNoScatterRT accumulator = \
        prep_IntegralStructNoScatterRT(tau, integrated_source)

    if accumulator.nfreq < 1:
        raise RuntimeError("Problem with initializing the accumulator!")

    cdef double rest_freq3 = rest_freq_Hz*rest_freq_Hz*rest_freq_Hz
    cdef double source_func_numerator = (2*HPLANCK_CGS * rest_freq3 /
                                         (C_CGS * C_CGS))

    # temproray variables used within the loop:
    cdef LineProfileStruct prof
    cdef double profile_val, alpha_center, stim_emission_correction
    cdef double source_func_cgs
    cdef double* alpha_nu_cgs = <double*> PyMem_Malloc(sizeof(double) * nfreq)

    for pos_i in range(num_segments):

        # Using equations 1.78 and 1.79 of Rybicki and Lighman to get
        # - absorption coefficient (with units of cm**-1), including the
        #   correction for stimulated-emission
        # - the source function
        # - NOTE: there are some weird ambiguities in the frequency dependence
        #   in these equations. These are discussed below.

        # first compute alpha at the line-center (ignoring broadening profile)
        # & then compute alpha as a function of frequency:
        stim_emission_correction = (1.0 - n2g1_div_n1g2[pos_i])
        alpha_center = (HPLANCK_CGS * rest_freq_Hz * ndens_1[pos_i] * B12_cgs *
                        stim_emission_correction) / (4*np.pi)

        # construct the profile for the current gas-element
        prof = prep_LineProfHelper(rest_freq_Hz, doppler_parameter_b[pos_i],
                                   vLOS[pos_i])
        for freq_i in range(nfreq):
            profile_val = eval_line_profile(obs_freq[freq_i], rest_freq_Hz,
                                            prof)
            alpha_nu_cgs[freq_i] = alpha_center * profile_val

        source_func_cgs = (
            source_func_numerator / ((1.0/n2g1_div_n1g2[pos_i]) - 1.0)
        )

        # FREQUENCY AMBIGUITIES:
        # - in Rybicki and Lighman, the notation used in equations 1.78 and 1.79
        #   suggest that all frequencies used in computing linear_absorption and
        #   source_function should use the observed frequency (in the
        #   reference-frame where gas-parcel has no bulk motion)
        # - currently, we use the line's central rest-frame frequency everywhere
        #   other than in the calculation of the line profile.
        #
        # In the absorption-coefficient, I think our choice is well-motivated!
        # -> if you look back at the derivation the equation 1.74, it seems
        #    that the leftmost frequency should be the rest_freq (it seems like
        #    they dropped this along the way and carried it through to 1.78)
        # -> in the LTE case, they don't use rest-freq in the correction for
        #    stimulated emission (eqn 1.80). I think what we do in the LTE case
        #    is more correct.
        #
        # Overall, I suspect the reason that Rybicki & Lightman play a little
        # fast and loose with frequency dependence is the sort of fuzzy
        # assumptions made when deriving the Einstein relations. (It also helps
        # them arive at result that source-function is a black-body in LTE)
        #
        # At the end of the day, it probably doesn't matter much which
        # frequencies we use (the advantage of our approach is we can put all
        # considerations of LOS velocities into the calculation of the line
        # profile)

        # now that we know the source-function and the absorption coefficient,
        # let's compute the integral(s) for the current section of the array
        update_IntegralStructNoScatterRT(accumulator, alpha_nu_cgs,
                                         source_func_cgs, dz[pos_i])

    # do some cleanup!
    clean_IntegralStructNoScatterRT(accumulator)
    PyMem_Free(alpha_nu_cgs)

    # these output arrays were updated within accumulator!
    return integrated_source, tau
