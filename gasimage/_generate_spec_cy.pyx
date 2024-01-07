import numpy as np
import unyt

cimport numpy as np
cimport cython

from libc.math cimport exp as exp_f64
from libc.math cimport sqrt as sqrt_f64
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# CLEANUP ToDo List:
# -> unify the interface a little more between _generate_ray_spectrum_cy and
#    _generate_noscatter_spectrum_cy
# -> add support for NdensStrategy.SpecialLegacySpinFlipCase to
#    _generate_noscatter_spectrum_cy
# -> simplify the doppler-parameter stuff... it's definitely too complicated
#    we basically just have 2 branches:
#    1. The branch where we compute b from kinetic temperature on the fly
#    2. The branch where we precompute b on the grid
# -> I think it would be sensible to consolidate all of the operations
#    involving the kinetic temperature
#
# Optimization ideas:
# -> it would make sense to make some kind of grid struct that tracks pointers
#    to the 3D quantities (this is a can of worms... do we use memoryviews? Or,
#    do we force all "fields" to have consistent strides?)
#    - under this picture, we would skip the whole step of extracting into a 1D
#      buffer. We would instead directly have the code extract the values from
#      the 3D grid as they are needed
#    - this would help with the hypothetical case where we eventually introduce
#      higher order numerical integration with trilinear interpolation
# -> of course this would all be easier to do if we started shifting this logic
#    to c++ (for use of templates...)

cdef extern from "cpp/stuff.hpp":

    # these are all some constants defined in the included header
    double INV_SQRT_PI
    double QUARTER_DIV_PI
    double C_CGS
    double HPLANCK_CGS
    double MH_CGS
    double KBOLTZ_CGS

    struct C_LineProps:
        int g_lo
        int g_hi
        double freq_Hz # rest frequency
        double energy_lo_erg # energy of the lower state
        double A21_Hz
        double B12_cgs

    struct LinInterpPartitionFn:
        const double* log10_T
        const double* log10_partition
        long len

    double eval_partition_fn(const LinInterpPartitionFn& pack,
                             double T_val)

    struct Ndens_And_Ratio:
        double ndens_1
        double n2g1_div_n1g2

    Ndens_And_Ratio ndens_and_ratio_from_partition(
        LinInterpPartitionFn partition_fn_pack, double kinetic_T,
        double ndens_ion_species, C_LineProps line_props)

    struct LineProfileStruct:
        double norm
        double neg_half_div_sigma2
        double emit_freq_factor

    LineProfileStruct prep_LineProfHelper(double rest_freq,
                                          double doppler_parameter_b,
                                          double velocity_offset)

    double eval_line_profile(double obs_freq, double rest_freq,
                             LineProfileStruct prof)

    struct RayAlignedProps:
        long num_segments
        const double* dz
        const double* vLOS
        const double* ndens
        const double* kinetic_T
        const double* precomputed_doppler_parameter_b

    void optically_thin_21cm_ray_spectrum_impl(
        const C_LineProps line_props, const long nfreq, const double* obs_freq,
        const RayAlignedProps ray_aligned_data, double* out)

    int generate_noscatter_spectrum_impl(
        C_LineProps line_props, const long nfreq, const double* obs_freq,
        const RayAlignedProps ray_aligned_data,
        const LinInterpPartitionFn partition_fn_pack,
        double* out_integrated_source, double* out_tau)

cdef C_LineProps get_LineProps_struct(object line_props) except *:
    # this converts the LineProps class into C_LineProps
    cdef C_LineProps out
    out.g_lo = line_props.g_lo
    out.g_hi = line_props.g_hi
    out.freq_Hz = line_props.freq_Hz
    out.energy_lo_erg = line_props.energy_lo_erg
    out.A21_Hz = line_props.A_Hz
    out.B12_cgs = line_props.B_absorption_cgs
    return out

cdef RayAlignedProps get_RayAlignedProps(
    long num_segments,
    const double[::1] dz, const double[::1] vLOS, const double[::1] ndens,
    const double[::1] kinetic_T,
    const double[::1] precomputed_doppler_parameter_b
) except *:

    cdef RayAlignedProps out
    out.num_segments = num_segments
    out.dz = &dz[0]
    out.vLOS = &vLOS[0]
    out.ndens = &ndens[0]
    if kinetic_T is None:
        out.kinetic_T = NULL
    else:
        out.kinetic_T = &kinetic_T[0]

    if precomputed_doppler_parameter_b is None:
        out.precomputed_doppler_parameter_b = NULL
    else:
        out.precomputed_doppler_parameter_b = (
            &precomputed_doppler_parameter_b[0]
        )
    return out

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

def _generate_opticallythin_spectrum_cy(object line_props,
                                        const double[::1] obs_freq,
                                        const double[::1] vLOS,
                                        const double[::1] ndens_HI_n1state,
                                        const double[::1] doppler_parameter_b,
                                        const double[::1] dz,
                                        double[::1] out,
                                        const double[::1] kinetic_T = None):
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
    line_props : LineProps
        Encodes details about the line transition
    obs_freq : ndarray, shape(nfreq,)
        Array of frequencies to perform radiative transfer at (in Hz).
    vLOS : ndarray, shape(ngas,)
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
    out : ndarray, shape(nfreq,)
        Array to hold the outputs
    """
    cdef C_LineProps c_line_props = get_LineProps_struct(line_props)

    cdef RayAlignedProps ray_data = get_RayAlignedProps(
        num_segments = dz.shape[0],
        dz = dz, vLOS = vLOS, ndens = ndens_HI_n1state, kinetic_T = kinetic_T,
        precomputed_doppler_parameter_b = doppler_parameter_b
    )
    optically_thin_21cm_ray_spectrum_impl(c_line_props,
                                          obs_freq.shape[0], &obs_freq[0],
                                          ray_data, &out[0])
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


from enum import Enum

class NdensStrategy(Enum):
    """
    This specifies the "strategy" when it comes to number density.

    To be a little more clear:
      - this specifies the interpretation of the ndens field
      - it also specifies how we compute the ratio of the number densities
        of the absorbers and emitters
    """

    IonNDensGrid_LTERatio = 1
    # we have implemented the above case
    # -> the grid has a field for the total number density of the ion species.
    # -> the user must provide a partition function; it is used to compute the
    #    number density of the absorber (assuming LTE)
    # -> the number-density ratio is assumed

    # These are 2 variants I hope to support in the future...
    #AbsorberNDensGrid_LTERatio = 2
    #AbsorberNDensGrid_ArbitraryRatio = 3
    # -> this case would be nice to have... This is the most generic case!

    SpecialLegacySpinFlipCase = 4
    # this is what we have traditionally used in the optically-thin case
    # -> we effectively assume that the number density field gives the
    #    number-density of Hydrogen in the electronic ground state (n = 1).
    #    This is the sum of the number densities in the spin 0 and spin 1 states
    # -> we assume that the ratio of number densities is directly given by the
    #    statistical weights...


def generate_ray_spectrum(grid, spatial_grid_props,
                          full_ray_start, full_ray_uvec, line_props,
                          legacy_optically_thin_spin_flip,
                          particle_mass_in_grams, obs_freq,
                          ndens_strat, ndens_field, partition_func = None,
                          doppler_parameter_b = None,
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

    if ndens_strat == NdensStrategy.IonNDensGrid_LTERatio:
        assert partition_func is not None

    # Prefetch some quantities and do some work to figure out unit conversions
    # ahead of time:

    tmp_ndens = grid[ndens_field]
    # get factor that must be multiplied by this ndens to convert to cm**-3
    ndens_to_cc_factor = float(tmp_ndens.uq.to('cm**-3').v)
    ndens_grid = tmp_ndens.ndview

    # load in velocity information:

    tmp_vx = grid['gas', 'velocity_x']
    tmp_vy = grid['gas', 'velocity_y']
    tmp_vz = grid['gas', 'velocity_z']
    assert ((tmp_vx.units == tmp_vy.units) and (tmp_vx.units == tmp_vx.units))
    # compute the factor that must be multiplied by velocity to convert to cm/s
    vel_to_cm_per_s_factor = float(tmp_vx.uq.to('cm/s').v)
    # now, get versions of velocity components without units
    vx, vy, vz = tmp_vx.ndview, tmp_vy.ndview, tmp_vz.ndview

    if legacy_optically_thin_spin_flip:
        assert ndens_strat == NdensStrategy.SpecialLegacySpinFlipCase
    else:
        assert out is None
        if ndens_strat == NdensStrategy.SpecialLegacySpinFlipCase:
            raise NotImplementedError("THIS CASE ISN'T SUPPORTED YET!")

    # in certain cases, we don't need to load the temperature field... It may
    # be useful to avoid loading it in these cases (to handle cases where the
    # temperature field isn't defined)
    kinetic_T = grid['gas', 'temperature']
    kinetic_T_to_K_factor = float(kinetic_T.uq.to('K').v)

    # currently, this function can only be used for the 21 spin-flip transition
    # -> consequently, we ALWAY specify that the particle mass is that of a
    #    Hydrogen atom!
    doppler_parameter_b = _construct_Doppler_Parameter_Approach(
        grid, approach = doppler_parameter_b, particle_mass_in_grams = MH_CGS
    )

    return _generate_ray_spectrum(
        legacy_optically_thin_spin_flip = legacy_optically_thin_spin_flip,
        spatial_grid_props = spatial_grid_props,
        full_ray_start = full_ray_start, full_ray_uvec = full_ray_uvec,
        line_props = line_props, ndens_strat = ndens_strat,
        obs_freq_Hz = _obs_freq_Hz_view,
        vx = vx, vy = vy, vz = vz,
        vel_to_cm_per_s_factor = vel_to_cm_per_s_factor,
        doppler_parameter_b = doppler_parameter_b,
        ndens_grid = ndens_grid,
        ndens_to_cc_factor = ndens_to_cc_factor,
        kinetic_T_grid = kinetic_T.ndview,
        kinetic_T_to_K_factor = kinetic_T_to_K_factor,
        partition_func = partition_func,
        out = out)


def _generate_ray_spectrum(bint legacy_optically_thin_spin_flip,
                           object spatial_grid_props,
                           object full_ray_start, object full_ray_uvec,
                           object line_props, object ndens_strat,
                           double[::1] obs_freq_Hz,
                           double[:,:,::1] vx, double[:,:,::1] vy,
                           double[:,:,::1] vz,
                           double vel_to_cm_per_s_factor,
                           FusedDopplerParameterType doppler_parameter_b,
                           double[:,:,::1] ndens_grid,
                           double ndens_to_cc_factor,
                           double[:,:,::1] kinetic_T_grid,
                           double kinetic_T_to_K_factor,
                           object partition_func,
                           object out):
    # this function sure has a lot of arguments...

    cdef Py_ssize_t nrays = full_ray_uvec.shape[0]
    cdef Py_ssize_t i, j

    cdef Py_ssize_t nfreq = obs_freq_Hz.shape[0]
    if legacy_optically_thin_spin_flip:
        out_shape = (nrays, nfreq)
    else:
        out_shape = (nrays,  2*nfreq)

    if out is None:
        out = np.empty(out_shape, dtype = 'f8')
    else:
        assert out.shape == out_shape
        assert out.dtype == np.dtype('f8')

    cdef double[:,::1] out_view = out

    # declaring types used within the (nested) loop:
    cdef long i0, i1, i2, num_cells
    cdef const double[::1] cur_uvec_view
    cdef long[:,:] idx3D_view

    # allocate 1d buffers used to hold the various quantities along the ray
    cdef long max_num = max_num_intersections(spatial_grid_props.grid_shape)
    cdef dict npy_buffers = dict(
        (k, np.empty(shape = (max_num,), dtype = 'f8'))
        for k in ['dz_cm', 'vlos', 'ndens', 'kinetic_T', 'doppler_parameter'])

    cdef double[::1] dz_cm = npy_buffers['dz_cm']
    cdef double[::1] vlos = npy_buffers['vlos']
    cdef double[::1] ndens = npy_buffers['ndens']
    cdef double[::1] kinetic_T = npy_buffers['kinetic_T']
    cdef double[::1] cur_doppler_parameter_b = npy_buffers['doppler_parameter']

    cdef RayAlignedProps ray_data = get_RayAlignedProps(
        num_segments = 0,
        dz = dz_cm, vLOS = vlos, ndens = ndens, kinetic_T = kinetic_T,
        precomputed_doppler_parameter_b = cur_doppler_parameter_b
    )

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

            tmp_idx, tmp_dz = traverse_grid(line_uvec = ray_uvec,
                                            line_start = ray_start,
                                            spatial_props = spatial_grid_props)
            idx3D_view = tmp_idx

            # convert dz to cm to avoid problems later
            tmp_dz *= cm_per_length_unit

            num_cells = tmp_dz.shape[0]
            ray_data.num_segments = num_cells

            doppler_parameter_b.get_vals_cm_per_s(
                idx3Darr = idx3D_view, out = cur_doppler_parameter_b
            )

            # read the quantities along the ray into the 1d buffer
            # -> explicitly access the buffers through the memoryviews rather
            #    than the pointers stored in the ray_data struct, because the
            #    pointers are all `const double*`, to denote that they are
            #    read-only to the actual ray-tracing functions
            cur_uvec_view = ray_uvec
            with cython.boundscheck(False), cython.wraparound(False):
                for j in range(ray_data.num_segments):
                    # convert dz to cm to avoid problems later
                    dz_cm[j] = tmp_dz[j] 

                    # extract the jth element along the ray
                    i0 = idx3D_view[0,j]
                    i1 = idx3D_view[1,j]
                    i2 = idx3D_view[2,j]

                    # should probably confirm correctness of velocity sign
                    vlos[j] = (
                        cur_uvec_view[0] * vx[i0,i1,i2] +
                        cur_uvec_view[1] * vy[i0,i1,i2] +
                        cur_uvec_view[2] * vz[i0,i1,i2]
                    ) * vel_to_cm_per_s_factor

                    ndens[j] = (ndens_grid[i0,i1,i2] * ndens_to_cc_factor)
                    kinetic_T[j] = (kinetic_T_grid[i0,i1,i2] *
                                    kinetic_T_to_K_factor)

            if legacy_optically_thin_spin_flip:
                # SANITY CHECK:
                assert ndens_strat == NdensStrategy.SpecialLegacySpinFlipCase
                
                _generate_opticallythin_spectrum_cy(
                    line_props = line_props,
                    obs_freq = obs_freq_Hz, vLOS = vlos,
                    ndens_HI_n1state = ndens,
                    doppler_parameter_b = cur_doppler_parameter_b,
                    dz = dz_cm[:num_cells],
                    out = out_view[i,:])
            else:
                # sanity check!
                assert ndens_strat == NdensStrategy.IonNDensGrid_LTERatio
                
                _generate_noscatter_spectrum_cy(
                    line_props = line_props,
                    obs_freq = obs_freq_Hz,
                    vLOS = vlos,
                    ndens = ndens,
                    kinetic_T = kinetic_T,
                    doppler_parameter_b = cur_doppler_parameter_b,
                    dz = dz_cm[:num_cells],
                    level_pops_arg = partition_func,
                    out_integrated_source = out_view[i,:nfreq],
                    out_tau = out_view[i, nfreq:])
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
        out_l = [
            {'integrated_source' : out[i,:nfreq], 'total_tau' : out[i,nfreq:]}
            for i in range(nrays)
        ]
        return out_l


####### DOWN HERE WE DEFINE STUFF RELATED TO FULL (NO-SCATTER) RT

class NdensRatio:
    hi_div_lo: np.ndarray

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


def _generate_noscatter_spectrum_cy(object line_props,
                                    const double[::1] obs_freq,
                                    const double[::1] vLOS,
                                    const double[::1] ndens,
                                    const double[::1] kinetic_T,
                                    const double[::1] doppler_parameter_b,
                                    const double[::1] dz,
                                    object level_pops_arg,
                                    double[::1] out_integrated_source,
                                    double[::1] out_tau):
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
    line_props : LineProps
        Encodes details about the line transition
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
    integrated_source: ndarray, shape(nfreq,)
        Holds the integrated_source function as a function of frequency. This 
        is also the total intensity if there is no background intensity. If
        the background intensity is given by ``bkg_intensity`` (an array where
        theat varies with frequency), then the total intensity is just
        ``bkg_intensity*np.exp(-tau[:, -1]) + integrated_source``.
    optical_depth: `numpy.ndarray`, shape(nfreq,ngas+1)
        Holds the integrated optical depth as a function of frequency computed
        at the edge of each ray-segment. We define optical depth such that it 
        is zero at the observer,
        and such that it increases as we move backwards along the ray (i.e. it
        increases with distance from the observer)
    """
    cdef C_LineProps c_line_props = get_LineProps_struct(line_props)

    cdef PyLinInterpPartitionFunc partition_fn
    cdef LinInterpPartitionFn partition_fn_pack

    cdef RayAlignedProps ray_data = get_RayAlignedProps(
        num_segments = dz.shape[0],
        dz = dz, vLOS = vLOS, ndens = ndens, kinetic_T = kinetic_T,
        precomputed_doppler_parameter_b = doppler_parameter_b
    )

    if isinstance(level_pops_arg, PyLinInterpPartitionFunc):
        partition_fn = <PyLinInterpPartitionFunc>level_pops_arg
        partition_fn_pack = partition_fn.get_partition_fn_struct()

        errcode = generate_noscatter_spectrum_impl(
            c_line_props, obs_freq.shape[0], &obs_freq[0], ray_data,
            partition_fn_pack, &out_integrated_source[0], &out_tau[0])
    else:
        raise ValueError("unallowed/untested level_pops_arg")

    if errcode != 0:
        raise RuntimeError("SOMETHING WENT WRONG")
