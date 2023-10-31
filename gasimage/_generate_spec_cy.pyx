import numpy as np
import unyt

cimport numpy as np
cimport cython

from libc.math cimport exp as exp_f64

DEF _INV_SQRT_PI = 0.5641895835477563
DEF _QUARTER_DIV_PI = 0.07957747154594767

# define some CONSTANTS equal to some yt constants
# yt.units.c_cgs =
DEF _C_CGS = 29979245800.0
# = yt.units.h_cgs
DEF _H_CGS = 6.62606957e-27


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
    cdef double temp = _C_CGS / (rest_freq * doppler_parameter_b)

    # now fill in the LineProfileStruct
    cdef LineProfileStruct out
    out.norm = _INV_SQRT_PI * temp
    out.neg_half_div_sigma2 = -1.0 * temp * temp
    out.emit_freq_factor = 1.0 / (1.0 + velocity_offset/_C_CGS)
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
                                double[::1] out):
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
                _H_CGS * rest_freq * n1* A10_Hz * profile_val * _QUARTER_DIV_PI
            )
            out[j] += cur_jnu * cur_dz

    return out

from ._ray_intersections_cy import traverse_grid
from .generate_ray_spectrum import _calc_doppler_parameter_b
from .rt_config import default_spin_flip_props
from .utils.misc import check_consistent_arg_dims
    
def generate_ray_spectrum(grid, grid_left_edge, grid_right_edge,
                          cell_width, grid_shape, cm_per_length_unit,
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


    cdef Py_ssize_t nrays = full_ray_uvec.shape[0]
    if out is not None:
        assert out.shape == (int(nrays), obs_freq.size)
    else:
        out = np.empty(shape = (int(nrays), obs_freq.size), dtype = np.float64)

    vx_vals = grid['gas', 'velocity_x']
    vy_vals = grid['gas', 'velocity_y']
    vz_vals = grid['gas', 'velocity_z']

    assert str(obs_freq.units) == 'Hz'
    cdef const double[::1] _obs_freq_view = obs_freq.ndview
    cdef double[:,::1] _out_view = out

    spin_flip_props = default_spin_flip_props()
    cdef double _A10_Hz = float(spin_flip_props.A10_quantity.to('Hz').v)
    cdef double _rest_freq_Hz = float(spin_flip_props.freq_quantity.to('Hz').v)

    cdef Py_ssize_t i

    try:
        for i in range(nrays):
            ray_start = full_ray_start[i,:]
            ray_uvec = full_ray_uvec[i,:]

            tmp_idx, dz = traverse_grid(
                line_uvec = ray_uvec,
                line_start = ray_start,
                grid_left_edge = grid_left_edge,
                cell_width = cell_width,
                grid_shape = grid_shape
            )

            idx = (tmp_idx[0], tmp_idx[1], tmp_idx[2])

            # convert dz to cm to avoid problems later
            dz *= cm_per_length_unit
            dz = unyt.unyt_array(dz, 'cm')

            # compute the velocity component. We should probably confirm
            # correctness of the velocity sign
            vlos = (ray_uvec[0] * vx_vals[idx] +
                    ray_uvec[1] * vy_vals[idx] +
                    ray_uvec[2] * vz_vals[idx]).to('cm/s')

            if doppler_parameter_b is None:
                # it might be more sensible to make doppler_parameter_b into a
                # field
                cur_doppler_parameter_b = _calc_doppler_parameter_b(
                    grid,idx).to('cm/s')
            else:
                cur_doppler_parameter_b = doppler_parameter_b.to('cm/s')

            ndens_HI_n1state = grid[ndens_HI_n1state_field][idx].to('cm**-3')

            _generate_ray_spectrum_cy(
                obs_freq = _obs_freq_view,
                velocities = vlos.ndview,
                ndens_HI_n1state = ndens_HI_n1state.ndview,
                doppler_parameter_b = cur_doppler_parameter_b.ndview,
                rest_freq = _rest_freq_Hz,
                dz = dz.ndarray_view(),
                A10_Hz = _A10_Hz,
                out = _out_view[i,:])
    except:
        print('There was a problem!')
        pairs = [('line_uvec', ray_uvec),
                 ('line_start', ray_start),
                 ('grid_left_edge', grid_left_edge),
                 ('cell_width', cell_width)]
        for name, arr in pairs:
            arr_str = np.array2string(arr, floatmode = 'unique')
            print(f'{name} = {arr_str}')
        print(f'grid_shape = {np.array2string(np.array(grid_shape))}')
        raise
    return out
