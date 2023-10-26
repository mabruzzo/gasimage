import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport exp as exp_double

DEF _INV_SQRT_PI = 0.5641895835477563
DEF _QUARTER_DIV_PI = 0.07957747154594767

# define some CONSTANTS equal to some yt constants
# yt.units.c_cgs =
DEF _C_CGS = 29979245800.0
# = yt.units.h_cgs
DEF _H_CGS = 6.62606957e-27

cdef struct LineProfileStruct:
    # norm is the normalization factor of the line profile
    double norm
    # half_div_sigma2 is used in the exponential
    double half_div_sigma2
    # multiply emit_freq_factor by obs_freq to get the emision frequency
    double emit_freq_factor

@cython.cdivision(True)
cdef inline LineProfileStruct prep_LineProfHelper(double rest_freq,
                                                  double doppler_parameter_b,
                                                  double velocity_offset):# nogil:
    cdef double temp = _C_CGS / (rest_freq * doppler_parameter_b)
    cdef LineProfileStruct out
    out.norm = _INV_SQRT_PI * temp
    out.half_div_sigma2 = temp * temp
    out.emit_freq_factor = 1.0 / (1.0 + velocity_offset/_C_CGS)
    return out

cdef inline double eval_line_profile(double obs_freq, double rest_freq,
                                     LineProfileStruct prof): #nogil:
    cdef double emit_freq = obs_freq * prof.emit_freq_factor
    cdef double delta_nu = (emit_freq - rest_freq)
    cdef double exponent = (-1.0 * delta_nu * delta_nu * prof.half_div_sigma2)
    return prof.norm * exp_double(exponent)

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

    import unyt

    if velocity_offset is None:
        velocity_offset = doppler_parameter_b * 0.0
    else:
        assert velocity_offset.shape == doppler_parameter_b.shape

    out = np.empty((obs_freq.size, doppler_parameter_b.size), dtype = '=f8')

    velocity_offset = velocity_offset.to('cm/s').v
    doppler_parameter_b = doppler_parameter_b.to('cm/s').v
    rest_freq = rest_freq.to('Hz').v
    obs_freq = obs_freq.to('Hz').v

    cdef LineProfileStruct prof
    cdef Py_ssize_t num_obs_freq = obs_freq.size
    cdef Py_ssize_t i, j

    for i in range(doppler_parameter_b.size):
        prof = prep_LineProfHelper(rest_freq, doppler_parameter_b[i],
                                   velocity_offset[i])
        for j in range(num_obs_freq):
            out[j,i] = eval_line_profile(obs_freq[j], rest_freq, prof)
    return out / unyt.Hz


# Einstein A coefficient (in Hz):
DEF _A10 = 2.85e-15


@cython.boundscheck(False)
@cython.wraparound(False)
def _generate_ray_spectrum_cy(const double[:] obs_freq,
                              const double[:] velocities,
                              const double[:] ndens_HI_n1state,
                              const double[:] doppler_parameter_b,
                              double rest_freq,
                              const double[:] dz,
                              double[:] out):
    """
    obs_freq and rest_freq should be in units should be in units of Hz
    velocities and doppler_parameter_b should be in units of cm/s
    ndens_HI_n1state should be in units of cm^{-3}
    dz should be in units of cm
    
    out should be the same size as obs_freq
    """

    cdef Py_ssize_t num_cells = dz.size
    cdef Py_ssize_t num_obs_freq = obs_freq.size
    cdef Py_ssize_t i, j

    for j in range(num_obs_freq):
        out[j] = 0

    cdef double n1, cur_dz, profile_val, cur_jnu
    cdef LineProfileStruct prof

    for i in range(num_cells):
        n1 = 0.75*ndens_HI_n1state[i]
        cur_dz = dz[i]
        prof = prep_LineProfHelper(rest_freq, doppler_parameter_b[i],
                                   velocities[i])
        for j in range(num_obs_freq):
            profile_val = eval_line_profile(obs_freq[j], rest_freq, prof)
            cur_jnu = (
                _H_CGS * rest_freq * n1* _A10 * profile_val * _QUARTER_DIV_PI
            )
            out[j] += cur_jnu * cur_dz
    return out

    
