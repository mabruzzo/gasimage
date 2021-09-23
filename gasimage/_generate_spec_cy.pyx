import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport exp as exp_double

DEF _INV_SQRT_PI = 0.3183098861837907
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
                                                  double doppler_v_width,
                                                  double velocity_offset) nogil:
    cdef double temp = _C_CGS / (rest_freq * doppler_v_width)

    cdef LineProfileStruct out
    out.norm = _INV_SQRT_PI * temp
    out.half_div_sigma2 = temp * temp
    out.emit_freq_factor = 1.0 / (1.0 + velocity_offset/_C_CGS)
    return out

cdef inline double eval_line_profile(double obs_freq, double rest_freq,
                                     LineProfileStruct prof) nogil:
    cdef double emit_freq = obs_freq * prof.emit_freq_factor
    cdef double delta_nu = (emit_freq - rest_freq)
    cdef double exponent = (-1.0 * delta_nu * delta_nu * prof.half_div_sigma2)
    return prof.norm * exp_double(exponent)

# Einstein A coefficient (in Hz):
DEF _A10 = 2.85e-15


@cython.boundscheck(False)
@cython.wraparound(False)
def _generate_ray_spectrum_cy(const double[:] obs_freq,
                              const double[:] velocities,
                              const double[:] ndens_HI,
                              const double[:] doppler_v_width,
                              double rest_freq,
                              const double[:] dz,
                              double[:] out):
    """
    obs_freq and rest_freq should be in units should be in units of Hz
    velocities and doppler_v_width should be in units of cm/s
    ndens_HI should be in units of cm^{-3}
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
        n1 = 0.75*ndens_HI[i]
        cur_dz = dz[i]
        prof = prep_LineProfHelper(rest_freq, doppler_v_width[i],
                                   velocities[i])
        for j in range(num_obs_freq):
            profile_val = eval_line_profile(obs_freq[j], rest_freq, prof)
            cur_jnu = (
                _H_CGS * rest_freq * n1* _A10 * profile_val * _QUARTER_DIV_PI
            )
            out[j] += cur_jnu * cur_dz
    return out

    
