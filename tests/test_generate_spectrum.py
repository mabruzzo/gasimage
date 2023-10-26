import numpy as np
import unyt

from gasimage.generate_ray_spectrum import line_profile

def _simplest_line_profile_impl(obs_freq, doppler_parameter_b,
                                rest_freq, velocity_offset = None):
    """
    this is just a dummy implementation

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

    if velocity_offset is None:
        velocity_offset = doppler_parameter_b * 0.0
    else:
        assert velocity_offset.shape == doppler_parameter_b.shape

    out = np.empty((obs_freq.size, doppler_parameter_b.size), dtype = '=f8')
    for j in range(doppler_parameter_b.size):
        # compute what Rybicki and Lightman call the "Doppler width", and
        # represent with the variable $delta\nu_D$
        delta_nu = doppler_parameter_b[j] * rest_freq / unyt.c_cgs

        def line_window(f):
            tmp  = -(f-rest_freq)**2 / (delta_nu)**2
            return ( np.exp(tmp.to('dimensionless').v) /
                     (delta_nu * np.sqrt(np.pi)) )

        bulk_vlos = velocity_offset[j]

        for i in range(obs_freq.size):
            # convert from the observed frequency, f, to f_bulkrest the
            # frequency emitted/absorbed in the frame where the bulk gas
            # velocity is zero
            #
            # f = f_bulkrest * (1 + bulk_vlos / c)
            # f_bulkrest = f / (1 + bulk_vlos / c)
            f_bulkrest = obs_freq[i] / (1 + bulk_vlos / unyt.c_cgs)

            out[i,j] = line_window(f_bulkrest).to('1/Hz').v
    return out / unyt.Hz

def _test_freq_response(rtol, fn = _simplest_line_profile_impl,
                        shift_freq_by_sigma = None):

    nsigma_integral_pairs = [(1,0.682689492137086), (2,0.954499736103642),
                             (3,0.997300203936740)]

    rest_freq = 1.4204058E+09 * unyt.Hz
    T = unyt.unyt_array(1e4, 'K')
    _doppler_parameter_b = np.sqrt(2 * unyt.kboltz_cgs * T / unyt.mh_cgs)
    _stddev_freq_profile = (rest_freq * _doppler_parameter_b /
                            (np.sqrt(2) * unyt.c_cgs)).to('Hz')

    freq_arr = np.linspace(rest_freq - _stddev_freq_profile,
                           rest_freq + _stddev_freq_profile,
                           num = 201)

    if shift_freq_by_sigma is None:
        v_offset = None
    else:
        freq_arr += _stddev_freq_profile * shift_freq_by_sigma
        v_offset = (np.array([1,1,1]) *
                    _doppler_parameter_b * shift_freq_by_sigma / np.sqrt(2))

    doppler_parameter_b = _doppler_parameter_b / np.array([1,2,3])

    profile = fn(obs_freq = freq_arr,
                 doppler_parameter_b = doppler_parameter_b,
                 rest_freq = rest_freq,
                 velocity_offset = v_offset)
    print(f'\n{fn.__name__}')

    nsigma = 1
    delta_freq = np.diff(freq_arr)[0].v

    for i, (n_stddev, expected_integral) in enumerate(nsigma_integral_pairs):
        integral = np.trapz(profile[:,i].v, dx = delta_freq)
        print(integral, np.abs(integral - expected_integral)/expected_integral,
              expected_integral)
        np.testing.assert_allclose(
            actual = integral, desired = expected_integral, rtol=rtol, atol = 0,
            err_msg = ("problem with comparison of integral out to "
                       f"+/- {n_stddev} of the profile computed by "
                       f"{fn.__name__} with a {shift_freq_by_sigma} sigma "
                       "shift in the central frequency")
        )

def test_dummy_func():
    _test_freq_response(rtol = 8e-06, fn = _simplest_line_profile_impl,
                        shift_freq_by_sigma = None)
    _test_freq_response(rtol = 3e-05, fn = _simplest_line_profile_impl,
                        shift_freq_by_sigma = 1)
    _test_freq_response(rtol = 4e-05, fn = _simplest_line_profile_impl,
                        shift_freq_by_sigma = -1)

def test_line_profile():
    _test_freq_response(rtol = 8e-06, fn = line_profile,
                        shift_freq_by_sigma = None)
    _test_freq_response(rtol = 3e-05, fn = line_profile,
                        shift_freq_by_sigma = 1)
    _test_freq_response(rtol = 4e-05, fn = line_profile,
                        shift_freq_by_sigma = -1)
