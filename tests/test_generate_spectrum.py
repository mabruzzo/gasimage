
import datetime
import numpy as np
import unyt

from gasimage import default_spin_flip_props
from gasimage.generate_ray_spectrum import (
    _generate_ray_spectrum_py,
    line_profile,
    _generate_noscatter_spectrum,
    blackbody_intensity_cgs
)
from gasimage._generate_spec_cy import (
    full_line_profile_evaluation, _generate_ray_spectrum_cy,
    _generate_noscatter_spectrum_cy
)

from gasimage.rt_config import (
    default_halpha_props, crude_H_partition_func
)


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

    doppler_parameter_b = _doppler_parameter_b / np.array([1,2,3])

    if shift_freq_by_sigma is None:
        v_offset = None
    else:
        freq_arr += _stddev_freq_profile * shift_freq_by_sigma
        v_offset = (np.ones(shape = doppler_parameter_b.shape) *
                    _doppler_parameter_b * shift_freq_by_sigma / np.sqrt(2))

    t1 = datetime.datetime.now()
    profile = fn(obs_freq = freq_arr,
                 doppler_parameter_b = doppler_parameter_b,
                 rest_freq = rest_freq,
                 velocity_offset = v_offset)
    t2 = datetime.datetime.now()
    print(f'\n{fn.__name__}, elapsed: {t2-t1}')

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


def test_line_profile_cython():
    _test_freq_response(rtol = 8e-06, fn = full_line_profile_evaluation,
                        shift_freq_by_sigma = None)
    _test_freq_response(rtol = 3e-05, fn = full_line_profile_evaluation,
                        shift_freq_by_sigma = 1)
    _test_freq_response(rtol = 4e-05, fn = full_line_profile_evaluation,
                        shift_freq_by_sigma = -1)

def _gen_test_data(rest_freq, nfreq = 201, ngas = 100, zero_vlos = False,
                   nominal_dz = 0.25 * unyt.pc,
                   seed = 12345, n_freq_stddev = 1):
    rng = np.random.default_rng(seed = seed)
    T = unyt.unyt_array(1e4, 'K') * rng.uniform(low = 0.75, high = 1.25,
                                                size = ngas)

    pressure_div_kboltz = unyt.unyt_quantity(1e3, 'K/cm**3')
    # assume all gas is neutral Hydrogen in electronic ground state (just
    # to get some basic numbers)
    ndens = pressure_div_kboltz / T

    doppler_parameter_b = np.sqrt(2 * unyt.kboltz_cgs * T / unyt.mh_cgs)
    _stddev_freq_profile = (rest_freq * doppler_parameter_b.max() /
                            (np.sqrt(2) * unyt.c_cgs)).to('Hz')

    freq_arr = np.linspace(rest_freq - n_freq_stddev*_stddev_freq_profile,
                           rest_freq + n_freq_stddev*_stddev_freq_profile,
                           num = nfreq)

    vlos = doppler_parameter_b.min() * rng.uniform(low = 0.5, high = 1.5,
                                                   size = ngas)
    if zero_vlos:
        vlos *= 0
    dz = np.ones(shape = T.shape, dtype = '=f8') * nominal_dz

    return {'T': T, 'ndens' : ndens, 'obs_freq' : freq_arr, 'vlos': vlos,
            'dz' : dz, 'doppler_parameter_b' : doppler_parameter_b}

def _test_generate_ray_spectrum(nfreq = 201, ngas = 100, zero_vlos = False,
                                seed = 12345, rtol = 1e-15, atol = 0):
    print('\nray-gen comparison')

    rest_freq = default_spin_flip_props().freq_quantity

    fn_name_pairs = [(_generate_ray_spectrum_py, 'python version'),
                     (_generate_ray_spectrum_cy, 'optimized cython version')]

    data = _gen_test_data(rest_freq, nfreq = nfreq, ngas = ngas,
                          zero_vlos = zero_vlos, seed = seed)

    if True:

        kwargs = dict(
            obs_freq = data['obs_freq'].to('Hz'),
            velocities = data['vlos'].to('cm/s'),
            ndens_HI_n1state = data['ndens'].to('cm**-3'),
            doppler_parameter_b = data['doppler_parameter_b'].to('cm/s'),
            dz = data['dz'].to('cm'),
        )

        out_l = []

        for fn, fn_name in fn_name_pairs:
            out_l.append( np.zeros(shape = data['obs_freq'].shape,
                                   dtype = '=f8') )
            if fn == _generate_ray_spectrum_py:
                my_kwargs = dict(
                    A10 = default_spin_flip_props().A10_quantity,
                    rest_freq = default_spin_flip_props().freq_quantity,
                    only_spontaneous_emission = True,
                    level_pops_from_stat_weights = True,
                    ignore_natural_broadening = True,
                    **kwargs)
            else:
                my_kwargs = dict(
                    ( (k,arr.ndarray_view()) for k,arr in kwargs.items() )
                )
                my_kwargs['A10_Hz'] = float(
                    default_spin_flip_props().A10_quantity.to('Hz').v)
                my_kwargs['rest_freq'] = float(
                    default_spin_flip_props().freq_quantity.to('Hz').v)
            t1 = datetime.datetime.now()
            fn(out = out_l[-1], **my_kwargs)
            t2 = datetime.datetime.now()
            print(f'{fn_name}, elapsed: {t2-t1}')

        py_version, cy_version = out_l

        np.testing.assert_allclose(
            actual = cy_version, desired = py_version, rtol=rtol, atol = atol,
            err_msg = ("results of the python and optimized cython versions"
                       "of ray-spectrum-generation function disagree")
        )

def test_generate_ray_spectrum():
    # the max atol is ridiculously small!

    # this first one is a little redundant with the rest, but its presence
    # seems to speed up subsequent timings
    _test_generate_ray_spectrum(nfreq = 201, ngas = 100, zero_vlos = False,
                                seed = 987924, rtol = 3e-12)
    
    _test_generate_ray_spectrum(nfreq = 201, ngas = 30, zero_vlos = False,
                                seed = 987924, rtol = 3e-12)
    _test_generate_ray_spectrum(nfreq = 201, ngas = 100, zero_vlos = False,
                                seed = 12345, rtol = 2e-12)
    _test_generate_ray_spectrum(nfreq = 201, ngas = 100, zero_vlos = True,
                                seed = 34432, rtol = 4e-15)

def plot_helper(obs_freq, dz_vals, integrated_source, total_tau, debug_info,
                plot_debug = False):
    import matplotlib.pyplot as plt

    I_unit = (r'$\left[\frac{{\rm erg}}{{\rm cm}^{2}\ {\rm s}\ '
              r'{\rm ster}\ {\rm Hz}}\right]$')

    wave = (unyt.c_cgs/obs_freq).to('nm')

    if not plot_debug:
        fig,ax_arr = plt.subplots(2,1, figsize = (4,8), sharex = True)
        ax_arr[0].plot(wave, integrated_source)
        ax_arr[1].plot(wave, total_tau)
        ax_arr[1].set_xlabel('wavelength (nm)')
        ax_arr[0].set_ylabel(r'$I_\nu(z)$ ' + I_unit)
        ax_arr[1].set_ylabel(r'$\tau_\nu$')
    else:
        tau, source_func = debug_info

        tau_midpoint = 0.5*(tau[:,:-1] + tau[:,1:])
        fig, ax_arr = plt.subplots(3,1, figsize = (4,8))
        ax_arr[0].plot(source_func[0,:])

        center_f = obs_freq.size//2
        for freq_ind in [0,center_f//4, center_f//2, 3*center_f//4, center_f]:
            ax_arr[1].plot(tau[freq_ind, :])
            label = (r'$\lambda$ = ' +
                     f'{float(wave[freq_ind].to("nm").v):.3f} nm')
            ax_arr[2].plot(tau_midpoint[freq_ind, :], source_func[freq_ind,:],
                           label = label)

        ax_arr[0].set_ylabel(r'$S_\nu(z)$ ' + I_unit)
        ax_arr[1].set_ylabel(r'$\tau_\nu(z)$')
        for i in range(2):
            ax_arr[i].set_xlabel('depth')
        ax_arr[2].set_ylabel(r'$S_\nu(\tau_\nu)$ ' + I_unit)
        ax_arr[2].set_xlabel(r'$\tau_\nu$')

        ax_arr[2].set_xlim(0,2)
        ax_arr[2].legend()
    fig.tight_layout()
    plt.show()
    
    

def _test_generate_noscatter_ray_spectrum(nfreq = 401, ngas = 100,
                                          zero_vlos = True,
                                          nominal_dz = (0.25/100) * unyt.pc,
                                          seed = 12345):

    line_props = default_halpha_props()
    partition_func = crude_H_partition_func(electronic_state_max = 20)

    data = _gen_test_data(rest_freq = line_props.freq_quantity, nfreq = nfreq,
                          ngas = ngas, zero_vlos = zero_vlos, seed = seed,
                          nominal_dz = nominal_dz, n_freq_stddev = 4)

    integrated_source, total_tau, debug_info = _generate_noscatter_spectrum(
        line_props = line_props,
        obs_freq = data["obs_freq"], vLOS = data['vlos'].to('cm/s'),
        ndens = data["ndens"], kinetic_T = data["T"],
        doppler_parameter_b = data["doppler_parameter_b"],
        dz = data["dz"],
        level_pops_arg = partition_func,
        return_debug = True)

    # in thermodynamic equilibrium, the source function is just a blackbody
    # -> the way things are currently implemented, we expect the blackbody at
    #    the central frequency to match the source function at all frequencies
    # -> the alternative would be to compute the blackbody at all frequencies
    thermodynamic_beta = 1.0/(unyt.kboltz_cgs * data["T"]).to("erg").v
    expected = blackbody_intensity_cgs(line_props.freq_Hz,
                                       thermodynamic_beta)
    source_func = debug_info[1]
    for i in range(data["obs_freq"].size):
        np.testing.assert_allclose(source_func[i,:].v, expected, rtol = 2e-15,
                                   atol = 0)

    #plot_helper(data["obs_freq"], data["dz"], integrated_source, total_tau,
    #            debug_info, plot_debug = False)



def test_generate_noscatter_ray_spectrum():

    _test_generate_noscatter_ray_spectrum(zero_vlos = True,
                                          nominal_dz = (0.25/100) * unyt.pc)

    _test_generate_noscatter_ray_spectrum(zero_vlos = False,
                                          nominal_dz = (0.25/100) * unyt.pc)
    
    _test_generate_noscatter_ray_spectrum(zero_vlos = True,
                                          nominal_dz = (0.25/20) * unyt.pc)

    # very high column density
    _test_generate_noscatter_ray_spectrum(zero_vlos = True,
                                          nominal_dz = (0.25) * unyt.pc)

def _test_cmp_generate_ray_spectrum(nfreq = 401, ngas = 100,
                                    zero_vlos = True,
                                    nominal_dz = (0.25/100) * unyt.pc,
                                    seed = 12345, rtol = 1e-15, atol = 0):
    print('\nray-gen comparison')

    
    line_props = default_halpha_props()
    partition_func = crude_H_partition_func(electronic_state_max = 20)
    rest_freq = line_props.freq_quantity

    fn_name_pairs = [
        (_generate_noscatter_spectrum, 'python version'),
        (_generate_noscatter_spectrum_cy, 'optimized cython version')]

    data = _gen_test_data(rest_freq = line_props.freq_quantity, nfreq = nfreq,
                          ngas = ngas, zero_vlos = zero_vlos, seed = seed,
                          nominal_dz = nominal_dz, n_freq_stddev = 4)


    kwargs = dict(
        line_props = line_props,
        obs_freq = data["obs_freq"], vLOS = data['vlos'].to('cm/s'),
        ndens = data["ndens"], kinetic_T = data["T"],
        doppler_parameter_b = data["doppler_parameter_b"],
        dz = data["dz"],
        level_pops_arg = partition_func,
    )

    integrated_source_l = []
    tau_l = []

    for fn, fn_name in fn_name_pairs:
        
        t1 = datetime.datetime.now()
        integrated_source, tau_tot = fn( **kwargs)
        t2 = datetime.datetime.now()
        if isinstance(integrated_source, unyt.unyt_array):
            integrated_source = integrated_source.v
        integrated_source_l.append(integrated_source)
        tau_l.append(tau_tot)
        print(f'{fn_name}, elapsed: {t2-t1}')

    for name, out_l in [("integrated_source",integrated_source_l),
                        ("tau_tot", tau_l)]:
        py_version, cy_version = out_l

        np.testing.assert_allclose(
            actual = cy_version, desired = py_version, rtol=rtol, atol = atol,
            err_msg = (f"'{name}'-results produced by the python and optimized "
                       "cython versions disagree")
        )


def test_cmp_generate_noscatter_ray_spectrum():

    _test_cmp_generate_ray_spectrum(zero_vlos = True,
                                    nominal_dz = (0.25/100) * unyt.pc,
                                    rtol = 2e-13)

    _test_cmp_generate_ray_spectrum(zero_vlos = False,
                                    nominal_dz = (0.25/100) * unyt.pc,
                                    rtol = 3e-11)
    
    _test_cmp_generate_ray_spectrum(zero_vlos = False,
                                    nominal_dz = (0.25/20) * unyt.pc,
                                    rtol = 3e-11)

    # very high column density
    _test_generate_noscatter_ray_spectrum(zero_vlos = False,
                                          nominal_dz = (0.25) * unyt.pc)

# an excellent test to add would be checking the curve of growth!
