import numpy as np
import unyt

from ._ray_intersections_cy import traverse_grid
from .rt_config import default_spin_flip_props
from .utils.misc import _has_consistent_dims

_inv_sqrt_pi = 1.0/np.sqrt(np.pi)

def line_profile(rest_freq, obs_freq, doppler_parameter_b,
                 velocity_offset = None):
    """
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

    # temp is equivalent to 1./(sqrt(2) * sigma)
    temp = unyt.c_cgs / (rest_freq * doppler_parameter_b)

    norm = _inv_sqrt_pi * temp
    half_div_sigma2 = temp*temp
    
    # need to check this correction!
    if velocity_offset is None:
        emit_freq = obs_freq[:,np.newaxis]
    else:
        assert velocity_offset.shape == doppler_parameter_b.shape
        v_div_c_plus_1 = 1 + velocity_offset/unyt.c_cgs
        emit_freq = obs_freq[:,np.newaxis]/(v_div_c_plus_1.to('dimensionless').v)

    delta_nu = (emit_freq - rest_freq)
    delta_nu_sq = delta_nu*delta_nu
    exponent = (-1*delta_nu_sq*half_div_sigma2)

    return norm*np.exp(exponent.to('dimensionless').v)

def _generate_ray_spectrum_py(obs_freq, velocities, ndens_HI_n1state,
                              doppler_parameter_b, dz, rest_freq,
                              A10 = 2.85e-15*unyt.Hz,
                              only_spontaneous_emission = True,
                              level_pops_from_stat_weights = True,
                              ignore_natural_broadening = True,
                              out = None):
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

    While the result is an ordinary numpy array, the units should be understood
    to be erg/(cm**2 * Hz * s * steradian).

    Parameters
    ----------
    obs_freq
        Array of frequencies to perform radiative transfer at.
    velocities
        Velocities of the gas in cells along the ray
    ndens_HI_n1state
        The number density of neutral hydrogen in cells along the ray
    doppler_parameter_b
        The Doppler parameter (aka Doppler broadening parameter) of the gas in
        cells along the ray, often abbreviated with the variable ``b``. This
        has units consistent with velocity. ``b/sqrt(2)`` specifies the
        standard deviation of the line-of-sight velocity component. When this
        quantity is multiplied by ``rest_freq/unyt.c_cgs``, you get what 
        Rybicki and Lightman call the "Doppler width". This alternative
        quantity is a factor of ``sqrt(2)`` larger than the standard deviation
        of the frequency profile.
    dz
        The distance travelled by the ray through each cell.
    rest_freq
        The rest-frame frequency of the transition
    A10
        The einstein coefficient for spontaneous emission
    only_spontaneous_emission: bool, optional
        Ignore absorption coefficient (and corrections for stimulated emssion)
    level_pops_from_stat_weights:  bool, optional
        When True, we directly approximate the level populations with the ratio
        of the statistical weights.
    ignore_natural_broadening: bool, optional
        When True, we only consider gaussian doppler broadening.

    Notes
    -----
    The level populations of HI energy states are quantified by the "spin 
    temperature" ($T_s$) which is related to level populations according to:
       $\frac{n_1}{n_0} = \frac{g_1}{g_0} \exp(-h \nu_s / (k_B T_s))$
    in which:
     - $n_1$ and $n_0$ specify the number densities of HI in the levels with
       total spin equal to $1$ & $0$
     - $g_1$ and $g_0$ specify the statitical weights of levels 1 and 0
     - $\nu$ is the rest-frame frequency of the transition

    The spin temperature can be linked or entirely independent of the kinetic
    temperature of the gas (i.e. "every-day" temperature). If there's a 
    black-body radiation field, then it could also hypothetically be linked to
    the temperature of the blackbody (There's a whole field of observational
    cosmology that studies the evolution in $T_s$).

    When ``level_pops_from_statistical_weights`` is ``True``, we effectively 
    assume that $\frac{n_1}{n_0} = \frac{g_1}{g_0}$. In other words, we assume
    that $\exp(-h \nu_s / (k_B T_s)) \approx 1$. This is a pretty good
    assumption since $h\nu_s/k_B \approx 0.068\, {\rm K}$. The relative error
    from this approximation is under $1 \%$ for $T_s \geq 7\, {\rm K}$ and 
    under $0.1 \%$ for $T_s \geq 70\, {\rm K}$.
    """

    if (only_spontaneous_emission and level_pops_from_stat_weights and
        ignore_natural_broadening):
        # compute ndens in state 1, by combining info from following points:
        # - since level_pops_from_stat_weights is True: n1/n0 == g1/g0
        # - for 21cm transition, g1 = 3 and g0 = 1 
        # - since n0 and n1 are the only states, n0 + n1 == ndens
        # -> Putting these together: n1/3 + n1 == ndens OR 4*n1/3 == ndens
        n1 = 0.75 * ndens_HI_n1state

        # compute the line-profile
        profiles = line_profile(rest_freq = rest_freq,
                                obs_freq = obs_freq,
                                doppler_parameter_b = doppler_parameter_b,
                                velocity_offset = velocities)

        # now compute the specific (monochromatic) emission coefficient at a
        # number of frequencies.
        # -> a useful aside: the numerator in the following equation dictates
        #    how much energy is emitted in volume dV, frequency-range dnu over
        #    time dt.
        # -> The numerator specifies how much energy is emitted in all
        #    directions. To determine how much energy is emitted per solid
        #    angle dOmega, we divide by the total solid angle (4pi steradians).
        j_nu = (unyt.h_cgs * rest_freq * n1 * A10 * profiles /
                unyt.unyt_quantity(4*np.pi, 'steradian'))
        # now integrate j_nu over the length of the ray:
        integrated = (j_nu*dz).sum(axis=1)

        if out is not None:
            out[:] = integrated.to('erg/(cm**2 * Hz * s * steradian)').v
            return out
        else:
            return integrated.to('erg/(cm**2 * Hz * s * steradian)').v
    elif (not only_spontaneous_emission) and ignore_natural_broadening:
        raise RuntimeError("This branch is untested (& it needs T_spin)")
        return _generate_general_spectrum(
            T_spin = None, # = 100.0*unyt.K
            obs_freq = obs_freq, velocities = velocities,
            ndens_HI_n1state = ndens_HI_n1state,
            doppler_parameter_b = doppler_parameter_b, dz = dz, rest_freq = rest_freq,
            A10 = A10,
            level_pops_from_stat_weights = evel_pops_from_stat_weights
        )
    else:
        raise RuntimeError("support hasn't been added for this configuration")

def _calc_doppler_parameter_b(grid,idx):
    """
    Compute the Doppler parameter (aka Doppler broadening parameter) of the gas
    in cells specified by the inidices idx.

    Note
    ----
    The Doppler parameter had units consistent with velocity and is often
    represented by the variable ``b``. ``b/sqrt(2)`` specifies the standard
    deviation of the line-of-sight velocity component.

    For a given line-transition with a rest-frame frequency ``rest_freq``,
    ``b * rest_freq/unyt.c_cgs`` specifies what Rybicki and Lightman call the
    "Doppler width". The "Doppler width" divided by ``sqrt(2)`` is the standard
    deviation of the line-profile for the given transition.
    """

    T_field, mmw_field = ('gas','temperature'), ('gas','mean_molecular_weight')
    T_vals, mmw_vals = grid[T_field], grid[mmw_field]

    # previously, a bug in the units caused a really hard to debug error (when
    # running the program in parallel. So now we manually check!
    if not _has_consistent_dims(T_vals, unyt.dimensions.temperature):
        raise RuntimeError(f"{T_field!r}'s units are wrong")
    elif not _has_consistent_dims(mmw_vals, unyt.dimensions.dimensionless):
        raise RuntimeError(f"{mmw_field!r}'s units are wrong")

    return np.sqrt(2*unyt.kb_cgs * T_vals[idx] / (mmw_vals[idx] * unyt.mh_cgs))

def generate_ray_spectrum_legacy(grid, grid_left_edge, grid_right_edge,
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

    nrays = full_ray_uvec.shape[0]
    if out is not None:
        assert out.shape == ((nrays,) + obs_freq.shape)
    else:
        out = np.empty(shape = (nrays, obs_freq.size),
                       dtype = np.float64)
    assert str(obs_freq.units) == 'Hz'

    vx_vals = grid['gas', 'velocity_x']
    vy_vals = grid['gas', 'velocity_y']
    vz_vals = grid['gas', 'velocity_z']

    spin_flip_props = default_spin_flip_props()

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

            out[i,:] = _generate_ray_spectrum_py(
                obs_freq = obs_freq, velocities = vlos,
                ndens_HI_n1state = grid[ndens_HI_n1state_field][idx],
                doppler_parameter_b = cur_doppler_parameter_b,
                rest_freq = spin_flip_props.freq_quantity,
                dz = dz,
                A10 = spin_flip_props.A10_quantity,
                only_spontaneous_emission = True,
                level_pops_from_stat_weights = True,
                ignore_natural_broadening = True)
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


# ==========================================================================
# the following function was written at one point when I really dug into the
# details of radiative transfer. It is not tested and probably has bugs. It
# would also take some work to parallelize...
# ==========================================================================

def _generate_general_spectrum(T_spin, obs_freq, velocities, ndens_HI_n1state,
                               doppler_parameter_b, dz, rest_freq,
                               A10 = 2.85e-15*unyt.Hz,
                               level_pops_from_stat_weights = True):
    g1_div_g0 = 3

    exp_term = 1.0
    if not level_pops_from_statistical_weights:
        exp_term = np.exp((-unyt.h_cgs * rest_freq /
                           (unyt.kboltz_cgs * Tspin)).to('dimensionless').v)
    n0 = ndens_HI_n1state / ( 1 + g1_div_g0 * exp_term)

    n1 = ndens_HI_n1state - n0
    B10 = 0.5 * A10 * unyt.c_cgs**2 / (unyt.h_cgs * rest_freq**3)
    B01 = g1_div_g0 * B10

    profiles = line_profile(obs_freq = obs_freq,
                            doppler_parameter_b = doppler_parameter_b,
                            rest_freq = rest_freq,
                            velocity_offset = velocities)
    # absorption coefficient with correction for stimulated emission
    # (This has units of length**-1
    alpha_nu = (unyt.h_cgs * rest_freq * (n0 * B01 - n1 * B10) *
                line_profiles) / (4 * np.pi)

    source_func = n1 * A10 / (n0 * B01 - n1 * B10)
    # in reality, source_func is j_nu / alpha_nu
    # -> when computing j_nu and alpha_nu from the Einstein coefficients,
    #    you wind up dividing by 4pi in both cases. But in the former,
    #    you're really dividing by 4pi steradian.
    # -> While the factors of 4pi cancel out, the units don't. I need to
    #    add them in
    source_func /= unyt.steradian

    # compute tau_nu values at edges between cells
    tau_nu = np.empty(shape = (alpha_nu.shape[0], alpha_nu.shape[1]+1),
                      dtype = alpha_nu.dtype)
    tau_nu[:,0] = 0.0 # when ray enters, optical depth is zero
    tau_nu[:,1:] = (alpha_nu*dz).sum(axis=1).to('dimensionless').v

    # NOTE: max_tau_nu must be known at the observer. For this reason, this
    # algorithm isn't as easily parallelized
    max_tau_nu = tau_nu[:,-1]

    # intensity at tau = 0
    intensity0 = unyt.unyt_array(np.zeros(shape = obs_freq.shape,
                                          dtype = alpha_nu.dtype),
                                 'erg/(cm**2 * Hz * s * steradian)')

    out = unyt.unyt_array(np.zeros(shape = obs_freq.shape,
                                   dtype = alpha_nu.dtype),
                          'erg/(cm**2 * Hz * s * steradian)')

    for nu_ind in range(obs_freq.size):
        # in the following, I drop the nu subscript from every variable
        # I(max_tau) = I(0) * exp(-max_tau) +
        #       int_0^tau (exp(-(max_tau - tau'))*S(tau))dtau
        #
        # Now, let's break the integral up into n parts, where
        n = source_func.shape[1]
        # For frequency-index nu_ind, if we assume that the source function
        # has a constant value of source_func[nu_ind, i] between
        # tau_nu[nu_ind:i] and tau_nu[nu_ind:i], then the integral
        # becomes:
        #   sum_{i=0}^{n-1} [exp(-max_tau_nu) * source_func[nu_ind, i] *
        #                    (exp(tau_nu[nu_ind:i+1] - tau_nu[nu_ind:i])]
        #
        # and the original equation becomes:
        # I(max_tau) = exp(-max_tau) *
        #       [I(0) +
        #        sum_{i=0}^{n-1} [source_func[nu_ind, i] *
        #                        (exp(tau_nu[nu_ind:i+1] - tau_nu[nu_ind:i])
        #       ]

        intensity_tau0 = out.uq * 0.0
        out[nu_ind] = np.exp(-max_tau[nu_ind]) * (
            intensity_tau0 +
            np.sum(source_func[nu_ind, :] *
                   np.exp(np.diff(tau_nu[nu_ind,:])))
        )

    return out, max_tau_nu
