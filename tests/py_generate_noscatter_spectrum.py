# this is basically a python implementation of the generate_noscatter_spectrum
# -> it was originally in the gasimage module, but it's not really being used
#    except for testing purposes.
# -> for that reason, we've moved it here

from datetime import datetime

import numpy as np
import unyt

from gasimage._generate_spec_cy import _calc_doppler_parameter_b, NdensRatio
from gasimage.generate_ray_spectrum import line_profile
from gasimage._ray_intersections_cy import traverse_grid

def blackbody_intensity_cgs(freq_Hz, thermodynamic_beta_cgs):
    h_cgs = unyt.h_cgs.v
    c_cgs = unyt.c_cgs.v
    numerator = 2 * h_cgs * (freq_Hz*freq_Hz*freq_Hz) / (c_cgs * c_cgs)
    denominator = np.exp(h_cgs * freq_Hz * thermodynamic_beta_cgs) - 1.0
    return numerator / denominator


def _generate_noscatter_spectrum(line_props, obs_freq, vLOS, ndens, kinetic_T,
                                 doppler_parameter_b, dz, level_pops_arg, *,
                                 return_debug = False):
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
    vLOS : unyt.unyt_array, shape(ngas,)
        Velocities of the gas in cells along the ray (in cm/s).
    ndens : unyt.unyt_array, shape(ngas,)
        The number density in cells along the ray (in cm**-3). The exact
        meaning of this argument depends on ``level_pops_arg``.
    kinetic_T : unyt.unyt_array, shape(ngas,)
        Specifies the kinetic temperatures along the ray
    doppler_parameter_b : unyt.unyt_array, shape(ngas,)
        The Doppler parameter (aka Doppler broadening parameter) of the gas in
        cells along the ray, often abbreviated with the variable ``b``. This
        has units consistent with velocity. ``b/sqrt(2)`` specifies the
        standard deviation of the line-of-sight velocity component. When this
        quantity is multiplied by ``rest_freq/unyt.c_cgs``, you get what 
        Rybicki and Lightman call the "Doppler width". This alternative
        quantity is a factor of ``sqrt(2)`` larger than the standard deviation
        of the frequency profile.
    dz : unyt.unyt_array, shape(ngas,)
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

    t_list = []
    last_t = datetime.now()

    kboltz_cgs = unyt.kboltz_cgs.v
    h_cgs = unyt.h_cgs.v

    rest_freq = line_props.freq_quantity

    # consider 2 states: states 1 and 2.
    # - State2 is the upper level and State1 is the lower level
    # - B12 multiplied by average intensity gives the rate of absorption
    # - A21 gives the rate of spontaneous emission

    B12_cgs = unyt.unyt_quantity(line_props.B_absorption_cgs,
                                 'Hz*steradian*cm**2/erg')
    A21_cgs = unyt.unyt_quantity(line_props.A_Hz, 'Hz')

    g_1, g_2 = float(line_props.g_lo), float(line_props.g_hi)

    energy_1_erg = line_props.energy_lo_erg
    restframe_energy_photon_erg = (rest_freq * unyt.h_cgs).to('erg').v

    if callable(level_pops_arg):
        # assume LTE
        ndens_ion_species = ndens
        partition_func = level_pops_arg
        # thermodynamic beta
        beta_cgs = 1.0 / (kinetic_T * unyt.kboltz_cgs).to('erg').v

        # in this case, treat ndens as number density of particles described by
        # partition function
        ndens_1 = (ndens_ion_species * g_1 * np.exp(-energy_1_erg * beta_cgs)
                   / partition_func(kinetic_T))

        # n1/n2 = (g1/g2) * exp(restframe_energy_photon_cgs * beta_cgs)
        # n2/n1 = (g2/g1) * exp(-restframe_energy_photon_cgs * beta_cgs)
        # (n2*g1)/(n1*g2) = exp(-restframe_energy_photon_cgs * beta_cgs)
        n2g1_div_n1g2 = np.exp(-1 * restframe_energy_photon_erg * beta_cgs)

    else:
        raise RuntimeError("UNTESTED")
        assert isinstance(level_pops_arg, NdensRatio)
        # in this case, treat ndens as number density of lower state
        ndens_1 = ndens
        ndens2_div_ndens1 = level_pop_arg.hi_div_lo
        n2g1_div_n1g2 = ndens2_div_ndens1 * g_1 /g_2

    t_list.append(('Boltzmann-stuff', datetime.now() - last_t))
    last_t = datetime.now()

    # compute the line_profiles
    profiles = line_profile(obs_freq = obs_freq,
                            doppler_parameter_b = doppler_parameter_b,
                            rest_freq = rest_freq,
                            velocity_offset = vLOS)

    t_list.append(('profile calculation', datetime.now() - last_t))
    last_t = datetime.now()

    # Using equations 1.78 and 1.79 of Rybicki and Lighman to get
    # - absorption coefficient (with units of cm**-1), including the correction
    #   for stimulated-emission
    # - the source function
    # - NOTE: there are some weird ambiguities in the frequency dependence in
    #   these equations. These are discussed below.
    stim_emission_correction = (1.0 - n2g1_div_n1g2)
    alpha_nu = (unyt.h_cgs * rest_freq * ndens_1 * B12_cgs *
                stim_emission_correction * profiles) / (4*np.pi)
    # the division by steradian comes from the fact we divide by 4pi
    alpha_nu = (alpha_nu / unyt.steradian).to('cm**-1')

    rest_freq3= rest_freq*rest_freq*rest_freq
    tmp = (1.0/n2g1_div_n1g2) - 1.0
    source_func = (2*unyt.h_cgs * rest_freq3 / (unyt.c_cgs * unyt.c_cgs)) / tmp
    # the formula for the source function has an implicit steradian^-1 term
    # we manually add in
    source_func = (source_func/unyt.steradian).to(
        'erg/(cm**2 * Hz * s * steradian)')

    # convert source_func from 1D to 2D. The way we have constructed the source
    # function has no frequency dependence (other than on the central
    # rest-frame frequency)
    source_func = np.broadcast_to(source_func[None,:], shape = alpha_nu.shape,
                                  subok = True)

    # FREQUENCY AMBIGUITIES:
    # - in Rybicki and Lighman, the notation used in equations 1.78 and 1.79
    #   suggest that all frequencies used in computing linear_absorption and
    #   source_function should use the observed frequency (in the
    #   reference-frame where gas-parcel has no bulk motion)
    # - currently, we use the line's central rest-frame frequency everywhere
    #   other than in the calculation of the line profile.
    #
    # In the absorption-coefficient, I think our choice is well-motivated!
    # -> if you look back at the derivation the equation 1.74, it looks seems
    #    that the leftmost frequency should be the rest_freq (it seems like
    #    they dropped this along the way and carried it through to 1.78)
    # -> in the LTE case, they don't use rest-freq in the correction for
    #    stimulated emission (eqn 1.80). I think what we do in the LTE case is
    #    more correct.
    #
    # Overall, I suspect the reason that Rybicki & Lightman play a little fast
    # and loose with frequency dependence is the sort of fuzzy assumptions made
    # when deriving the Einstein relations. (It also helps them arive at result
    # that source-function is a black-body in LTE)
    #
    # At the end of the day, it probably doesn't matter much which frequencies
    # we use (the advantage of our approach is we can put all considerations of
    # LOS velocities into the calculation of the line profile)

    t_list.append(('computing absorption & source', datetime.now() - last_t))
    last_t = datetime.now()

    dz = dz.to('cm')

    tau, integrated_source = solve_noscatter_rt(source_function = source_func,
                                                absorption_coef = alpha_nu,
                                                dz = dz)
    t_list.append(('integral', datetime.now() - last_t))

    if False:
        print("python integration profiling:")
        for name, t in t_list:
            print(f'->{name}: {t}')

    if return_debug:
        return integrated_source, tau[:,-1], (tau, source_func)
    else:
        return integrated_source, tau[:,-1]

def solve_noscatter_rt(source_function, absorption_coef, dz):
    """
    Solves for the optical depth and the integrated source_function diminished
    by absorption.

    Each input arg is an array. For each arg (other than ``bkg_intensity``) the
    index of the trailing axis corresponds to position along a ray. 
      - `arr[...,0]` specifies the value at the location closest to the observer
      - `arr[...,-1]` specifies the value at the location furthest from the
        observer.
    To put it another way, light moves from high index to low index. 
    Alternatively, as we increase index, we move "backwards along the ray.

    Parameters
    ----------
    source_function: `unyt.unyt_array`, shape(nfreq,ngas)
        The source function
    absorption_coef: `unyt.unyt_array`, shape(nfreq,ngas)
        The linear absorption coefficient
    dz : `unyt.unyt_array`, shape(ngas,)
        The distance travelled by the ray through each cell (in cm).
    bkg_intensity : `unyt.unyt_array`, shape(nfreq,)
        The background intensity

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

    # part 1: solve for tau
    #
    # we defined tau so that it is increasing as we move away from the observer
    #
    # NOTE: higher indices of dz are also further from observer
    nfreq = source_function.shape[0]

    starting_tau = 0.0
    tau = np.empty(shape = (nfreq, dz.size + 1), dtype = 'f8')

    tau[:, 0] = starting_tau
    tau[:, 1:] = starting_tau + np.cumsum(
        (absorption_coef * dz).to('dimensionless').ndview, axis = 1
    )

    # we are effectively solving the following integral (dependence on
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

    # multiply diffs by -1 since it computes exp(-𝜏_(i+1)) - exp(-𝜏_i) 
    diffs = -np.diff(np.exp(-tau), axis = 1)
    integral_term = (source_function * diffs).sum(axis = 1)

    return tau, integral_term
