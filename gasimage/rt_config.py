import dataclasses
import numpy as np
import unyt

from gasimage._generate_spec_cy import PyLinInterpPartitionFunc

@dataclasses.dataclass(frozen = True)
class LineProperties:
    """
    Encodes data about a line-transition

    Notes
    -----
    - to support natural-broadening we would need other quantities (like
      oscillator strength)
    - if we ever want to support multiple lines in a single calculation, we
      may want to consider somehow making these instances lighter weight!
    - to make things more self documenting, we may want to track cases where
      we are looking at hyperfine transitions between levels
    """

    freq_Hz: float # the rest-frame frequency for the transition
    A_Hz: float  # Einstein coefficient for spontaneous emission
    g_lo: int
    g_hi: int
    energy_lo_erg: float # sometimes called the excitation potential

    def __post_init__(self):
        assert 0 < self.freq_Hz < float('inf')
        assert 0 < self.A_Hz < float('inf')
        assert (0 < self.g_lo) and isinstance(self.g_lo, int)
        assert (0 < self.g_hi) and isinstance(self.g_hi, int)
        assert 0 <= self.energy_lo_erg < float('inf')

    @property
    def A_quantity(self):
        return unyt.unyt_quantity(self.A_Hz, "Hz")

    @property
    def freq_quantity(self):
        return unyt.unyt_quantity(self.freq_Hz, "Hz")

    @property
    def B_absorption_cgs(self): return B_coefs(self)[0]

    @property
    def B_stimemiss_cgs(self):  return B_coefs(self)[1]

    @property
    def energy_lo_quantity(self):
        return unyt.unyt_quantity(self.energy_lo_erg,'erg')

def B_coefs(lineprop):
    # returns the pair of Einstein B coefficients:
    #    (B_absorption, B_stimulated_emission)
    #
    # if we are considering 2 states, 1 & 2, (1 has lower energy so absorption
    # causes a transition from 1 to 2), then:
    # -> B_absorption coefficient is called B12
    # -> B_stimulated_emission coefficient is called B21
    A21, freq = lineprop.A_quantity, lineprop.freq_quantity
    g1, g2 = lineprop.g_lo, lineprop.g_hi

    B_stimulated_emission = (0.5 * A21 * (unyt.c_cgs*unyt.c_cgs) /
                             (unyt.h_cgs * (freq*freq*freq))).in_cgs().v
    B_absorption = g2 * B_stimulated_emission / g1
    return (B_absorption, B_stimulated_emission)

def default_spin_flip_props():
    # note for the future: the
    # -> following linked paper notes that the natural width of the 21 cm line
    #    is negligibly small
    #        https://ui.adsabs.harvard.edu/abs/2019MNRAS.483..593K
    # -> the wikipedia page on the 21cm line (aka H line) elaborates on this a
    #    little.
    #    -> Apparently, the uncertainty principle links lifetime with the
    #       uncertainty of the energy level (the uncertainty in the energy
    #       level is linked to the natural width).
    #    -> Because the transition has such a long lifetime the spectral line
    #       has an extremely small natural width
    return LineProperties(
        freq_Hz = 1.4204058E+09, # not sure where this is from
        A_Hz = 2.85e-15,       # mentioned in Liszt (2001) A&A 371, 698-707
        g_lo = 1, # statistical weight of state with total-spin = 0
        g_hi = 3, # statistical weight of state with total-spin = 1
        energy_lo_erg = 0.0
    )

def air_to_vacuum(air_wavelength):
    """
    Converts wavelengths measured in the air to the corresponding wavelength
    measured in the vacuum.

    Notes
    -----
    I only know this is important from the time I spent working on Korg (the
    stellar spectral synthesis code). As in Korg, we adopt the formulas
    provided by the documentation on the VALD 
    [website](https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion).
    Note: code is adopted from Korg.

    To convert from vacuum to air, the documentation talks about using the
    formula from D. Morton (2000), which originally comes from Birch and Downs
    (1994). The documentation then explains that the opposite conversion (the
    calculation performed by this function) is somewhat more difficult, while
    maintaing high precision. They ultimately decide on using the formula 
    derived by N. Piskunov. That's what we also use.
    explains that deriving the inverse calculation is somewhat non-trivial.
    """

    assert (air_wavelength > unyt.unyt_quantity(200, 'nm')).all()

    s = 1e4/air_wavelength.to('Angstrom').ndview
    s2 = s*s

    # n is the refraction index
    n = (1 + 0.00008336624212083 +
         0.02408926869968 / (130.1065924522 - s2) +
         0.0001599740894897 / (38.92568793293 - s2) )

    return n * air_wavelength

def builtin_halpha_props():

    # This information comes from NIST. They provide a couple line transitions.
    # We picked the transition consistent with what Wikipedia cites (other
    # transitions, are also shown -- and I don't totally understand the
    # differences)

    # https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=H+I&limits_type=0&low_w=650&upp_w=660&unit=1&submit=Retrieve+Data&de=0&I_scale_type=1&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on

    air_wavelength = 6562.79 * unyt.angstrom
    vacuum_wavelength = air_to_vacuum(air_wavelength)

    assert vacuum_wavelength.units == unyt.angstrom
    freq_Hz = (unyt.c_cgs.ndview * 1e8) / (vacuum_wavelength.ndview)

    A_Hz = 4.4101e7

    energy_lo_wavenumber = 82259.158 / unyt.cm
    energy_lo_erg = float(((unyt.h_cgs * unyt.c_cgs)
                           * energy_lo_wavenumber).to('erg').v)

    # energy_lo_erg is roughly 13.6 eV - (13.6/n_lo^2)

    # for hydrogen, the statistical weight is 2n^2
    return LineProperties(freq_Hz = freq_Hz, A_Hz = A_Hz,
                          g_lo = 2 * (2**2), g_hi = 2 * (3**2),
                          energy_lo_erg = energy_lo_erg)


# WE COULD PROBABLY PUT THIS SOMEWHERE ELSE

def _build_simple_H_partition_func_table(*, pressure = None,
                                         electronic_state_max = None):

    _BOHR_RADIUS = unyt.unyt_quantity(5.2917721090380e-11, 'm').to('cm')
    _RYDBERG_E = unyt.unyt_quantity(2.179872361103542e-18, 'Joule').to('erg')

    log_T_vals = np.linspace(0.,8.0,321)
    T_vals = (10.0**log_T_vals) * unyt.K

    if electronic_state_max is None:
        assert pressure is not None
        # following 9.40 from Rybicki & Lightman, the max orbital value
        # considered at each temperature is limited by interpaticle orbit:
        #   n_max^2 *_BOHR_RADIUS * Z**(-1)~ndens
        #   n_max approx sqrt(Z/_BOHR_RADIUS) ndens**(-1/6)
        # in practice, the densities are much too low for this to work well!

        Z = 1.0 # nuclear charge
        # compute number-density at each temperature
        ndens = pressure / (unyt.kboltz_cgs * T_vals)
        nmax = np.floor(
            np.sqrt(Z/_BOHR_RADIUS) * (ndens)**(-1/6.0)
        ).to('dimensionless').v.astype(np.int64)
            
    else:
        assert pressure is None
        assert int(electronic_state_max) == electronic_state_max
        assert electronic_state_max > 0
        nmax = np.ones(T_vals.shape, dtype = 'i8')
        nmax *= int(electronic_state_max)
    nvals = np.arange(1.0, nmax[-1]+1.0)

    g_vals = 2 * (nvals * nvals)
    factor = (1.0 - 1.0 / (nvals*nvals))
    energies = _RYDBERG_E*factor

    beta = 1.0 / (unyt.kboltz_cgs * T_vals)

    interp_points = np.log10(np.array([
        np.sum(g_vals[:nmax[i]] *
               np.exp(-(energies[:nmax[i]] * beta[i]).to('dimensionless').v
               ))
        for i in range(T_vals.size)
    ]))

    return PyLinInterpPartitionFunc(log_T_vals, interp_points)

def crude_H_partition_func(electronic_state_max = 20):
    """
    Returns a crude Hydrogen Partition function.
    """
    assert int(electronic_state_max) == electronic_state_max
    assert electronic_state_max > 0

    return _build_simple_H_partition_func_table(
        electronic_state_max = electronic_state_max)
