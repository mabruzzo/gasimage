import dataclasses
import unyt

@dataclasses.dataclass(frozen = True)
class SpinFlipProperties:
    """Encodes data about the spin-flip transition
    """

    freq_Hz: float # the rest-frame frequency for the transition
    A10_Hz: float  # Einstein coeficient for spontaneous emission
    g0: int # statistical weight of state with total-spin = 0
    g1: int # statistical weight of state with total-spin = 1

    # to support natural-broadening we would need other quantities (like
    # oscillator strength...)

    def __post_init__(self):
        assert (self.g0 == 1) and (self.g1 == 3) # not up for debate
        assert 0 < self.freq_Hz < float('inf')
        assert 0 < self.A10_Hz < float('inf')

    @property
    def A10_quantity(self):
        return unyt.unyt_quantity(self.A10_Hz, "Hz")

    @property
    def freq_quantity(self):
        return unyt.unyt_quantity(self.freq_Hz, "Hz")


def default_spin_flip_props():
    return SpinFlipProperties(
        freq_Hz = 1.4204058E+09, # not sure where this is from
        A10_Hz = 2.85e-15,       # mentioned in Liszt (2001) A&A 371, 698-707
        g0 = 1,
        g1 = 3
    )

# In the future, we could conceivably support other types of Radiative Transfer
# For now, we tailor the config object to HI observations since we can make
# some simplifying assumptions

#class RTPhysicsConfigHI:
#    """
#    Object used to track configuration of the physics of HI (21 cm)
#    radiative-transfer (and any assumptions).
#
#    Note
#    ----
#    Users should generally avoid relying upon the internal state of this 
#    class (it is subject to change).
#
#    The only interactions that users are expected to have with this object are
#    to create it with ``rt_physics_config_HI`` and then pass the resulting
#    object into optically_thin_ppv.
#    """
#    def __init__(self):
#        # when this is False, then we need to 
#        self.level_pops_from_stat_weights = level_pops_from_stat_weights
#
#        # actual physics quantites
#        self.rest_freq_Hz = 1.4204058E+09
#        self.g0_g1_pair = (1, 3) # the statistical_weights
#        self.
#
#    def level_populations_from_statistical_weights(self):
#        return self._level_populations_from_statistical_weights
