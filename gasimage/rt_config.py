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
