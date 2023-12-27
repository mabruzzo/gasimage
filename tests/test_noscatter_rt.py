import numpy as np
import unyt

from gasimage.generate_ray_spectrum import blackbody_intensity_cgs
import pytest

def solve_rt(source_function, absorption_coef, dz, bkg_intensity):
    """
    Solve the full equation of radiative transfer.

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
    spectra: ndarray, shape(nfreq,)
        Holds the intensity as a function of frequency.

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

      I_\nu (\tau_\nu=0) =  I_\nu(\tau_\nu)\, e^{-\tau_\nu} - f,

    where :math:`f`, the integral term, is given by:

    .. math::

      f = \int_0^{\tau_\nu}  S_\nu(\tau_\nu^\prime)\, e^{-\tau_\nu^\prime}\, d\tau_\nu^\prime.

    """

    # part 1: solve for tau
    #
    # we defined tau so that it is increasing as we move away from the observer
    #
    # NOTE: higher indices of dz are also further from observer
    nfreq = source_function.shape[0]

    starting_tau = 0.0
    tau = np.empty(shape = (nfreq, dz.size + 1), dtype = 'f8')
    for freq_ind in range(nfreq):
        tau[freq_ind, 0] = starting_tau
        tau[freq_ind, 1:] = starting_tau + np.cumsum(
            (absorption_coef[freq_ind] * dz).to('dimensionless').ndview
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

    exp_neg_tau = np.exp(-tau)
    integral_term = np.zeros(shape=(nfreq,), dtype = 'f8') * source_function.uq
    for freq_i in range(nfreq):
        running_sum = 0.0 * source_function.uq
        for pos_i in range(dz.size):
            running_sum += (
                source_function[freq_i,pos_i] *
                (exp_neg_tau[freq_i, pos_i] - exp_neg_tau[freq_i, pos_i+1])
            )
        integral_term[freq_i] = running_sum

    return tau, bkg_intensity*exp_neg_tau[:, -1] + integral_term

_INTENSITY_UNIT_CGS = 'erg/(cm**2 * Hz * s * steradian)'

_intensity_quan_cgs = lambda x: unyt.unyt_quantity(float(x),
                                                   _INTENSITY_UNIT_CGS)

# I have no idea what a reasonable number is, for the source function! We could
# try evaluating the blackbody equation to get something sensible...

def _test_function(nfreq, ngas = 30, seed = 12345,
                  nominal_source_func = _intensity_quan_cgs(1.0),
                  nominal_absorption_coef = unyt.unyt_quantity(1e-8, 'cm**-1'),
                  nominal_dz = unyt.unyt_quantity(3e18,'cm') # roughly 1 pc!
                  ):
    rng = np.random.default_rng(seed = seed)

    source_func = (
        nominal_source_func* rng.uniform(low=0.7, high=1.2, size = nfreq*ngas)
    ).reshape(nfreq,ngas)

    absorption_coef = (
        nominal_absorption_coef * rng.uniform(low=np.nextafter(0.0,1.0),
                                              high=1.0, size = nfreq*ngas)
    ).reshape(nfreq,ngas)

    dz = nominal_dz * rng.uniform(low =np.nextafter(0.0, 1.0), high = 1.0,
                                  size = ngas)

    bkg_intensity = source_func[:,0] * 0.0

    taus, intensity = solve_rt(source_func, absorption_coef, dz,
                               bkg_intensity)

    return taus, intensity


"""
def plot():
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,1)

    nfreq = 2
    taus, intensity = test_function(
        2, ngas = 30, seed = 12345,
    )

    for i in range(nfreq):
        ax.plot(taus[i])
    plt.show()

plot()
"""
#from gasimage.rt_config import default_halpha_props
#print('%e' % default_halpha_props().freq_Hz)

def build_analytic_evaluator_():
    import sympy # we need sympy for unyt anyway, not too much of an issue to
                 # depend on this
    tau_final, Ibkg = sympy.symbols('tau_final, Ibkg', real = True)

    x, a, b, c = sympy.symbols('x, a, b, c', real = True)
    source_func_expr = a*x*x + b*x + c

    rslt = (
        Ibkg * sympy.exp(-tau_final) +
        sympy.integrate(source_func_expr * sympy.exp(-(tau_final-x)),
                        (x,0,tau_final)) # integrate x from 0 to tau_final
    )

    def source_fn(tau_val, *, a_coef = _intensity_quan_cgs(0),
                  b_coef = _intensity_quan_cgs(0),
                  c_coef = _intensity_quan_cgs(0)):
        out = source_func_expr.subs(
            {x : tau_val,
             a : float(a_coef.to(_INTENSITY_UNIT_CGS).v),
             b : float(b_coef.to(_INTENSITY_UNIT_CGS).v),
             c : float(c_coef.to(_INTENSITY_UNIT_CGS).v)}
        )
        return unyt.unyt_quantity(float(out), _INTENSITY_UNIT_CGS)

    def intensity_fn(optical_depth, *, a_coef = _intensity_quan_cgs(0),
                     b_coef = _intensity_quan_cgs(0),
                     c_coef = _intensity_quan_cgs(0),
                     I_bkg_val = _intensity_quan_cgs(0)):
        assert optical_depth >= 0
        out = rslt.subs({a : float(a_coef.to(_INTENSITY_UNIT_CGS).v),
                         b : float(b_coef.to(_INTENSITY_UNIT_CGS).v),
                         c : float(c_coef.to(_INTENSITY_UNIT_CGS).v),
                         Ibkg : float(I_bkg_val.to(_INTENSITY_UNIT_CGS).v),
                         tau_final : optical_depth})
        return unyt.unyt_quantity(float(out), _INTENSITY_UNIT_CGS)
    return source_fn, intensity_fn

_ANALYTIC_SRCFN, _ANALYTIC_SOLN = build_analytic_evaluator_()
_ANALYTIC_SOLN.__doc__ = """
    Computes the analytic solution for radiative transfer along a ray where:
      - there is no scattering
      - the source function has the analytic form ``a * tau**2 + b * tau + c``

    We have explicitly used the version of the radiative transfer equation
    described by Rybicki and Lightman, in order to act as a more rigorous test
    on our real implementation. In this implementation, optical depth increases
    as light propogates along the ray. It has a minimum at the start of the ray
    and a maximum at the observer.

    Parameters
    ----------
    optical_depth: float
        Total optical depth along the ray
    a_coef,b_coef,c_coef: unyt.unyt_quantity
        Source function coefficients
    I_bkg_val: unyt.unyt_quantity
        The intensity at the ray's origin
    """


class AnalyticSetup:
    # This is setup in terms of the Rybicki & Lightman definition of optical
    # depth (it increases as light propagates... It has a max at the observer)
    
    def __init__(self, tau_total, *, a_coef = _intensity_quan_cgs(0),
                 b_coef = _intensity_quan_cgs(0),
                 c_coef = _intensity_quan_cgs(0)):
        self.source_coefs = {"a_coef" : a_coef,
                             "b_coef" : b_coef,
                             "c_coef" : c_coef}
        self.total_optical_depth = tau_total

    def evaluate_intensity_analytic(self, I_bkg_val = _intensity_quan_cgs(0)):
        return _ANALYTIC_SOLN(optical_depth = self.total_optical_depth,
                              I_bkg_val = I_bkg_val, **self.source_coefs)

    def discretize_analytic(self, dz_vals, *, left_close_to_obs = True):

        # we make a bunch of assumptions

        linear_absorption_coef = np.ones(dz_vals.shape, 'f8')
        # for now, lets assume that the absorption coefficient is constant
        linear_absorption_coef *= self.total_optical_depth / dz_vals.sum()

        # compute the amount optical depth changes in each segment
        delta_tau_mag =  np.abs(linear_absorption_coef * dz_vals)\
                           .to('dimensionless').ndview

        # compute the optical depth at the left edge of each ray segment
        tau_left_edges = np.empty((dz_vals.size + 1,), 'f8')
        tau_left_edges[0] = 0.0
        tau_left_edges[1:] = np.cumsum(delta_tau_mag)

        # won't be exact, but should be close
        #assert tau_right_edges == self.total_optical_depth
        tau_centers = tau_left_edges[:-1] + 0.5*delta_tau_mag
        #print(tau_centers)
        source_func = unyt.unyt_array(np.empty(dz_vals.shape, 'f8'),
                                      _INTENSITY_UNIT_CGS)
        for i in range(source_func.size):
            if left_close_to_obs:
                source_func[i] = _ANALYTIC_SRCFN(
                    self.total_optical_depth - tau_centers[i],
                    **self.source_coefs
                )
            else:
                source_func[i] = _ANALYTIC_SRCFN(
                    tau_centers[i],
                    **self.source_coefs
                )
        return linear_absorption_coef, source_func

def compare(tau_total = 1.0, num = 20,
            a_coef = _intensity_quan_cgs(5.0),
            b_coef = _intensity_quan_cgs(4.0),
            c_coef = _intensity_quan_cgs(3.0),):
    
    tmp = AnalyticSetup(tau_total = tau_total,
                        a_coef = a_coef, b_coef = b_coef, c_coef = c_coef)

    total_width = unyt.unyt_quantity(3e19, 'cm')
    
    dz_vals = np.ones(num) * (total_width / num)
    absorption_coef, sourcefn = tmp.discretize_analytic(
        dz_vals, left_close_to_obs = True
    )

    bkg_intensity = unyt.unyt_array([0.0], _INTENSITY_UNIT_CGS)

    taus, intensity = solve_rt(
        sourcefn[None,:], absorption_coef[None,:],
        dz_vals, bkg_intensity = bkg_intensity)

    desired = tmp.evaluate_intensity_analytic(bkg_intensity)
    
    return (np.abs(intensity[0] - desired)/desired)


def test_convergence():
    nums = [8,16,32,64,128]

    # in this case, the source function is constant. We don't actually expect
    # convergence here. Everything should just be the right answer
    rel_errs = np.array([
        compare(tau_total = 20.0, num = num,
                a_coef = _intensity_quan_cgs(0.0),
                b_coef = _intensity_quan_cgs(0.0),
                c_coef = _intensity_quan_cgs(5.0))
        for num in nums
    ])
    assert (rel_errs <= 2e-16).all()

    # here source function is a linear function of tau
    rel_errs = np.array([
        compare(tau_total = 2.0, num = num,
                a_coef = _intensity_quan_cgs(0.0),
                b_coef = _intensity_quan_cgs(5.0),
                c_coef = _intensity_quan_cgs(5.0))
        for num in nums
    ])

    factors = np.array([rel_errs[i]/rel_errs[i+1] for i in range(len(nums)-1)])
    assert (factors >= 2.0).all()
    assert rel_errs[0] < 3e-3

    # here source function is a quadratic function of tau
    rel_errs = np.array([
        compare(tau_total = 2.0, num = num,
                a_coef = _intensity_quan_cgs(5.0),
                b_coef = _intensity_quan_cgs(5.0),
                c_coef = _intensity_quan_cgs(5.0))
        for num in nums
    ])

    factors = np.array([rel_errs[i]/rel_errs[i+1] for i in range(len(nums)-1)])
    assert (factors >= 2.0).all()
    assert rel_errs[0] < 6e-3


def test_blackbody():
    astropy_modeling = pytest.importorskip("astropy.modeling")

    temperature = 5778*unyt.K

    wavelengths = np.geomspace(100.0, 3000.0, num = 1001) * unyt.nm
    frequencies = (unyt.c_cgs/wavelengths).to('Hz')
    intensities = blackbody_intensity_cgs(
        frequencies.to('Hz').ndview,
        (1.0 / (unyt.kboltz * temperature)).in_cgs().v
    )

    bb = astropy_modeling.models.BlackBody(temperature.to_astropy())
    ref_intensity = unyt.unyt_array.from_astropy(bb(wavelengths.to_astropy()))
    ref_intensity = ref_intensity.to('erg/(Hz*cm**2 * s * steradian)').v

    np.testing.assert_allclose(actual = intensities, desired = ref_intensity,
                               rtol = 1.6e-6, atol = 0.0)
