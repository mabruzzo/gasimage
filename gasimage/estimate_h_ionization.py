import numpy as np
import yt

"""
We can estimate this given the mean molecular weight $\mu$. The quation for $\mu$ is:

$\mu = \frac{m_{\rm H} n_{\rm H} + m_{\rm He} n_{\rm He} + 
             \bar{m}_{\rm metal} n_{\rm metal} + m_{\rm e} n_{\rm e}}{
             m_{\rm H}(n_{\rm H} + n_{\rm He} + n_{\rm metal} + n_{\rm e})}$
             
We'll apply 2 simplifications:
1. We'll ignore mass contributions of electrons
2. We'll assume that only $n_{\rm H}$ is ionized. This is probably reasonable since Hydrogen makes up ${\sim}90\%$ of all atoms/ions by number. Let's define $f$ as the ionization fraction of Hydrogen. This means that $n_{\rm e} = f n_{\rm H}$.

After applying these simplifications, we have:

$\mu \sim \frac{m_{\rm H} n_{\rm H} + m_{\rm He} n_{\rm He} + \bar{m}_{\rm metal} n_{\rm metal}}{
                m_{\rm H}((1+f)n_{\rm H} + n_{\rm He} + n_{\rm metal})}$

Let's define $A_i$ as the atomic weight of a species in amu. This reduces the equation to:

$\mu \sim \frac{n_{\rm H} + 4 n_{\rm He} + \bar{A}_{\rm metal} n_{\rm metal}}{
               (1+f)n_{\rm H} + n_{\rm He} + n_{\rm metal}}$

We can re-arrange the equation to solve for $f$:

$(1+f)n_{\rm H} \sim 
    \frac{n_{\rm H} + 4 n_{\rm He} + \bar{A}_{\rm metal} n_{\rm metal}}{\mu}
    - n_{\rm He} - n_{\rm metal}$
"""
def estimate_hydrogen_ionizations(rho, eint, eos,
                                  metal_mass_frac,
                                  mu_metal = 16,
                                  internal_constant = False,
                                  mmw = None):
    # TODO: in the future, potentially decouple this from gascloudtool

    # this was copied from the notebook where I was looking at different
    # potential sets of initial conditions
    # By Convention: X = H mass fraction
    #                Y = He mass fraction
    #                Z = metal mass fraction
    # eos.calculate_mu may make the usage of internal constants inconsistent


    X = eos._my_chem.HydrogenFractionByMass
    assert metal_mass_frac is not None
    Z = metal_mass_frac
    Y = 1. - X - Z
    assert Y > 0

    if internal_constant:
        raise NotImplemented()
    else:
        mH=yt.units.mh

    n_H = (X*rho/mH).to('cm**-3').v
    n_He = (Y*rho/(4*mH)).to('cm**-3').v
    n_metal = (Z*rho/(mu_metal*mH)).to('cm**-3').v

    if mmw is None:
        assert rho.flags['C_CONTIGUOUS']
        assert eint.flags['C_CONTIGUOUS']
        mmw = eos.calculate_mu(rho.ravel(order = 'C'),eint.ravel(order = 'C'),
                               metal_mass_frac)
        assert mmw.flags['C_CONTIGUOUS']
        mmw.shape = rho.shape
    n_total = (rho/(mmw*mH)).to('cm**-3').v
    n_e = n_total - n_H - n_He - n_metal

    #w = np.logical_and(~np.isnan(n_e),
    #                   ~np.isnan(n_H+n_He))
    #assert (n_e <= (n_H+2*n_He))[w].all()
    w_more_e_than_prim = n_e > (n_H+2*n_He)

    # we probably need the Saha Equation or something to better estimate
    # the fractional ionization
    #
    # instead we will return:
    #    - the H ionization assuming minimal HeII and HeIII
    #    - the H ionization assuming nHII/nH = nHeII/nHe and (minimal HeIII)

    mask = np.isnan(n_e)

    minimum_fraction = np.empty(dtype = rho.dtype,
                                shape = rho.shape)
    maximum_fraction = np.empty(dtype = rho.dtype,
                                shape = rho.shape)

    minimum_fraction[mask] = np.nan
    maximum_fraction[mask] = np.nan

    if mask.all():
        unmasked = slice(None)
    else:
        unmasked = ~mask

    # compute minimum_fraction
    #  - this is when H and He have the same single ionization fraction
    #    (and He is not doubly ionized)
    minimum_fraction[unmasked] = n_e[unmasked]/(n_H[unmasked]+n_He[unmasked])

    # compute maximum fraction (Hydrogen is as ionized as possible)
    maximum_fraction[unmasked] = n_e[unmasked]/n_H[unmasked]

    minimum_fraction[minimum_fraction > 1] = 1
    maximum_fraction[maximum_fraction > 1] = 1


    minimum_fraction[w_more_e_than_prim] = 1
    maximum_fraction[w_more_e_than_prim] = 1

    return minimum_fraction,maximum_fraction

def create_nHI_field(ds, eos, metal_mass_frac,
                     use_min_ion_frac = True,
                     field_name = ('gas', 'H_p0_number_density'),
                     mu_metal = 16,
                     add_field_kwargs = {}):
    # this currently ignores the local metallicity
    # it would be far more self-consistent to include that!
    
    target_units = None
    if add_field_kwargs.get('units','auto') != 'auto':
        target_units = add_field_kwargs['units']

    def _nH_I(field,data):
        eint = data['internal_energy']
        assert eint.units.same_dimensions_as(
            ds.unit_system['specific_energy']
        )

        try:
            mmw = data['mean_molecular_weight']
        except yt.utilities.exceptions.YTFieldNotFound:
            mmw = None

        min_ifrac, max_ifrac = estimate_hydrogen_ionizations(
            rho = data['density'], eint = eint,
            eos = eos, metal_mass_frac = metal_mass_frac,
            mu_metal = mu_metal, mmw = mmw,
            internal_constant = False
        )

        if use_min_ion_frac:
            ifrac = min_ifrac
        else:
            ifrac = max_ifrac
        nHI = ((data['density']*eos._my_chem.HydrogenFractionByMass * 
                (1.0-ifrac))/yt.units.mh)
        if target_units is None:
            return nHI
        else:
            return nHI.to(target_units)

    assert 'function' not in add_field_kwargs
    _kwargs = add_field_kwargs.copy()

    _default_keyword_values = (
        ('sampling_type', 'local'),
        ('units', 'auto'),
        ('dimensions', yt.units.dimensions.number_density),
        ('take_log', True)
    )
    for k,v in _default_keyword_values:
        if k not in _kwargs.keys():
            _kwargs[k] = v

    ds.add_field(field_name, function = _nH_I, **_kwargs)
