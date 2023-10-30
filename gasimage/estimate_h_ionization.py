import warnings

import numpy as np
import unyt
import yt

"""
This module provides routines for estimating the ionization fraction, $f$ for Hydrogen.
In these calculations, let's assume we know the mean-molecular-weight for Hydrogen and Helium (and free electrons originating from them). We call this quantity $\mu_{\rm primoridial}$. We can estimate $f_{H}$ given the mean molecular weight $\mu_{\rm primoridial}$, which is defined as:

$$
  \mu_{\rm primoridial} = 
      \frac{m_{\rm H} n_{\rm H} + m_{\rm He} n_{\rm He} + m_{\rm e} n_{\rm e}}
           {m_{\rm H}(n_{\rm H} + n_{\rm He} + n_{\rm e})}
$$

We'll ignore mass contributions of electrons and assume that $m_{\rm He}$ is exactly $4 m_{\rm H}$.

$$
  \mu_{\rm primoridial} = 
      \frac{n_{\rm H} + 4 n_{\rm He}}
           {n_{\rm H} + n_{\rm He} + n_{\rm e}}
$$

In practice, it's useful to invert this equation to solve for $n_{\rm e}$:

$$
  n_{\rm e} = 
      \frac{n_{\rm H} + 4 n_{\rm He}}
           {\mu_{\rm primoridial}} - n_{\rm H} - n_{\rm He}
$$


We've also already made the implicit assumption that the only free electrons in the above equation come from ionized hydrogen and ionized helium. Let's define $f$ as the fraction of ionized Hydrogen and $g n_{\rm He}$ as the number density of electrons from Helium, where $0 \leq g \leq 2$. 

Basically, we get $f$ by solving $n_{\rm e} = f n_{\rm H} + g n_{\rm He}$. We can consider a few limiting cases:


Max $f$ case:
-------------

In this regime we assume that as many of the primordial free-electrons are sourced from hydrogen as possible (this is probably a reasonable assumption since the universe is over $90%$ Hydrogen by number density.

We can compute $f$ as:

$$
     \begin{cases} 
        n_{\rm e} / n_{\rm H} & n_{\rm e} < n_{\rm H} \\
        1                     & {\rm otherwise}
     \end{cases}
$$

Equal single primordial ionization:
-----------------------------------

In this case, we assume that helium becomes singly ionized at the same rate as hydrogen , and there is no double-ionization of helium until after hydrogen is completely ionized.

In this scenario, $g = f$ when $n_{\rm e} \leq n_{\rm H} + n_{\rm He}$. Thus, $f$ is given by:

$$
  \begin{cases} 
        n_{\rm e} / (n_{\rm He} + n_{\rm H}) & n_{\rm e} < (n_{\rm H} + n_{\rm He} \\
        1                                    & {\rm otherwise}
  \end{cases}
$$

I suspect the real-world case is somewhere between this regime and the max-$f$ case


Min $f$ case:
-------------

This is not a very physical case, since we are assuming that Helium will be doubly ionized before any Hydrogen is ionized. Since the max number density contributed by Helium is $\min(2 n_{\rm He}, n_{\rm e})$ the value of $f$ is given by:

$$
  \begin{cases} 
        (n_{\rm e} - 2 n_{\rm He}) / n_{\rm H} & n_{\rm e} > 2 n_{\rm He} \\
        0                                      & {\rm otherwise}
  \end{cases}
$$
"""


def _grackle_dflts(parname):
    if parname == 'HydrogenFractionByMass':
        # ASSUMPTION: grackle was used with `primordial_chemistry == 0`
        #             (if we used grackle in any other configuration. We'd
        #             already explicitly know the ionization fraction)
        #
        # NOTE: when you first grackle's chemistry_data initially stores
        #       0.76 in the HydrogenFractionByMass field. But, if grackle
        #       is used with primordial_chemistry = 0, the value is later
        #       updated to the following:
        return 1. / (1. + 0.1 * 3.971)
    elif parname == 'SolarMetalFractionByMass':
        # unless you used a really old grackle version (around version 2.1),
        # the following is the SolarMetalFractionByMass used in grackle
        return 0.01295
    elif parname == 'MU_METAL':
        return 16.0 # NOTE: this isn't actually a parameter in grackle, but 
                    #       this is the value assumed throughout the codebase
    else:
        raise ValueError(parname)

def _check_ionization_case_kwarg(case, *, force_err = False):
    _case_choices = ('equal-single-primordial-ionization',
                     'max_H_ionization', 'min_H_ionization')
    if force_err or (case not in _case_choices):
        raise ValueError("the ionization_case keyword argument must be one of "
                         f"{_case_choices!r}, not {case}")

def _estimate_grackle_rho_HI_helper(rho_primordial, mu_primordial,
                                    ionization_case = 'max_H_ionization'):
    """
    Helper function for estimating the mass density of neutral Hydrogen
    assuming that grackle was to compute the mean-molecular-weight (with the 
    primordial_chemistry flag set to zero)
    
    This function is called after the primordial mass density (mass density of
    Hydrogen, Helium, and electrons) and primordial mean molecular weight have
    been computed. NOTE: we assume that the electron-mass is negligible.

    In other words, the mass-density from metals and the contribution, to the
    mmw from metals (including free electrons contributed by the  metals) have
    already been removed. NOTE: The removal of these contributions directly
    mirrors the internal workings of Grackle

    Parameters
    ----------
    rho_primordial : `unyt.unyt_array`
        Specifies the total mass-density of the electrons, Hydrogen, & Helium.
    mu_primordial : `np.ndarray`
        Specifies the mean-molecular weight of the electrons, Hydrogen, & 
        Helium, as computed by Grackle.
    ionization_case : str
        This specifies assumptions about ionization. Should be one of:
    
            - 'max_H_ionization'  specifies that as many electrons as possible 
              are sourced from Hydrogen.
            - 'equal-single-primordial-ionization' specifies that when the
              number density of electrons is less than the combined number
              density of Hydrogen & Helium, the fraction of Helium that is
              singly ionized Helium is equal to the Hydrogen ionization 
              fraction (and no Helium is doubly ionizied). In the other case, 
              all Hydrogen is ionized.
            - 'min_H_ionization' specifies that as many electrons are
              contributed by Helium as possible (i.e. all Helium will be 
              doubly-ionized before any Hydrogen is ionized).
        
    Notes
    -----
    Historically, there is an inconsistency in the values of hydrogen-mass and
    kboltz units internally by Grackle, and the values of those constants
    employed by ``pygrackle`` and ``unyt``. I'm pretty sure that it doesn't
    really affect the calculations within the function itself - I think all
    occurences of Hydrogen-mass within this function effectively cancel out.
    With that said, the discrepancy will almost certainly affect  the value 
    passed as ``mu_primordial``.
    """
    mH = unyt.mh_cgs
    mHe = 4 * mH # assumption is consistent with grackle

    h_mass_frac = _grackle_dflts('HydrogenFractionByMass')
    # we effectively assume that electron mass fraction is zero (as is
    # consistent with grackle)
    
    rho_H = h_mass_frac * rho_primordial
    ndens_H = rho_H / mH
    ndens_He = (1.0 - h_mass_frac) * rho_primordial / mHe

    ndens_electron = (rho_primordial / (mH * mu_primordial)) - ndens_He - ndens_H
    # this is probably not strictly necessary, but I could definitely imagine
    # it coming up
    ndens_electron.ndview[ndens_electron.ndview < 0] = 0

    f_H_ion = np.empty(shape = ndens_electron.shape, dtype = 'f8')
    if ionization_case == 'max_H_ionization':
        # in this case, Hydrogen provides as much of the electrons as
        # possible
        w = ndens_electron <= ndens_H
        f_H_ion[w] = (ndens_electron[w] / ndens_H[w]).to('dimensionless').ndview
        f_H_ion[~w] = 1.0 # in this case, there are more electrons than H atoms
    elif ionization_case == 'equal-single-primordial-ionization':
        # in this case, when there are fewer electrons than the combination of
        # Hydrogen + Helium, assume that the fraction of singly-ionized Helium
        # is equal to the fraction of ionized Hydrogen (also assume there is no
        # doubly-ionized Helium)
        ndens_H_and_He = ndens_H + ndens_He
        w = ndens_electron <= ndens_H_and_He
        f_H_ion[w] = (ndens_electron[w] /
                      ndens_H_and_He[w]).to('dimensionless').ndview
        f_H_ion[~w] = 1.0
    elif ionization_case == 'min_H_ionization':
        # in this case, Helium provides as much of the electrons as possible. A
        # single Helium atom can contribute up to 2 electrons

        # the masking here follows a slightly different pattern in this branch
        w = ndens_electron > (2 * ndens_He)
        f_H_ion[w] = ( (ndens_electron[w] - (2 * ndens_He[w])) / 
                       ndens_H[w]).to('dimensionless').ndview
        f_H_ion[~w] = 0.0 # this is NOT a typo!
    else:
        _check_ionization_case_kwarg(ionization_case, force_err = True)

    # just to be safe, lets ensure f_H_ion is forced to have values between 0
    # and 1
    f_H_ion = np.clip(f_H_ion, a_min = 0.0, a_max = 1.0) # NOTE: this preserves
                                                         #       NaNs

    # f_H_ion * rho_H gives mass density of H II, we want mass density of H I
    return (1.0 - f_H_ion) * rho_H

def estimate_grackle_ndens_HI(rho, mmw_vals, rho_metal, *,
                              ionization_case = 'max_H_ionization'):
    """
    Estimate the number density of neutral hydrogen given the mean molecular
    weight assuming that grackle was used to compute the mean-molecular-weight
    (with the primordial chemistry flag set to zero)

    Parameters
    ----------
    rho : `unyt.unyt_array`
        Specifies the total mass-density of the gas
    mmw_vals : `unyt.unyt_array`
        Specifies the mean-molecular weight, as computed by Grackle.
    rho_metal : `unyt.unyt_array`, Optional
        Specifies the metal-mass-fraction.
    ionization_case : str
        This specifies assumptions about ionization. Should be one of:
    
            - 'max_H_ionization'  specifies that as many electrons as possible 
              are sourced from Hydrogen.
            - 'equal-single-primordial-ionization' specifies that when the
              number density of electrons is less than the combined number
              density of Hydrogen & Helium, the fraction of Helium that is
              singly ionized Helium is equal to the Hydrogen ionization 
              fraction (and no Helium is doubly ionizied). In the other case, 
              all Hydrogen is ionized.
            - 'min_H_ionization' specifies that as many electrons are
              contributed by Helium as possible (i.e. all Helium will be 
              doubly-ionized before any Hydrogen is ionized).

    Notes
    -----
    Historically, there is an inconsistency in the values of hydrogen-mass
    and kboltz units internally by Grackle, and the values of those 
    constants employed by ``pygrackle`` and ``unyt``
    """
    _check_ionization_case_kwarg(ionization_case)

    # If you are trying to lookup the algorithm used by Grackle to compute
    # mmw, be aware that I fixed a bug in the routines for computing mmw 
    # (when primordial_chemistry == 0.0) a while back.

    # When Grackle is run wih primordial_chemistry == 0.0, they first use
    # Cloudy-tables to compute the mean-molecular-weight (mu) of the 
    # primordial species (namely Hydrogen, Helium & electrons) and then
    # they effectively use the following equation to get the effective mu
    # of the entire gas.
    #
    # mu_eff = rho / (mH * (ndens_primordial + ndens_metal))
    #                           OR
    # mu_eff = rho / ( (rho_primordial / mu_primordial) + 
    #                  (rho_metal / mu_metal)             )
    #
    # We can rearange this equation to get:
    #
    # rho / mu_eff = (rho_primordial / mu_primordial) + (rho_metal / mu_metal)
    #                           OR
    # rho_primordial / mu_primordial = (rho / mu_eff) - (rho_metal / mu_metal)
    #
    # Finally, we get the equation for mu_primordial:
    #
    # mu_primordial = rho_primordial / ( (rho / mu_eff) - 
    #                                    (rho_metal / mu_metal) )
    mu_eff = mmw_vals
    rho_primordial = rho - rho_metal
    mu_metal = _grackle_dflts('MU_METAL')
    mu_primordial = rho_primordial / ( (rho / mu_eff) - 
                                       (rho_metal / mu_metal) )

    # we want to compute mu_primordial:
    rho_HI = _estimate_grackle_rho_HI_helper(rho_primordial = rho_primordial,
                                             mu_primordial = mu_primordial,
                                             ionization_case = ionization_case)
    return rho_HI / unyt.mh_cgs

def _coerce_missing_units(ds, field_data, field_name):
    # this really shouldn't be necessary (especially after
    # I fix the yt-frontend!)
    if not field_data.units.is_dimensionless:
        return field_data
    elif field_name == ('enzoe', 'temperature'):
        return unyt.unyt_array(field_data.ndview, 'K')
    elif field_name == ('enzoe', 'metal_density'):
        return ds.arr(
            field_data.ndview,
            ds.field_info['enzoe','density'].units)
    else:
        raise RuntimeError(f'unrecognized field: {field_name}')

def _check_metallicity_AND_metal_dens_args(globally_fixed_metallicity = None, 
                                           metal_density_field = None):
    if (globally_fixed_metallicity is None) and (metal_density_field is None):
        raise ValueError("You must specify either globally_fixed_metallicity "
                         "or metal_density_field")
    elif (globally_fixed_metallicity is None) == (metal_density_field is None):
        raise ValueError("You can't specify both globally_fixed_metallicity "
                         "and metal_density_field")
    elif ( (globally_fixed_metallicity is not None) and
           (not (0.0 <= globally_fixed_metallicity)) ):
        raise ValueError("globally_fixed_metallicity must be non-negative")


def create_nHI_field_gracklemmw(ds,
                                field_name = ('gas', 'H_p0_number_density'), *,
                                globally_fixed_metallicity = None, 
                                metal_density_field = None,
                                ionization_case = 'max_H_ionization',
                                add_field_kwargs = {}):
    """
    Adds a yt-field to the specified dataset `ds` that specifies the number
    density of neutral Hydrogen
    
    Parameters
    ----------
    ds
        A yt dataset
    field_name: tuple of str, Optional
        Specifies the name of the output field
    globally_fixed_metallicity : float, Optional
        When specified, the metal mass density of each cell is set to
        ``rho * globally_fixed_metallicity * SolarMetalFractionByMass``,
        where ``rho`` is the total mass density and 
        ``SolarMetalFractionByMass`` is the value adopted from Grackle.
        When specified, ``metal_density_field`` must be ``None``.
    metal_density_field : tuple of str, Optional
        Specifies the name of the field to use to specify the metal mass density.
        When specified, ``globally_fixed_metallicity`` must be ``None``.
    ionization_case : str
        This specifies assumptions about ionization. Should be one of:
    
            - 'max_H_ionization'  specifies that as many electrons as possible 
              are sourced from Hydrogen.
            - 'equal-single-primordial-ionization' specifies that when the
              number density of electrons is less than the combined number
              density of Hydrogen & Helium, the fraction of Helium that is
              singly ionized Helium is equal to the Hydrogen ionization 
              fraction (and no Helium is doubly ionizied). In the other case, 
              all Hydrogen is ionized.
            - 'min_H_ionization' specifies that as many electrons are
              contributed by Helium as possible (i.e. all Helium will be 
              doubly-ionized before any Hydrogen is ionized).
    add_field_kwargs : dict, Optional
        kwargs to forward ``ds.add_field``.
    """
    
    _check_ionization_case_kwarg(ionization_case)
    _check_metallicity_AND_metal_dens_args(
        globally_fixed_metallicity = globally_fixed_metallicity, 
        metal_density_field = metal_density_field)

    # define the function actually used to derive the field
    target_units = 'cm**-3'
    if add_field_kwargs.get('units','') == 'auto':
        target_units = None
    elif 'units' in add_field_kwargs:
        target_units = add_field_kwargs['units']

    def _nH_I(field,data):
        # load the mass density field
        rho = data['gas','density']

        # prepare rho_metal
        if globally_fixed_metallicity is not None:
            zsolar = _grackle_dflts('SolarMetalFractionByMass')
            rho_metal = rho * globally_fixed_metallicity * zsolar
        else:
            rho_metal = data[metal_density_field]
            # sometimes we forget to add the density units
            if ('enzoe', 'metal_density') == metal_density_field:
                rho_metal = _coerce_missing_units(data.ds, rho_metal, 
                                                  metal_density_field)

        # prepare the mean-molecular-weight

        # TODO: generalize to ('gas','temperature')
        T_vals = data['enzoe', 'temperature']
        T_vals = _coerce_missing_units(
            ds, T_vals, ('enzoe', 'temperature'))

        eint = data['enzoe', 'internal_energy']
        gm1 = ds.gamma - 1.0
        mmw_vals = (T_vals * unyt.kboltz_cgs / 
                    (gm1 * eint * unyt.mh_cgs) ).to('dimensionless')

        # finally estimate the number density of neutral Hydrogen
        out = estimate_grackle_ndens_HI(rho = rho, mmw_vals = mmw_vals,
                                        rho_metal = rho_metal,
                                        ionization_case = ionization_case)
        if target_units is None:
            return out
        else:
            return out.to(target_units)
        
    assert 'function' not in add_field_kwargs
    _kwargs = add_field_kwargs.copy()

    _default_keyword_values = (
        ('sampling_type', 'local'),
        ('units', target_units),
        ('dimensions', unyt.dimensions.number_density),
        ('take_log', True)
    )
    for k,v in _default_keyword_values:
        if k not in _kwargs.keys():
            _kwargs[k] = v

    ds.add_field(field_name, function = _nH_I, **_kwargs)

class EnzoEUsefulFieldAdder:
    """
    A convenience class used to help define necessary fields when using an
    Enzo-E simulation.

    The following fields are all related to defining 

    Parameters
    ----------
    ndens_HI_field_name: tuple of str, Optional
        Specifies the name of the output field
    globally_fixed_metallicity : float, Optional
        When specified, the metal mass density of each cell is set to
        ``rho * globally_fixed_metallicity * SolarMetalFractionByMass``,
        where ``rho`` is the total mass density and 
        ``SolarMetalFractionByMass`` is the value adopted from Grackle.
        When specified, ``metal_density_field`` must be ``None``.
    metal_density_field : tuple of str, Optional
        Specifies the name of the field to use to specify the metal mass density.
        When specified, ``globally_fixed_metallicity`` must be ``None``.
    ionization_case : str
        This specifies assumptions about ionization. Should be one of:
    
            - 'max_H_ionization'  specifies that as many electrons as possible 
              are sourced from Hydrogen.
            - 'equal-single-primordial-ionization' specifies that when the
              number density of electrons is less than the combined number
              density of Hydrogen & Helium, the fraction of Helium that is
              singly ionized Helium is equal to the Hydrogen ionization 
              fraction (and no Helium is doubly ionizied). In the other case, 
              all Hydrogen is ionized.
            - 'min_H_ionization' specifies that as many electrons are
              contributed by Helium as possible (i.e. all Helium will be 
              doubly-ionized before any Hydrogen is ionized).
    ndens_HI_add_field_kwargs : dict, Optional
        kwargs to forward ``ds.add_field``.
    """

    def __init__(self, *,
                 ndens_HI_field_name = ('gas', 'H_p0_number_density'),
                 globally_fixed_metallicity = None, 
                 metal_density_field = None,
                 ionization_case = 'max_H_ionization',
                 ndens_HI_add_field_kwargs = {}):

        _check_ionization_case_kwarg(ionization_case)
        _check_metallicity_AND_metal_dens_args(
            globally_fixed_metallicity = globally_fixed_metallicity, 
            metal_density_field = metal_density_field)

        self.create_nHI_field_gracklemmw_kwargs = {
            'field_name' : HI_field_name,
            'globally_fixed_metallicity' : globally_fixed_metallicity,
            'metal_density_field' : metal_density_field,
            'ionization_case' : ionization_case,
            'add_field_kwargs' : ndens_HI_add_field_kwargs
        }

    def __call__(self, ds):
        # just as a convenience, sure the ('gas','temperature') field exists
        # create ('gas','temperature') field (with correct units!)
        #
        # I should really work on improving the yt-frontend to make sure it's
        # defined automatically for us!

        if ( (('enzoe','temperature') in ds.field_info) and
             (('gas','temperature') not in ds.field_info) ):
            print("The ('gas','temperature') doesn't exist. So we're defining "
                  "it using the ('enzoe','temperature') field")

            def _temperature(field, data):
                field = ('enzoe', 'temperature')
                # ('enzoe', 'temperature') may be labelled as dimensionless
                if data[field].units.is_dimensionless:
                    return data[field] * unyt.K
                return data[field]

            ds.add_field(('gas', 'temperature'), function = _temperature,
                         sampling_type = 'local', take_log = True, units = 'K')

        elif ('gas','temperature') not in ds.field_info:
            warnings.warn("The ('gas','temperature') field doesn't exist and "
                          "can't easily be defined. You are probably going to "
                          "run into issues during ray-tracing")

        # define the number-density of Hydrogen
        create_nHI_field_gracklemmw(ds,
                                    **self.create_nHI_field_gracklemmw_kwargs)
