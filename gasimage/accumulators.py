"""
Define Accumulator Strategies

An "Accumulator Strategy" is represpresented by the `AccumStratT` typing-stub
that we define down below. Essentially, it is a configurable type that
describes the approach for creating some kind of an "image" of gas.

- At this time, all strategies involve ray-casting. Current implemented types
perform radiative transfer with different levels of rigor. 
- Another strategy could hypothetically perform a projection...

The fundamental idea is that an AccumStratT is a configurable type that 
parameterizes the commands that must be executed. An important note is that an
AccumStrat should NOT store any of the state of the calculation. All state
should be stored externally. At the time of writing this, that external state
is essentially a dictionary of arrays...

Rather than thinking of AccumStratT as an abstract base class, it's more of a
"protocol". At the time of writing this blurb,

- an AccumStratT instance provides the following attributes (usually defined as
  class attributes):
   -> the `name` attribute (a pretty name to identify the AccumStrat - in the
      future, we may use this in some kind of factory function)
   -> the `simple_elementwise_sum_consolidate` attrbute (specifies whether we
      can perform a really simple optimization when consolidating the results
      from individual ray-subdomain combinations)
- an AccumStratT must provide the following methods:
   -> do_work
   -> get_rslt_props
   -> validate_intermediate_rslt
   -> consolidate
   -> post_process_rslt
"""

from collections.abc import Sequence
from typing import Any, Callable, Dict, Tuple, TypeVar

import numpy as np
import unyt

from .generate_ray_spectrum import generate_ray_spectrum_legacy
from ._generate_spec_cy import NdensStrategy, generate_ray_spectrum
from .ray_traversal._misc_cy import SpatialGridProps
from .rt_config import LineProperties, default_spin_flip_props
from .utils.misc import _has_consistent_dims

# define the actual AccumStratT typing stub
AccumStratT = TypeVar("AccumStratT")

def _validate_basic_quan_props(accum, rslt):
    # this is borrowed and reworked from pyvsf

    quan_props = accum.get_rslt_props()
    assert len(quan_props) == len(rslt)

    if (len(quan_props) == 1) and isinstance(rslt, np.ndarray):
        # for backwards compatibility
        name, dtype, shape = quan_props[0]
        if rslt.dtype != dtype:
            raise ValueError(
                f"the intermediate quantity, '{name}', for the {accum.name} "
                f"accumulator should have a dtype of {dtype}, not of "
                "{rslt.dtype}"
            )
        elif rslt.shape != shape:
            raise ValueError(
                f"the intermediate quantity, '{name}', for the {accum.name} "
                f"accumulator should have a shape of {shape}, not of "
                f"{rslt.shape}"
            )
        return None

    for name, dtype, shape in quan_props:
        if name not in rslt:
            raise ValueError(
                f"The result for the '{accum.name}' accumulator is missing a "
                f"quantity called '{name}'"
            )
        elif rslt[name].dtype != dtype:
            raise ValueError(
                f"the {name} quantity for the {accum.name} accumulator should "
                f"have a dtype of {dtype}, not of {rslt[name].dtype}"
            )
        elif rslt[name].shape != shape:
            raise ValueError(
                f"the {name} quantity for the {accum.name} accumulator should "
                f"have a shape of {shape}, not of {rslt[name].shape}"
            )


_DFLT_ndens_HI_n1 = ('gas', 'H_p0_number_density')

class OpticallyThinAccumStrat:
    """
    Represents the configuration of the operations associated with optically
    thin Radiative Transfer

    At the moment, this can only be configured to work with the spin-flip
    transition.

    This is intended to be immutable!
    """

    # instance attributes:
    obs_freq_Hz: np.ndarray
    use_cython_gen_spec: bool
    ndens_HI_n1state_field: Tuple[str,str]
    particle_mass_in_grams: float
    doppler_parameter_b: Any

    # class attributes:
    name = "opticallyThin21cm"
    simple_elementwise_sum_consolidate = True

    def __init__(self, *, obs_freq: unyt.unyt_array,
                 use_cython_gen_spec: bool,
                 ndens_HI_n1state_field: Tuple[str,str],
                 doppler_parameter_b: Any,
                 particle_mass_in_grams: float = float(unyt.mh_cgs.v)):
        assert np.ndim(obs_freq) == 1
        assert (obs_freq.size > 0) and (obs_freq.ndview > 0).all()
        # make obs_freq_Hz as immutable as possible
        self.obs_freq_Hz = obs_freq.to('Hz').ndview.copy()
        self.obs_freq_Hz.flags['WRITEABLE'] = False

        self.use_cython_gen_spec = use_cython_gen_spec
        self.ndens_HI_n1state_field = ndens_HI_n1state_field
        self.particle_mass_in_grams = float(particle_mass_in_grams)
        assert self.particle_mass_in_grams > 0
        self.doppler_parameter_b = doppler_parameter_b

    def do_work(self, grid, spatial_grid_props, full_ray_start, full_ray_uvec):

        obs_freq = unyt.unyt_array(self.obs_freq_Hz, 'Hz')
        out = np.empty(shape = (full_ray_uvec.shape[0], obs_freq.size),
                       dtype = 'f8')

        if self.use_cython_gen_spec:
            generate_ray_spectrum(
                grid = grid,
                spatial_grid_props = spatial_grid_props,
                full_ray_start = full_ray_start,
                full_ray_uvec = full_ray_uvec,
                line_props = default_spin_flip_props(),
                legacy_optically_thin_spin_flip = True,
                obs_freq = obs_freq,
                particle_mass_in_grams = self.particle_mass_in_grams,
                ndens_strat = NdensStrategy.SpecialLegacySpinFlipCase,
                ndens_field = self.ndens_HI_n1state_field,
                partition_func = None,
                doppler_parameter_b = self.doppler_parameter_b,
                out = out,
            )
        else:
            generate_ray_spectrum_legacy(
                grid = grid, spatial_grid_props = spatial_grid_props,
                full_ray_start = full_ray_start, full_ray_uvec = full_ray_uvec,
                obs_freq = obs_freq,
                ndens_HI_n1state_field = self.ndens_HI_n1state_field,
                out = out,
                doppler_parameter_b = self.doppler_parameter_b
            )

        return out

    def get_rslt_props(self):
        shape = (np.size(self.obs_freq_Hz),)
        return [('intensity', np.float64, shape)]

    def validate_intermediate_rslt(self, rslt):
        _validate_basic_quan_props(self, rslt)

    def consolidate(self, vals):
        # this is not necessary since self.simple_elementwise_sum_consolidate
        # is True. We only really implement this for testing purposes

        name, dtype, shape = self.get_rslt_props()[0]
        out = {name : np.zeros(dtype = dtype, shape = shape)}

        for val in vals:
            # for vals is just a list of arrays (this is frankly a little odd)
            out[name][:] += val
        return out
        

    def post_process_rslt(self, out):
        out['intensity'] = unyt.unyt_array(
            out['intensity'], 'erg/(cm**2 * Hz * s * steradian)')

def consolidate_noscatter_rtchunks(chunk_itr):
    """
    Consolidate the results of noscatter RT

    Parameters
    ----------
    chunk_itr: iterator or iterable
        An iterator or iterable of dictionaries holding the results of
        radiative transfer for chunks of a simulation domain along a single
        ray. The elements are ordered with increasing distance from the
        observer. Each dictionary should have entries associated with 2 keys:
        `"integrated_source"` & `"total_tau`. These entries respectively track
        the integrated source function through the chunk and the optical depth
        through the chunk.
    """
    chunk_itr = iter(chunk_itr)

    try:
        first_chunk_rslt = next(chunk_itr)
    except StopIteration:
        raise ValueError(
            "consolidate_noscatter_rtchunks must be called with an iterable "
            "that contains at least 1 entry") from None

    assert len(first_chunk_rslt) == 2 # sanity check!
    out = {"integrated_source" : first_chunk_rslt["integrated_source"],
           "total_tau" : first_chunk_rslt["total_tau"]}

    # iterator over the remaining chunks
    for cur_chunk in chunk_itr:
        # step 1: update out["integrated_source"] by adding the integrated
        #  source term from cur_chunk, after attenuating it based on the total
        #  optical depth between the observer & the near-edge of current chunk
        out["integrated_source"][:] +=  (np.exp(-out["total_tau"][:]) *
                                         cur_chunk["integrated_source"])

        # step 2: update out["total_tau"] so that it includes the optical depth
        #  produced by the current chunk AND all chunks closer to the observer
        out["total_tau"][:] += cur_chunk["total_tau"]
    return out

class NoScatterRTAccumStrat:
    """
    Represents the configuration of the operations associated with Radiative 
    ransfer

    This is intended to be immutable!
    """

    # instance attributes:
    obs_freq_Hz: np.ndarray
    ignore_vLOS: bool
    line_props: LineProperties
    species_mass_g: float
    partition_func: Callable[[unyt.unyt_array], np.ndarray]
    ndens_field: Tuple[str,str] # specified the number density of the species
                                # described by the partition function

    # class attributes:
    name = "noScatterRT"
    simple_elementwise_sum_consolidate = False

    def __init__(self, *, obs_freq: unyt.unyt_array,
                 line_props: LineProperties, species_mass_g: float,
                 partition_func: Callable[[unyt.unyt_array], np.ndarray],
                 ndens_field: Tuple[str,str],
                 ignore_vLOS: bool = False):
        assert np.ndim(obs_freq) == 1
        assert (obs_freq.size > 0) and (obs_freq.ndview > 0).all()
        # make obs_freq_Hz as immutable as possible
        self.obs_freq_Hz = obs_freq.to('Hz').ndview.copy()
        self.obs_freq_Hz.flags['WRITEABLE'] = False

        self.line_props = line_props
        self.species_mass_g = species_mass_g
        self.partition_func = partition_func
        self.ndens_field = ndens_field
        self.ignore_vLOS = ignore_vLOS

    def do_work(self, grid, spatial_grid_props, full_ray_start, full_ray_uvec):

        obs_freq = unyt.unyt_array(self.obs_freq_Hz, 'Hz')
        
        out = generate_ray_spectrum(
            grid = grid, spatial_grid_props = spatial_grid_props,
            full_ray_start = full_ray_start, full_ray_uvec = full_ray_uvec,
            line_props = self.line_props,
            legacy_optically_thin_spin_flip = False,
            particle_mass_in_grams = self.species_mass_g, obs_freq = obs_freq,
            ndens_strat = NdensStrategy.IonNDensGrid_LTERatio,
            ndens_field = self.ndens_field,
            partition_func = self.partition_func,
            doppler_parameter_b = 'normal',
            ignore_vLOS = self.ignore_vLOS,
            out = None)
        return out

    def get_rslt_props(self):
        shape = (np.size(self.obs_freq_Hz),)
        return [('integrated_source', np.float64, shape),
                ('total_tau',         np.float64, shape)]

    def validate_intermediate_rslt(self, rslt):
        _validate_basic_quan_props(self, rslt)

    def consolidate(self, rslt_iter):
       return consolidate_noscatter_rtchunks(rslt_iter)

    def post_process_rslt(self, out):
        out['integrated_source'] = unyt.unyt_array(
            out['integrated_source'], 'erg/(cm**2 * Hz * s * steradian)')

def freq_from_v_channels(v_channels, line_props):
    if not _has_consistent_dims(v_channels, unyt.dimensions.velocity):
        raise ValueError("v_channels has the wrong units")
    rest_freq = line_props.freq_quantity
    return (rest_freq * (1 + v_channels/unyt.c_cgs)).to('Hz')

def configure_single_line_rt(line_props, species_mass_g, ndens_field, *,
                             kind = 'noscatter', observed_freq = None,
                             v_channels = None, partition_func = None,
                             doppler_parameter_b = 'normal',
                             ignore_vLOS = False):
    """
    Generate the configured accumulation strategy for radiative transfer.

    Parameters
    ----------
    line_props : LineProps
        Encodes details about the line transition
    species_mass_g : float
        Specifies the mass of the species in grams
    ndens_field : tuple of 2 strings
        Specifies the name of the yt-field that specifies the number density of
        the relevant particle species.
        - when `kind == 'noscatter'`, the specified field should correspond to
          the number-density used in the partition function (so that this field
          and the parition function can be used to compute the number-density
          in individual energy states).
        - when `kind` is either `'noscatter21cm'` or `'opticallyThin21cm'`,
          the specified field should correspond to the total number density of
          Hydrogen in the electronic ground state (the electronic state n = 1).
          In other words, it includes the number density of hydrogen in either
          the spin-up or spin-down state.
    kind : {'noscatter', 'noscatter21cm', 'opticallyThin21cm'}
        Specifies the radiative transfer strategy
    observed_freq : unyt.unyt_array, Optional
        Array of frequencies to perform radiative transfer at (in Hz). Either
        this kwarg or the v_channels must be provided (but not both).
    v_channels : unyt.unyt_array, Optional
        Array of velocity channels that we are interested in computing the
        intensities at. Either this kwarg or the v_channels must be provided
        (but not both).
    partition_func : PyLinInterpPartitionFunc, optional
        Specifies the partition function of the species associated with the
        transition.
    doppler_parameter_b: `unyt.unyt_quantity` or `str`
        Optional parameter that can be used to specify the Doppler parameter
        (aka Doppler Broadening parameter) assumed for every cell of the
        simulation. A value of 'normal' will use the correct formula (this is
        the preferred behavior) for computing the value from the local 
        temperature. When this is passed 'legacy', we will use the incorrect
        legacy formula (it involves using mean-molecular-mass instead of
        aborber/emitter mass). When not specified, this is computed from the
        local temperature (and mean-molecular-weight). To avoid any ambiguity,
        this quantity has units consistent with velocity, this quantity is
        commonly represented by the variable ``b``, and ``b/sqrt(2)`` specifies
        the standard deviation of the line-of-sight velocity component. Note
        that ``b * rest_freq / (unyt.c_cgs * sqrt(2))`` specifies the standard
        deviation of the line-profile for the transition that occurs at a rest
        frequency of ``rest_freq``.
    """
    if not isinstance(line_props, LineProperties):
        # in the future, we could be more flexible about this...
        raise TypeError("line_props must be an instance of LineProperties")

    if (observed_freq is None) == (v_channels is None):
        raise ValueError("One of the `observed_freq` & `v_channels` kwargs is "
                         "required. It's also incorrect to provide both ")
    elif observed_freq is None:
        observed_freq = freq_from_v_channels(v_channels, line_props)
    else:
        observed_freq = observed_freq

    assert ndens_field is not None

    if kind == 'noscatter':
        assert partition_func is not None
        assert doppler_parameter_b == 'normal'

        return NoScatterRTAccumStrat(
            obs_freq = freq_from_v_channels(v_channels, line_props),
            line_props = line_props,
            species_mass_g = species_mass_g,
            partition_func = partition_func,
            ndens_field = ndens_field,
            ignore_vLOS = ignore_vLOS
        )
    elif kind == 'noscatter21cm':
        raise NotImplementedError("this hasn't been tested quite yet")
    elif kind == 'opticallyThin21cm':
        raise RuntimeError("This hasn't been tested quite yet")
        assert ignore_vLOS == False

        return OpticallyThinAccumStrat(
            obs_freq = observed_freq,
            use_cython_gen_spec = True,
            ndens_HI_n1state_field = ndens_field,
            doppler_parameter_b = doppler_parameter_b,
            particle_mass_in_grams = particle_mass_in_grams
        )
    else:
        raise ValueError("kind must be 'noscatter', 'noscatter21cm', or "
                         "'opticallyThin21cm'")
