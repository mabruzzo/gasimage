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

from typing import Any, Callable, Dict, Tuple, TypeVar

import numpy as np
import unyt

from .generate_ray_spectrum import generate_ray_spectrum_legacy
from ._generate_spec_cy import NdensStrategy, generate_ray_spectrum
from .ray_traversal.spatial_grid_props import SpatialGridProps
from .rt_config import LineProperties, default_spin_flip_props

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
    doppler_parameter_b: Any

    # class attributes:
    name = "opticallyThin21cm"
    simple_elementwise_sum_consolidate = True

    def __init__(self, *, obs_freq: unyt.unyt_array,
                 use_cython_gen_spec: bool,
                 ndens_HI_n1state_field: Tuple[str,str],
                 doppler_parameter_b: Any):
        assert np.ndim(obs_freq) == 1
        assert (obs_freq.size > 0) and (obs_freq.ndview > 0).all()
        # make obs_freq_Hz as immutable as possible
        self.obs_freq_Hz = obs_freq.to('Hz').ndview.copy()
        self.obs_freq_Hz.flags['WRITEABLE'] = False

        self.use_cython_gen_spec = use_cython_gen_spec
        self.ndens_HI_n1state_field = ndens_HI_n1state_field
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
                particle_mass_in_grams = unyt.mh_cgs.v,
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
                 ndens_field: Tuple[str,str]):
        assert np.ndim(obs_freq) == 1
        assert (obs_freq.size > 0) and (obs_freq.ndview > 0).all()
        # make obs_freq_Hz as immutable as possible
        self.obs_freq_Hz = obs_freq.to('Hz').ndview.copy()
        self.obs_freq_Hz.flags['WRITEABLE'] = False

        self.line_props = line_props
        self.species_mass_g = species_mass_g
        self.partition_func = partition_func
        self.ndens_field = ndens_field

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
        
