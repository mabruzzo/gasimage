
from typing import Any, Dict, Callable, Tuple

import numpy as np
import unyt

from .generate_ray_spectrum import generate_ray_spectrum_legacy
from ._generate_spec_cy import (
    generate_ray_spectrum,
    generate_noscatter_ray_spectrum
)
from .rt_config import LineProperties

class SpatialGridProps:
    """
    This collects spatial properties of the current block (or grid) of the
    dataset

    Notes
    -----
    The original motivation for doing this was to allow rescaling of the grid
    in adiabatic simulations. It's unclear whether that functionality still
    works properly (it definitely hasn't been tested in all contexts).
    """
    cm_per_length_unit : float
    grid_shape: np.ndarray
    left_edge: np.ndarray
    right_edge: np.ndarray
    cell_width: np.ndarray

    def __init__(self, *, cm_per_length_unit: float,
                 grid_shape: np.ndarray,
                 grid_left_edge: unyt.unyt_array,
                 grid_right_edge: unyt.unyt_array,
                 length_unit: str,
                 rescale_factor: float = 1.0):

        assert cm_per_length_unit > 0
        self.cm_per_length_unit = cm_per_length_unit

        assert grid_shape.shape == (3,) and (grid_shape > 0).all()
        assert issubclass(grid_shape.dtype.type, np.integer)
        self.grid_shape = grid_shape.copy() # copy it so it owns the data

        assert grid_left_edge.shape == grid_right_edge.shape == (3,)
        self.left_edge = grid_left_edge.to(length_unit).v * rescale_factor
        self.right_edge = grid_right_edge.to(length_unit).v * rescale_factor
        self.cell_width = (
            (self.right_edge - self.left_edge) / np.array(grid_shape)
        )

        for attr in ['grid_shape', 'left_edge', 'right_edge', 'cell_width']:
            getattr(self,attr).flags['WRITEABLE'] = False
            assert getattr(self,attr).flags['OWNDATA'] == True # sanity check!

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
    # todo remove misc_kwargs and replace with more details
    misc_kwargs: Dict[str, Any]

    # class attributes:
    name = "opticallyThin21cm"
    simple_elementwise_sum_consolidate = True

    def __init__(self, *, obs_freq: unyt.unyt_array,
                 use_cython_gen_spec: bool,
                 misc_kwargs: Dict[str, Any] = {}):
        assert np.ndim(obs_freq) == 1
        assert (obs_freq.size > 0) and (obs_freq.ndview > 0).all()
        # make obs_freq_Hz as immutable as possible
        self.obs_freq_Hz = obs_freq.to('Hz').ndview.copy()
        self.obs_freq_Hz.flags['WRITEABLE'] = False

        self.use_cython_gen_spec = use_cython_gen_spec
        self.misc_kwargs = misc_kwargs
        for key in ['obs_freq', 'grid', 'out', 'ray_start', 'ray_uvec',
                    'full_ray_start', 'full_ray_uvec']:
            if key in misc_kwargs:
                raise ValueError(f"'{key}' should not be a key of misc_kwargs")

    def do_work(self, grid, spatial_grid_props, full_ray_start, full_ray_uvec):

        obs_freq = unyt.unyt_array(self.obs_freq_Hz, 'Hz')
        out = np.empty(shape = (full_ray_uvec.shape[0], obs_freq.size),
                       dtype = 'f8')

        if self.use_cython_gen_spec:
            generate_ray_spectrum(
                grid = grid, spatial_grid_props = spatial_grid_props,
                full_ray_start = full_ray_start, full_ray_uvec = full_ray_uvec,
                obs_freq = obs_freq, out = out, **self.misc_kwargs
            )
        else:
            generate_ray_spectrum_legacy(
                grid = grid, spatial_grid_props = spatial_grid_props,
                full_ray_start = full_ray_start, full_ray_uvec = full_ray_uvec,
                obs_freq = obs_freq, out = out, **self.misc_kwargs
            )

        return out

    def get_rslt_props(self):
        shape = (np.size(self.obs_freq_Hz),)
        return [('intensity', np.float64, shape)]

    def validate_intermediate_rslt(self, rslt):
        _validate_basic_quan_props(self, rslt)

    def consolidate(self, vals):
        raise RuntimeError("Not necessary")

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

        out = generate_noscatter_ray_spectrum(
            grid = grid, spatial_grid_props = spatial_grid_props,
            full_ray_start = full_ray_start, full_ray_uvec = full_ray_uvec,
            obs_freq = obs_freq, line_props = self.line_props,
            partition_func = self.partition_func,
            particle_mass_g = self.species_mass_g,
            ndens_field = self.ndens_field)

        return out

    def get_rslt_props(self):
        shape = (np.size(self.obs_freq_Hz),)
        return [('integrated_source', np.float64, shape),
                ('total_tau',         np.float64, shape)]

    def validate_intermediate_rslt(self, rslt):
        _validate_basic_quan_props(self, rslt)

    def consolidate(self, rslt_iter):
       consolidate_noscatter_rtchunks(chunk_itr)

    def post_process_rslt(self, out):
        out['integrated_source'] = unyt.unyt_array(
            out['integrated_source'], 'erg/(cm**2 * Hz * s * steradian)')
        
