from typing import Any, Dict, Callable, Tuple

import numpy as np
import unyt

from .generate_ray_spectrum import generate_ray_spectrum_legacy
from ._generate_spec_cy import generate_ray_spectrum
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
    commutative_consolidate = True

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
    commutative_consolidate = False

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
        out = np.empty(shape = (full_ray_uvec.shape[0], obs_freq.size),
                       dtype = 'f8')

        

        return out
