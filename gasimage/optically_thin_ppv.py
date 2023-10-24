import datetime
from typing import Dict, List, Set, Tuple

import numpy as np
import unyt
import yt

try:
    from schwimmbad import SerialPool as _DummySerialPool
except ImportError:

    class _DummySerialPool:
        def map(self, func, iterable, callback = None):
            for result in map(func, iterable):
                if callback is not None:
                    callback(result)
                yield result

import gc

from ._ray_intersections_cy import ray_box_intersections, traverse_grid
from .generate_ray_spectrum import generate_ray_spectrum
from .ray_collection import ConcreteRayList
from .utils.misc import _has_consistent_dims

class Worker:
    """
    This does most of the ray-casting heavy lifting.

    Essentially, the constructor takes in a bunch of configuration. Then the
    __call__ is invoked directing the program to perform raytracing for a
    subset of rays through a particular region of space. The result is then
    returned and the caller ultimately stiches the result back together.

    In more detail, this class exists for the sake of facillitating
    parallelization using a worker-pool.
    - when calling a worker-pool, the managing process specifies a function
      that needs to be executed and provides an iterable of arguments that
      should be passed to that function. The pool first sends the function to
      each of the worker (serialization is done via pickle). Then the pool
      passes arguments to each worker, executes the function with the argument
      and then sends the results back to the primary process.
    - in principle, the ray-casting could all be accomplished with a normal
      function, by adding more arguments to the function. Since a lot of the
      arguments wouldn't change between function calls, we would be wasting a
      bunch of time serializing/unserializing those arguments.
    - Thus, we define this class instead of a function that stores those
      unchanging arguments. We effectively cache these unchanging arguments.

    TODO: improve this documentation
    TODO: explain somewhat roundabout handling of ds_initializer
    """
    def __init__(self, ds_initializer, obs_freq, length_unit_name,
                 rescale_length_factor, generate_ray_spectrum_kwargs):
        assert np.ndim(obs_freq) == 1

        self.ds_initializer = ds_initializer
        self.obs_freq = obs_freq
        self.length_unit_name = length_unit_name
        self.rescale_length_factor = rescale_length_factor

        for key in ['obs_freq', 'grid', 'grid_left_edge', 'grid_right_edge',
                    'cell_width', 'grid_shape', 'out', 'ray_start', 'ray_uvec']:
            if key in generate_ray_spectrum_kwargs:
                raise ValueError(f"'{key}' should not be a key of "
                                 "generate_ray_spectrum_kwargs")
        self.generate_ray_spectrum_kwargs = generate_ray_spectrum_kwargs


    def __call__(self, elem):
        grid_index, ray_start, ray_uvec = elem
        assert (ray_uvec.ndim == 2) and ray_uvec.shape[1] == 3

        out = np.empty(shape = (ray_uvec.shape[0], self.obs_freq.size),
                       dtype = np.float64)

        # load the dataset
        _ds_initializer = self.ds_initializer
        if isinstance(_ds_initializer, yt.data_objects.static_output.Dataset):
            ds = _ds_initializer
        else:
            ds = _ds_initializer()

        # load properties about the current block of the dataset
        grid = ds.index.grids[grid_index]
        grid_shape = ds.index.grid_dimensions[grid_index]
        left_edge \
            = ds.index.grid_left_edge[grid_index].to(self.length_unit_name).v
        right_edge \
            = ds.index.grid_right_edge[grid_index].to(self.length_unit_name).v
        left_edge *= self.rescale_length_factor
        right_edge *= self.rescale_length_factor
        cell_width = ((right_edge - left_edge)/np.array(grid.shape))

        # now actually process the rays
        for i in range(ray_uvec.shape[0]):
            cur_ray_start = ray_start[i,:]
            cur_ray_uvec = ray_uvec[i,:]

            # generate the spectrum
            generate_ray_spectrum(
                grid, grid_left_edge = left_edge, grid_right_edge = right_edge,
                cell_width = cell_width, grid_shape = grid_shape,
                ray_start = cur_ray_start, ray_uvec = cur_ray_uvec,
                obs_freq = self.obs_freq, out = out[i,:],
                **self.generate_ray_spectrum_kwargs
            )
        grid.clear_data()
        del grid
        gc.collect()

        return grid_index, out


def _top_level_grid_indices(ds):
    """
    Creates a 3D array holding the indices of the top level blocks

    Note: we use the term block and grid interchangably
    """
    assert ds.index.max_level == 0
    n_blocks = len(ds.index.grids)

    # there's probably a better way to do the following:
    root_block_width = (ds.index.grid_right_edge - ds.index.grid_left_edge)[0]
    to_nearest_int = lambda arr: np.trunc(arr+0.5).astype(np.int32)

    blocks_per_axis = to_nearest_int(
        (ds.domain_width / root_block_width).in_cgs().v
    )
    block_loc_array = np.empty(shape=blocks_per_axis, dtype = np.int32)
    assert block_loc_array.size == n_blocks

    block_array_indices = to_nearest_int(
        ((ds.index.grid_left_edge - ds.domain_left_edge) /
         root_block_width).in_cgs().v
    )

    block_loc_array[tuple(block_array_indices.T)] = np.arange(n_blocks)
    return block_loc_array

class RayGridAssignments:
    # for the sake of clarity, let's list & annotate the instance variables of
    # this class (These are not class variables!)

    # a dictionary that associates sequences of grid-ids to ray-indices
    # -> the keys are a unique list of grid-ids (this needs to be a tuple so
    #    that it can be hashed)
    # -> each key is associated with a list of ray ids (or indexes) that pass
    #    through the sequence of grid-ids.
    # -> a ray id cannot be associated with more than one grid-id-sequence
    _sequence_table: Dict[Tuple[int,...],List[int]]

    # this set lists all subgrid ids that are associated with one or more rays
    _subgrids_with_rays: Set[int]

    def __init__(self, ds, ray_list, units,
                 rescale_length_factor = 1.0):
        subgrid_array = _top_level_grid_indices(ds)
        domain_left_edge = (
            ds.domain_left_edge.to(units).v * rescale_length_factor
        )
        domain_right_edge = (
            ds.domain_right_edge.to(units).v * rescale_length_factor
        )
        cell_width = ((domain_right_edge - domain_left_edge)/
                      np.array(subgrid_array.shape))
        subgrid_array_shape = subgrid_array.shape

        self._sequence_table = {}

        self._subgrids_with_rays = set()

        ray_uvec_l = ray_list.get_ray_uvec()

        for i in range(len(ray_list)):
            ray_start = ray_list.ray_start_codeLen[i,:]
            ray_uvec = ray_uvec_l[i,:]

            intersect_dist = ray_box_intersections(
                line_start = ray_start, line_uvec = ray_uvec, 
                left_edge = domain_left_edge, right_edge = domain_right_edge)
            if len(intersect_dist) < 2:
                continue
            idx,_ = traverse_grid(line_uvec = ray_uvec,
                                  line_start = ray_start,
                                  grid_left_edge = domain_left_edge,
                                  cell_width = cell_width,
                                  grid_shape = subgrid_array_shape)
            subgrid_sequence = tuple(subgrid_array[idx[0],idx[1],idx[2]])

            self._subgrids_with_rays.update(subgrid_sequence)

            if subgrid_sequence in self._sequence_table:
                self._sequence_table[subgrid_sequence].append(i)
            else:
                self._sequence_table[subgrid_sequence] = [i]
        
    def rays_associated_with_subgrid(self, subgrid_id):
        """
        Returns the indicies of rays that are associated with the given
        subgrid_id
        """
        out = []
        if subgrid_id not in self._subgrids_with_rays:
            return out
        for subgrid_sequence, ray_ids in self._sequence_table.items():
            if subgrid_id in subgrid_sequence:
                out = out + ray_ids
        return out



def optically_thin_ppv(v_channels, ray_collection, ds,
                       ndens_HI_n1state = ('gas', 'H_p0_number_density'),
                       *, doppler_v_width = None, use_cython_gen_spec = False,
                       rescale_length_factor = None, pool = None):
    """
    Generate a mock ppv image (position-position-velocity image) of a
    simulation using a ray-tracing radiative transfer algorithm that assumes 
    that the gas is optically thin.

    Parameters
    ----------
    v_channels: 1D `unyt.unyt_array`
        A monotonically increasing array of velocity channels.
    ray_collection:
        A collection of rays. This should be an instance of one of the classes
        defined in this package
    ds: `yt.data_objects.static_output.Dataset` or `SnapDatasetInitializer`
        The dataset or an object that initializes the dataset from which the
        image is constructed. If ds is the dataset, itself, this function
        cannot be parallelized (due to issues with pickling). Currently, the 
        dataset must be unigrid.
    ndens_HI_n1state: 2-tuple of strings
        The name of the yt-field holding the number density of H I atoms
        (neutral Hydrogen) in the electronic ground state (i.e. the electron is
        in the n=1 orbital). By default, this is the number density of all
        H I atoms. This approximation is discussed below in the notes.
    doppler_v_width: `unyt.unyt_quantity`, Optional
        Optional parameter that can be used to specify the doppler width at
        every cell in the resulting image. When not specified, this is computed
        from the local line of sight velocity.
    use_cython_gen_spec: bool, optional
        Generate the spectrum using the cython implementation. This is
        currently experimental (and should not be used until the results are 
        confirmed to be consistent with the python implementation).
    rescale_length_factor: float, Optional
        When not `None`, the width of each cell is multiplied by this factor.
        (This effectively holds the position of the simulation's origin fixed
        in place).
    pool: Optional
        A taskpool from the schwimmbad package can be specified for this
        argument to facillitate parallelization.

    Notes
    -----
    By default, ``ndens_HI_n1state`` is set to the yt-field specifying the
    number density of all neutral Hydrogen. When this field is accurate, and
    the electron-energy level populations are in LTE, this is probably a fairly
    decent approximation. While it may overestimate the fraction of neutral
    Hydrogen at n=1 at higher temperatures (above a few times $10^4\, {\rm K}$),
    most of the hydrogen should be ionized at these temperatures.

    We assumed that v<<c. This lets us:
    - neglect the transverse redshift
    - approximate the longitudinal redshift as freq_obs = rest_freq*(1+vel/c)

    TODO: In the future, it would be nice to be able to specify a bulk velocity
          for the gas.
    TODO: In the future, we might want to consider adding support for using a 
          Voigt line profile
    TODO: Revisit treatment of velocity channels. I think we currently just
          compute the opacity at the nominal velocity of the channel. In
          reality, I think velocity channels are probably more like bins
    """
    if rescale_length_factor is not None:
        assert rescale_length_factor > 0 and np.ndim(rescale_length_factor) == 0
    else:
        rescale_length_factor = 1.0

    pool = _DummySerialPool() if pool is None else pool

    is_mpi_root_proc = (not hasattr(pool,'is_master')) or pool.is_master()

    # use the code units (since the cell widths are usually better behaved)

    # do a bunch of setup!
    if is_mpi_root_proc:
        # this branch is always executed by the root process driving the
        # program. (It's only not called by non-root processes if the user is
        # using mpi -- mpi suport still needs to be tested)

        # now get an instance of the dataset
        if isinstance(ds, yt.data_objects.static_output.Dataset):
            if not isinstance(pool, schwimmbad.SerialPool):
                raise ValueError("When ds is a dataset object, pool must be "
                                 "assigned a schwimmbad.SerialPool instance.")
            else:
                my_ds = ds
        else:
            my_ds = ds()

        if hasattr(ray_collection, 'domain_edge_sanity_check'):
            _l = rescale_length_factor * my_ds.domain_left_edge
            _r = rescale_length_factor * my_ds.domain_right_edge
            ray_collection.domain_edge_sanity_check(_l,_r)

        # create the iterator
        # use the code units (since the cell widths are usually better behaved)
        length_unit_name = 'code_length'
        length_unit_quan = my_ds.quan(1.0, 'code_length')

        if isinstance(ray_collection, ConcreteRayList):
            ray_list = ray_collection
        else:
            ray_list = ray_collection.as_concrete_ray_list(length_unit_quan)

        print('Constructing RayGridAssignments')
        subgrid_ray_map = RayGridAssignments(
            my_ds, ray_list = ray_list,
            units = length_unit_name,
            rescale_length_factor = rescale_length_factor
        )

        def generator():
            for grid_ind in range(my_ds.index.grids.size):
                ray_idx =subgrid_ray_map.rays_associated_with_subgrid(grid_ind)
                if len(ray_idx) == 0:
                    continue
                else:
                    print('grid_index = ', grid_ind, ' num_rays = ',
                          len(ray_idx))

                ray_start_codeLen, ray_uvec = \
                    ray_list.get_selected_raystart_rayuvec(ray_idx)
                yield grid_ind, ray_start_codeLen, ray_uvec


        # create the worker
        rest_freq = 1.4204058E+09*unyt.Hz,
        worker = Worker(
            ds_initializer = ds,
            obs_freq = (rest_freq*(1+v_channels/unyt.c_cgs)).to('Hz'),
            length_unit_name = length_unit_name,
            rescale_length_factor = rescale_length_factor,
            generate_ray_spectrum_kwargs = {
                'rest_freq' : rest_freq,
                'cm_per_length_unit' : length_unit_quan.to('cm').v,
                'doppler_v_width' : doppler_v_width,
                'ndens_HI_n1state_field' : ndens_HI_n1state,
                'use_cython_gen_spec' : use_cython_gen_spec,
            }
        )

        # create the output array and callback function
        out_shape = (v_channels.size,) + ray_collection.shape
        out = np.zeros(shape = out_shape)

        # flatten the out array down to 2-dimensions:
        out_2D = out.view()
        out_2D.shape = (v_channels.size, -1)

        def callback(rslt):
            # this callback function actually takes the result of each call to
            # Worker.__call__ (once they have been communicated back to the
            # primary thread) and consolidates them in the output array.
            # - for testing this pipeline with a parallel-pool, it might be
            #   convenient to add an option to force the contributions to be
            #   added in a deterministic order
            # - note: the order should always be deterministic in serial.
            grid_index, ray_spectra = rslt
            ray_idx = subgrid_ray_map.rays_associated_with_subgrid(grid_index)
            for i,ray_ind in enumerate(ray_idx):
                out_2D[:,ray_ind] += ray_spectra[i,:]
    else:
        # this branch is almost never executed. The one exception is when the
        # program is executed using an mpi-pool. In that case all non-root
        # processes execute this path (they essentially do nothing...). Those
        # processes should receive all of the information they need to do the
        # work through the pool (just like processes in multiprocessing.Pool.
        #
        # NOTE: mpi suport still needs to be tested
        generator = None
        worker = None
        callback = None
        out = None

    # the bulk of the work occurs here:
    _tstart = datetime.datetime.now()
    print('begin raycasting -- start time: {_tstart.time()}')

    for elem in pool.map(worker, generator(), callback = callback):
        continue

    _tend = datetime.datetime.now()
    print('finished raycasting\n'
          f'-> start, end time: {_tstart.time()}, {_tend.time()}\n'
          f'-> elapsed time: {_tend - _tstart}')

    # attach units and massage the output shape
    out_units = 'erg/(cm**2 * Hz * s * steradian)'
    if len(ray_collection.shape) == 1:
        assert out_2D.shape == (v_channels.size, 1)
        return unyt.unyt_array(out_2D[:,0], out_units)
    else:
        return unyt.unyt_array(out, out_units)

_intensity_dim = (
    unyt.dimensions.energy /
    (unyt.dimensions.area * unyt.dimensions.frequency * unyt.dimensions.time *
     unyt.dimensions.solid_angle)
)

def convert_intensity_to_Tb(ppv, v_channels,
                            rest_freq = 1.4204058E+09*unyt.Hz):
    """
    Convert intensity ppv array to brightness temperature

    Notes
    -----
    The correctness of this calculation is not entirely clear
    """

    if not _has_consistent_dims(ppv, _intensity_dim):
        raise ValueError("ppv has the wrong units")
    elif not _has_consistent_dims(rest_freq, unyt.dimensions.frequency):
        raise ValueError("rest_freq doesn't have appropriate dimensions")

    if v_channels is None:
        freq = rest_freq
    elif np.shape(v_channels) != np.shape(ppv)[:1]:
        raise ValueError("when not None, the number of v_channels should "
                         "match np.shape(ppv)[0]")
    elif not _has_consistent_dims(v_channels, unyt.dimensions.velocity):
        raise ValueError("v_channels has the wrong units")
    else:
        # assume that velocity << c
        obs_freq = (rest_freq*(1+v_channels/unyt.c_cgs)).to('Hz')
        freq = obs_freq
        new_shape = [1 for elem in ppv.shape]
        new_shape[0] = v_channels.size
        freq.shape = tuple(new_shape)

    # take some care with this conversion:
    #
    # - Recall that brightness temperature is the temperature that we'd plug
    #   into the black-body radiation equation,
    #                2 * h * nu**3              1
    #      B_nu(T) = ------------- *  -----------------------
    #                     c**2        exp(h*nu/(k_B * T)) - 1
    #   to recover the B_nu equal I_nu (Note that B_nu has the same units
    #   as I_nu).
    #
    # - since we're in the limit where h*nu << k_B*T, we can use the
    #   Rayleigh-Jeans law:
    #     B_nu = 2 * nu^2 * k_B * T / c**2
    #
    # - in each of the above equations, the RHS is missing an implicit
    #   multiplication by `(1/steradian)`.

    return (0.5*(unyt.c_cgs / freq)**2 * ppv * unyt.steradian /
            unyt.kboltz_cgs).to('K')
