import datetime
from functools import partial
import gc
from typing import Any, Dict, List, Optional, Set, Tuple

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

from .accumulators import (
    AccumStratT,
    OpticallyThinAccumStrat,
    SpatialGridProps
)
from ._ray_intersections_cy import ray_box_intersections, traverse_grid
from .ray_collection import ConcreteRayList
from .utils.misc import _has_consistent_dims
from .rt_config import default_spin_flip_props

def _first_or_default(itr, default=None):
    # returns the first element of the iterator/iterable or a default value.
    return next(iter(itr), default)

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
    def __init__(self, ds_initializer, length_unit_name, cm_per_length_unit,
                 rescale_length_factor, accum_strat):

        self.ds_initializer = ds_initializer
        self.length_unit_name = length_unit_name
        self.cm_per_length_unit = cm_per_length_unit
        self.rescale_length_factor = rescale_length_factor
        self.accum_strat = accum_strat

    def __call__(self, elem):
        grid_index, ray_start, ray_uvec = elem
        assert (ray_uvec.ndim == 2) and ray_uvec.shape[1] == 3

        # load the dataset
        _ds_initializer = self.ds_initializer
        if isinstance(_ds_initializer, yt.data_objects.static_output.Dataset):
            ds = _ds_initializer
        else:
            ds = _ds_initializer()

        # load properties about the current block of the dataset
        grid = ds.index.grids[grid_index]

        # sanity check!
        assert (np.array(grid.shape) ==
                ds.index.grid_dimensions[grid_index]).all()

        spatial_grid_props = SpatialGridProps(
            cm_per_length_unit = self.cm_per_length_unit,
            grid_shape = ds.index.grid_dimensions[grid_index],
            grid_left_edge = ds.index.grid_left_edge[grid_index],
            grid_right_edge = ds.index.grid_right_edge[grid_index],
            length_unit = self.length_unit_name,
            rescale_factor = self.rescale_length_factor
        )

        # now actually process the rays
        out = self.accum_strat.do_work(
            grid = grid, spatial_grid_props = spatial_grid_props,
            full_ray_start = ray_start, full_ray_uvec = ray_uvec
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

    # this associates the number of intersected grids with a ray-id
    # -> note this may not need to be an attribute
    # -> we could potentially make a factory method that returns both this list
    #    and RayGridAssignments together
    _num_intersected_subgrids: List[int]

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

        length = len(ray_list)
        self._num_intersected_subgrids = [0 for i in range(length)]

        ray_uvec_l = ray_list.get_ray_uvec()

        for i in range(length):
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

            # record that the current ray is associated with subgrid_sequence

            self._subgrids_with_rays.update(subgrid_sequence)

            if subgrid_sequence in self._sequence_table:
                self._sequence_table[subgrid_sequence].append(i)
            else:
                self._sequence_table[subgrid_sequence] = [i]

            # record the number of intersected subgrids
            self._num_intersected_subgrids[i] = len(subgrid_sequence)

    def num_subgrid_intersections(self):
        return self._num_intersected_subgrids

    def rays_associated_with_subgrid(self, subgrid_id,
                                     pair_with_seqindx = False):
        """
        Returns the indicies of rays that are associated with the given
        subgrid_id
        """
        if subgrid_id not in self._subgrids_with_rays:
            return []
        out = []
        if pair_with_seqindx:
            for subgrid_sequence, ray_ids in self._sequence_table.items():
                if subgrid_id not in subgrid_sequence:
                    continue
                seqpos = subgrid_sequence.index(subgrid_id)
                out = out + [(ray_id, seqpos) for ray_id in ray_ids]
        else:
            for subgrid_sequence, ray_ids in self._sequence_table.items():
                if subgrid_id in subgrid_sequence:
                    out = out + ray_ids

        return out

class CallbackFunctor:
    """
    Configurable callback function that is passed the result of each call to
    `Worker.__call__` (the result is first communicated back to the primary
    thread and then this is invoked on the primary thread)

    This basically supports 2 modes of operation:
      1. When `accum_strat.simple_elementwise_sum_consolidate` is True and
         there is only a single output array, this special mode can be used. In
         this case, `self.temporary` is unused (it's set to `None`). Instead,
         the callback will directly update values within
         `self.single_sum_accum_2Darr`, which is the only array contained by
         the `out_dict_2D` that is passed to the constructor
      2. This is the more general mode of operation. In this case, the callback
         uses `self.temporary` variable to amass (or consolidate) all of the
         results from every relevant ray-subgrid pair. After the program is
         done executing `Worker.__call__`, the entries of `self.temporary` are
         consolidated in a separate step.

    To summarize, the first case makes use of `self.single_sum_accum_2Darr`
    while `self.temporary` has a value of `None`. The latter case makes use of
    `self.temporary` and `self.single_sum_accum_2Darr` has a value of `None`
    """
    # declare the instance variables:
    accum_strat: AccumStratT
    subgrid_ray_map: RayGridAssignments
    single_sum_accum_2Darr: Optional[np.ndarray]
    temporary: Optional[List[List[Any]]]
    

    def __init__(self, accum_strat: AccumStratT,
                 out_dict_2D: Dict[str, np.ndarray],
                 subgrid_ray_map: RayGridAssignments,
                 force_general_mode: bool = False):
        self.accum_strat, self.subgrid_ray_map = accum_strat, subgrid_ray_map
        self.single_sum_accum_2Darr, self.temporary = None, None

        if (accum_strat.simple_elementwise_sum_consolidate and
            not force_general_mode):
            if len(out_dict_2D) != 1:
                raise NotImplementedError(
                    "The implementation needs to be revisited to some degree: "
                    "an invariant was violated.")
            self.single_sum_accum_2Darr = _first_or_default(
                out_dict_2D.values()).view() # we use view for explicitness
        else:
            # reserve space in `self.temporary` for each relevant ray-subdomain
            # pair (that will be returned by the worker)
            self.temporary = []
            for pair in enumerate(subgrid_ray_map.num_subgrid_intersections()):
                ray_id, num_intersections = pair
                #assert len(self.temporary) == ray_id # sanity check!
                self.temporary.append([None for i in range(num_intersections)])

    def __call__(self, rslt):
        grid_index, outputs = rslt

        if self.temporary is None:
            # - for testing this pipeline with a parallel-pool, it might be
            #   convenient to add an option to force the contributions to
            #   be added in a deterministic order
            # - note: the order should always be deterministic in serial.

            ray_idx = self.subgrid_ray_map.rays_associated_with_subgrid(
                grid_index, pair_with_seqindx = False)
            ray_spectra = outputs
            for i,ray_ind in enumerate(ray_idx):
                self.single_sum_accum_2Darr[:,ray_ind] += ray_spectra[i,:]
        else:
            pairs = self.subgrid_ray_map.rays_associated_with_subgrid(
                grid_index, pair_with_seqindx = True)

            for i, (ray_ind, seqindx) in enumerate(pairs):
                # each entry in temporary corresponds to a distinct ray!
                # -> we are ordering the outputs for each ray so that the
                #    output for the nearest grid_index comes first (I think
                #    -- although, it's possible the order is reversed)
                self.temporary[ray_ind][seqindx] = outputs[i]

def generate_image(accum_strat, ray_collection, ds, *,
                   force_general_consolidation = False,
                   rescale_length_factor = None, pool = None):
    """
    Generate a mock ppv image (position-position-velocity image) of a
    simulation using a ray-casting radiative transfer algorithm.

    Parameters
    ----------
    accum_strat: AccumStratT
        Specifies the details of the imaging. NOTE: we may want not to have the
        user specify this directly.
    ray_collection:
        A collection of rays. This should be an instance of one of the classes
        defined in this package
    ds: `yt.data_objects.static_output.Dataset` or `SnapDatasetInitializer`
        The dataset or an object that initializes the dataset from which the
        image is constructed. If ds is the dataset, itself, this function
        cannot be parallelized (due to issues with pickling). Currently, the 
        dataset must be unigrid.
    force_general_consolidation: bool, Optional
        When `False` (the default), certain accumulation strategies may use
        a faster approach for consolidation. Note: the order of consolidation
        may vary based on the value of this parameter (so the exact values of
        the result may change slightly). This primarily exists for testing
        purposes.
    rescale_length_factor: float, Optional
        When not `None`, the width of each cell is multiplied by this factor.
        (This effectively holds the position of the simulation's origin fixed
        in place).
    pool: Optional
        A taskpool from the schwimmbad package can be specified for this
        argument to facillitate parallelization.

    Notes
    -----
    We assumed that v<<c. This lets us:
    - neglect the transverse redshift
    - approximate the longitudinal redshift as freq_obs = rest_freq*(1+vel/c)

    Todo
    ----
    - In the future, it would be nice to be able to specify a bulk velocity
      for the gas.
    - In the future, we might want to consider adding support for using a 
      Voigt line profile
    - Revisit treatment of velocity channels. I think we currently just
      compute the opacity at the nominal velocity of the channel. In reality, I
      think velocity channels are probably more like bins

    (It may make more sense to decouple these todo notes from this particular
    function...)
    """
    if rescale_length_factor is not None:
        if np.all(rescale_length_factor <= 0):
            raise ValueError("When specified, rescale_length_factor must be a "
                             "positive value")
        elif np.ndim(rescale_length_factor) != 0:
            raise ValueError("When specified, rescale_length_factor must be a "
                             "scalar value")
        assert float(rescale_length_factor) == rescale_length_factor
        rescale_length_factor = float(rescale_length_factor)
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
            if not isinstance(pool, _DummySerialPool):
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
        worker = Worker(ds_initializer = ds,
                        length_unit_name = length_unit_name,
                        cm_per_length_unit = float(length_unit_quan.to('cm').v),
                        rescale_length_factor = rescale_length_factor,
                        accum_strat = accum_strat)

        # create the output!
        rslt_props = accum_strat.get_rslt_props()
        
        out_dict = {}
        out_dict_2D = {} # this holds views of the entries in out_dict that
                         # have been flattened down to 2D
        for name, dtype, shape in rslt_props:
            assert isinstance(shape, tuple) and len(shape) == 1 # sanity check
            out_shape = shape + ray_collection.shape
            if accum_strat.simple_elementwise_sum_consolidate:
                out_dict[name] = np.zeros(shape = out_shape, dtype = dtype)
            else:
                fill_value = np.nan if dtype == np.float64 else 0
                out_dict[name] = np.full(shape = out_shape, dtype = dtype,
                                         fill_value = fill_value)

            out_dict_2D[name] = out_dict[name].view()
            out_dict_2D[name].shape = (shape[0], -1)

        # construct the callback functor that is passed the result of each call
        # to  `Worker.__call__` (the functor is invoked on the primary thread)
        callback = CallbackFunctor(
            accum_strat = accum_strat, out_dict_2D = out_dict_2D,
            subgrid_ray_map = subgrid_ray_map,
            force_general_mode = force_general_consolidation)

        
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
    print(f'begin raycasting -- start time: {_tstart.time()}')

    for elem in pool.map(worker, generator(), callback = callback):
        continue

    _tend = datetime.datetime.now()
    print('finished raycasting\n'
          f'-> start, end time: {_tstart.time()}, {_tend.time()}\n'
          f'-> elapsed time: {_tend - _tstart}')

    # consolidate the contents of callback.temporary to fill in out_dict_2D
    if callback.temporary is None:
        # do nothing -- this is a special case where the callback bypassed
        # callback.temporary and directly filled out_dict_2D
        pass
    else:
        # use the contents of temporary to fill in out_dict_2D
        temporary = callback.temporary

        _tstart_con = datetime.datetime.now()

        print(f'begin consolidation -- start time: {_tstart_con.time()}')
        for ray_ind, ray_rslts in enumerate(temporary):
            consolidated = accum_strat.consolidate(ray_rslts)
            for name, _, _ in rslt_props:
                out_dict_2D[name][...,ray_ind] = consolidated[name]

        _tend_con = datetime.datetime.now()
        print('finished consolidation\n'
              f'-> start, end time: {_tstart_con.time()}, {_tend_con.time()}\n'
              f'-> elapsed time: {_tend_con - _tstart_con}')

    # do any post-processing (this is where units will be attached)
    accum_strat.post_process_rslt(out_dict)

    # massage the output shape
    if ray_collection.shape == (1,):
        for name, _, shape in rslt_props:
            assert out_dict_2D[name].shape == (shape[0],1)
            out_dict[name] = np.squeeze(out_dict[name])
    return out_dict

def freq_from_v_channels(v_channels, line_props):
    if not _has_consistent_dims(v_channels, unyt.dimensions.velocity):
        raise ValueError("v_channels has the wrong units")
    rest_freq = line_props.freq_quantity
    return (rest_freq * (1 + v_channels/unyt.c_cgs)).to('Hz')

# define the legacy interface for backwards compatability!
def optically_thin_ppv(v_channels, ray_collection, ds,
                       ndens_HI_n1state = ('gas', 'H_p0_number_density'),
                       *, doppler_parameter_b = None,
                       use_cython_gen_spec = True,
                       force_general_consolidation = False,
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
    doppler_parameter_b: `unyt.unyt_quantity`, Optional
        Optional parameter that can be used to specify the Doppler parameter
        (aka Doppler Broadening parameter) assumed for every cell of the
        simulation. When not specified, this is computed from the local
        temperature (and mean-molecular-weight). To avoid any ambiguity, this
        quantity has units consistent with velocity, this quantity is commonly
        represented by the variable ``b``, and ``b/sqrt(2)`` specifies the
        standard deviation of the line-of-sight velocity component. Note that
        ``b * rest_freq / (unyt.c_cgs * sqrt(2))`` specifies the standard
        deviation of the line-profile for the transition that occurs at a rest
        frequency of ``rest_freq``.
    use_cython_gen_spec: bool, optional
        Generate the spectrum using the faster cython implementation. Default
        is True.
    force_general_consolidation: bool, Optional
        When `False` (the default), a faster consolidation strategy will be
        used. When `True`, a slower, but more general purpose strategy will be
        used. The order of consolidation may vary based on the value of this
        parameter (so the exact values of the result may change slightly)
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

    See Also
    --------
    generate_image: the underlying function that does most of the heavy lifting
    """

    # create the accum_strat
    line_props = default_spin_flip_props()
    accum_strat = OpticallyThinAccumStrat(
        obs_freq = freq_from_v_channels(v_channels, line_props),
        use_cython_gen_spec = use_cython_gen_spec,
        misc_kwargs = {
            'rest_freq' : line_props.freq_quantity,
            'doppler_parameter_b' : doppler_parameter_b,
            'ndens_HI_n1state_field' : ndens_HI_n1state,
        }
    )

    out_dict = generate_image(
        accum_strat = accum_strat, ray_collection = ray_collection, ds = ds,
        force_general_consolidation = force_general_consolidation,
        rescale_length_factor = rescale_length_factor, pool = pool
    )

    # pull the array out of the dictionary for backwards compatability
    if len(out_dict) != 1:
        raise RuntimeError("Sanity-check failed! generate_image is expected "
                           "to produce a dict with just one key-value pair "
                           f"when passed a {accum_strat.name} accum strategy")
    else:
        return _first_or_default(out_dict.values())
    

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
    The correctness of this calculation is not entirely clear. The main 
    question is: should we use a single frequency for all channels or a
    separate for each channel? Currently we do the latter.
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

def convert_Tb_to_intensity(ppv_Tb, v_channels,
                            rest_freq = 1.4204058E+09*unyt.Hz):
    """
    Convert brightness temperature ppv array to brightness temperature

    This is the inverse of ``convert_intensity_to_Tb``.
    """
    if not _has_consistent_dims(ppv_Tb, unyt.dimensions.temperature):
        raise ValueError("ppv_Tb has the wrong units")
    elif not _has_consistent_dims(rest_freq, unyt.dimensions.frequency):
        raise ValueError("rest_freq doesn't have appropriate dimensions")

    if v_channels is None:
        freq = rest_freq
    elif np.shape(v_channels) != np.shape(ppv_Tb)[:1]:
        raise ValueError("when not None, the number of v_channels should "
                         "match np.shape(ppv_Tb)[0]")
    elif not _has_consistent_dims(v_channels, unyt.dimensions.velocity):
        raise ValueError("v_channels has the wrong units")
    else:
        obs_freq = (rest_freq*(1+v_channels/unyt.c_cgs)).to('Hz')
        freq = obs_freq
        new_shape = [1 for elem in ppv_Tb.shape]
        new_shape[0] = v_channels.size
        freq.shape = tuple(new_shape)

    out = 2 * ppv_Tb * unyt.kboltz_cgs * (freq/unyt.c_cgs)**2 / unyt.steradian
    return out.to('erg/(cm**2 * Hz * s * steradian)')
