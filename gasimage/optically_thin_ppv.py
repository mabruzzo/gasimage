import numpy as np
import yt

import schwimmbad

import gc

from ._ray_intersections_cy import ray_box_intersections, traverse_grid
from ._generate_spec_cy import _generate_ray_spectrum_cy

_inv_sqrt_pi = 1.0/np.sqrt(np.pi)

def line_profile(obs_freq, doppler_v_width, rest_freq,
                 velocity_offset = None):
    # freq_arr is a 1D array
    # doppler_v_width is a 1D array (it may have a different
    # shape from freq_arr)
    # rest_freq is a scalar

    # temp is equivalent to 1./(sqrt(2) * sigma)
    temp = (
        unyt.c_cgs/(rest_freq*doppler_v_width)
    )

    norm = _inv_sqrt_pi * temp
    half_div_sigma2 = temp*temp
    
    # need to check this correction!
    if velocity_offset is None:
        emit_freq = obs_freq[:,np.newaxis]
    else:
        assert velocity_offset.shape == doppler_v_width.shape
        v_div_c_plus_1 = 1 + velocity_offset/unyt.c_cgs
        emit_freq = obs_freq[:,np.newaxis]/(v_div_c_plus_1.to('dimensionless').v)

    delta_nu = (emit_freq - rest_freq)
    delta_nu_sq = delta_nu*delta_nu
    exponent = (-1*delta_nu_sq*half_div_sigma2)

    return norm*np.exp(exponent.to('dimensionless').v)

def _generate_ray_spectrum_py(obs_freq, velocities, ndens_HI,
                              doppler_v_width, rest_freq, dz,
                              out = None):
    _A10 = 2.85e-15*unyt.Hz
    n1 = 0.75*ndens_HI # need spin temperature to be more exact
    profiles = line_profile(obs_freq = obs_freq,
                            doppler_v_width = doppler_v_width,
                            rest_freq = rest_freq,
                            velocity_offset = velocities)
    j_nu = unyt.h_cgs * rest_freq *n1* _A10 * profiles/(4*np.pi)
    integrated = (j_nu*dz).sum(axis=1)

    if True:
        # need to think more about the units
        # there may be an implicit dependence on the solid angle
        if out is not None:
            out[:] = integrated.to('erg/cm**2').v
            return out
        else:
            return integrated.to('erg/cm**2').v
    if False:
        n0 = 0.25*ndens_HI
        g1_div_g0 = 3
        rest_wave = unyt.c_cgs/rest_freq
        Tspin = 100.0*unyt.K
        stim_correct = 1.0-np.exp(-0.0682/Tspin.to('K').v)
        alpha_nu = n0*g1_div_g0*_A10*rest_wave**2 * stim_correct * profiles/(8*np.pi)
        optical_depth = (alpha_nu*dz).sum(axis=1)
        assert out is None
        return integrated,optical_depth

def _calc_doppler_v_width(grid,idx):
    return np.sqrt(2*unyt.kb_cgs * grid['temperature'][idx]/
                  (grid['mean_molecular_weight'][idx]*unyt.mh_cgs))

def generate_ray_spectrum(grid, grid_left_edge, grid_right_edge,
                          cell_width, grid_shape, cm_per_length_unit,
                          ray_start, ray_uvec,
                          rest_freq, obs_freq, doppler_v_width = None,
                          ndens_HI_field = ('gas', 'H_p0_number_density'),
                          use_cython_gen_spec = False,
                          out = None):

    # do NOT use grid to access length-scale information. This will really mess
    # some things up related to rescale_length_factor

    if out is not None:
        assert out.shape == obs_freq.shape
    else:
        out = np.empty_like(obs_freq)
    assert str(obs_freq.units) == 'Hz'

    vx_field,vy_field,vz_field = (
        ('gas','velocity_x'), ('gas','velocity_y'), ('gas','velocity_z')
    )


    if True:
        try:
            tmp_idx, dz = traverse_grid(
                line_uvec = ray_uvec,
                line_start = ray_start,
                grid_left_edge = grid_left_edge,
                cell_width = cell_width,
                grid_shape = grid_shape
            )
        except:
            print('There was a problem!')
            pairs = [('line_uvec', ray_uvec),
                     ('line_start', ray_start),
                     ('grid_left_edge', grid_left_edge),
                     ('cell_width', cell_width)]
            for name, arr in pairs:
                arr_str = np.array2string(arr, floatmode = 'unique')
                print(f'{name} = {arr_str}')
            print(f'grid_shape = {np.array2string(np.array(grid_shape))}')
            raise

        idx = (tmp_idx[0], tmp_idx[1], tmp_idx[2])

        # convert dz to cm to avoid problems later
        dz *= cm_per_length_unit
        dz = yt.YTArray(dz, 'cm')

        # compute the velocity component. We should probably confirm
        # correctness of the velocity sign
        vlos = (ray_uvec[0] * grid[vx_field][idx] +
                ray_uvec[1] * grid[vy_field][idx] +
                ray_uvec[2] * grid[vz_field][idx]).to('cm/s')

        if doppler_v_width is None:
            # it might be more sensible to make doppler_v_width into a field
            cur_doppler_v_width = _calc_doppler_v_width(grid,idx).to('cm/s')
        else:
            cur_doppler_v_width = doppler_v_width.to('cm/s')

        ndens_HI = grid[ndens_HI_field][idx].to('cm**-3')

        if use_cython_gen_spec:
            # There were bugs in this version of the method
            raise RuntimeError(
                "The cythonized version has not been properly tested (it may "
                "not return correct results)"
            )
            _generate_ray_spectrum_cy(
                obs_freq = obs_freq.ndarray_view(),
                velocities = vlos.ndarray_view(),
                ndens_HI = ndens_HI.ndarray_view(),
                doppler_v_width = cur_doppler_v_width.ndarray_view(), 
                rest_freq = float(rest_freq.v),
                dz = dz.ndarray_view(),
                out = out)
        else:
            out[:] = _generate_ray_spectrum_py(
                obs_freq = obs_freq, velocities = vlos, 
                ndens_HI = grid[ndens_HI_field][idx],
                doppler_v_width = cur_doppler_v_width, 
                rest_freq = rest_freq, dz = dz)
    return out

class Worker:
    def __init__(self, ds_initializer, obs_freq, length_unit_name,
                 ray_start, rescale_length_factor,
                 generate_ray_spectrum_kwargs):
        assert np.ndim(obs_freq) == 1

        self.ds_initializer = ds_initializer
        self.obs_freq = obs_freq
        self.length_unit_name = length_unit_name
        self.ray_start = ray_start
        self.rescale_length_factor = rescale_length_factor

        for key in ['obs_freq', 'grid', 'grid_left_edge', 'grid_right_edge',
                    'cell_width', 'grid_shape', 'out', 'ray_start', 'ray_uvec']:
            if key in generate_ray_spectrum_kwargs:
                raise ValueError(f"'{key}' should not be a key of "
                                 "generate_ray_spectrum_kwargs")
        self.generate_ray_spectrum_kwargs = generate_ray_spectrum_kwargs


    def __call__(self, elem):
        grid_index, ray_stop_locs = elem
        assert (ray_stop_locs.ndim == 2) and ray_stop_locs.shape[1] == 3

        out = np.empty(shape = (ray_stop_locs.shape[0], self.obs_freq.size),
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
        for i in range(ray_stop_locs.shape[0]):
            cur_ray_stop = ray_stop_locs[i,:]

            ray_vec = cur_ray_stop - self.ray_start
            ray_uvec = (ray_vec/np.sqrt((ray_vec*ray_vec).sum()))

            # generate the spectrum
            generate_ray_spectrum(
                grid, grid_left_edge = left_edge, grid_right_edge = right_edge,
                cell_width = cell_width, grid_shape = grid_shape,
                ray_start = self.ray_start, ray_uvec = ray_uvec,
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
    # Helper class that keeps track of which root-grids that each ray
    # intersects with.
    def __init__(self, ds, ray_start, ray_stop_l, units,
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

        for i, cur_ray_stop in enumerate(ray_stop_l):
            ray_vec = cur_ray_stop - ray_start
            ray_uvec = (ray_vec/np.sqrt((ray_vec*ray_vec).sum()))

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
        print(len(self._sequence_table))
        
    def rays_associated_with_subgrid(self, subgrid_id):
        out = []
        if subgrid_id not in self._subgrids_with_rays:
            return out
        for subgrid_sequence, ray_ids in self._sequence_table.items():
            if subgrid_id in subgrid_sequence:
                out = out + ray_ids
        return out


def optically_thin_ppv(v_channels, ray_start, ray_stop, ds,
                       ndens_HI_field = ('gas', 'H_p0_number_density'),
                       doppler_v_width = None,
                       use_cython_gen_spec = False,
                       rescale_length_factor = None,
                       pool = None):
    """
    Generate a mock ppv image of a simulation using a ray-tracing radiative
    transfer algorithm that assumes that the gas is optically thin.

    Parameters
    ----------
    v_channels: 1D `unyt.unyt_array`
        A monotonically increasing array of velocity channels.
    ray_start: 1D `unyt.unyt_array`
        A (3,) array that specifies the observer's location (with respect to
        the dataset's coordinate system). Currently, this must be located 
        outside of the simulation domain.
    ray_stop: 3D `unyt.unyt_array`
        A (m,n,3) array specifying the location where each ray stops.
    ds: `yt.data_objects.static_output.Dataset` or `SnapDatasetInitializer`
        The dataset or an object that initializes the dataset from which the
        image is constructed. If ds is the dataset, itself, this function
        cannot be parallelized (due to issues with pickling). Currently, the 
        dataset must be unigrid.
    ndens_HI_field
        The name of the field holding the number density of H I atoms
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

    if pool is None:
        pool = schwimmbad.SerialPool()

    is_root = (not hasattr(pool,'is_master')) or pool.is_master()

    if is_root:
        # now get an instance of the dataset
        if isinstance(ds, yt.data_objects.static_output.Dataset):
            if not isinstance(pool, schwimmbad.SerialPool):
                raise ValueError("When ds is a dataset object, pool must be "
                                 "assigned a schwimmbad.SerialPool instance.")
            else:
                my_ds = ds
        else:
            my_ds = ds()

        _l = rescale_length_factor * my_ds.domain_left_edge
        _r = rescale_length_factor * my_ds.domain_right_edge
        if np.logical_and(ray_start >= _l, ray_start <= _r).all():
            raise RuntimeError('We can potentially relax this in the future.')
    
        # use the code units (since the cell widths are usually better behaved)
        length_unit_name = 'code_length'
        length_unit_quan = my_ds.quan(1.0, 'code_length')

        # there is no need to rescale ray_stop
        ray_start = ray_start.to('cm').v /length_unit_quan.to('cm').v
        ray_stop = ray_stop.to('cm').v / length_unit_quan.to('cm').v
        assert ray_start.shape == (3,)

        # create the iterator
        assert ray_stop.shape[-1] == 3
        ray_stop_2D = ray_stop.view()
        ray_stop_2D.shape = (-1,3)
        assert ray_stop_2D.flags['C_CONTIGUOUS']

        print('Constructing RayGridAssignments')
        subgrid_ray_map = RayGridAssignments(
            my_ds, ray_start, ray_stop_l = ray_stop_2D,
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

                ray_stop_loc = np.copy(ray_stop_2D[ray_idx,:])
                yield grid_ind, ray_stop_loc


        # create the worker
        rest_freq = 1.4204058E+09*unyt.Hz,
        worker = Worker(
            ds_initializer = ds,
            obs_freq = (rest_freq*(1+v_channels/unyt.c_cgs)).to('Hz'),
            length_unit_name = length_unit_name,
            ray_start = ray_start,
            rescale_length_factor = rescale_length_factor,
            generate_ray_spectrum_kwargs = {
                'rest_freq' : rest_freq,
                'cm_per_length_unit' : length_unit_quan.to('cm').v,
                'doppler_v_width' : doppler_v_width,
                'ndens_HI_field' : ndens_HI_field,
                'use_cython_gen_spec' : use_cython_gen_spec,
            }
        )
        

        # create the output array and callback function
        out_shape = (v_channels.size,) + ray_stop.shape[:-1]
        out = np.zeros(shape = out_shape)

        # flatten the out array down to 2-dimensions:
        out_2D = out.view()
        out_2D.shape = (v_channels.size, -1)

        def callback(rslt):
            grid_index, ray_spectra = rslt
            ray_idx = subgrid_ray_map.rays_associated_with_subgrid(grid_index)
            for i,ray_ind in enumerate(ray_idx):
                out_2D[:,ray_ind] += ray_spectra[i,:]
    else:
        generator = None
        worker = None
        callback = None
        out = None

    print('begin processing')
    for elem in pool.map(worker, generator(), callback = callback):
        continue
    print('finished processing')

    # need to think more about the about output units (specifically,
    # think about dependence on solid angle)
    out_units = 'erg/cm**2'
    if np.ndim(ray_stop) == 1:
        assert out_2D.shape == (v_channels.size, 1)
        return yt.YTArray(out_2D[:,0], out_units)
    else:
        return yt.YTArray(out, out_units)
    

def convert_intensity_to_Tb(ppv, rest_freq = 1.4204058E+09*unyt.Hz,
                            v_channels = None):
    if v_channels is not None:
        # assume that velocity << c
        obs_freq = (rest_freq*(1+v_channels/unyt.c_cgs)).to('Hz')
        freq = obs_freq
        assert v_channels.shape == ppv.shape[:1]
        new_shape = [1 for elem in ppv.shape]
        new_shape[0] = v_channels.size
        freq.shape = tuple(new_shape)

    else:
        freq = rest_freq

    return (0.5*(unyt.c_cgs/freq)**2 * ppv/unyt.kboltz_cgs).to('K')
