import numpy as np
import yt

import gc

from .ray_intersections import ray_box_intersections, traverse_grid

_inv_sqrt_pi = 1.0/np.sqrt(np.pi)

def line_profile(obs_freq, doppler_v_width, rest_freq,
                 velocity_offset = None):
    # freq_arr is a 1D array
    # doppler_v_width is a 1D array (it may have a different
    # shape from freq_arr)
    # rest_freq is a scalar

    # temp is equivalent to 1./(sqrt(2) * sigma)
    temp = (
        yt.units.c_cgs/(rest_freq*doppler_v_width)
    )

    norm = _inv_sqrt_pi * temp
    half_div_sigma2 = temp*temp
    
    # need to check this correction!
    if velocity_offset is None:
        emit_freq = obs_freq[:,np.newaxis]
    else:
        assert velocity_offset.shape == doppler_v_width.shape
        v_div_c_plus_1 = 1 + velocity_offset/yt.units.c_cgs
        emit_freq = obs_freq[:,np.newaxis]/(v_div_c_plus_1.to('dimensionless').v)

    delta_nu = (emit_freq - rest_freq)
    delta_nu_sq = delta_nu*delta_nu
    exponent = (-1*delta_nu_sq*half_div_sigma2)

    return norm*np.exp(exponent.to('dimensionless').v)

def _generate_spectrum(obs_freq, velocities, ndens_HI,
                       doppler_v_width, rest_freq, dz):
    _A10 = 2.85e-15*yt.units.Hz
    n1 = 0.75*ndens_HI # need spin temperature to be more exact
    profiles = line_profile(obs_freq = obs_freq,
                            doppler_v_width = doppler_v_width,
                            rest_freq = rest_freq,
                            velocity_offset = velocities)
    j_nu = yt.units.h_cgs * rest_freq *n1* _A10 * profiles/(4*np.pi)
    integrated = (j_nu*dz).sum(axis=1)

    if True:
        # need to think more about the units
        # there may be an implicit dependence on the solid angle
        return integrated.to('erg/cm**2').v
    if False:
        n0 = 0.25*ndens_HI
        g1_div_g0 = 3
        rest_wave = yt.units.c_cgs/rest_freq
        Tspin = 100.0*yt.units.K
        stim_correct = 1.0-np.exp(-0.0682/Tspin.to('K').v)
        alpha_nu = n0*g1_div_g0*_A10*rest_wave**2 * stim_correct * profiles/(8*np.pi)
        optical_depth = (alpha_nu*dz).sum(axis=1)
        return integrated,optical_depth

def _calc_doppler_v_width(grid,idx):
    return np.sqrt(2*yt.units.kb * grid['temperature'][idx]/
                  (grid['mean_molecular_weight'][idx]*yt.units.mh))

def generate_ray_spectrum(grid, grid_left_edge, grid_right_edge,
                          cell_width, grid_shape, length_unit,
                          ray_start, ray_uvec,
                          rest_freq, obs_freq, doppler_v_width = None,
                          ndens_HI_field = ('gas', 'H_p0_number_density'),
                          out = None):
    if out is not None:
        assert out.shape == obs_freq.shape
    else:
        out = np.empty_like(rest_freq)
    
    vx_field,vy_field,vz_field = (
        ('gas','velocity_x'), ('gas','velocity_y'), ('gas','velocity_z')
    )

    #print(grid_left_edge, grid_right_edge, ray_start, ray_uvec)
    intersect_dist = ray_box_intersections(
        line_start = ray_start, line_uvec = ray_uvec, 
        left_edge = grid_left_edge, 
        right_edge = grid_right_edge)
    #print(intersect_dist)
    if len(intersect_dist) < 2:
        out[:] = 0.0
    else:
        try:
            tmp_idx, dz = traverse_grid(
                line_uvec = ray_uvec,
                line_start = ray_start,
                grid_left_edge = grid_left_edge,
                cell_width = cell_width,
                grid_shape = grid_shape)
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
            #out[:] = np.nan
            #return out

        idx = (tmp_idx[0], tmp_idx[1], tmp_idx[2])
        #print(idx)

        # convert dz to cm to avoid problems later
        dz = dz * length_unit.to('cm')

        # compute the velocity component. We should probably confirm
        # correctness of the velocity sign
        vlos = (ray_uvec[0] * grid[vx_field][idx] +
                ray_uvec[1] * grid[vy_field][idx] +
                ray_uvec[2] * grid[vz_field][idx])

        if doppler_v_width is None:
            # it would probably be more sensible to make doppler_v_width
            # into a field
            cur_doppler_v_width = _calc_doppler_v_width(grid,idx)
        else:
            cur_doppler_v_width = doppler_v_width

        # we should come back to this and handle it properly in the future
        out[:] = _generate_spectrum(obs_freq = obs_freq,
                                    velocities = vlos, 
                                    ndens_HI = grid[ndens_HI_field][idx],
                                    doppler_v_width = cur_doppler_v_width, 
                                    rest_freq = rest_freq, 
                                    dz = dz)
    return out

def optically_thin_ppv(v_channels, ray_start, ray_stop, ds,
                       ndens_HI_field = ('gas', 'H_p0_number_density'),
                       doppler_v_width = None):
    """


    Notes
    -----
    We assumed that v<<c. This let's us:
    - neglect the transverse redshift
    - approximate the longitudinal redshift as freq_obs = rest_freq*(1+vel/c)

    In the future, it would be nice to be able to specify a bulk velocity for 
    the gas.
    """
    if np.logical_and(ray_start >= ds.domain_left_edge,
                      ray_start <= ds.domain_right_edge).all():
        raise RuntimeError()
    
    # use the code units (since the cell widths are usually better behaved)
    length_unit_name = 'code_length'
    length_unit_quan = ds.quan(1.0, 'code_length')

    ray_start = ray_start.to('cm').v /length_unit_quan.to('cm').v
    ray_stop = ray_stop.to('cm').v / length_unit_quan.to('cm').v
    assert ray_start.shape == (3,)

    assert ray_stop.shape[-1] == 3
    ray_stop_2D = ray_stop.view()
    ray_stop_2D.shape = (-1,3)
    assert ray_stop_2D.flags['C_CONTIGUOUS']

    # create the output array
    out_shape = (v_channels.size,) + ray_stop.shape[:-1]
    out = np.zeros(shape = out_shape)

    # flatten the out array down to 2-dimensions:
    out_2D = out.view()
    out_2D.shape = (v_channels.size, -1)

    rest_freq = 1.4204058E+09*yt.units.Hz
    obs_freq = (rest_freq*(1+v_channels/yt.units.c_cgs)).to('Hz')

    # this is a contiguous array where the spectrum for a single
    # ray gets stored
    _temp_buffer = np.empty(shape = (v_channels.size,),
                            dtype = np.float64)
    for grid_ind in range(ds.index.grids.size):
        grid = ds.index.grids[grid_ind]
        print(grid, grid_ind)
        grid_shape = ds.index.grid_dimensions[grid_ind]
        left_edge = ds.index.grid_left_edge[grid_ind].to(length_unit_name).v
        right_edge = ds.index.grid_right_edge[grid_ind].to(length_unit_name).v
        cell_width = ((right_edge - left_edge)/np.array(grid.shape))

        for i, cur_ray_stop in enumerate(ray_stop_2D):

            ray_vec = cur_ray_stop - ray_start
            ray_uvec = (ray_vec/np.sqrt((ray_vec*ray_vec).sum()))

            # generate the spectrum
            generate_ray_spectrum(grid, grid_left_edge = left_edge,
                                  grid_right_edge = right_edge,
                                  cell_width = cell_width, 
                                  grid_shape = grid_shape,
                                  ray_start =ray_start, ray_uvec = ray_uvec,
                                  length_unit = length_unit_quan,
                                  rest_freq = rest_freq,
                                  obs_freq = obs_freq,
                                  doppler_v_width = doppler_v_width,
                                  ndens_HI_field = ndens_HI_field,
                                  out = _temp_buffer)
            out_2D[:,i] += _temp_buffer
        
        grid.clear_data()
        del grid
        gc.collect()

    # need to think more about the about output units (specifically,
    # think about dependence on solid angle)
    out_units = 'erg/cm**2'
    if np.ndim(ray_stop) == 1:
        assert out_2D.shape == (v_channels.size, 1)
        return yt.YTArray(out_2D[:,0], out_units)
    else:
        return yt.YTArray(out, out_units)

def convert_intensity_to_Tb(ppv, rest_freq = 1.4204058E+09*yt.units.Hz,
                            v_channels = None):
    if v_channels is not None:
        # assume that velocity << c
        obs_freq = (rest_freq*(1+v_channels/yt.units.c_cgs)).to('Hz')
        freq = obs_freq
        assert v_channels.shape == ppv.shape[:1]
        new_shape = [1 for elem in ppv.shape]
        new_shape[0] = v_channels.size
        freq.shape = tuple(new_shape)

    else:
        freq = rest_freq

    return (0.5*(yt.units.c_cgs/freq)**2 * 
            ppv/yt.units.kboltz_cgs).to('K')
