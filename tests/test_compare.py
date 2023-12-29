import os.path
import datetime

import numpy as np
import unyt
import yt


from gasimage._ray_intersections_cy import ray_box_intersections
from gasimage.generate_image import freq_from_v_channels, generate_image

from gasimage.accumulators import NoScatterRTAccumStrat
from gasimage.rt_config import builtin_halpha_props, crude_H_partition_func
from gasimage.ray_collection import ConcreteRayList
from gasimage.utils.testing import assert_allclose_units

from py_generate_noscatter_spectrum import (
    _generate_noscatter_spectrum as py_generate_noscatter_spectrum
)
from test_full import _dummy_create_field_callback

def ray_values_startend(ds, start_end_pairs, fields = [],
                        find_indices = False):

    # even if fields is empty, it always contains an entry call dl, which is
    # the length through a cell

    max_level = np.amax(ds.index.grid_levels)

    if find_indices:
        grid_left_edge = ds.domain_left_edge.to('code_length').ndview
        cell_width = (
            ds.domain_width/ds.domain_dimensions).to('code_length').ndview
    else:
        grid_left_edge, cell_width = None, None

    for start,end in start_end_pairs:
        cur_fields = {}
        start, end = np.asanyarray(start), np.asanyarray(end)
        total_length = np.linalg.norm(end - start)

        num_matching_components = np.count_nonzero(start == end)
        if (num_matching_components == 3) or (total_length == 0.0):
            raise AssertionError("START AND END POINTS ARE COLOCATED")
        elif False and (num_matching_components == 2):
            if start[0] != end[0]:
                cast_ax, plane_coord = 0, (start[1],start[2])
            elif start[1] != end[1]:
                cast_ax, plane_coord = 1, (start[2],start[0])
            else:
                cast_ax, plane_coord = 2, (start[0],start[1])
            ray = ds.ortho_ray(axis = cast_ax, coords = plane_coord)

            l,r = ds.domain_left_edge[cast_ax], ds.domain_right_edge[cast_ax]
            if isinstance(l, unyt.unyt_array):
                l = l.to('code_length').v
            if isinstance(l, unyt.unyt_array):
                r = r.to('code_length').v
            assert l >= min(start[cast_ax], end[cast_ax])
            assert r <= max(start[cast_ax], end[cast_ax])

            if max_level != 0:
                raise RuntimeError("Problems may arise with AMR")

            pos_fields = ['x','y','z']
            sorted_idx = np.argsort(ray[pos_fields[cast_ax]])
            dl = ray[['dx','dy','dz'][cast_ax]]

            #cur_fields
            #raise RuntimeError(list(zip(ray['dx'],ray['dy'],ray['dz'])))
        else:
            ray = ds.ray(start, end)

            pos_fields = [('index', ax) for ax in 'xyz'] 
            sorted_idx = np.argsort(ray["t"])
            dl = ray['dts'][sorted_idx] * total_length

        if isinstance(dl, unyt.unyt_array):
            if dl.units.is_dimensionless:
                dl = dl.ndview
            else:
                dl = dl.to('code_length')
        for field in fields:
            cur_fields[field] = ray[field][sorted_idx]
        cur_fields['dl'] = dl

        if find_indices:
            indices = np.empty(shape = (3, sorted_idx.size), dtype = int)
            for i, pos_field in enumerate(pos_fields):
                left_offset = (
                    ray[pos_field][sorted_idx].to('code_length').ndview -
                    grid_left_edge[i] - 0.5 * cell_width[i])
                indices[i,:] = np.trunc(
                    left_offset/cell_width[i] + 0.5).astype(int)
            cur_fields['indices'] = indices
        yield cur_fields

def _zip_vectors(a,b):
    arg_shapes = []
    for arg in [a,b]:
        arg_shape = np.shape(arg)
        arg_ndim = len(arg_shape) # == np.ndim(arg)
        if arg_ndim != 1 and arg_ndim != 2:
            raise ValueError("each argument must be 1d or 2d")
        elif arg_shape[-1] != 3:
            raise ValueError("the last axis of each argument must have a "
                             "length of 3")
        arg_shapes.append(arg_shape)

    if np.ndim(a) == 1 and np.ndim(b) == 1:
        yield a,b
    elif np.ndim(a) == 1:
        for b_elem in b:
            yield a,b_elem
    elif np.ndim(b) == 1:
        for a_elem in a:
            yield a_elem, b
    elif arg_shapes[0] != arg_shapes[1]:
        raise ValueError("when both arguments are 2d, they must have "
                         "identical shapes") 
    else:
        yield from zip(a, b)

def _to_uvec(vec):
    vec = np.asanyarray(vec)
    if (vec == 0).all():
        raise ValueError()
    return vec / np.linalg.norm(vec)

def ray_values(ds, ray_start, ray_vec, fields = [],
               find_indices = False, include_uvec = False):

    # even if fields is empty, it always contains an entry call dl, which is
    # the length through a cell

    left_edge, right_edge = ds.domain_left_edge, ds.domain_right_edge

    pairs = []
    uvec_l = []
    for cur_ray_start, cur_ray_vec in _zip_vectors(ray_start, ray_vec):
        uvec = _to_uvec(cur_ray_vec)

        intersect_dists = ray_box_intersections(cur_ray_start, uvec,
                                                left_edge = left_edge,
                                                right_edge = right_edge)
        assert len(intersect_dists) > 0

        # intentionally overshoot the back edge of the grid!
        cur_ray_stop = cur_ray_start + uvec * 1.2 * np.amax(intersect_dists)
        pairs.append((cur_ray_start, cur_ray_stop))

    itr = ray_values_startend(ds, start_end_pairs = pairs, fields = fields,
                              find_indices = find_indices)

    yield from itr

def _get_ray_start_vec(ds):
    start = ds.arr(
        [-1.17144820e+07, -3.82624020e+22, -2.34289641e+06], 'cm'
    ).to('code_length').ndview

    stop = ds.arr(
        [[[-1.14763702e-01,  1.39017361e+01,  8.51251161e-16],
          [-1.30065396e-01,  1.39016649e+01,  8.51251161e-16],
          [-1.45367046e-01,  1.39015847e+01,  8.51251161e-16],
          [-1.60668647e-01,  1.39014957e+01,  8.51251161e-16],
          [-1.75970194e-01,  1.39013978e+01,  8.51251161e-16],
          [-1.91271681e-01,  1.39012910e+01,  8.51251161e-16],
          [-2.06573103e-01,  1.39011752e+01,  8.51251161e-16],
          [-2.21874455e-01,  1.39010506e+01,  8.51251161e-16]],

         [[-1.14763682e-01,  1.39017316e+01,  1.53018593e-02],
          [-1.30065374e-01,  1.39016604e+01,  1.53018593e-02],
          [-1.45367022e-01,  1.39015803e+01,  1.53018593e-02],
          [-1.60668620e-01,  1.39014913e+01,  1.53018593e-02],
          [-1.75970164e-01,  1.39013933e+01,  1.53018593e-02],
          [-1.91271648e-01,  1.39012865e+01,  1.53018593e-02],
          [-2.06573068e-01,  1.39011708e+01,  1.53018593e-02],
          [-2.21874418e-01,  1.39010462e+01,  1.53018593e-02]],

         [[-1.14763624e-01,  1.39017183e+01,  3.06037134e-02],
          [-1.30065308e-01,  1.39016471e+01,  3.06037134e-02],
          [-1.45366948e-01,  1.39015669e+01,  3.06037134e-02],
          [-1.60668538e-01,  1.39014779e+01,  3.06037134e-02],
          [-1.75970075e-01,  1.39013800e+01,  3.06037134e-02],
          [-1.91271551e-01,  1.39012732e+01,  3.06037134e-02],
          [-2.06572963e-01,  1.39011574e+01,  3.06037134e-02],
          [-2.21874305e-01,  1.39010328e+01,  3.06037134e-02]],

         [[-1.14763527e-01,  1.39016960e+01,  4.59055571e-02],
          [-1.30065198e-01,  1.39016248e+01,  4.59055571e-02],
          [-1.45366825e-01,  1.39015447e+01,  4.59055571e-02],
          [-1.60668402e-01,  1.39014557e+01,  4.59055571e-02],
          [-1.75969926e-01,  1.39013577e+01,  4.59055571e-02],
          [-1.91271389e-01,  1.39012509e+01,  4.59055571e-02],
          [-2.06572788e-01,  1.39011352e+01,  4.59055571e-02],
          [-2.21874117e-01,  1.39010106e+01,  4.59055571e-02]],

         [[-1.14763391e-01,  1.39016649e+01,  6.12073854e-02],
          [-1.30065044e-01,  1.39015936e+01,  6.12073854e-02],
          [-1.45366652e-01,  1.39015135e+01,  6.12073854e-02],
          [-1.60668212e-01,  1.39014245e+01,  6.12073854e-02],
          [-1.75969717e-01,  1.39013266e+01,  6.12073854e-02],
          [-1.91271163e-01,  1.39012198e+01,  6.12073854e-02],
          [-2.06572544e-01,  1.39011040e+01,  6.12073854e-02],
          [-2.21873855e-01,  1.39009794e+01,  6.12073854e-02]],

         [[-1.14763216e-01,  1.39016248e+01,  7.65091928e-02],
          [-1.30064846e-01,  1.39015536e+01,  7.65091928e-02],
          [-1.45366431e-01,  1.39014735e+01,  7.65091928e-02],
          [-1.60667967e-01,  1.39013844e+01,  7.65091928e-02],
          [-1.75969449e-01,  1.39012865e+01,  7.65091928e-02],
          [-1.91270871e-01,  1.39011797e+01,  7.65091928e-02],
          [-2.06572229e-01,  1.39010640e+01,  7.65091928e-02],
          [-2.21873517e-01,  1.39009393e+01,  7.65091928e-02]],

         [[-1.14763003e-01,  1.39015758e+01,  9.18109744e-02],
          [-1.30064604e-01,  1.39015046e+01,  9.18109744e-02],
          [-1.45366160e-01,  1.39014245e+01,  9.18109744e-02],
          [-1.60667668e-01,  1.39013355e+01,  9.18109744e-02],
          [-1.75969122e-01,  1.39012376e+01,  9.18109744e-02],
          [-1.91270515e-01,  1.39011307e+01,  9.18109744e-02],
          [-2.06571845e-01,  1.39010150e+01,  9.18109744e-02],
          [-2.21873104e-01,  1.39008904e+01,  9.18109744e-02]],

         [[-1.14762750e-01,  1.39015180e+01,  1.07112725e-01],
          [-1.30064317e-01,  1.39014468e+01,  1.07112725e-01],
          [-1.45365841e-01,  1.39013666e+01,  1.07112725e-01],
          [-1.60667315e-01,  1.39012776e+01,  1.07112725e-01],
          [-1.75968734e-01,  1.39011797e+01,  1.07112725e-01],
          [-1.91270095e-01,  1.39010729e+01,  1.07112725e-01],
          [-2.06571390e-01,  1.39009571e+01,  1.07112725e-01],
          [-2.21872616e-01,  1.39008325e+01,  1.07112725e-01]]], 'kpc'
    ).to('code_length').ndview[:2,:2,:]

    stop = stop.reshape(-1,3)

    vec = stop - np.broadcast_to(start,shape = stop.shape)
    #print(vec)
    #print()
    #print(np.linalg.norm(vec, ord = 2, axis = 1))
    return start, vec

def _get_ray_start_vec2(ds):

    start = np.array([[-60.0,  +0.125, +0.125],
                      [+60.0,  +0.125, +0.125],
                      [+0.125,-20.0,   +0.125],
                      [+0.125,+20.0,   +0.125],
                      [+0.125, +0.125,-20.0],
                      [+0.125, +0.125,+20.0]])
    vec = np.array([[+1.0,  0.0,  0.0],
                    [-1.0,  0.0,  0.0],
                    [ 0.0, +1.0,  0.0],
                    [ 0.0, -1.0,  0.0],
                    [ 0.0,  0.0, +1.0],
                    [ 0.0,  0.0, -1.0]])

    idx = slice(2,None) # we clip the 1st 2 entries because it roughly triples
                        # the test's duration
    return start[idx,:], vec[idx,:]


from debug_plotting import plot_ray_spectrum, plot_rel_err

# We are going to setup 2

def test_compare_full_noscatter_rt(indata_dir):
    enzoe_sim_path = os.path.join(
        indata_dir, ('X100_M1.5_HD_CDdftCstr_R56.38_logP3_Res8/cloud_07.5000/'
                     'cloud_07.5000.block_list')
    )

    line_props = builtin_halpha_props()
    partition_func = crude_H_partition_func(electronic_state_max = 20)

    v_channels = unyt.unyt_array(np.arange(-170,180,0.736125), 'km/s')
    obs_freq = freq_from_v_channels(v_channels, line_props = line_props)

    accum_strat = NoScatterRTAccumStrat(
        obs_freq = obs_freq,
        line_props = line_props,
        species_mass_g = unyt.mh_cgs.to('g').v,
        partition_func = partition_func,
        ndens_field = ('gas', 'H_p0_number_density')
    )

    ds = yt.load(enzoe_sim_path)
    _dummy_create_field_callback(ds, use_trident_ion = True)

    # the errors are much larger when we use _get_ray_start_vec(ds)
    # -> I suspect this is because the rays in that case are on angles and
    #    there is a small amount of disagreement with what yt returns... Plus
    #    the distances are less-accurate with the YTRay appraoch
    # -> it's worth noting that the discrepancies seem largest furthest away
    #    from the line-center, which is actually comforting...
    start, vec = _get_ray_start_vec2(ds)
    if start.size < vec.size:
        start = np.tile(start, (vec.shape[0],1))
    elif start.size > vec.size:
        vec = np.tile(vec, (start.shape[0],1))

    ray_collection = ConcreteRayList(
        ray_start_codeLen = start, ray_vec = vec
    )

    actual_rslt = generate_image(accum_strat, ray_collection, ds)

    alt_rslt = _dumber_full_noscatter_rt(accum_strat, ray_collection, ds)

    if False:
        for rslt in [actual_rslt, alt_rslt]:
            plot_ray_spectrum(obs_freq, dz_vals = None,
                              integrated_source = rslt['integrated_source'],
                              total_tau = rslt['total_tau'],
                              rest_freq = line_props.freq_quantity)
        plot_rel_err(obs_freq, actual_rslt, alt_rslt)

    assert_allclose_units(
        actual = actual_rslt['integrated_source'],
        desired = alt_rslt['integrated_source'], rtol = 3e-9, atol=0,
        err_msg = ("error occured while comparing the integrated_source with "
                   "the result from a simpler (slower) implementation")
    )

    np.testing.assert_allclose(
        actual = actual_rslt['total_tau'], desired = alt_rslt['total_tau'],
        rtol = 2e-10, atol=0,
        err_msg = ("error occured while comparing the total optical depth with "
                   "the result from a simpler (slower) implementation")
    )

def _dumber_full_noscatter_rt(accum_strat, concrete_ray_list, ds):
    # this is basically a dumbed down implementation of the noscatter rt
    # -> here, accum_strat is mostly just a way to encode information

    assert isinstance(concrete_ray_list, ConcreteRayList)

    start = concrete_ray_list.ray_start_codeLen
    vec = concrete_ray_list.get_ray_uvec()

    n_rays = concrete_ray_list.shape[0]
    integrated_source = unyt.unyt_array(
        np.full(shape =(accum_strat.obs_freq_Hz.size, n_rays),
                dtype = 'f8', fill_value = np.nan),
        'erg/(cm**2*sr)')
    integrated_tau = np.full(shape =(accum_strat.obs_freq_Hz.size, n_rays),
                             dtype = 'f8', fill_value = np.nan)

    fields = [accum_strat.ndens_field, ('gas', 'temperature'),
              ('gas','mean_molecular_weight'), ('gas', 'velocity_x'),
              ('gas', 'velocity_y'), ('gas', 'velocity_z')]
    itr = ray_values(ds, ray_start = start, ray_vec = vec,
                     fields = fields, find_indices = False)
    for i, ray_data in enumerate(itr):
        print(i)
        t1 = datetime.datetime.now()
        uvec = _to_uvec(vec[i])
        vLOS = (uvec[0] * ray_data['gas','velocity_x'] +
                uvec[1] * ray_data['gas','velocity_y'] +
                uvec[2] * ray_data['gas','velocity_z']).to('cm/s')

        if False: # this is the older "legacy" behavior (it's wrong!)
            doppler_parameter_b = np.sqrt(
                2.0 * unyt.kboltz_cgs * ray_data['gas','temperature'] /
                (ray_data['gas','mean_molecular_weight'] * unyt.mh_cgs)
            ).to('cm/s')
        else:
            doppler_parameter_b = np.sqrt(
                2.0 * unyt.kboltz_cgs * ray_data['gas','temperature'] /
                (accum_strat.species_mass_g * unyt.g)
            ).to('cm/s')

        tmp = py_generate_noscatter_spectrum(
            line_props = accum_strat.line_props,
            obs_freq = unyt.unyt_array(accum_strat.obs_freq_Hz, 'Hz'),
            vLOS = vLOS,
            ndens = ray_data[accum_strat.ndens_field],
            kinetic_T = ray_data['gas','temperature'],
            doppler_parameter_b = doppler_parameter_b,
            dz = ds.arr(ray_data['dl'],'code_length').to('cm'),
            level_pops_arg = accum_strat.partition_func
        )
        integrated_source[:,i] = tmp[0]
        integrated_tau[:,i] = tmp[1]
        t2 = datetime.datetime.now()
        print(t2 - t1)

    return {'integrated_source' : integrated_source,
            'total_tau' : integrated_tau}
