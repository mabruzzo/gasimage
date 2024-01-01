import os.path
import datetime

import numpy as np
import unyt
import yt



from gasimage.generate_image import freq_from_v_channels, generate_image

from gasimage.accumulators import NoScatterRTAccumStrat
from gasimage.rt_config import builtin_halpha_props, crude_H_partition_func
from gasimage.ray_collection import ConcreteRayList
from gasimage.utils.testing import assert_allclose_units
from gasimage._generate_spec_cy import _generate_noscatter_spectrum_cy

from py_generate_noscatter_spectrum import (
    _generate_noscatter_spectrum as py_generate_noscatter_spectrum
)
from ray_testing_utils import ray_values, _to_uvec
from test_full import _dummy_create_field_callback


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
    ).to('code_length').ndview[:,:,:]

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

def _test_compare_full_noscatter_rt(indata_dir, aligned_rays):
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

    if not aligned_rays:
        # the errors are MUCH larger in this case!
        # -> I suspect this is because the rays in that case are on angles and
        #    there is a small amount of disagreement with what yt returns...
        #    Plus the distances are less-accurate with the YTRay appraoch
        # -> it's worth noting that the discrepancies seem largest furthest away
        #    from the line-center, which is actually comforting...
        start, vec = _get_ray_start_vec(ds)
        if False: # highlight problematic case!
            #start = np.array([start])
            vec = vec[4:6, :]
            print(vec)
        rtol_intensity, atol_intensity = 1.e-6, 1.e-6
        rtol_tau, atol_tau = 1e-5, 1.4e-5
        case_name = 'perspective-rays'
    else:
        start, vec = _get_ray_start_vec2(ds)
        rtol_intensity, atol_intensity = 3e-9, 0.0
        rtol_tau, atol_tau = 2e-10, .0
        case_name = 'aligned-rays'

    if start.size < vec.size:
        start = np.tile(start, (vec.shape[0],1))
    elif start.size > vec.size:
        vec = np.tile(vec, (start.shape[0],1))

    ray_collection = ConcreteRayList(
        ray_start_codeLen = start, ray_vec = vec
    )

    actual_rslt = generate_image(accum_strat, ray_collection, ds)
    alt_rslt = _dumber_full_noscatter_rt(accum_strat, ray_collection, ds,
                                         ray_values_kind = 'yt')

    debug = False

    if debug:
        plot_rel_err(obs_freq, actual_rslt, alt_rslt)
    elif False:
        calc_rdiff = lambda k: (actual_rslt[k] - alt_rslt[k]) / alt_rslt[k]
        calc_adiff = lambda k: (actual_rslt[k] - alt_rslt[k])

        start = ray_collection.ray_start_codeLen
        vec = ray_collection.get_ray_uvec()
        rdiff = calc_rdiff('integrated_source')
        adiff = calc_adiff('integrated_source')

        # find ray_ind where max is largest
        ray_i = np.argmax(np.abs(rdiff).max(axis=0))
        # find the index of that ray's spectra where rdiff is largest
        rel_max_i = np.argmax(np.abs(rdiff[:,ray_i]))
        print(ray_i, rel_max_i)

        print(calc_rdiff('integrated_source').shape)
        print(
            'largest rdiff is ray:', ray_i,
            '\n  vec:', vec[ray_i],
            '\n  rdiff maximized at freq of:', obs_freq[rel_max_i],
            '\n    -> wave = ', (unyt.c_cgs/obs_freq[rel_max_i]).to('nm'),
            '\n    -> rdiff intensity= ', rdiff[rel_max_i, ray_i],
            '\n    -> adiff intensity= ', adiff[rel_max_i, ray_i],
            '\n    -> rdiff total_tau= ',
                calc_rdiff('total_tau')[rel_max_i, ray_i],
            '\n    -> adiff total_tau= ',
                calc_adiff('total_tau')[rel_max_i, ray_i],
            '\n    -> alt integrated_source = ',
                alt_rslt['integrated_source'][rel_max_i, ray_i],
            '\n    -> alt total_tau = ',
                alt_rslt['total_tau'][rel_max_i, ray_i] 
        )

        plot_rel_err(obs_freq, actual = actual_rslt, other = alt_rslt,
                     ray_ind = ray_i)

    assert_allclose_units(
        actual = actual_rslt['integrated_source'],
        desired = alt_rslt['integrated_source'], rtol = rtol_intensity,
        atol=atol_intensity,
        err_msg = ("error occured while comparing the integrated_source with "
                   "the result from a simpler (slower) implementation -- when "
                   f"using {case_name} rays")
    )

    np.testing.assert_allclose(
        actual = actual_rslt['total_tau'], desired = alt_rslt['total_tau'],
        rtol = rtol_tau, atol=atol_tau,
        err_msg = ("error occured while comparing the total optical depth with "
                   "the result from a simpler (slower) implementation -- when "
                   f"using {case_name} rays")
    )

def _dumber_full_noscatter_rt(accum_strat, concrete_ray_list, ds,
                              ray_values_kind = 'yt'):
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
              ('index', 'x'), ('index', 'y'), ('index','z'),
              ('gas','mean_molecular_weight'), ('gas', 'velocity_x'),
              ('gas', 'velocity_y'), ('gas', 'velocity_z')]
    itr = ray_values(ds, ray_start = start, ray_vec = vec,
                     fields = fields, find_indices = True,
                     kind = ray_values_kind)
    t1 = datetime.datetime.now()
    for i, ray_data in enumerate(itr):
        if i == 0:
            first = unyt.unyt_array([ray_data['index', ax][0] for ax in 'xyz'])
            last = unyt.unyt_array([ray_data['index', ax][-1] for ax in 'xyz'])
            print('vec:', vec[i], 'first:', first, 'last:', last)
            print(ray_data['indices'].shape)
        uvec = _to_uvec(vec[i])
        vLOS = (uvec[0] * ray_data['gas','velocity_x'] +
                uvec[1] * ray_data['gas','velocity_y'] +
                uvec[2] * ray_data['gas','velocity_z']).to('cm/s')


        doppler_parameter_b = np.sqrt(
            2.0 * unyt.kboltz_cgs * ray_data['gas','temperature'] /
            (accum_strat.species_mass_g * unyt.g)
        ).to('cm/s')

        if True:
            tmp = _generate_noscatter_spectrum_cy(
                line_props = accum_strat.line_props,
                obs_freq = unyt.unyt_array(accum_strat.obs_freq_Hz, 'Hz').ndview,
                vLOS = vLOS.to('cm/s').ndview,
                ndens = ray_data[accum_strat.ndens_field].to('cm**-3').ndview,
                kinetic_T = ray_data['gas','temperature'].to('K').ndview,
                doppler_parameter_b = doppler_parameter_b.to('cm/s').ndview,
                dz = ds.arr(ray_data['dl'],'code_length').to('cm').ndview,
                level_pops_arg = accum_strat.partition_func
            )
        else:
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


def test_compare_full_noscatter_rt_alignedrays(indata_dir):
    _test_compare_full_noscatter_rt(indata_dir, aligned_rays = True)

def test_compare_full_noscatter_rt_perspectiverays(indata_dir):
    _test_compare_full_noscatter_rt(indata_dir, aligned_rays = False)
