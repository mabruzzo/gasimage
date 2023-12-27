from contextlib import ExitStack
import os

import numpy as np
import unyt

from gasimage.optically_thin_ppv import convert_intensity_to_Tb
from gasimage.snapdsinit import SnapDatasetInitializer
from gasimage.utils.fits import write_to_fits, read_cube_from_fits
from gasimage.utils.generate_image import generate_image_arr
from gasimage.utils.testing import assert_allclose_units

_REF_REST_FREQ = np.float32(1.4204058E+09)*unyt.Hz

def _dummy_create_field_callback(ds):
    # this is a simple function to make define the ('gas','temperature') &
    # ('gas', 'H_p0_number_density') fields. Our calculation of the latter
    # involves some massive simplifying assumptions!

    # create ('gas','temperature') field (with correct units!)
    def _temperature(field, data):
        field = ('enzoe', 'temperature')
        # ('enzoe', 'temperature') may be labelled as dimensionless
        if data[field].units.is_dimensionless:
            return data[field] * unyt.K
        return data[field]
    ds.add_field(('gas', 'temperature'), function = _temperature,
                 sampling_type = 'local', take_log = True,
                 units = 'K')

    # create nH_I field
    def _nH_I(field,data):
        HydrogenFractionByMass, ifrac = 0.72, 0.0 # fairly arbitrary values
        rho_HI = data['density']* HydrogenFractionByMass * (1.0-ifrac)
        n_HI = rho_HI/unyt.mh_cgs
        return n_HI
    ds.add_field(('gas', 'H_p0_number_density'), function = _nH_I,
                 sampling_type = 'local', units = 'auto', take_log = True,
                 dimensions = unyt.dimensions.number_density)

def _create_raw_ppv(enzoe_sim_path,
                    sky_delta_latitude_arr_deg,
                    sky_longitude_arr_deg, v_channels,
                    obs_distance = 12.4*unyt.kpc,
                    sky_latitude_ref_deg = 0.0,
                    domain_theta_rad = np.pi/2,
                    domain_phi_rad = 3*np.pi/2,
                    use_cython_gen_spec = True,
                    force_general_consolidation = False, nproc = 1):
    # before use_cython_gen_spec existed, it was effectively False
    ds_loader = SnapDatasetInitializer(
        enzoe_sim_path, setup_func = _dummy_create_field_callback)

    with ExitStack() as stack:
        # setup pool for parallelism, if applicable ...
        if (int(nproc) != nproc) or nproc < 1:
            raise ValueError("nproc must be a positive integer")
        elif nproc == 1:
            pool = None
        else:
            import schwimmbad
            pool = stack.enter_context(schwimmbad.MultiPool(processes = nproc))

        ppv_arr = generate_image_arr(
            ds_loader,
            sky_delta_latitude_arr_deg = sky_delta_latitude_arr_deg,
            sky_longitude_arr_deg = sky_longitude_arr_deg,
            obs_distance = obs_distance, v_channels = v_channels,
            ndens_HI_n1state = ('gas', 'H_p0_number_density'),
            rescale_length_factor = 1,
            use_cython_gen_spec = use_cython_gen_spec,
            force_general_consolidation = force_general_consolidation,
            pool = pool
        )

    assert np.isfinite(ppv_arr).all(), "SANITY CHECK"

    ppv_Tb = convert_intensity_to_Tb(ppv_arr, rest_freq = _REF_REST_FREQ,
                                     v_channels = v_channels)
    return ppv_arr, ppv_Tb

def _create_raw_ppv_and_save_fits(enzoe_sim_path, out_fname,
                                  sky_delta_latitude_arr_deg,
                                  sky_longitude_arr_deg, v_channels,
                                  obs_distance = 12.4*unyt.kpc,
                                  sky_latitude_ref_deg = 0.0,
                                  domain_theta_rad = np.pi/2,
                                  domain_phi_rad = 3*np.pi/2,
                                  use_cython_gen_spec = True,
                                  clobber_file = False, nproc = 1):
    # before use_cython_gen_spec existed, it was effectively False

    _, ppv_Tb = _create_raw_ppv(
        enzoe_sim_path = enzoe_sim_path,
        sky_delta_latitude_arr_deg = sky_delta_latitude_arr_deg,
        sky_longitude_arr_deg = sky_longitude_arr_deg,
        v_channels = v_channels,
        obs_distance = obs_distance,
        sky_latitude_ref_deg = sky_latitude_ref_deg,
        domain_theta_rad = domain_theta_rad,
        domain_phi_rad = domain_phi_rad,
        use_cython_gen_spec = use_cython_gen_spec,
        nproc = 1)

    assert np.isfinite(ppv_Tb).all(), "SANITY CHECK"

    write_to_fits(out_fname, ppv_Tb,
                  sky_longitude_arr_deg = sky_longitude_arr_deg,
                  sky_delta_latitude_arr_deg = sky_delta_latitude_arr_deg,
                  sky_latitude_ref_deg = sky_latitude_ref_deg,
                  v_channels = v_channels, rest_freq = _REF_REST_FREQ,
                  writeto_kwargs = {'overwrite' : clobber_file})

def _compare_results(cur_fname, ref_fname, err_msg, **kw):
    # note: I think that writing to disk encodes the data to 32bits (even
    # though most calculations are in 64 bits), so minor changes won't
    # necessarily show up.

    cur_arr, cur_cubeinfo = read_cube_from_fits(cur_fname,
                                                default_v_units = None)
    ref_arr, ref_cubeinfo = read_cube_from_fits(ref_fname,
                                                default_v_units = None)
    # I suspect that the cubeinfo objects may not be exactly equal if the
    # reference file is fairly old because at one point we changed,
    # what the name of the velocity axis is recorded as.
    # -> the old choice didn't specify whether velocity scales linearly with
    #    wavelength or frequency. Our new choice now specifies that it scales
    #    linearly with frequency (I really don't think it matters unless the
    #    cloud moves at relativistic speeds)

    # We can still do some crude sanity-checks on the cubeinfo:
    # NOTE: don't expect the input ra,dec, and v_channels info to be exactly
    #       equal to the stuff encoded in the cubeinfo object. While the
    #       cubeinfo objects will losslessly round-trip, coercing the inputs
    #       into a cubeinfo object is somewhat lossy
    ref_ra, ref_dec = ref_cubeinfo.get_ra_dec(ref_arr.shape, units = 'deg',
                                              ndim = 1, unyt_arr = True)
    ref_v_channels = ref_cubeinfo.get_v_channels(ref_arr.shape, units = 'm/s',
                                                 ndim = 1, unyt_arr = True)
    cur_ra, cur_dec = cur_cubeinfo.get_ra_dec(cur_arr.shape, units = 'deg',
                                              ndim = 1, unyt_arr = True)
    cur_v_channels = cur_cubeinfo.get_v_channels(cur_arr.shape, units = 'm/s',
                                                 ndim = 1, unyt_arr = True)
    assert (cur_ra == ref_ra).all()
    assert (cur_dec == ref_dec).all()
    assert (cur_v_channels == ref_v_channels).all()

    assert_allclose_units(actual = cur_arr, desired = ref_arr,
                          err_msg = err_msg, **kw)

def _test_full_raytrace_perspective_helper(answer_test_config, indata_dir,
                                           base_fname = 'result.fits',
                                           override_kwargs = {}):
    # this is used to help define multiple tests

    # indata_dir is the value set by the --indata-dir command line argument
    enzoe_sim_path = os.path.join(
        indata_dir, ('X100_M1.5_HD_CDdftCstr_R56.38_logP3_Res16/cloud_07.5000/'
                     'cloud_07.5000.block_list')
    )

    # answer_test_config.save_answer_dir is set based on the name of this
    #    function and (usually) by --save-answer-dir cli option
    out_fname = os.path.join(answer_test_config.save_answer_dir, base_fname)

    kwargs = dict(
        sky_delta_latitude_arr_deg = np.arange(-0.5, 0.51, 0.0666667),
        sky_longitude_arr_deg = np.arange(-1.5, 0.51, 0.0666667),
        v_channels = unyt.unyt_array(np.arange(-170,180,0.736125), 'km/s'),
        obs_distance = unyt.unyt_quantity(12.4, 'kpc'),
        sky_latitude_ref_deg = 0.0,
        domain_theta_rad = np.pi/2, domain_phi_rad = 3*np.pi/2,
        use_cython_gen_spec = True,
        nproc = 1
    )

    for k,v in override_kwargs.items():
        if k not in kwargs:
            raise RuntimeError(f"invalid kwarg specified: {k}")
        else:
            kwargs[k] = v

    # generate an undegraded mock image
    _create_raw_ppv_and_save_fits(
        enzoe_sim_path = enzoe_sim_path, out_fname = out_fname,
        clobber_file = True, **kwargs
    )

    if answer_test_config.ref_answer_dir is not None:
        ref_fname = os.path.join(answer_test_config.ref_answer_dir, base_fname)
        if not os.path.isfile(ref_fname):
            AssertionError(f"couldn't find previous test result at {ref_fname}")
        _compare_results(
            cur_fname = out_fname,
            ref_fname = ref_fname,
            err_msg = "The results should be EXACTLY the same",
            rtol = 0.0, atol = 0.0
        )


def test_full_raytrace_perspective(answer_test_config, indata_dir):
    # when pytest executes this function, it automatically generates values for
    #     answer_test_config, indata_dir
    # by passing the values produced by the functions of the same name found in
    # conftest.py (and decorated with @pytest.fixture).
    # -> these values are informed by the --save-answer-dir, --ref-answer-dir,
    #    --indata-dir options that get passed on the command-line

    # older versions of this test were effectively run with
    # use_cython_gen_spec = False. But, because the calculations are done
    # in 64-bit precision, and the comparisons are done after coercing to
    # 32-bit precision, the tests still pass!

    _test_full_raytrace_perspective_helper(
        answer_test_config, indata_dir, base_fname = 'result.fits',
        override_kwargs = dict(use_cython_gen_spec = True)
    )

def test_full_raytrace_perspective_alt(answer_test_config, indata_dir):
    # just like test_full_raytrace_perspective, but with some arbitrary
    # differences

    # older versions of this test were effectively run with
    # use_cython_gen_spec = False. But, because the calculations are done
    # in 64-bit precision, and the comparisons are done after coercing to
    # 32-bit precision, the tests still pass!

    _test_full_raytrace_perspective_helper(
        answer_test_config, indata_dir, base_fname = 'result.fits',
        override_kwargs = dict(obs_distance = unyt.unyt_quantity(15, 'kpc'),
                               domain_theta_rad = 2*np.pi/3,
                               domain_phi_rad = np.pi/3,
                               use_cython_gen_spec = True,
        )
    )

def _test_compare_gen_spec(indata_dir, cython_legacy_comparison,
                           rtol, atol):
    enzoe_sim_path = os.path.join(
        indata_dir, ('X100_M1.5_HD_CDdftCstr_R56.38_logP3_Res8/cloud_07.5000/'
                     'cloud_07.5000.block_list')
    )

    ppv_arrs = []

    if cython_legacy_comparison:
        kwargs_pair = [
            {'use_cython_gen_spec' : True},
            {'use_cython_gen_spec' : False},
        ]
        err_msg = ("failure in comparison of ppvs built with cython-based "
                   "spectrum builder and the legacy spectrum builder")
    else:
        kwargs_pair = [
            {'force_general_consolidation' : True},
            {'force_general_consolidation' : False},
        ]
        err_msg = ("failure in comparison of ppvs built with the generic "
                   "consolidation strategy and faster legacy strategy")

    for kwargs in kwargs_pair:
        print(f'\nstart run with {kwargs}')
        ppv, _ = _create_raw_ppv(
            enzoe_sim_path = enzoe_sim_path,
            sky_delta_latitude_arr_deg = np.arange(0.0, 0.25, 0.0333333),
            sky_longitude_arr_deg = np.arange(0.25, 0.51,     0.0333333),
            v_channels = unyt.unyt_array(np.arange(-170,180,0.736125), 'km/s'),
            obs_distance = unyt.unyt_quantity(12.4, 'kpc'),
            sky_latitude_ref_deg = 0.0,
            domain_theta_rad = np.pi/2, domain_phi_rad = 3*np.pi/2,
            nproc = 1, **kwargs
        )
        ppv_arrs.append(ppv)

    ppv_newer, ppv_older_approach = ppv_arrs

    assert_allclose_units(
        actual = ppv_newer, desired = ppv_older_approach,
        err_msg = ("failure in comparison of ppvs built with cython-based "
                   "spectrum builder and the legacy spectrum builder"),
        rtol = rtol, atol = atol)

def test_compare_cython_legacy_gen_spec(indata_dir):
    _test_compare_gen_spec(indata_dir, cython_legacy_comparison = True,
                           rtol = 4.e-12, atol = 0.0)

# here we compare the traditional consolidation strategy (we directly
# consolidate in the output array) vs the more general-purpose approach
def test_compare_consolidation_strat(indata_dir):
    _test_compare_gen_spec(indata_dir, cython_legacy_comparison = False,
                           rtol = 0.0, atol = 0.0)


# other useful tests for the future:
# - rays outside of the domain
# - explicitly test parallelism
# - maybe compare against axis-aligned...
