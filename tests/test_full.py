from contextlib import ExitStack
import os

import numpy as np
import unyt

from gasimage.optically_thin_ppv import convert_intensity_to_Tb
from gasimage.snapdsinit import SnapDatasetInitializer
from gasimage.utils.fits import write_to_fits
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

def _create_raw_ppv_and_save_fits(enzoe_sim_path, out_fname,
                                  sky_delta_latitude_arr_deg,
                                  sky_longitude_arr_deg, v_channels,
                                  obs_distance = 12.4*unyt.kpc,
                                  sky_latitude_ref_deg = 0.0,
                                  domain_theta_rad = np.pi/2,
                                  domain_phi_rad = 3*np.pi/2,
                                  clobber_file = False, nproc = 1):

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
            pool = pool
        )

    assert np.isfinite(ppv_arr).all(), "SANITY CHECK"

    ppv_Tb = convert_intensity_to_Tb(ppv_arr, rest_freq = _REF_REST_FREQ,
                                     v_channels = v_channels)

    assert np.isfinite(ppv_Tb).all(), "SANITY CHECK"

    write_to_fits(out_fname, ppv_Tb,
                  sky_longitude_arr_deg = sky_longitude_arr_deg,
                  sky_delta_latitude_arr_deg = sky_delta_latitude_arr_deg,
                  sky_latitude_ref_deg = sky_latitude_ref_deg,
                  v_channels = v_channels, rest_freq = _REF_REST_FREQ,
                  writeto_kwargs = {'overwrite' : clobber_file})

def read_ppv_from_disk(fname):
    # returns a position-position-velocity datacube where:
    #  - axis 0 of the output array varies in velocity
    #  - axis 1 of the output array varies in declination (or sky latitude)
    #  - axis 2 of the output array varies in right ascension (or sky longitude)
    #
    # This sort of output implicitly assumes a cartesian projection.

    from astropy.io import fits
    
    with fits.open(fname, mode = 'readonly') as hdul:
        if len(hdul) != 1:
            raise ValueError("only equipped to handle fits files with one "
                             "HDU (Header Data Unit)") 
        hdu = hdul[0]
        hdr = hdu.header

        # sanity check (note that the WCS object stores things in reverse
        # compared to the numpy shape ordering)

        data = hdu.data[...]
        if hdr.get('BUNIT', None) == 'K (Tb)':
            data = unyt.unyt_array(data, 'K')
        return data

def _compare_results(cur_fname, ref_fname, err_msg, **kw):
    cur_arr = read_ppv_from_disk(cur_fname)
    ref_arr = read_ppv_from_disk(ref_fname)
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
    
    _test_full_raytrace_perspective_helper(answer_test_config, indata_dir,
                                           base_fname = 'result.fits')

def test_full_raytrace_perspective_alt(answer_test_config, indata_dir):
    # just like test_full_raytrace_perspective, but with some arbitrary
    # differences

    _test_full_raytrace_perspective_helper(
        answer_test_config, indata_dir, base_fname = 'result.fits',
        override_kwargs = dict(obs_distance = unyt.unyt_quantity(15, 'kpc'),
                               domain_theta_rad = 2*np.pi/3,
                               domain_phi_rad = np.pi/3,
        )
    )

# other useful tests for the future:
# - rays outside of the domain
# - explicitly test parallelism
# - maybe compare against axis-aligned...
