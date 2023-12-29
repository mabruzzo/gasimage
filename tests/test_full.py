from contextlib import ExitStack
import os

import numpy as np
import unyt
import yt

from gasimage.optically_thin_ppv import (
    convert_intensity_to_Tb,
    optically_thin_ppv
)
from gasimage.ray_collection import perspective_ray_grid
from gasimage.snapdsinit import SnapDatasetInitializer
from gasimage.utils.fits import write_to_fits, read_cube_from_fits
from gasimage.utils.testing import assert_allclose_units

_REF_REST_FREQ = np.float32(1.4204058E+09)*unyt.Hz

def _dummy_create_field_callback(ds, use_trident_ion = False):
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

    if use_trident_ion:
        # let's be explicit about exactly which ionization table is used
        import trident
        trident.add_ion_fields(
            ds, ions=['H I'],
            ionization_table = os.path.expanduser('~/.trident/hm2012_hr.h5')
        )
    else:

        # create nH_I field
        def _nH_I(field,data):
            HydrogenFractionByMass, ifrac = 0.72, 0.0 # fairly arbitrary values
            rho_HI = data['density']* HydrogenFractionByMass * (1.0-ifrac)
            n_HI = rho_HI/unyt.mh_cgs
            return n_HI
        ds.add_field(('gas', 'H_p0_number_density'), function = _nH_I,
                     sampling_type = 'local', units = 'auto', take_log = True,
                     dimensions = unyt.dimensions.number_density)

def generate_image_arr(ds_initializer, v_channels, sky_delta_latitude_arr_deg,
                       sky_longitude_arr_deg, obs_distance,
                       sky_latitude_ref_deg = 0.0,
                       domain_theta_rad = np.pi/2, domain_phi_rad = 3*np.pi/2,
                       ndens_HI_n1state = ('gas', 'H_p0_number_density'),
                       doppler_parameter_b = None,
                       use_cython_gen_spec = True,
                       force_general_consolidation = False,
                       rescale_length_factor = None, pool = None):
    """
    THIS IS FOR TESTING PURPOSES - this was a legacy function written a long
    time ago! We could probably delete this function!

    Generate a mock ppv image of a simulation using a ray-tracing radiative
    transfer algorithm that assumes that the gas is optically thin.

    The resulting image has a pixel for each combination of `v_channel`, 
    `sky_delta_latitude_deg`, and `sky_longitude_arr_deg`.

    Parameters
    ----------
    ds_initializer:
         This can either be a yt-dataset or a dataset initialization object.
         The latter is required if you want to use a parallel pool (since you
         can't pickle these datasets)
    rescale_length_factor: float, optional
        When specified, the width of each cell in the simulation is multiplied 
        by this factor. (This effectively holds the position of the 
        simulation's origin fixed in place).
    v_channels: 1D `unyt.unyt_array`
        A monotonically increasing array of velocity channels.
    sky_delta_latitude_arr_deg: 1D np.ndarray
        Array of sky-latitude locations (in degrees), where rays should be 
        traced. These are measured with respect to the sky-latitude of the ray
        between the observer and the reference point, from the observer's
        perspective. That reference sky-latitude is specified by 
        sky_latitude_ref_deg.
    sky_longitude_arr_deg: 1D np.ndarray
        Array of sky-longitude locations (in degrees), where rays should be 
        traced. For reference, the sky-longitude of the ray between the
        observer and the reference point is always assumed to be zero because
        it has no effect on the resulting mercator projection and can be 
        arbitrarily changed later.
    obs_distance : unyt.unyt_quantity
        Distance between the observer and the reference point (which currently 
        coincides with the center of the simulation domain).
    sky_latitude_ref_deg: float, optional
        specify the latitude on the sky (from the observer's perspective) that 
        the primary ray passes through. This can have significant implications 
        for the mercator projection. The default value is 0.0 (corresponding to
        the celestial equator). 
    domain_theta_rad, domain_phi_rad: float
        Alongside obs_distance, these quantities specify the location of the 
        observer in spherical coordinates relative to the reference point
        (which is currently assumed to coincide with the center of the
        simulation domain). Note that the spherical coordinates assuming that 
        the x,y, and z directions align with the simulation's native coordinate
        axes (for reference, the x,y,z directions in the observer's frame are
        generally different).
    ndens_HI_n1state
        The name of the yt-field holding the number density of H I atoms
        (neutral Hydrogen) in the electronic ground state (i.e. the electron is
        in the n=1 orbital). See the documentation for `optically_thin_ppv` for
        more discussion about the motivation for the default value.
    rescale_length_factor: float, Optional
        When not `None`, the width of each cell is multiplied by this factor.
        (This effectively holds the position of the simulation's origin fixed
        in place).
    pool: Optional
        A taskpool from the schwimmbad package can be specified for this 
        argument to facillitate parallelization.

    Notes
    -----
    Presently, this function assumes that the "reference point" coincides with
    the center of the simulation. It might be nice to use an arbitrary
    reference point instead (all of the helper functions should already support
    this, but this should be tested). If that is done, some decisions will need
    to be made in order to avoid ambiguities with 

    `rescale_length_factor` is provided for rescaling the sizes of adiabatic
    simulations (since they are scale-free).
    """
    # produce a mock image (not initially in terms of brightness temperature)
    if isinstance(ds_initializer, yt.data_objects.static_output.Dataset):
        _ds = ds_initializer
        assert pool is None # can't parallelize!
    else:
        _ds = ds_initializer()

    if not (-90.0 < sky_latitude_ref_deg < 90.0):
        # testing will need to be required when it's equal to 90.0
        raise ValueError("sky_latitude_ref_deg must be between -90.0 & + 90.0")
    else:
        assert (sky_latitude_ref_deg + sky_delta_latitude_arr_deg.min()) > -90.0
        assert (sky_latitude_ref_deg + sky_delta_latitude_arr_deg.max()) < 90.0

    if sky_latitude_ref_deg != 0.0:
        raise ValueError("the sky_latitude_ref_deg argument is untested with "
                         "a value other than 0.0")

    if rescale_length_factor is not None:
        assert rescale_length_factor > 0 and np.ndim(rescale_length_factor) == 0

    # determine the starting and ending points of all rays
    ray_collection = perspective_ray_grid(
        sky_latitude_ref_deg = sky_latitude_ref_deg,
        observer_distance = obs_distance,
        sky_delta_latitude_arr_deg = sky_delta_latitude_arr_deg,
        sky_longitude_arr_deg = sky_longitude_arr_deg,
        domain_theta_rad = domain_theta_rad,
        domain_phi_rad = domain_phi_rad,
        ds = _ds, rescale_length_factor = rescale_length_factor
    )

    out = optically_thin_ppv(
        v_channels, ray_collection = ray_collection, ds = ds_initializer,
        ndens_HI_n1state = ndens_HI_n1state,
        doppler_parameter_b = doppler_parameter_b,
        use_cython_gen_spec = use_cython_gen_spec,
        rescale_length_factor = rescale_length_factor,
        force_general_consolidation = force_general_consolidation,
        pool = pool
    )
    return out


def _create_raw_ppv(enzoe_sim_path,
                    sky_delta_latitude_arr_deg,
                    sky_longitude_arr_deg, v_channels,
                    obs_distance = 12.4*unyt.kpc,
                    sky_latitude_ref_deg = 0.0,
                    domain_theta_rad = np.pi/2,
                    domain_phi_rad = 3*np.pi/2,
                    use_cython_gen_spec = True,
                    doppler_parameter_b = None,
                    force_general_consolidation = False,
                    nproc = 1):
    # -> before use_cython_gen_spec existed, it was effectively False
    # -> to get the behavior equivalent to before doppler_parameter_b was
    #    a parameter of this function, pass it 'legacy'
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
            doppler_parameter_b = doppler_parameter_b,
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
                                  doppler_parameter_b = None,
                                  clobber_file = False, nproc = 1):
    # -> before use_cython_gen_spec existed, it was effectively False
    # -> to get the behavior equivalent to before doppler_parameter_b was
    #    a parameter of this function, pass it 'legacy'

    _, ppv_Tb = _create_raw_ppv(
        enzoe_sim_path = enzoe_sim_path,
        sky_delta_latitude_arr_deg = sky_delta_latitude_arr_deg,
        sky_longitude_arr_deg = sky_longitude_arr_deg,
        v_channels = v_channels,
        obs_distance = obs_distance,
        sky_latitude_ref_deg = sky_latitude_ref_deg,
        domain_theta_rad = domain_theta_rad,
        domain_phi_rad = domain_phi_rad,
        doppler_parameter_b = doppler_parameter_b,
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
        doppler_parameter_b = 'legacy', # only use this so we don't have to
                                        # change our test-answers
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
            doppler_parameter_b = 'normal', # only use this so we don't have to
                                            # change our test-answers
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
                           rtol = 5.e-12, atol = 0.0)

# here we compare the traditional consolidation strategy (we directly
# consolidate in the output array) vs the more general-purpose approach
def test_compare_consolidation_strat(indata_dir):
    _test_compare_gen_spec(indata_dir, cython_legacy_comparison = False,
                           rtol = 0.0, atol = 0.0)


# other useful tests for the future:
# - rays outside of the domain
# - explicitly test parallelism
# - maybe compare against axis-aligned...
