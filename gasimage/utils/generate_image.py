import numpy as np
import unyt
import yt

from gasimage.ray_collection import perspective_ray_grid
from gasimage.optically_thin_ppv import optically_thin_ppv


# this is mostly for testing-purposes!
# - a lot of recent refactoring will hopefully have made this function a lot
#   less necessary

def generate_image_arr(ds_initializer, v_channels, sky_delta_latitude_arr_deg,
                       sky_longitude_arr_deg, obs_distance,
                       sky_latitude_ref_deg = 0.0,
                       domain_theta_rad = np.pi/2, domain_phi_rad = 3*np.pi/2,
                       ndens_HI_n1state = ('gas', 'H_p0_number_density'),
                       use_cython_gen_spec = True,
                       force_general_consolidation = False,
                       rescale_length_factor = None, pool = None):
    """
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
        doppler_parameter_b = None,
        use_cython_gen_spec = use_cython_gen_spec,
        rescale_length_factor = rescale_length_factor,
        force_general_consolidation = force_general_consolidation,
        pool = pool
    )
    return out
