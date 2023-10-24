import numpy as np
import unyt
from .utils.misc import check_consistent_arg_dims

def _magnitude(vec): # handles unyt.unyt_array better than np.linalg
    assert vec.ndim == 1
    return np.sqrt(vec*vec).sum()

def _dot(matrix,vector):
    # to do, replace this with an actual numpy function
    assert vector.shape == (3,)
    if matrix.shape == (3,3):
        return (matrix[:,0] * vector[0] + matrix[:,1] * vector[1] + 
                matrix[:,2] * vector[2])
    elif matrix.shape == (3,):
        return (matrix * vector).sum()
    else:
        raise RuntimeError()

def rotation_matrix(theta, ax):
    # the result is a matrix rot
    #
    # _dot(rot,vec) represents a counter-clockwise rotation
    # of the vector by angle theta (when the `ax`-axis points 
    # towards the observer)
    #
    # alternatively, _dot(rot,vec) represents the new 
    # coordinates of vec after a clockwise rotation of the
    # axes by angle theta (when the `ax`-axis points 
    # towards the observer)

    if ax == 'x':
        out = ((1.0,           0.0,            0.0),
               (0.0, np.cos(theta), -np.sin(theta)),
               (0.0, np.sin(theta),  np.cos(theta)))
    elif ax == 'y':
        out = (( np.cos(theta), 0.0, np.sin(theta)),
               (           0.0, 1.0,           0.0),
               (-np.sin(theta), 0.0, np.cos(theta)))
    elif ax == 'z':
        out = ((np.cos(theta), -np.sin(theta),  0.0),
               (np.sin(theta),  np.cos(theta),  0.0),
               (          0.0,            0.0,  1.0))
    return np.array(out)

def rotate_from_obs_axes(points, observer_latitude_rad, domain_theta_rad,
                         domain_phi_rad):
    """
    Returns the values of points (initially specified in the observer's 
    coordinate system) after rotating the coordinate axes to match the
    orientation of the domain's coordinate axes. 

    Note that the origin of the coordinate system remains unchanged.

    Parameters
    ----------
    points: np.ndarray
        This is either a single 3D vector, with shape (3,) or an array
        of N 3D vectors, with shape (N,3).

    Notes
    -----
    This could be significantly optimized
    """
    # in observer's coordinate axis ray to reference point lies
    # in x-z plane
    
    if points.shape == (3,):
        points = np.array([points.value]) * points.unit_quantity
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    
    points_uq = points.unit_quantity
    
    points_value = points.value

    # the observer's coordinates axes are defined such that:
    # - the ray through the reference point is strictly in the 
    #   x_obs-z_obs plane (y_obs is zero everywhere along the ray)
    # - observer_latitude_rad gives the latitude of the ray through
    #   the reference point
    # - y_obs lies in the plane created by x_domain-y_domain
    # - unless observer_latitude_rad = +-pi/2, every point along 
    #   the ray has x_obs>=0
    
    # rotate the x_obs-z_obs axis such that any point that 
    # originally had a latitude of observer_latitude_rad (in the 
    # observer's coordinate axes), now has a latitude
    # of 0 (with the updated coordinates). This is equivalent to 
    # rotating the coordinate axes clockwise by angle, 
    # `observer_latitude_rad`
    # the new x_obs axis is now parallel to the ray passing through
    # the reference point
    y_rotation_1 = rotation_matrix(observer_latitude_rad,'y')

    domain_latitude = np.pi/2 - domain_theta_rad
    # rotate the x_obs-z_obs axes (again) of the observer's coordinate 
    # system clockwise by angle domain_latitude about the y_obs-axis.
    # Doing this will align z_obs with z_domain and will make x_obs
    # point in the plane formed by x_domain-y_domain
    y_rotation_2 = rotation_matrix(domain_latitude, 'y')

    # finally, rotate (the current) x_obs-y_obs axes clockwise around 
    # z_obs (which now is parallel to z_domain) by 
    # angle np.pi + domain_phi_rad. This finishes aligning
    # x_obs,y_obs,z_obs with x_domain,y_domain,z_domain
    z_rotation = rotation_matrix(np.pi + domain_phi_rad, 'z')

    out = np.empty_like(points_value)
    for i in range(points_value.shape[0]):
        # first rotate the 
        temp = _dot(y_rotation_1, points_value[i,:])
        temp2 = _dot(y_rotation_2, temp)
        out[i,:] = _dot(z_rotation, temp2)
    return out * points_uq

def _construct_radial_unit_vector(theta,phi):
    if np.shape(theta) == () and np.shape(phi) == ():
        _theta = theta
        _phi = phi
    else:
        _phi,_theta = np.meshgrid(phi,theta)

    xhat = np.sin(_theta)*np.cos(_phi)
    yhat = np.sin(_theta)*np.sin(_phi)
    zhat = np.cos(_theta)
    unit_vectors = np.stack([xhat,yhat,zhat], axis = -1)
    return unit_vectors

def _convert_spherical_to_cartesian(r,theta_rad,phi_rad):
    # this uses the physics convention
    # theta = angle down from z
    # phi = angle from x (in x-y plane)
    #
    # potentially look into using function from yt or astropy
    unit_vectors = _construct_radial_unit_vector(theta_rad,phi_rad)
    return r*unit_vectors


def _find_obs_ray_end_points(ds, sky_latitude_ref_deg, 
                             observer_distance,
                             sky_delta_latitude_arr_deg,
                             sky_longitude_arr_deg,
                             rescale_length_factor = None):

    # when rescale_length_factor is not None, the width of each cell is
    # multiplied by this factor. (This effectively holds the position of the
    # simulation's origin fixed in place).

    # choose an ending point that is sure to be farther than 
    # the furthest domain edge
    longest_width = np.sqrt(np.square(ds.domain_width).sum())

    if rescale_length_factor is not None:
        longest_width *= rescale_length_factor

    sky_theta_ref = np.deg2rad(90.0 - sky_latitude_ref_deg)
    sky_delta_theta = np.deg2rad(90.0 - sky_delta_latitude_arr_deg)
    sky_theta_arr = np.deg2rad(90.0 - sky_delta_latitude_arr_deg)
    sky_phi_arr = np.deg2rad(sky_longitude_arr_deg)

    # we just need to find the end points of these rays (we want to
    # intentionally overshoot).
    dist = (observer_distance+2*longest_width)

    return _convert_spherical_to_cartesian(r = dist,
                                           theta_rad = sky_theta_arr,
                                           phi_rad = sky_phi_arr)


def convert_observer_to_domain_points(points_to_transform,
                                      observer_distance,
                                      sky_latitude_ref_deg,
                                      domain_reference_point,
                                      domain_theta_rad,
                                      domain_phi_rad):
    """
    Convert the points from the observer's coordinate system to the domain's
    coordinate system.

    Each vector should be ordered (x,y,z)

    Parameters
    ----------
    points_to_transform: np.ndarray
        Array of N 3D vectors, with shape (N,3) to be transformed.
    observer_distance: unyt.unyt_quantity
        Distance of the observer from the reference point.
    sky_latitude_ref_deg: float
        The latitude of the reference point from the observer's perspective in 
        degrees. When this is positive, it denotes that many degrees above the 
        observer's x-axis. When negative it denotes the number of degrees below
        the observer's x-axis. When zero, the reference point lies along the 
        observer's x-axis.
    domain_reference_point: np.ndarray
        Position of the reference point in the simulation domain (relative to 
        the domain's origin). This is specified in the simulation's coordinate 
        system.
    domain_theta_rad, domain_phi_rad: float
        Spherical coordinates of the observer relative to the reference point.

    Returns
    -------
    converted_ray_origin
        The observer's position in the domain's coordinate system
    converted_points
        The converted points

    Notes
    -----
    A signficant amount of time elapsed between writing this function and adding
    this docstring. I believe that this function has only been rigorously
    tested with `sky_latitude_ref_deg=0`. This variable was put into place to 
    get correct spacing of rays under various projections (i.e. mercator) when
    observing objects at high sky latitudes. Since the sky_longitude doesn't 
    affect mercator distributions, that has been assumed to be zero.

    ToDo
    ----
    Make this a class that can perform forward and reverse transformations
    """

    check_consistent_arg_dims(points_to_transform, unyt.dimensions.length,
                              "points_to_transform")
    check_consistent_arg_dims(observer_distance, unyt.dimensions.length,
                              "observer_distance")
    check_consistent_arg_dims(domain_reference_point, unyt.dimensions.length,
                              "domain_reference_point")

    # this function was originally just written to transform the end points of
    # rays. But, it can be used to transform arbitrary points.
    ray_end_points = points_to_transform

    # in the observer's frame, get cartesian position of ref point
    sky_theta_ref = np.deg2rad(90.0 - sky_latitude_ref_deg)
    sky_phi_ref = 0.0
    _temp = _construct_radial_unit_vector(sky_theta_ref,sky_phi_ref)
    ref_point_rel_to_obs = _temp * observer_distance

    observer_latitude_rad = np.deg2rad(sky_latitude_ref_deg)

    # recompute the ref point's position (relative to the observer's
    # origin) after rotating the observer's coordinate axes to match
    # the orientation of the domain's coordinate axes
    kwargs_rotate_from_obs_axes = {
        'observer_latitude_rad' : np.deg2rad(sky_latitude_ref_deg),
        'domain_theta_rad' : domain_theta_rad,
        'domain_phi_rad' : domain_phi_rad
    }

    rotated_ref_point_rel_to_obs =  rotate_from_obs_axes(
        points = ref_point_rel_to_obs, **kwargs_rotate_from_obs_axes
    )

    def _convert_from_observer_coord(observer_pos):
        # now perform rotations on observer's frame so that the coordinate
        # system matches the orientation of the domain coordinate system
        rotated_observer_pos = rotate_from_obs_axes(
            points = observer_pos,**kwargs_rotate_from_obs_axes
        )

        # now convert observer_pos so that it's measured with respect to 
        # reference point (using the domain cartesian axes)
        translated_temp = rotated_observer_pos - rotated_ref_point_rel_to_obs

        # now convert translated_temp so that it's measured with respect to 
        # the domain origin (using the original cartesian axes)
        return translated_temp + domain_reference_point


    converted_ray_origin = _convert_from_observer_coord(
        np.array([0.0,0.0,0.0])*unyt.cm
    )[0]
    flattened_end_points = ray_end_points.view()
    if ray_end_points.shape != (3,):
        flattened_end_points.shape = (-1,3)
    converted_ray_end_points = _convert_from_observer_coord(
        flattened_end_points
    )
    if ray_end_points.shape != (3,):
        converted_ray_end_points.shape = ray_end_points.shape
    return converted_ray_origin, converted_ray_end_points
