import warnings

import numpy as np
import unyt

from .ray_creation import \
    convert_observer_to_domain_points, _find_obs_ray_end_points, \
    _magnitude
from .utils.misc import _has_consistent_dims

def _is_np_ndarray(obj): # confirm its an instance, but not a subclass
    return obj.__class__ is np.ndarray

def _convert_vec_l_to_uvec_l(vec_l):
    assert (vec_l.shape[0] >= 1) and  (vec_l.shape[1:] == (3,))
    mag_square = (vec_l*vec_l).sum(axis=1)
    if (mag_square == 0.0).any():
        raise RuntimeError("there's a case where the vector has 0 magnitude")
    return vec_l/np.sqrt(mag_square[np.newaxis].T)

def _check_ray_args(arg0, arg1, name_pair,
                    expect_3D_arg0 = False,
                    expect_1D_arg1 = False,
                    check_type = True):
    arg0_name, arg1_name = name_pair
    if check_type and not _is_np_ndarray(arg0):
        raise TypeError(
            f"{arg0_name} must be a np.ndarray, not '{type(arg0).__name__}'")
    elif check_type and not _is_np_ndarray(arg1):
        raise TypeError(
            f"{arg1_name} must be a np.ndarray, not '{type(arg1).__name__}'")
    elif arg0.shape[-1] != 3:
        raise ValueError(f"{arg0_name}'s shape, {arg0.shape}, is invalid. "
                         "The last element must be 3")
    elif arg1.shape[-1] != 3:
        raise ValueError(f"{arg1_name}'s shape, {arg1.shape}, is invalid. "
                         "The last element must be 3")
    elif (any(e < 1 for e in arg0.shape[:-1]) or
          any(e < 1 for e in arg1.shape[:-1])):
        raise RuntimeError("not clear how this situation could occur")

    if expect_3D_arg0 and arg0.ndim != 3:
        raise ValueError(f"{arg0_name} must have 3 dims, not {arg0.ndim}")
    elif (not expect_3D_arg0) and (arg0.ndim != 2):
        raise ValueError(f"{arg0_name} must have 2 dims, not {arg0.ndim}")

    if expect_1D_arg1 and (arg1.ndim != 1):
        raise ValueError(f"{arg1_name} must have 1 dim, not {arg1.ndim}")
    elif (not expect_1D_arg1) and (arg1.ndim !=2):
        raise ValueError(f"{arg1_name} must have 2 dims, not {arg1.ndim}")

    if ((not expect_3D_arg0) and (not expect_1D_arg1) and
        (arg1.shape != arg0.shape)):
        raise ValueError(f"{arg0_name} and {arg1_name} have a shape mismatch, "
                         f"{arg0.shape} and {arg1.shape}")

    # might want to enforce the following in the future
    #if arg0.strides[-1] != arg0.dtype.itemsize:
    #    raise ValueError(f"{arg0_name} must be contiguous along axis -1")
    #elif arg1.strides[-1] != arg1.dtype.itemsize:
    #    raise ValueError(f"{arg1_name} must be contiguous along axis -1")


class ConcreteRayList:
    """
    Immutable list of rays.

    Note
    ----
    We choose to store ray_vec instead of ray_uvec for the following reasons:

    - It's difficult to check that a floating-point unit-vector is actually a
      unit vector.

    - usually, you're better off just doing the normalization when you use the
      unit vector. But if you do the normalization on the unit vector, then you
      can wind up with slightly different values. Functionally, this shouldn't
      change things, but it could complicate testing...

    TODO
    ----
    we should move away from assuming that the ray_start values in this class
    have units of 'code_length'. This introduces some headaches.
    - In the long-term, it may be useful to keep this pickleable, so we should
      probably not store a unyt object directly inside this class (at least,
      we don't want a unyt object depending on sim-specific units). Maybe we
      should just store the name of the units.
    """

    def __init__(self, ray_start_codeLen, ray_vec):

        _check_ray_args(ray_start_codeLen, ray_vec,
                        ("ray_start_codeLen", "ray_vec"))

        self.ray_start_codeLen = ray_start_codeLen
        self._ray_vec = ray_vec

        problems = (np.square(ray_vec).sum(axis = 1) == 0)
        if problems.any():
            raise RuntimeError(f"ray_vec at indices {np.where(tmp)[0]} "
                               "specify 0-vectors")

    @classmethod
    def from_start_stop(cls, ray_start_codeLen, ray_stop_codeLen):
        _check_ray_args(ray_start_codeLen, ray_stop_codeLen,
                        ("ray_start_codeLen", "ray_stop_codeLen"))
        return cls(ray_start_codeLen = ray_start_codeLen,
                   ray_vec = ray_stop_codeLen - ray_start_codeLen)

    def __repr__(self):
        return (f'{type(self).__name__}({self.ray_start_codeLen!r}, '
                f'{self._ray_vec!r})')

    def __len__(self):
        return self.ray_start_codeLen.shape[0]

    @property
    def shape(self):
        return (len(self.shape),)

    def get_ray_uvec(self):
        return _convert_vec_l_to_uvec_l(self._ray_vec)

    # this is temporarily commented out since we'll get funky behavior when
    # self.ray_start_codeLen is not actually in code-units
    # -> in principle, this can happen if as_concrete_ray_list(arg), with an
    #    argument other than 1 code-length
    #def domain_edge_sanity_check(self, self.left, self.right):
    #    if np.logical_and(self.ray_start_codeLen >= _l,
    #                      self.ray_start_codeLen <= _r).all():
    #        raise RuntimeError('We can potentially relax this in the future.')

    def get_selected_raystart_rayuvec(self, idx):
        ray_start = self.ray_start_codeLen[idx, :]
        ray_start.flags['WRITEABLE'] = False

        ray_uvec = self.get_ray_uvec()[idx, :]
        ray_uvec.flags['WRITEABLE'] = False

        # if self.ray_start_codeLen.strides[0] OR self._ray_vec.strides[0] is
        # equal to 0, we may be able to get clever (and reduce the size of the
        # returned arrays... - this could be useful when they are dispatched as
        # messages)
        return ray_start, ray_uvec


class PerspectiveRayGrid2D:
    """
    A 2D Grid of rays that all trace to a single observer.

    Users should avoid constructing this directly. Instead, they should 
    construct instances via build_perspective_grid.

    Parameters
    ----------
    ray_start: 1D `unyt.unyt_array`
        A (3,) array that specifies the observer's location (with respect to
        the dataset's coordinate system). Currently, this must be located
        outside of the simulation domain.
    ray_stop: 3D `unyt.unyt_array`
        A (m,n,3) array specifying the location where each ray stops.
    """
    def __init__(self, ray_start, ray_stop):
        _check_ray_args(ray_stop, ray_start, ("ray_stop", "ray_start"),
                        expect_3D_arg0 = True, expect_1D_arg1 = True,
                        check_type = False)

        assert _has_consistent_dims(ray_start, unyt.dimensions.length)
        assert _has_consistent_dims(ray_stop, unyt.dimensions.length)

        self.ray_start = ray_start
        self.ray_stop = ray_stop

    @property
    def shape(self):
        return self.ray_stop.shape[:-1]

    def domain_edge_sanity_check(self, left, right):
        if np.logical_and(self.ray_start >= left,
                          self.ray_start <= right).all():
            raise RuntimeError('We can potentially relax this in the future.')

    def as_concrete_ray_list(self, length_unit_quan):
        # at the moment, length_unit_quan should really specify code_length

        assert _has_consistent_dims(length_unit_quan, unyt.dimensions.length)

        # TODO: consider trying out np.broadcast_to to attempt to reduce space
        #       (while the result won't be contiguous along axis 0, it should
        #       still be contiguous along axis 1)


        # TODO: with some refactoring, we could probably avoid rescaling
        #       self.ray_stop

        # the current convention is for length_unit_quan to specify the
        # code_length
        ray_start = self.ray_start.to('cm').v /length_unit_quan.to('cm').v
        ray_stop = self.ray_stop.to('cm').v / length_unit_quan.to('cm').v

        _ray_stop_2D = ray_stop.view()
        _ray_stop_2D.shape = (-1,3)
        assert _ray_stop_2D.flags['C_CONTIGUOUS']

        num_list_entries = _ray_stop_2D.shape[0]

        return ConcreteRayList.from_start_stop(
            np.tile(ray_start, (num_list_entries, 1)),
            _ray_stop_2D
        )



def perspective_ray_grid(sky_latitude_ref_deg, observer_distance,
                         sky_delta_latitude_arr_deg,
                         sky_longitude_arr_deg,
                         domain_theta_rad,
                         domain_phi_rad,
                         *, ds, rescale_length_factor = None):
    """
    Construct a regularly spaced 2d grid of rays that all trace back to a 
    single observer.

    The rays are sampled in sky_latitude (like declination) and sky_longitude
    (like right_ascension). The spacing in these samples are constant, you will
    have a mercator projection

    Parameters
    ----------
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
    observer_distance : unyt.unyt_quantity
        Distance between the observer and the reference point (which currently 
        coincides with the origin of the simulation domain).
    sky_latitude_ref_deg: float, optional
        specify the latitude on the sky (from the observer's perspective) that 
        the primary ray passes through. This can have significant implications 
        for the mercator projection. The default value is 0.0 (corresponding to
        the celestial equator). 
    domain_theta_rad, domain_phi_rad: float
        Alongside obs_distance, these quantities specify the location of the 
        observer in spherical coordinates relative to the reference point
        (which is currently assumed to coincide with the origin of the
        simulation domain). Note that the spherical coordinates assuming that 
        the x,y, and z directions align with the simulation's native coordinate
        axes (for reference, the x,y,z directions in the observer's frame are
        generally different).
    ds
        The simulation from which the output coordinate system is taken. In
        the near future, this will no longer be necessary.
    rescale_length_factor: float, Optional
        When not `None`, the width of each cell is multiplied by this factor.
        (This effectively holds the position of the simulation's origin fixed
        in place).
    """

    assert ds.coordinates.axis_order == ('x', 'y', 'z')

    # the following is somewhat roundabout...
    end_points = _find_obs_ray_end_points(
        ds, sky_latitude_ref_deg = sky_latitude_ref_deg,
        observer_distance = observer_distance,
        sky_delta_latitude_arr_deg = sky_delta_latitude_arr_deg,
        sky_longitude_arr_deg = sky_longitude_arr_deg,
        rescale_length_factor = rescale_length_factor
    )

    ray_start, ray_stop = convert_observer_to_domain_points(
        points_to_transform = end_points,
        observer_distance = observer_distance,
        sky_latitude_ref_deg = sky_latitude_ref_deg,
        domain_reference_point = ds.arr([0.0,0.0,0.],'cm'),
        domain_theta_rad = domain_theta_rad,
        domain_phi_rad = domain_phi_rad
    )
    return PerspectiveRayGrid2D(ray_start, ray_stop)


class ParallelRayGrid2D:
    # in this class, positioning of ray_start is not actually all that
    # important. Neglecting numerical effects, moving the ray_start
    # closer/further from the domain shouldn't have any practical effect...
    #
    # Maybe we should add a flag encoding whether the user really values the
    # ray_start values...
    def __init__(self, ray_start, ray_vec):
        _check_ray_args(ray_start, ray_vec, ("ray_start", "ray_vec"),
                        expect_3D_arg0 = True, expect_1D_arg1 = True,
                        check_type = False)

        assert _has_consistent_dims(ray_start, unyt.dimensions.length)
        assert _is_np_ndarray(ray_vec) # should be unitless

        self.ray_start = ray_start
        self._ray_vec = ray_vec

    @property
    def shape(self):
        return self.ray_start.shape[:-1]

    def domain_edge_sanity_check(self, left, right):
        _ray_start_2D = self.ray_start.view()
        _ray_start_2D.shape = (-1,3)
        for i in range(_ray_start_2D.shape[0]):
            cur_ray_start = _ray_start_2D[i,:]
            if np.logical_and(cur_ray_start >= left,
                              cur_rat_start <= right).all():
                raise RuntimeError('We can potentially relax this in the '
                                   'future.')

    def as_concrete_ray_list(self, length_unit_quan):
        # at the moment, length_unit_quan should really specify code_length

        assert _has_consistent_dims(length_unit_quan, unyt.dimensions.length)

        # TODO: consider trying out np.broadcast_to to attempt to reduce space
        #       (while the result won't be contiguous along axis 0, it should
        #       still be contiguous along axis 1)

        # the current convention is for length_unit_quan to specify the
        # code_length
        ray_start = self.ray_start.to('cm').v /length_unit_quan.to('cm').v

        _ray_start_2D = ray_start.view()
        _ray_start_2D.shape = (-1,3)
        assert _ray_start_2D.flags['C_CONTIGUOUS']

        num_list_entries = _ray_start_2D.shape[0]

        return ConcreteRayList(_ray_start_2D,
                               np.tile(self._ray_vec, (num_list_entries, 1)))


def _build_2D_grid_of_3D_points(x_vals, y_vals, z_vals, unit_choice = None):
    # this needs to be cleaned up. I probably got the output-shape wrong!

    if unit_choice is None:
        unit_choice = x_vals.units

    x_vals = x_vals.to(unit_choice).v
    y_vals = y_vals.to(unit_choice).v
    z_vals = z_vals.to(unit_choice).v

    # merged has a shape (x_vals.size, y_vals.size, 1, 3). Convert to
    # (x_vals.size, y_vals.size, 3)

    assert (x_vals.ndim == 0) and (y_vals.ndim == 1) and (z_vals.ndim == 1)
    yy,zz = np.meshgrid(y_vals, z_vals, indexing = 'xy')
    merged = np.stack([np.broadcast_to(x_vals,yy.shape),yy,zz], axis = -1)
    return unyt.unyt_array(merged, unit_choice)



def parallel_ray_grid(delta_im_x, delta_im_y,
                      domain_theta_rad, domain_phi_rad,
                      *, ds, observer_distance = None):
    """
    Construct a collection of rays that are all parallel to each other.

    You can imagine placing an observer relative to a simulation at an 
    arbitrary distance. This creates a grid of rays that are parallel to a line
    drawn between the observer and the simulation reference point (currently
    fixed at the origin).

    If ``delta_im_x``, ``delta_im_y`` are specified in terms of angles, the 
    ``observer_distance`` argument is used to convert them to length-units
    using the small angle approximation: 
        `` angle / radians = transverse_displacement / dist ``

    In the limit where ``observer_distance`` is much, much larger than the 
    simulation domain, differences between the resulting object and the object
    produced by ``perspective_ray_grid`` should disappear.

    Parameters
    ----------
    delta_im_x, delta_im_y: `unyt.unyt_array`
        1D arrays of x and y positions that specify where rays should be 
        traced. 
        along which rays
    domain_theta_rad, domain_phi_rad: float
        These quantities specify the location of the observer in spherical 
        coordinates relative to the reference point (which is currently assumed
        to coincide with the origin of the simulation domain).
    ds
        The simulation from which the output coordinate system is taken. In
        the near future, this will no longer be necessary.
    observer_distance : ``unyt.unyt_quantity``, optional
        Distance between the observer and the reference point (which currently 
        coincides with the origin of the simulation domain). This is only
        needed when ``delta_im_x`` and ``delta_im_y`` are have units associated
        with angles.

    Note
    ----
    This follows slightly different conventions
    """

    # NOTE: we don't really need to know much about the simulation domain other
    #       than its size. A potential optimization could be to just allow
    #       users to pass that information in...

    # TODO: maybe add support for rescale_length_factor?
    rescale_length_factor = None

    # Prologue: argument checking/coercion

    assert (observer_distance is None or
            (_has_consistent_dims(observer_distance, unyt.dimensions.length) and
             (observer_distance.v > 0)))

    assert np.ndim(delta_im_x) == 1 and np.shape(delta_im_x)[0] > 0
    assert np.ndim(delta_im_y) == 1 and np.shape(delta_im_y)[0] > 0

    if (_has_consistent_dims(delta_im_x, unyt.dimensions.angle) and
        _has_consistent_dims(delta_im_y, unyt.dimensions.angle)):
        if observer_distance is None:
            raise ValueError("observer_distance must not be None when "
                             "delta_im_x and delta_im_y are angles")
        delta_im_x = delta_im_x.to('radian').v * observer_distance
        delta_im_y = delta_im_y.to('radian').v * observer_distance
    elif not (_has_consistent_dims(delta_im_x, unyt.dimensions.length) and
              _has_consistent_dims(delta_im_y, unyt.dimensions.length)):
        raise ValueError("delta_im_x and delta_im_y must both be distances or "
                         "angles")

    # come up with an effective distance to the observer. Since the rays
    # are parallel, the observer's position doesn't really matter, as long
    # as all rays end outside of the frame. For simplicity, we come up with
    # an arbitrarily sized min_distance
    assert ds.coordinates.axis_order == ('x', 'y', 'z')
    min_dist = 1.1 * _magnitude(ds.domain_left_edge - ds.domain_right_edge)
    if (observer_distance is not None) and (observer_distance > min_dist):
        eff_obs_dist = observer_distance
    else:
        eff_obs_dist = min_dist

    if True:
        # strive for consistency with the parallel-ray case:
        # -> the domain-reference point is located at positive x. Specifically,
        #    it can be found at (x,y,z) = (+eff_obs_dist, 0, 0)
        # -> image_x is similar to values of sky-longitude... (so aligned with
        #    the y-axis)
        # -> image_y is similar to values of sky-latitude... (so aligned with
        #    z-axis)

        ray_start_obs_frame = _build_2D_grid_of_3D_points(
            x_vals = 0.0*unyt.km, y_vals = delta_im_x, z_vals = delta_im_y,
            unit_choice = delta_im_x.units)

        domain_reference_point = unyt.unyt_array([0.0,0.0,0.],'cm')

        effective_observer_loc, ray_start = convert_observer_to_domain_points(
            points_to_transform = ray_start_obs_frame,
            observer_distance = eff_obs_dist,
            sky_latitude_ref_deg = 0.0,
            domain_reference_point = domain_reference_point,
            domain_theta_rad = domain_theta_rad,
            domain_phi_rad = domain_phi_rad
        )

        # all rays run parallel to the ray connecting effective_observer_loc
        ray_vec = (domain_reference_point - effective_observer_loc).v

        return ParallelRayGrid2D(ray_start, ray_vec)
    else:
        raise RuntimeError("UNTESTED")
        # probably would be useful to do something more similar to yt

        # come up with (or user specifies normal vector) - the vector between
        # observer and the domain-reference-point

        # in domain reference frame:
        observer_loc = eff_obs_dist.v * np.array([
            np.sin(domain_theta_rad) * np.cos(domain_phi_rad), # x
            np.sin(domain_theta_rad) * np.sin(domain_phi_rad), # y
            np.cos(domain_theta_rad) # z
        ])

        normal_vec = -1 * normal_vec / _magnitude(normal_vec)
        north_vec = None # user-specified...

        im_y_uvec = (
            north_vec  -
            np.dot(north_vec, normal_vec) * normal_vec / _magnitude(normal_vec)
        )
        im_y_uvec = im_y_uvec / magnitude(im_y_vec)

        # might want to check sign of im_x_uvec...
        im_x_uvec = np.cross(normal_vec, im_y_uvec)
        im_x_uvec = magntiude(im_x_uvec)

        ray_start_unit= delta_im_x.units
        ray_start = unyt.unyt_array(
            np.empty((delta_im_x.size, delta_im_y.size, 3), dtype = np.float64),
            units = ray_start_unit
        )

        for i in range(delta_im_x.size):
            for j in range(delta_im_y.size):
                ray_start[i,j,:] = (observer_loc + im_x_uvec * delta_im_x[i] + 
                                    im_y_uvec * delta_im_y[i]).to(ray_start_unit)
        return ParallelRayGrid2D(ray_start = ray_start,
                                 ray_vec = normal_vec)
