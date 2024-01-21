import numpy as np
cimport numpy as np
import cython
cimport cython
np.import_array()


cdef extern from "math.h":
    cdef double ceil(double arg)
    cdef double floor(double arg)

# _MAX_DOUBLE = np.finfo(np.float64).max
DEF _MAX_DOUBLE = 1.7976931348623157e+308
DEF _INIT_MAX_INTERSECT_D = 1.7976931348623157e+308

# _MIN_DOUBLE = np.finfo(np.float64).min
DEF _MIN_DOUBLE = -1.7976931348623157e+308
DEF _INIT_MIN_INTERSECT_D = -1.7976931348623157e+308

cdef struct IntersectionPair:
    double min_t
    double max_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef IntersectionPair _ray_box_intersections(const double[:] line_start,
                                             const double[:] line_uvec,
                                             const double[:] left_edge,
                                             const double[:] right_edge):
    # all of the arguments are supposed to have 3 elements
    # we also implicitly assume that line_uvec is not all zero

    cdef double min_intersect_d = _INIT_MIN_INTERSECT_D
    cdef double max_intersect_d = _INIT_MAX_INTERSECT_D

    cdef double axis_min_d
    cdef double axis_max_d
    cdef Py_ssize_t i

    for i in range(3):
        if line_uvec[i] == 0:
            if ((line_start[i] < left_edge[i]) or
                (line_start[i] >= right_edge[i])):
                # asymmetrical comparison is intentional

                # modify the values to indicate that there is
                # no intersection at all
                max_intersect_d = _INIT_MIN_INTERSECT_D
                min_intersect_d = _INIT_MAX_INTERSECT_D
                break
            else:
                continue
        elif line_uvec[i] > 0.0:
            axis_min_d = (left_edge[i] - line_start[i]) / line_uvec[i]
            axis_max_d = (right_edge[i] - line_start[i]) / line_uvec[i]
        else:
            axis_min_d = (right_edge[i] - line_start[i]) / line_uvec[i]
            axis_max_d = (left_edge[i] - line_start[i]) / line_uvec[i]

        min_intersect_d = max(axis_min_d, min_intersect_d) # not a typo!
        max_intersect_d = min(axis_max_d, max_intersect_d) # not a typo!

    # we could remove the following branches
    cdef IntersectionPair out
    if (max_intersect_d < min_intersect_d) or (max_intersect_d <= 0): 
        # We do need to check that max_intersect_d is not zero (when 
        # it's zero, the only point of intersection is on the right
        # edge
        # Unsure if we need to explicitly check that max_intersect_d
        # is non-negative.
        out.min_t = -1
        out.max_t = -1
    elif min_intersect_d == max_intersect_d: # intersection with left corner
        out.min_t = min_intersect_d
        out.max_t = -1
    elif min_intersect_d < 0: # ray starts inside of the box
        out.min_t = -1
        out.max_t = max_intersect_d
    else:
        out.min_t = min_intersect_d
        out.max_t = max_intersect_d
    return out

#def ray_box_intersections(double[:] line_start, double[:] line_uvec,
#                          double[:] left_edge, double[:] right_edge):
def ray_box_intersections(line_start, line_uvec,
                          left_edge, right_edge):
    """
    This assumes that the each face of the box runs parallel to the
    coordinate axes

    Parameters
    ----------
    line_start: np.ndarray
        An array of shape (3,) representing the starting point of the line
    line_uvec: np.ndarray
        An array of shape (3,) that holds the unit vector that parameterizes 
        the line. This should be unitless
    left_edge: np.ndarray
        An array of shape (3,) that holds the lower left corner of the box
    right_edge: np.ndarray
        An array of shape (3,) that holds the upper right corner of the box

    Returns
    -------
    distances: list of floats
        An array of up to 2 elements that specify the intersection locations.

    Algorithm
    ---------
    The ordering of coordinates doesn't matter as long as its consistent

    Following the explanation of the Wikipedia article on line-plane 
    intersections:
      - The line can be parameterized as: line_start + line_uvec*d (here, d 
        represents distance since it's multiplied by a unit vector)
      - Consider a plane with normal vector, norm_vec, and that includes the
        point, plane_point
      - The line intersects the plane at a distance:
          d = ( ((plane_point - line_start) @ norm_vec) /
                 (norm_vec @ line_uvec))
        where the @ operator represents a dot product

    For this particular problem, norm_vec is aligned with an axis so that
    simplifies things. Consider the plane where the ith component of norm_vec
    is 1 or -1 (it doesn't matter since the negative sign cancels out) and 
    the other components are zero. The distance to intersection with
    that location is:
          d = (plane_point[i] - line_start[i]) / line_uvec[i]

    Notes
    -----
    The following paper details a faster approach if we just want to know 
    whether the ray intersects the box:
        Williams, Barrus, Morley, & Shirley: An Efficient and Robust Ray-Box 
        Intersection Algorithm. J. Graph. Tools 10(1): 49-54 (2005)
    It's likely that we could take elements of their algorithm to improve this
    implementation.
    """

    assert (line_uvec[0] != 0) or (line_uvec[1] != 1) or (line_uvec[2] != 2)

    assert (line_uvec != 0).any()
    assert line_start.shape == (3,)
    assert line_uvec.shape == (3,)
    assert left_edge.shape == (3,)
    assert right_edge.shape == (3,)

    cdef IntersectionPair pair = _ray_box_intersections(
        line_start.astype(np.float64, copy = False, order = 'C'),
        line_uvec.astype(np.float64, copy = False, order = 'C'),
        left_edge.astype(np.float64, copy = False, order = 'C'),
        right_edge.astype(np.float64, copy = False, order = 'C')
    )

    out = np.empty((2,), dtype = np.float64)
    cdef double[:] out_view = out

    if pair.min_t == -1:
        if pair.max_t == -1:
            return out[:0]
        else:
            out_view[0] = pair.max_t
            return out[:1]
    else:
        out_view[0] = pair.min_t
        if pair.max_t == -1:
            return out[:1]
        else:
            out_view[1] = pair.max_t
            return out

cdef inline double sign(double arg) nogil:
    return (0.0 < arg) - (arg < 0.0)

cdef inline double _floor_subtract_divide(double a, double b,
                                          double divisor):
    """
    Computes floor((a - b)/divisor).

    This correctly handles the case when a >> b or a << b
    """

    cdef double diff = a - b
    cdef double truncated_amount = a - (diff + b)
    # diff + truncated_amount gives the correct difference

    cdef double quotient = diff / divisor
    cdef double out = floor(quotient)
    # it's possible to refactor to avoid the branch
    if (out == quotient) and ((sign(truncated_amount) * sign(divisor)) < 0):
        return out - 1.0
    return out

cdef inline double _ceil_subtract_divide(double a, double b,
                                         double divisor):
    """
    Computes ceil((a - b)/divisor).

    This correctly handles the case when a >> b or a << b
    """
    cdef double diff = a - b
    cdef double truncated_amount = a - (diff + b)
    # diff + truncated_amount gives the correct difference

    cdef double quotient = diff / divisor
    cdef double out = ceil(quotient)
    if (out == quotient) and ((sign(truncated_amount) * sign(divisor)) > 0):
        return out + 1
    return out

def max_num_intersections(grid_shape):
    # conservative upper bound on the maximum number of grid intersections
    # the max number is actually smaller (and related to the max number of
    # intersectable cells by a diagonal line)
    return np.sum(grid_shape)

from typing import Tuple,Union
import unyt

def alt_build_spatial_grid_props(cm_per_length_unit: float,
                                 grid_shape: Union[Tuple[int,int,int],
                                                   np.ndarray],
                                 grid_left_edge: np.ndarray,
                                 cell_width: np.ndarray):

    # this is a dummy function that was created to help write tests since the
    # __init__ method of SpatialGridProps probably did a little too much

    grid_shape = np.array(grid_shape)

    spatial_props = SpatialGridProps(
        cm_per_length_unit = cm_per_length_unit,
        grid_shape = grid_shape,
        grid_left_edge = grid_left_edge,
        grid_right_edge = grid_left_edge + cell_width * grid_shape)

    # this is a little bit of a hack
    spatial_props.cell_width = cell_width.astype(dtype = 'f8', casting = 'safe',
                                                 copy = True)
    return spatial_props

def _coerce_grid_shape(grid_shape: Union[Tuple[int,int,int], np.ndarray]):
    _TYPE_ERR = (
        "grid_shape must be a 3-tuple of integers OR a numpy array of integers"
    )

    if isinstance(grid_shape, tuple): 
        if ((len(grid_shape) != 3) or
            not all(isinstance(e,int) for e in grid_shape)):
            raise TypeError(_TYPE_ERR)
        out = np.array(grid_shape, dtype = 'i8')
    elif isinstance(grid_shape, np.ndarray):
        if not issubclass(grid_shape.dtype.type, np.integer):
            raise TypeError(
                f"{_TYPE_ERR}; it's a numpy array with a different dtype")
        out = grid_shape
    else:
        raise TypeError(_TYPE_ERR)

    if not (out>=0).all():
        raise ValueError("All elements of grid_shape must be positive")
    return out

def _inspect_edge_args(grid_left_edge, grid_right_edge, allow_unyt = False):

    def _check(arg, arg_name):
        if not isinstance(arg, np.ndarray):
            raise TypeError(f"{arg_name} must be a numpy array")
        elif isinstance(arg, unyt.unyt_array) and not allow_unyt:
            raise TypeError(f"{arg_name} must NOT be a unyt_array")
        elif arg.shape != (3,):
            raise ValueError(f"{arg_name} must have a shape of (3,)")
        elif not np.isfinite(arg).all():
            raise ValueError(f"all elements in {arg_name} must be finite")
    _check(grid_left_edge, "grid_left_edge")
    _check(grid_right_edge, "grid_right_edge")

    if (grid_left_edge >= grid_right_edge).any():
        raise ValueError("each element in grid_right_edge should exceed the "
                         "corresponding element in grid_left_edge")



cdef assign_spatial_props_vals(SpatialGridPropsStruct* pack,
                               double cm_per_length_unit,
                               object grid_shape,
                               object left_edge,
                               object right_edge,
                               object cell_width):
    pack.cm_per_length_unit = cm_per_length_unit
    cdef Py_ssize_t i
    for i in range(3):
        pack.grid_shape[i] = grid_shape[i]
        pack.left_edge[i] = left_edge[i]
        pack.right_edge[i] = right_edge[i]
        pack.cell_width[i] = cell_width[i]

        pack.inv_cell_width[i] = 1.0 / cell_width[i]

cdef object create_3elem_array_cpy(double* ptr):
    return np.array([float(ptr[0]), float(ptr[1]), float(ptr[2])])

cdef class SpatialGridProps:
    """
    This collects spatial properties of the current block (or grid) of the
    dataset

    Notes
    -----
    The original motivation for doing this was to allow rescaling of the grid
    in adiabatic simulations. It's unclear whether that functionality still
    works properly (it definitely hasn't been tested in all contexts).
    """

    # note: we don't have a line with `cdef SpatialGridPropsStruct pack` here
    #       since it is declared in the associated header file

    @property
    def cm_per_length_unit(self):
        return self.pack.cm_per_length_unit

    @property
    def grid_shape(self):
        return np.array(
            [self.pack.grid_shape[0], self.pack.grid_shape[1],
             self.pack.grid_shape[2]]
        )

    @property
    def left_edge(self):
        return create_3elem_array_cpy(&self.pack.left_edge[0])

    @property
    def right_edge(self):
        return create_3elem_array_cpy(&self.pack.right_edge[0])

    @property
    def cell_width(self):
        return create_3elem_array_cpy(&self.pack.cell_width[0])

    @cell_width.setter
    def cell_width(self, value):
        assert np.ndim(value) == 1 and len(value) == 3
        for i in range(3):
            self.pack.cell_width[i] = value[i]
            self.pack.inv_cell_width[i] = 1.0 / value[i]

    def __getstate__(self):
        # used in the process of pickling (constructing the pickle data)
        out = {}
        for attr in ['cm_per_length_unit','grid_shape',
                     'left_edge', 'right_edge', 'cell_width']:
            out[attr] = getattr(self, attr)
        return out

    def __setstate__(self, state):
        # used in the process of unpickling
        assign_spatial_props_vals(
            pack = &(self.pack),
            cm_per_length_unit = state['cm_per_length_unit'],
            grid_shape = state['my_grid_shape'],
            left_edge = state['my_left_edge'],
            right_edge = state['my_right_edge'],
            cell_width = state['my_cell_width']
        )

    def __init__(self, *, cm_per_length_unit: float,
                 grid_shape: Union[Tuple[int,int,int], np.ndarray],
                 grid_left_edge: np.ndarray,
                 grid_right_edge: np.ndarray,
                 rescale_factor: float = 1.0):

        if cm_per_length_unit <= 0 or not np.isfinite(cm_per_length_unit):
            raise ValueError("cm_per_length_unit must be positive & finite")
        elif rescale_factor <= 0 or not np.isfinite(rescale_factor):
            raise ValueError("rescale factor must be positive & finite")
        cm_per_length_unit = cm_per_length_unit

        my_grid_shape = _coerce_grid_shape(grid_shape)

        _inspect_edge_args(grid_left_edge, grid_right_edge, allow_unyt = False)
        # in each of the following cases, we pass copy = False, since we know
        # that the input array is newly created!
        _kw = dict(order = 'C', dtype = 'f8', casting = 'safe', copy = False)
        my_left_edge = (grid_left_edge * rescale_factor).astype(**_kw)
        my_right_edge = (grid_right_edge * rescale_factor).astype(**_kw)
        my_cell_width = (
            (my_right_edge - my_left_edge) / grid_shape
        ).astype(**_kw)

        assign_spatial_props_vals(pack = &(self.pack),
                                  cm_per_length_unit = cm_per_length_unit,
                                  grid_shape = my_grid_shape,
                                  left_edge = my_left_edge,
                                  right_edge = my_right_edge,
                                  cell_width = my_cell_width)


    @classmethod
    def build_from_unyt_arrays(cls, *, cm_per_length_unit: float,
                               grid_shape: Union[Tuple[int, int, int],
                                                 np.ndarray],
                               grid_left_edge: unyt.unyt_array,
                               grid_right_edge: unyt.unyt_array,
                               length_unit: Union[str,unyt.Unit],
                               rescale_factor: float = 1.0):
        """
        This factory method is the preferred way to build a SpatialGridProps
        instance (it ensures that we are being self-consistent with units

        Notes
        -----
        This behavior used to be used by our constructor, but we found that to
        be a little limiting...

        Unclear whether this should be a separate function detached from
        SpatialGridProps
        """

        return cls(
            cm_per_length_unit = cm_per_length_unit,
            grid_shape = grid_shape,
            grid_left_edge = grid_left_edge.to(length_unit).ndview,
            grid_right_edge = grid_right_edge.to(length_unit).ndview,
            rescale_factor = rescale_factor)

    def __repr__(self):
        # Since we don't manually define a __str__ method, this will be called
        # by the built-in implementation
        #
        # at this moment, because the __init__ statement does a little too
        # much, this description won't be super useful
        return (
            "SpatialGridProps(\n"
            f"    cm_per_length_unit = {self.cm_per_length_unit!r},\n"
            f"    grid_shape = {np.array2string(self.grid_shape)},\n"
            "    grid_left_edge = "
            f"{np.array2string(self.left_edge, floatmode = 'unique')},\n"
            "    grid_right_edge = "
            f"{np.array2string(self.right_edge, floatmode = 'unique')},\n"
            "    cell_width = "
            f"{np.array2string(self.cell_width, floatmode = 'unique')},\n"
            ")"
        )

# these are here mostly just to make sure we perform these calculations
# consistently (plus, if we ever decide to change up this strategy, based on an
# internal change to the attributes, we just need to modify these

def get_grid_center(spatial_props):
    return 0.5 * (spatial_props.left_edge + spatial_props.right_edge)

def get_grid_width(spatial_props):
    return spatial_props.right_edge - spatial_props.left_edge

def spatial_props_from_ds(ds, length_unit = 'code_length', *,
                          rescale_factor = 1.0):
    """
    Construct a SpatialGridProps instance for a yt-dataset

    Parameters
    ----------
    ds
        The yt-dataset from which we will construct spatial properties
    length_unit: str
        This represents the length units associated with the quantities in the
        resulting object.
    rescale_length_factor: float, optional
        The width of each cell is multiplied by this factor. (This is intended
        for use with scale-free simulations, it effectively holds the position
        of the simulation's origin fixed in place).
    """
    if ds.coordinates.axis_order != ('x','y','z'):
        raise NotImplementedError("Currently, we only support simulations "
                                  "where axes are ordered ('x','y','z')")
    return SpatialGridProps(
        cm_per_length_unit = float(ds.quan(1.0, length_unit).to('cm').v),
        grid_shape = ds.domain_dimensions,
        grid_left_edge = ds.domain_left_edge.to(length_unit).ndview,
        grid_right_edge = ds.domain_right_edge.to(length_unit).ndview,
        rescale_factor = rescale_factor)
