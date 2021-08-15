import numpy as np
cimport numpy as np
import cython
np.import_array()


#cdef extern from "math.h":
#    cdef double ceil(double arg)
#    cdef double floor(double arg)

cdef double _INIT_MIN_INTERSECT_D = np.finfo(np.float64).min
cdef double _INIT_MAX_INTERSECT_D = np.finfo(np.float64).max

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

    #out = np.empty((2,), dtype = np.float64)
    #cdef double[:] out_view = out

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

    if (max_intersect_d < min_intersect_d) or (max_intersect_d <= 0): 
        # We do need to check that max_intersect_d is not zero (when 
        # it's zero, the only point of intersection is on the right
        # edge
        # Unsure if we need to explicitly check that max_intersect_d
        # is non-negative.
        return np.array([],dtype = np.float64)
    elif min_intersect_d == max_intersect_d:
        # intersection with a corner
        return np.array([min_intersect_d])
    elif min_intersect_d < 0: # ray starts inside of the box
        # should we differentiate this from corner-intersection?
        return np.array([max_intersect_d])
    else:
        return np.array([min_intersect_d,max_intersect_d])
