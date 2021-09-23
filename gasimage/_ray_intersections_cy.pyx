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

cdef double clip(double a, double a_min, double a_max):
    return min(max(a, a_min), a_max)


def _starting_index(const double[:] line_uvec, const double[:] line_start,
                    const double[:] grid_left_edge, const double[:] cell_width,
                    const double[:] grid_shape):
    # identify the index of the first cell-centered that the ray
    # intersects
    #
    #
    # All of the arguments are assumed to be 3-element arrays

    grid_right_edge_arr = np.empty((3,), dtype = np.float64)
    cdef double[:] grid_right_edge = grid_right_edge_arr

    

    # I think it's probably important that the right grid edge is computed from
    # grid_left_edge, cell_width, and grid_shape (to have more consistent
    # rounding handling)
    cdef Py_ssize_t i
    for i in range(3):
        grid_right_edge[i] = grid_left_edge[i] + cell_width[i] * grid_shape[i]

    cdef IntersectionPair intersect_t = _ray_box_intersections(
        line_start, line_uvec, grid_left_edge, grid_right_edge
    )

    # forward declare some temporary variables
    cdef double t_start
    line_entry_arr = np.empty((3,), dtype = np.float64)
    cdef double[:] line_entry = line_entry_arr
    cell_index_arr = np.empty((3,), dtype=np.intc)
    cdef int[:] cell_index = cell_index_arr

    if ((intersect_t.min_t == -1.0) or (intersect_t.max_t == -1)):
        # the ray starts within the box, just hits the lower left corner,
        # or never intersects the box
        raise NotImplementedError()
    else:
        t_start = intersect_t.min_t

        # compute location where line intersects the grid
        for i in range(3):
            line_entry[i] = clip(
                line_start[i] + line_uvec[i] * t_start,
                a_min = grid_left_edge[i],
                a_max = grid_right_edge[i],
            )

        # compute the first cell-centered index including the cell
        # compute the normalized_index:
        #     normalized_index = (line_entry - grid_left_edge) / cell_width
        # An integer valued normalized_index coincides with a cell face (a
        # value of 0 coincides with the left edge of the leftmost cell)
        #
        # normalized_index:    (i-1)     i     (i+1)
        #                        +-------+-------+
        #                        |       |       |
        # cell-centered index:   | (i-1) |   i   |
        #                        |       |       |
        #                        +-------+-------+
        # The starting cell-centered index for rays that enter the grid on 
        # a non-integer normalized_index is always floor(normalized_index)
        
        for i in range(3):
            if line_uvec[i] >= 0:
                # (handle line_uvec[i] == 0 AND line_uvec[i] > 0)
                #
                # if int(normalized_index) == normalized_index, the ray's starting
                # index is the cell for which the ray is on the LEFT edge
                #
                # index[i] = floor(normalized_index[i])
                #   = floor((line_entry[i] - grid_left_edge[i]) / cell_width[i])
                cell_index[i] = int(_floor_subtract_divide(
                    line_entry[i], grid_left_edge[i], cell_width[i]
                ))
            else: # inv_line_uvec[i] < 0
                # if int(normalized_index) == normalized_index, the ray's starting
                # index is the cell for which the ray is on the RIGHT edge
                #
                # index[i] = ceil(normalized_index[i]) - 1
                #   = ceil((line_entry[i] - grid_left_edge[i]) / cell_width[i]) - 1
                cell_index[i] = int(_ceil_subtract_divide(
                    line_entry[i], grid_left_edge[i], cell_width[i]
                )) - 1

    if (np.logical_or(cell_index_arr < 0,
                      cell_index_arr >= np.array(grid_shape)).any() or
        t_start < 0.0):
        print("Problems were encountered while determining the starting index.")
        print('line_start: ({}, {}, {})'.format(line_start[0], line_start[1],
                                                line_start[2]))
        print('line_uvec: ({}, {}, {})'.format(line_uvec[0], line_uvec[1],
                                               line_uvec[2]))
        print(t_start)

        raise AssertionError()

    return cell_index_arr, t_start


# the following can all be dramatically optimized
cdef struct CalcNextFaceTParams:
    double line_uvec
    double grid_left_edge
    double line_start
    double cell_width

cdef inline double _calc_next_face_t(int cur_cell_index,
                                     CalcNextFaceTParams params) nogil:
    # compute the distance to the next face from the 
    # start of the line

    # first, compute determine the index of the next face
    # note: face-index 0 corresponds to the left face of the
    # cell with index = 0

    cdef int next_face_ind
    if params.line_uvec > 0:
        next_face_ind = cur_cell_index + 1
    elif params.line_uvec < 0:
        next_face_ind = cur_cell_index
    else:
        raise RuntimeError()

    cdef double next_face_pos = (
        params.grid_left_edge + next_face_ind * params.cell_width
    )

    cdef double out = (
        (next_face_pos - params.line_start) / params.line_uvec
    )
    if out < 0:
        raise AssertionError()
    return out

def traverse_grid(line_uvec, line_start,
                  grid_left_edge, cell_width,
                  grid_shape):
    """
    Computes the grid indices that the ray intersects and computes
    the distance of the ray in each cell.
    
    This should not be called for a ray that just clips the corner
    
    Parameters
    ----------
    line_uvec
        Unit vector that runs parallel to the line
    line_start
        Location where the line enters the grid

    Returns
    -------
    indices: (3,N) array
        This is done to facillitate indexing of a numpy array
    distances: (N,) array
        The distance that a line travels through a given cell
    Notes
    -----
    None of the arguments should be an instance of a yt.YTArray
    
    In the future, consider encoding the cell indices as a 1d index
    in indices.
    """

    assert len(line_uvec) == 3
    assert not (line_uvec == 0).all()
    assert len(line_start) == 3
    assert len(grid_left_edge) == 3
    assert len(cell_width) == 3
    assert (cell_width > 0).all()
    assert len(grid_shape) == 3

    line_uvec = line_uvec.astype(np.float64, copy = False, order = 'C')
    line_start = line_start.astype(np.float64, copy = False, order = 'C')
    grid_left_edge = grid_left_edge.astype(np.float64, copy = False,
                                           order = 'C')
    cell_width = cell_width.astype(np.float64, copy = False,
                                   order = 'C')


    # be conservative about the max number of intersections
    # the max number is actually smaller (and related to the
    # max number of intersectable cells by a diagonal line)
    max_num = np.sum(grid_shape)

    # I'm using np.compat.long to be explicit that I want the "long" data type
    indices_arr = np.empty((3,max_num), np.compat.long, order = 'C')
    # tell the memory view that indices is C-contiguous
    cdef long[:, ::1] indices = indices_arr

    distances_arr = np.empty((max_num,), np.float64)
    cdef double[:] distances = distances_arr

    # compute the index of the very first cell that the ray intersects
    cell_index_arr, initial_cell_entry = _starting_index(
        line_uvec, line_start, grid_left_edge, cell_width, 
        np.array(grid_shape, dtype = np.float64)
    )
    cdef int[:] cell_index = cell_index_arr

    cdef Py_ssize_t axis

    # Prepare the structs that are used to find the t value at which the ray
    # will cross the next cell-face (along a given axis)
    cdef CalcNextFaceTParams[3] calc_face_t_params
    for axis in range(3):
        calc_face_t_params[axis].line_uvec = line_uvec[axis]
        calc_face_t_params[axis].line_start = line_start[axis]
        calc_face_t_params[axis].grid_left_edge = grid_left_edge[axis]
        calc_face_t_params[axis].cell_width = cell_width[axis]

    # now actually actually compute the t values corresponding to the locations
    # where the ray will cross the next cell face along each axis
    cdef double[3] next_face_t
    for axis in range(3):
        if line_uvec[axis] == 0:
            next_face_t[axis] = np.finfo(np.float64).max
        else:
            next_face_t[axis] = _calc_next_face_t(
                cur_cell_index = cell_index[axis],
                params = calc_face_t_params[axis]
            )

    # Next, fill the stop_index array. When the corresponding elements of
    # cell_index and stop_index are equal, we'll know that it's time to exit
    # the while-loop.
    cdef int[3] stop_index
    for axis in range(3):
        if line_uvec[axis] <= 0:
            stop_index[axis] = -1
        else:
            stop_index[axis] = grid_shape[axis]

    cdef double cell_entry_t
    cdef double cell_exit_t = initial_cell_entry
    cdef int num_cells = 0

    while ((stop_index[0] != cell_index[0]) and
           (stop_index[1] != cell_index[1]) and
           (stop_index[2] != cell_index[2])):
        # store the t value corresponding to the ray's entrance into the
        # current cell
        cell_entry_t = cell_exit_t
        # find the t value corresponding to ray's exit from the current cell
        cell_exit_t = min(min(next_face_t[0], next_face_t[1]),
                          next_face_t[2])

        # process current entry
        for axis in range(3):
            indices[axis, num_cells] = cell_index[axis]
        distances[num_cells] = cell_exit_t - cell_entry_t
        num_cells+=1

        # prepare for next loop entry
        for axis in range(3):
            if cell_exit_t == next_face_t[axis]:
                if line_uvec[axis] > 0:
                    cell_index[axis] += 1
                elif line_uvec[axis] < 0:
                    cell_index[axis] -= 1
                else:
                    raise RuntimeError()
                next_face_t[axis] = _calc_next_face_t(
                    cur_cell_index = cell_index[axis],
                    params = calc_face_t_params[axis]
                )
    for i in range(3):
        if not np.logical_and(indices_arr[i, :num_cells] < grid_shape[i],
                              indices_arr[i, :num_cells] >= 0).all():
            print('Problem dimension: i')
            print(indices[:, :num_cells])
            print(distances[:num_cells])
            raise AssertionError()
    if not (distances_arr[:num_cells] >= 0).all():
        print(indices[:, :num_cells])
        print(distances[:num_cells])
        raise AssertionError()

    return indices_arr[:, :num_cells],distances_arr[:num_cells]
