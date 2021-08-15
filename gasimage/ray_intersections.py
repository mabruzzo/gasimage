import numpy as np
import yt

def _unit_anyclose(a,b,rtol=1e-05, atol=1e-08):
    if np.size(b) == 0 or np.size(a) == 0:
        return np.array([],dtype = np.bool)
    else:
        a = np.asanyarray(a)
        b = np.asanyarray(b)
    #print(a)
    #print(b)
    diff = np.abs(a-b)
    atol = yt.YTQuantity(atol,diff.units)
    
    return diff <= (atol + rtol* b)


def ray_box_intersections(line_start, line_uvec, left_edge, right_edge,
                          atol = 1e-15, rtol = 1e-14):
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

    Notes
    -----
    The ordering of coordinates doesn't matter as long as its consistent

    Following the explanation of the Wikipedia article on line-plane intersections:
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
    """
    
    # this could be substantially optimized
    
    if isinstance(line_start,yt.YTArray):
        units = line_start.uq
        assert isinstance(left_edge, yt.YTArray)
        assert isinstance(right_edge, yt.YTArray)
        left_edge = left_edge.to(line_start.units).v
        right_edge = right_edge.to(line_start.units).v
        line_start = line_start.v
    else:
        assert not isinstance(left_edge, yt.YTArray)
        assert not isinstance(right_edge, yt.YTArray)
        units = None
        

    assert not isinstance(line_uvec,yt.YTArray)

    distances = []
    for corner in (left_edge, right_edge):
        for i in range(3): # iterate over faces that intersect corner
            # the normal vector for the current face has a value of
            # zero along each axis, except for axis i (where it's 1
            # or -1)

            if line_uvec[i] == 0:
                continue
            d = (corner[i] - line_start[i]) / line_uvec[i]
            if d < 0:
                continue

            if np.isclose(d,distances,atol=atol,rtol=rtol).any():
                continue
            intersection = line_start + line_uvec*d
            intersection[i] = corner[i] # this is essential
            if np.logical_and(
                np.logical_or(intersection > left_edge,
                              np.isclose(intersection,left_edge,atol=atol,rtol=rtol)),
                np.logical_or(intersection <= right_edge,
                              np.isclose(intersection,right_edge,atol=atol,rtol=rtol))).all():
                distances.append(d)
    distances = np.array(distances)
    distances.sort()
    #print(distances)
    assert len(distances) <= 2
    if units is None:
        return np.array(distances)
    else:
        return np.array(distances) * units

def _starting_index(line_uvec, line_start,
                    grid_left_edge, cell_width, 
                    grid_shape):
    # identify the index of the first cell-centered that the ray
    # intersects

    grid_right_edge = grid_left_edge + cell_width * grid_shape
    if np.logical_and(grid_left_edge < line_start,
                      grid_right_edge > line_start).all():
        # the ray starts within the box
        raise NotImplementedError()
    else:
        # assume that line_start is where the line intersects the grid

        # compute the normalized index. An integer value coincides with 
        # a cell face (a value of 0 coincides with the left edge of the
        # leftmost cell)
        normalized_index = ((line_start - grid_left_edge) / cell_width)
        #print(normalized_index)
        #print(int((line_start/cell_width - grid_left_edge/cell_width)[0]))

        # compute the first cell-centered index including the cell
        # normalized_index:    (i-1)     i     (i+1)
        #                        +-------+-------+
        #                        |       |       |
        # cell-centered index:   | (i-1) |   i   |
        #                        |       |       |
        #                        +-------+-------+
        # The starting cell-centered index for rays that enter the grid on 
        # a non-integer normalized_index is always floor(normalized_index)
        cell_index = np.empty((3,),dtype=np.int64)
        for i in range(3):
            if line_uvec[i] == 0:
                # if int(normalized_index) == normalized_index
                # the ray's starting index is the cell for which the ray
                # is on the LEFT edge
                cell_index[i] = int(np.floor(normalized_index[i]))
                assert cell_index[i] < grid_shape[i] # sanity check
            elif line_uvec[i] > 0:
                # if int(normalized_index) == normalized_index
                # the ray's starting index is the cell for which the ray
                # is on the LEFT edge
                cell_index[i] = int(np.floor(normalized_index[i]))
            else: # inv_line_uvec[i] < 0
                # if int(normalized_index) == normalized_index
                # the ray's starting index is the cell for which the ray
                # is on the RIGHT edge
                cell_index[i] = int(np.ceil(normalized_index[i])) - 1

                if cell_index[i] == grid_right_edge[i]:
                    # this can happen if floating point round-off errors are
                    # being troublesome
                    #
                    # in the future, once we have this function support points
                    # starting outside of the grid, we will clip this
                    eps = np.finfo(line_start.dtype).eps
                    if (np.abs(line_start[i] - grid_right_edge[i]) <
                        eps * np.abs(grid_right_edge)):
                        cell_index[i] -= 1
        dist_from_line_start = 0.0

    if (np.logical_or(cell_index < 0, cell_index >= np.array(grid_shape)).any() or
        dist_from_line_start < 0.0):
        print("Problems were encountered while determining the starting index.")
        print('line_start:', line_start)
        print('line_uvec:', line_uvec)
        print(f'grid_left_edge: {grid_left_edge} '
              f'grid_right_edge: {grid_right_edge}')
        print(grid_right_edge - line_start)
        print(line_start[0], grid_right_edge[0])
        print('starting index: ', cell_index)
        print('grid_shape: ', grid_shape)
        print(dist_from_line_start)
        raise AssertionError()

    return cell_index, dist_from_line_start



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
    
    # be conservative about the max number of intersections
    # the max number is actually smaller (and related to the
    # max number of intersectable cells by a diagonal line)
    max_num = np.sum(grid_shape)

    indices = np.empty((3,max_num), np.int64)
    distances = np.empty((max_num,), np.float64)

    delta_index = np.array(
        [int(np.sign(e)) for e in line_uvec],
        dtype = np.int64
    )

    def _calc_next_face_t(axis, cur_cell_index):
        # compute the distance to the next face from the 
        # start of the line

        # first, compute determine the index of the next face
        # note: face-index 0 corresponds to the left face of the
        # cell with index = 0

        if line_uvec[axis] > 0:
            next_face_ind = cur_cell_index[axis] + 1
        elif line_uvec[axis] < 0:
            next_face_ind = cur_cell_index[axis]
        else:
            raise RuntimeError()

        next_face_pos = (
            grid_left_edge[axis] + next_face_ind * cell_width[axis]
        )

        

        out = (next_face_pos - line_start[axis]) / line_uvec[axis]
        if out < 0:
            print('\naxis = ', axis)
            print('next_face_ind:', next_face_ind)
            print('next_face_pos:', next_face_pos)
            print('line_start:', line_start[axis])
            print('grid_left_edge:', grid_left_edge[axis])
            print('line_uvec:', line_uvec)
            print('cur_cell_index:', cur_cell_index[axis])
            raise AssertionError()
        return out


    # compute the index or the first cell that the ray
    # intersects
    cell_index, initial_cell_entry = _starting_index(
        line_uvec, line_start, grid_left_edge, cell_width, 
        grid_shape
    )

    next_face_t = np.empty((3,), dtype = np.float64)
    for axis in range(3):
        if line_uvec[axis] == 0:
            next_face_t[axis] = np.finfo(np.float64).max
        else:
            next_face_t[axis] = _calc_next_face_t(
                axis, cur_cell_index = cell_index
            )

    # when an element is equal to the corresponding entry of 
    # stop_index, exit the loop
    stop_index = np.empty((3,),dtype=np.int64)
    for axis in range(3):
        if line_uvec[axis] <= 0:
            stop_index[axis] = -1
        else:
            stop_index[axis] = grid_shape[axis]


    cell_exit_t = initial_cell_entry
    num_cells = 0

    while (stop_index != cell_index).all():
        cell_entry_t = cell_exit_t
        cell_exit_t = next_face_t.min()

        # process current entry
        indices[:, num_cells] = cell_index
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
                    axis, cur_cell_index = cell_index
                )
    for i in range(3):
        if not np.logical_and(indices[i, :num_cells] < grid_shape[i],
                              indices[i, :num_cells] >= 0).all():
            print('Problem dimension: i')
            print(indices[:, :num_cells])
            print(distances[:num_cells])
            raise AssertionError()
    if not (distances[:num_cells] >= 0).all():
        print(indices[:, :num_cells])
        print(distances[:num_cells])
        raise AssertionError()

    return indices[:, :num_cells],distances[:num_cells]
