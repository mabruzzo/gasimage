import numpy as np
import yt

from ._ray_intersections_cy import ray_box_intersections

"""
def ray_box_intersections(line_start, line_uvec, left_edge, right_edge):
    assert (line_uvec != 0).any()

    out = np.empty((2,), dtype = np.float64)
    
    problem = False
    result_dtype = np.result_type(line_start, line_uvec,
                                  left_edge, right_edge)
    _init_min_intersect_d = np.finfo(result_dtype).min
    _init_max_intersect_d = np.finfo(result_dtype).max
    
    min_intersect_d = np.finfo(np.float64).min
    max_intersect_d = np.finfo(np.float64).max


    for i in range(3):
        if line_uvec[i] == 0:
            if ((line_start[i] < left_edge[i]) or
                (line_start[i] >= right_edge[i])):
                # asymmetrical comparison is intentional

                # modify the values to indicate that there is
                # no intersection at all
                max_intersect_d = _init_min_intersect_d
                min_intersect_d = _init_max_intersect_d
                break
            else:
                continue
        elif line_uvec[i] > 0:
            axis_min_d = (left_edge[i] - line_start[i]) / line_uvec[i]
            axis_max_d = (right_edge[i] - line_start[i]) / line_uvec[i]
        else:
            axis_min_d = (right_edge[i] - line_start[i]) / line_uvec[i]
            axis_max_d = (left_edge[i] - line_start[i]) / line_uvec[i]

        if axis_min_d > min_intersect_d: # not a typo!
            min_intersect_d = axis_min_d

        if axis_max_d < max_intersect_d: # not a typo!
            max_intersect_d = axis_max_d

    if (max_intersect_d < min_intersect_d) or (max_intersect_d <= 0): 
        # We do need to check that max_intersect_d is not zero (when 
        # it's zero, the only point of intersection is on the right 
        # edge
        # Unsure if we need to explicitly check that max_intersect_d
        # is non-negative.
        return np.array([], dtype = np.float64)
    elif min_intersect_d == max_intersect_d:
        # intersection with a corner
        return np.array([min_intersect_d])
    elif min_intersect_d < 0: # ray starts inside of the box
        # should we differentiate this from corner-intersection?
        return np.array([max_intersect_d])
    else:
        return np.array([min_intersect_d,max_intersect_d])
"""

def _floor_subtract_divide(a, b, divisor):
    """
    Computes floor((a - b)/divisor).

    This correctly handles the case when a >> b or a << b
    """

    # perform a compensated difference
    diff = a - b
    truncated_amount = a - (diff + b)
    # diff + truncated_amount gives the correct difference

    quotient = diff / divisor
    out = np.floor(quotient)
    # it's possible to refactor to avoid the division on the following line
    if (out == quotient) and ((np.sign(truncated_amount) * np.sign(divisor)) < 0):
        return out - 1
    return out

def _ceil_subtract_divide(a, b, divisor):
    """
    Computes ceil((a - b)/divisor).

    This correctly handles the case when a >> b or a << b
    """
    # perform a compensated difference
    diff = a - b
    truncated_amount = a - (diff + b)

    quotient = diff / divisor
    out = np.ceil(quotient)
    if (out == quotient) and ((np.sign(truncated_amount) * np.sign(divisor)) > 0):
        return out + 1
    return out

def _starting_index(line_uvec, line_start,
                    grid_left_edge, cell_width,
                    grid_shape):
    # identify the index of the first cell-centered that the ray
    # intersects

    # I think it's probably important that we calculate the right grid edge
    # (even if we had access to the precomputed value)
    grid_right_edge = grid_left_edge + cell_width * grid_shape

    intersect_t = ray_box_intersections(line_start, line_uvec,
                                        grid_left_edge, grid_right_edge)
    if len(intersect_t) != 2:
        # the ray starts within the box, just hits the lower left corner,
        # or never intersects the box
        raise NotImplementedError()
    else:
        t_start = intersect_t[0]

        # compute location where line intersects the grid
        line_entry = np.clip(
            line_start + line_uvec * t_start,
            a_min = grid_left_edge,
            a_max = grid_right_edge,
        )

        # compute the normalized index. An integer value coincides with
        # a cell face (a value of 0 coincides with the left edge of the
        # leftmost cell)
        normalized_index = np.clip(
            ((line_entry - grid_left_edge) / cell_width),
            a_min = (0,0,0),
            a_max = np.array(grid_shape) + 1
        )
        #print(normalized_index)
        #print(int((line_entry/cell_width - grid_left_edge/cell_width)[0]))

        # compute the first cell-centered index including the cell
        # compute the normalized_index:
        #     normalized_index = (line_entry - grid_left_edge) / cell_width
        #
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

    if (np.logical_or(cell_index < 0, cell_index >= np.array(grid_shape)).any() or
        t_start < 0.0):
        print("Problems were encountered while determining the starting index.")
        print('line_start:', line_start)
        print('line_uvec:', line_uvec)
        print(f'grid_left_edge: {grid_left_edge} '
              f'grid_right_edge: {grid_right_edge}')
        print(grid_right_edge - line_entry)
        print(line_entry[0], grid_right_edge[0])
        print('starting index: ', cell_index)
        print('grid_shape: ', grid_shape)
        print(t_start)
        print(intersect_t)
        print('Entry: ', (line_start + line_uvec * intersect_t[0]))
        print('Exit: ', (line_start + line_uvec * intersect_t[1]))
        print('Normalized: ', (line_entry[2] == grid_right_edge[2]))

        print()

        raise AssertionError()

    return cell_index, t_start



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
