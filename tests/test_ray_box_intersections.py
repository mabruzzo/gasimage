from functools import partial

import numpy as np

from gasimage._ray_intersections_cy import ray_box_intersections

def _test_rb_intersections(line_start, line_uvec, left_edge, right_edge,
                           expected_result, zero_atol = None, rtol = None):

    line_start = np.asanyarray(line_start)
    line_uvec = np.asanyarray(line_uvec)
    left_edge = np.asanyarray(left_edge)
    right_edge = np.asanyarray(right_edge)
    expected_result = np.asanyarray(expected_result)
    
    if (expected_result< 0).any():
        raise ValueError('ill formed test')

    if (zero_atol is not None) and (zero_atol == 0):
        zero_atol = None
    elif (zero_atol is not None) and zero_atol < 0:
        raise ValueError("Ill formed test. zero_atol can't be negative")
    elif (rtol is not None) and rtol == 0:
        rtol = None
    elif (rtol is not None) and rtol < 0:
        raise ValueError("Ill formed test. rtol can't be negative")

    result = ray_box_intersections(line_start, line_uvec, 
                                   left_edge, right_edge)
    
    _arr_fmt = partial(np.array2string, floatmode = 'unique',
                       separator = ',')
    _suffix = (
        'for arguments: \n' +
        f'    line_start = {_arr_fmt(line_start)}' + '\n'
        f'    line_uvec  = {_arr_fmt(line_uvec)}' + '\n'
        f'    left_edge  = {_arr_fmt(left_edge)}' + '\n'
        f'    right_edge = {_arr_fmt(right_edge)}'
    )


    if len(expected_result) == 0 and len(result) == 0:
        return None
    elif len(expected_result) == 0:
        raise AssertionError(f'Expected no intersection. Found: {result}'
                             + '\n' + _suffix)
    elif len(expected_result) != len(result):
        raise AssertionError(f'Expected the result {expected_result}. '
                             f'Found: {result}' + '\n' + _suffix)

    assert len(expected_result) == len(result)

    for i in range(len(result)):
        if (expected_result[i] == 0) and (zero_atol is None):
            assert result[i] == 0
        elif (expected_result[i] == 0):
            assert result[i] <= zero_atol
        elif rtol is None:
            assert result[i] == expected_result[i]
        else:
            diff = np.abs(result - expected_result)[i]
            assert diff <= rtol*np.abs(expected_result[i])


def _sample_open_interval(a,b,generator = None):
    if b < a:
        raise ValueError('b must exceed a')

    dtype = np.double
    a = np.asanyarray(a,dtype = dtype)
    b = np.asanyarray(b,dtype = dtype)
    
    corrected_a = np.nextafter(a,-np.inf)
    if corrected_a == b:
        raise RuntimeError()

    difference = b - corrected_a
    if (difference + corrected_a) == b:
        difference = np.nextafter(difference,-np.inf)
        raise RuntimeError()

    # sample the open interval (a, b)
    if generator is None:
        tmp = np.random.random_sample()
    else:
        tmp = generator.random()

    return difference*tmp + corrected_a

def test_axis_aligned_rays(left_edge = (-60.0,-10.0,-10.0),
                           right_edge = (60.0, 10.0, 10.0)):
    
    generator = np.random.RandomState(seed =246573)
    
    width = np.array(right_edge) - np.array(left_edge) 

    for ax in range(3):
        
        for i in range(2):

            if i == 0:
                delta_ax = 1.0
                default_ax_edge = left_edge[ax]
                exit_ax_edge = right_edge[ax]
            else:
                delta_ax = -1.0
                default_ax_edge = right_edge[ax]
                exit_ax_edge = left_edge[ax]
                
            # setup line_uvec
            line_uvec = [0.,0.,0.]
            line_uvec[ax] = delta_ax

            default_line_start = np.array((0.0, 0.0, 0.0))
            for j in range(3):
                if j == ax:
                    default_line_start[ax] = default_ax_edge
                else:
                    default_line_start[j] = _sample_open_interval(
                        left_edge[j], right_edge[j],
                        generator = generator
                    )
                    
            transv_left_edge_starts = [
                default_line_start.copy() for i in range(3)
            ]

            counter = 0
            for j in range(3):
                if j == ax:
                    continue
                transv_left_edge_starts[counter][j] = left_edge[j]
                transv_left_edge_starts[-1][j] = left_edge[j]
                counter += 1

            for line_start in [default_line_start] + transv_left_edge_starts:
                # start the ray on the grid
                grid_aligned_start = line_start.copy()
                _test_rb_intersections(line_start = grid_aligned_start,
                                       line_uvec = line_uvec,
                                       left_edge = left_edge,
                                       right_edge = right_edge,
                                       expected_result = (0., width[ax]))

                offset = 0.5*width[ax]
                # start the ray outside of the grid (before the edge where it enters)
                outside_start = line_start.copy()
                outside_start[ax] -= delta_ax * offset
                _test_rb_intersections(line_start = outside_start,
                                       line_uvec = line_uvec,
                                       left_edge = left_edge,
                                       right_edge = right_edge,
                                       expected_result = (offset, offset + width[ax]))
            
                # start the ray inside of the grid
                inside_start = line_start.copy()
                inside_start[ax] += delta_ax * offset
                _test_rb_intersections(line_start = inside_start,
                                       line_uvec = line_uvec,
                                       left_edge = left_edge,
                                       right_edge = right_edge,
                                       expected_result = (width[ax] - offset,))

                # start the ray along ax on the edge where it should exit
                exit_edge_start = line_start.copy()
                exit_edge_start[ax] = exit_ax_edge
                _test_rb_intersections(line_start = exit_edge_start,
                                       line_uvec = line_uvec,
                                       left_edge = left_edge,
                                       right_edge = right_edge,
                                       expected_result = ())
                
                # start the ray after the edge where it nominally exits
                outside_exit_edge_start = line_start.copy()
                outside_exit_edge_start[ax] = exit_ax_edge + delta_ax * offset
                _test_rb_intersections(line_start = outside_exit_edge_start,
                                       line_uvec = line_uvec,
                                       left_edge = left_edge,
                                       right_edge = right_edge,
                                       expected_result = ())
                
        for j in range(3):
            if j == ax:
                continue
                
            starts = [default_line_start.copy() for _ in range(3)]
            
            starts[0][j] = left_edge[j] - width[j]
            starts[1][j] = right_edge[j]
            starts[2][j] = right_edge[j] + width[j]
            for line_start in starts:
                _test_rb_intersections(line_start = line_start,
                                       line_uvec = line_uvec,
                                       left_edge = left_edge,
                                       right_edge = right_edge,
                                       expected_result = ())
        # there are some additional cases without any intersections
        # that haven't been checked (its just extra permutations of
        # which axes cause the ray to start outside of the box)

if __name__ == '__main__':
    test_axis_aligned_rays(left_edge = (-60.0,-10.0,-10.0),
                           right_edge = (60.0, 10.0, 10.0))
