import numpy as np
import yt

from gasimage.ray_traversal import traverse_grid

from ray_testing_utils import ray_values_startend


# these all need to be replaced with actual tests that check correctness
# at the moment, they just check that the code doesn't crash


def test_traverse_grid():

    idx, dists = traverse_grid(line_uvec = np.array([-1.0, 0.0, 0]), 
                               line_start = np.array([60, 0, 0.]),
                               grid_left_edge = np.array([-60, 0.0, -10.]),
                               cell_width = np.array([1.0,1.0,1.0]),
                               grid_shape = (120,20,20))
    print(idx)

    idx, dists = traverse_grid(line_uvec = np.array([-1.0/np.sqrt(2), 1.0/np.sqrt(2),0]), 
                               line_start = np.array([60, 0, 0.]),
                               grid_left_edge = np.array([-60, -10, -10.]),
                               cell_width = np.array([1.0,1.0,1.0]),
                               grid_shape = (120,20,20))
    print(idx)
    print(dists)

    idx, dists = traverse_grid(line_uvec = np.array([-1.0/np.sqrt(2), -1.0/np.sqrt(2),0]), 
                               line_start = np.array([60, 0, 0.]),
                               grid_left_edge = np.array([-60, -10, -10.]),
                               cell_width = np.array([1.0,1.0,1.0]),
                               grid_shape = (120,20,20))
    print(idx)
    print(dists)


def test_traverse_grid2():
    # This file contains inputs that previously caused problems
    if False:
        # this demonstrates a particular problem
        line_uvec = np.array([-1.293895253754914e-16, 
                              -4.226182617406992e-01,
                              -9.063077870366500e-01])

        line_start = np.array([-5.3579909130206237e-16,
                               -1.7500526410557118e+00,
                               8.6469999999999985e+00])
        grid_left_edge = np.array([-51.882,
                                   -8.646999999999998,
                                   -8.646999999999998])
        cell_width = np.array([0.10808749999999999,
                               0.10808749999999999,
                               0.10808749999999999])
        grid_shape = (960, 160, 160)
        idx, dists = traverse_grid(line_uvec = line_uvec,
                                   line_start = line_start,
                                   grid_left_edge = grid_left_edge,
                                   cell_width = cell_width,
                                   grid_shape = grid_shape)
        print(idx)
        print(dists)

    if True:
        line_uvec = np.array([-0.7011924154900053,
                              -0.3090169943749472,
                              -0.6425244692980694])
        line_start = np.array([-22.499999999999996,
                               -9.915798031810596,
                               -6.277217808360678])
        grid_left_edge = np.array([-30., -10., -10.])
        cell_width = np.array([0.125, 0.125, 0.125])
        grid_shape = (60, 40, 40)
        idx, dists = traverse_grid(line_uvec = line_uvec,
                                   line_start = line_start,
                                   grid_left_edge = grid_left_edge,
                                   cell_width = cell_width,
                                   grid_shape = grid_shape)
        print(idx)



def test_traverse_grid_small_truncation_probs():
    # This a test that makes sure that we appropriately handle truncation
    # errors. I'm not completely sure that the answer is correct
    line_start = np.array([-5.37672641e-31, -1.75617212e-15,
                           1.43402336e+01])
    line_uvec = np.array([ 3.74939946e-32,  1.22464680e-16,
                           -1.00000000e+00])
    grid_left_edge = np.array([-7.5, -5.,0. ])
    grid_right_edge = np.array([0.0, 0.0, 5. ])
    cell_width = np.array([0.125, 0.125, 0.125])
    grid_shape = (60, 40, 40)
    idx, dists = traverse_grid(line_uvec = line_uvec,
                               line_start = line_start,
                               grid_left_edge = grid_left_edge,
                               cell_width = cell_width,
                               grid_shape = grid_shape)

    expected_idx = np.array(
        [[59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
          59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
          59, 59, 59, 59, 59, 59],
         [39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
          39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
          39, 39, 39, 39, 39, 39],
         [39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
          22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,
          5,  4,  3,  2,  1,  0]]
    )
    assert (expected_idx == idx).all()

#------------------------------------------------------------------------
# define some tests where we compare against results found with yt.YTRay
# -> in these tests, we currently need to specify the ray by providing its
#    start and end points
#------------------------------------------------------------------------

def traverse_grid_startend(start_end_pairs, grid_left_edge,
                           cell_width, grid_shape):

    out_pairs = []

    for start, end in start_end_pairs:
        line_start = np.array(start)
        line_vec = (np.array(end) - np.array(start))
        assert (line_vec != 0).any()
        line_uvec = line_vec / np.linalg.norm(line_vec)
    
        idx, dists = traverse_grid(
            line_uvec = line_uvec, 
            line_start = line_start,
            grid_left_edge = grid_left_edge,
            cell_width = cell_width,
            grid_shape = grid_shape)

        out_pairs.append((idx,dists))
    return out_pairs


def traverse_grid_startend_comparison(start_end_pairs, grid_left_edge,
                                      cell_width, grid_shape):
    data = {'Density' : np.zeros(dtype = 'f4', shape = grid_shape)}

    grid_right_edge = grid_left_edge + np.array(grid_shape) * cell_width
    bbox = np.stack([grid_left_edge, grid_right_edge], axis = 1)

    tmp_ds = yt.load_uniform_grid(data = data, domain_dimensions = grid_shape,
                                  bbox = bbox,
                                  periodicity = (False, False, False))

    itr = ray_values_startend(tmp_ds, start_end_pairs, fields = [],
                              find_indices = True)
    return [(elem['indices'], elem['dl']) for elem in itr]

def test_simple_traverse_grid_comparison():

    _1way_start_end_input_pairs = [
        ([80, 1.0, 2.0], [-60, 1.0, 2.0]),
        ([20, 0.0, 2.0], [20.0, 20.0, 2.0]),
        ([20, 1.0, -10.0], [20.0, 1.0, 10.0]),
        # need to improve the comparison function for the following case...
        #([20, 0.0, 2.0], [40, 20.0, 0.0])
    ]
    # reverse the directions
    start_end_pairs = []
    for pair in _1way_start_end_input_pairs:
        start_end_pairs.append(pair)
        start_end_pairs.append(pair[::-1])

    kwargs = {'start_end_pairs' : start_end_pairs,
              'grid_left_edge' : np.array([-60, 0.0, -10.]),
              'cell_width' : np.array([1.0,1.0,1.0]),
              'grid_shape' : (120,20,20) }

    rslt_ref = traverse_grid_startend_comparison(**kwargs)
    rslt_alt = traverse_grid_startend(**kwargs)

    for i, (start, end) in enumerate(kwargs['start_end_pairs']):
        # -> a more sophisticated case, would examine the case where we have
        #    slightly different numbers of cells
        # -> we would need to decide on the particular cases for raising an
        #    error:
        #    -> one thought is we could say its okay to be missing a particular
        #       cell if the path through that cell is essentially negligible
        #    -> an alternative related thought is we could compare the total
        #       length of the ray...
        #print(np.sum(rslt_alt[i][1]), np.sum(rslt_ref[i][1]))

        np.testing.assert_array_equal(
            x = rslt_alt[i][0], y = rslt_ref[i][0],
            err_msg = ("unequal intersection indices while comparing a "
                       f"ray that starts at {start} and ends at {end}")
        )
        np.testing.assert_allclose(
            actual = rslt_alt[i][1], desired = rslt_ref[i][1],
            rtol = 6e-15, atol = 0.0,
            err_msg = ("unequal intersection path-lengths while comparing a "
                       f"ray that starts at {start} and ends at {end}")
        )

# WE NEED TESTS OF LOTS OF EDGE CASES!

if __name__ == '__main__':
    test_traverse_grid_small_truncation_probs()
    test_traverse_grid2()
    test_simple_traverse_grid_comparison()
