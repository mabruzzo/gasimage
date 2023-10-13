import numpy as np

from gasimage.optically_thin_ppv import traverse_grid


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

if __name__ == '__main__':
    test_traverse_grid_small_truncation_probs()
    test_traverse_grid2()
