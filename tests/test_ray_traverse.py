import numpy as np

from gasimage.ray_intersections import traverse_grid


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

if __name__ == '__main__':
    test_traverse_grid2()
