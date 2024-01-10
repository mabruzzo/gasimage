"""
A lot of this comes from commit 395f74d8017df83b87f0956900437c79f8504a4c of
the yt-repository

here is the license for the code from the yt-repository. About halfway down the
page, we signal where we start writing new code...

yt is licensed under the terms of the Modified BSD License (also known as New
or Revised BSD), as follows:

Copyright (c) 2013-, yt Development Team
Copyright (c) 2006-2013, Matthew Turk <matthewturk@gmail.com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

Neither the name of the yt Development Team nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import numpy as np

cimport cython
from libc.math cimport fabs, floor

#cimport numpy as np
#ctypedef np.float64_t flt64_t

ctypedef double flt64_t

# the original yt-codebase had code similar to the following, but they defined
# an actual function, not a macro

cdef extern from *:
    """
    #define MY_FMIN(a, b) ((a) < (b)) ? (a) : (b)
    """

    flt64_t MY_FMIN(flt64_t a, flt64_t b)

cdef struct VolumeContainer:
    #-----------------------------------------------------------------------------
    # Encapsulates a volume container used for volume rendering.
    #
    #    n_fields       int              : The number of fields available to the volume renderer.
    #    data           flt64_t**   : The data within the volume container.
    #    mask           np.uint8_t*      : The mask of the volume container. It has dimensions one fewer in each direction than data.
    #    left_edge      flt64_t[3]  : The left edge of the volume container's bounding box.
    #    right_edge     flt64_t[3]  : The right edge of the volume container's bounding box.
    #    flt64_t   dds[3]           : The delta dimensions, such that dds[0] = ddx, dds[1] = ddy, dds[2] = ddz.
    #    flt64_t   idds[3]          : The inverse delta dimensions. Same as dds pattern, but the inverse. i.e. idds[0] = inv_ddx.
    #    dims           int[3]           : The dimensions of the volume container. dims[0] = x, dims[1] = y, dims[2] = z.
    #-----------------------------------------------------------------------------
    flt64_t left_edge[3]
    flt64_t right_edge[3]
    flt64_t dds[3]
    flt64_t idds[3]
    int dims[3]

#-----------------------------------------------------------------------------
# walk_volume(VolumeContainer *vc,  flt64_t v_pos[3], flt64_t v_dir[3], sampler_function *sample,
#             void *data, flt64_t *return_t = NULL, flt64_t max_t = 1.0)
#    vc        VolumeContainer*  : Pointer to the volume container to be traversed.
#    v_pos     flt64_t[3]   : The x,y,z coordinates of the ray's origin.
#    v_dir     flt64_t[3]   : The x,y,z coordinates of the ray's direction.
#    sample    sampler_function* : Pointer to the sampler function to be used.
#    return_t  flt64_t*     : Pointer to the final value of t that is still inside the volume container. Defaulted to NULL.
#    max_t     flt64_t      : The maximum value of t that the ray is allowed to travel. Defaulted to 1.0 (no restriction).
#
#    Note: 't' is not time here. Rather, it is a factor representing the difference between the initial point 'v_pos'
#             and the end point, which we might call v_end. It is scaled such that v_pos + v * t = v_pos at t = 0.0, and
#             v_end at t = 1.0. Therefore, if max_t is set to 1.0, there is no restriction on t.
#
# Written by the yt Development Team.
# Encapsulates the Amanatides & Woo "Fast Traversal Voxel Algorithm" to walk over a volume container 'vc'
# The function occurs in two phases, initialization and traversal.
# See: https://www.researchgate.net/publication/2611491_A_Fast_Voxel_Traversal_Algorithm_for_Ray_Tracing
# Returns: The number of voxels hit during the traversal phase. If the traversal phase is not reached, returns 0.
#-----------------------------------------------------------------------------
cdef int walk_volume(VolumeContainer *vc,
                     flt64_t* v_pos,
                     flt64_t* v_dir,
                     void (*sample)(flt64_t enter_t, flt64_t exit_t, int index[3], void *data),
                     void *data,
                     flt64_t *return_t = NULL,
                     flt64_t max_t = 1.0):
    cdef int cur_ind[3]
    cdef int step[3]
    cdef int x, y, i, hit, direction
    cdef flt64_t intersect_t = 1.1
    cdef flt64_t iv_dir[3]
    cdef flt64_t tmax[3]
    cdef flt64_t tdelta[3]
    cdef flt64_t exit_t = -1.0, enter_t = -1.0
    cdef flt64_t tl, temp_x, temp_y = -1
    if max_t > 1.0: max_t = 1.0
    direction = -1
    if vc.left_edge[0] <= v_pos[0] and v_pos[0] < vc.right_edge[0] and \
       vc.left_edge[1] <= v_pos[1] and v_pos[1] < vc.right_edge[1] and \
       vc.left_edge[2] <= v_pos[2] and v_pos[2] < vc.right_edge[2]:
        intersect_t = 0.0
        direction = 3
    for i in range(3):
        if (v_dir[i] < 0):
            step[i] = -1
        elif (v_dir[i] == 0.0):
            step[i] = 0
            continue
        else:
            step[i] = 1
        iv_dir[i] = 1.0/v_dir[i]
        if direction == 3: continue
        x = (i+1) % 3
        y = (i+2) % 3
        if step[i] > 0:
            tl = (vc.left_edge[i] - v_pos[i])*iv_dir[i]
        else:
            tl = (vc.right_edge[i] - v_pos[i])*iv_dir[i]
        temp_x = (v_pos[x] + tl*v_dir[x])
        temp_y = (v_pos[y] + tl*v_dir[y])
        if fabs(temp_x - vc.left_edge[x]) < 1e-10*vc.dds[x]:
            temp_x = vc.left_edge[x]
        elif fabs(temp_x - vc.right_edge[x]) < 1e-10*vc.dds[x]:
            temp_x = vc.right_edge[x]
        if fabs(temp_y - vc.left_edge[y]) < 1e-10*vc.dds[y]:
            temp_y = vc.left_edge[y]
        elif fabs(temp_y - vc.right_edge[y]) < 1e-10*vc.dds[y]:
            temp_y = vc.right_edge[y]
        if vc.left_edge[x] <= temp_x and temp_x <= vc.right_edge[x] and \
           vc.left_edge[y] <= temp_y and temp_y <= vc.right_edge[y] and \
           0.0 <= tl and tl < intersect_t:
            direction = i
            intersect_t = tl
    if enter_t >= 0.0: intersect_t = enter_t
    if not ((0.0 <= intersect_t) and (intersect_t < max_t)): return 0
    for i in range(3):
        # Two things have to be set inside this loop.
        # cur_ind[i], the current index of the grid cell the ray is in
        # tmax[i], the 't' until it crosses out of the grid cell
        tdelta[i] = step[i] * iv_dir[i] * vc.dds[i]
        if i == direction and step[i] > 0:
            # Intersection with the left face in this direction
            cur_ind[i] = 0
        elif i == direction and step[i] < 0:
            # Intersection with the right face in this direction
            cur_ind[i] = vc.dims[i] - 1
        else:
            # We are somewhere in the middle of the face
            temp_x = intersect_t * v_dir[i] + v_pos[i] # current position
            temp_y = ((temp_x - vc.left_edge[i])*vc.idds[i])
            # There are some really tough cases where we just within a couple
            # least significant places of the edge, and this helps prevent
            # killing the calculation through a segfault in those cases.
            if -1 < temp_y < 0 and step[i] > 0:
                temp_y = 0.0
            elif vc.dims[i] - 1 < temp_y < vc.dims[i] and step[i] < 0:
                temp_y = vc.dims[i] - 1
            cur_ind[i] =  <int> (floor(temp_y))
        if step[i] > 0:
            temp_y = (cur_ind[i] + 1) * vc.dds[i] + vc.left_edge[i]
        elif step[i] < 0:
            temp_y = cur_ind[i] * vc.dds[i] + vc.left_edge[i]
        tmax[i] = (temp_y - v_pos[i]) * iv_dir[i]
        if step[i] == 0:
            tmax[i] = 1e60
    # We have to jumpstart our calculation
    for i in range(3):
        if cur_ind[i] == vc.dims[i] and step[i] >= 0:
            return 0
        if cur_ind[i] == -1 and step[i] <= -1:
            return 0
    enter_t = intersect_t
    hit = 0
    while 1:
        hit += 1
        if tmax[0] < tmax[1]:
            if tmax[0] < tmax[2]:
                i = 0
            else:
                i = 2
        else:
            if tmax[1] < tmax[2]:
                i = 1
            else:
                i = 2
        exit_t = MY_FMIN(tmax[i], max_t)
        sample(enter_t, exit_t, cur_ind, data)
        cur_ind[i] += step[i]
        enter_t = tmax[i]
        tmax[i] += tdelta[i]
        if cur_ind[i] < 0 or cur_ind[i] >= vc.dims[i] or enter_t >= max_t:
            break
    if return_t != NULL: return_t[0] = exit_t
    return hit

# this is the end of the code from/inspired by yt...
#
# from this point and below all of the code was written for this down here is
# where the totally new code that I've written begins

from ._misc_cy import max_num_intersections, get_grid_center, get_grid_width

def _get_ray_stop(line_uvec, line_start, spatial_props):
    """
    this is a helper function. The whole point here is to come up with a
    ray-stop at a distance far enough away that we can be sure that it is 
    further than the furthest domain edge.

    At the moment, we are NOT worried about whether the ray actually intersects
    with the domain

    In the future, it would be better to come with start & end points of the 
    shortest ray that passes through both ends of the domain to ensure accuracy
    of the distance calculation.
    """

    # longest distance through the domain
    longest_width = np.linalg.norm(get_grid_width(spatial_props))

    ray_length = (
        np.linalg.norm(line_start - get_grid_center(spatial_props)) +
        2*longest_width
    )
    return line_start + line_uvec * ray_length

cdef struct PathTracker:
    long cell_count  # the number of encountered cells
    long* indices    # indices[i*3], indices[i*3+1], indices[i*3+2] denote
                     # the indices of the ith cell that the ray passes thru
    flt64_t* dt_arr  # dt_arr[i] "distance" (in the "t" parameterization) that
                     # a ray travels through the ith cell

cdef void sample_ray(flt64_t enter_t,flt64_t exit_t, int index[3], void *data):
    cdef PathTracker* ptr = <PathTracker*> data
    cdef long i = ptr.cell_count
    ptr.indices[i*3 + 0] = index[0]
    ptr.indices[i*3 + 1] = index[1]
    ptr.indices[i*3 + 2] = index[2]
    ptr.dt_arr[i] = (exit_t - enter_t)
    ptr.cell_count += 1


def traverse_grid(line_uvec, line_start, spatial_props):
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

    I have some qualms about the back & forth between our representations of
    rays.
    - Sometimes, we represent them with their start points and end points
    - Other times, we represent them with their start-points and unit vector
    
    Our conversion between these representations is a little concerning! We are
    definitely losing precision each time we do this! It's also a little
    concerning that the precision of the distances array returned by this
    function directly depends on the length of the array (and consequently on 
    the observer's distance!)
    """

    assert len(line_uvec) == 3
    assert not (line_uvec == 0).all()
    assert len(line_start) == 3

    ray_uvec = line_uvec.astype(np.float64, copy = False, order = 'C')
    ray_start = line_start.astype(np.float64, copy = False, order = 'C')

    cdef VolumeContainer vc
    for i in range(3):
        vc.left_edge[i]  = spatial_props.left_edge[i]
        vc.right_edge[i] = spatial_props.right_edge[i]
        vc.dds[i]        = spatial_props.cell_width[i]
        vc.idds[i]       = 1.0/spatial_props.cell_width[i]
        vc.dims[i]       = spatial_props.grid_shape[i]

    # we need to come up with an end point on the other side of the grid
    #intersect_dists = ray_box_intersections(ray_start, ray_uvec,
    #                                        left_edge = grid_left_edge,
    #                                        right_edge = grid_right_edge)

    # intentionally overshoot the back edge of the grid!
    ray_stop = _get_ray_stop(line_uvec, line_start, spatial_props)

    # get the max number of intersections
    max_num = max_num_intersections(spatial_props.grid_shape)

    # I'm using np.compat.long to be explicit that I want the "long" data type
    indices_arr = np.empty((max_num,3), np.compat.long, order = 'C')
    # tell the memory view that indices is C-contiguous
    cdef long[:, ::1] indices_view = indices_arr

    dt_arr = np.empty((max_num,), np.float64)
    cdef flt64_t[::1] dt_view = dt_arr

    cdef PathTracker ptracker
    ptracker.cell_count = 0
    ptracker.indices = &indices_view[0,0]
    ptracker.dt_arr = &dt_view[0]

    cdef flt64_t v_pos[3]
    cdef flt64_t v_dir[3]
    for i in range(3):
        v_pos[i] = ray_start[i]
        v_dir[i] = ray_stop[i] - ray_start[i]

    walk_volume(&vc, v_pos, v_dir, sample_ray, <void*> &ptracker)

    distances = dt_arr[:ptracker.cell_count]
    distances *= np.linalg.norm(ray_stop-ray_start)

    return (np.ascontiguousarray(indices_arr[:ptracker.cell_count,:].T),
            distances)
