"""
Define some functions to assist with testing voxel-traversal.

Frankly, this is all a little messy (and a little redundant). It could be
cleaned up
"""

import numpy as np
import unyt

from gasimage.generate_image import (
    itr_traverse_top_level_grids,
    fetch_grid_and_spatial_props
)
from gasimage.ray_collection import ConcreteRayList
from gasimage.ray_traversal import ray_box_intersections, traverse_grid

def coerce_unyt_to_npy(arr, units, equivalence = None,
                       passthru_dimensionless = False,
                       passthru_npy_array = False, can_return_view = False,
                       **in_units_kwargs):
    """
    This abstracts out a fairly simple operation that comes up a lot!
    """
    if isinstance(units, unyt.Unit):
        is_dimensionless = units.is_dimensonless
    else:
        is_dimensionless = units in ['dimensionless', '(dimensionless)']

    if isinstance(arr, unyt.unyt_array):
        if arr.units.is_dimensionless and passthru_dimensionless:
            if can_return_view:
                return arr.ndview
            return arr.value # make a copy

        if not isinstance(units, unyt.Unit):
            units = unyt.Unit(units, registry = arr.units.registry)

        if can_return_view and (units == arr.units):
            return arr.ndview
        return arr.to_value(units = units, equivalence = equivalence,
                            **in_units_kwargs) # always returns a copy
    elif isinstance(arr, np.ndarray) and (passthru_npy_array or
                                          is_dimensionless):
        if can_return_view:
            return arr
        return arr.copy()
    else:
        raise ValueError("unable to handle arr argument")

def ray_values_startend(ds, start_end_pairs, fields = [],
                        find_indices = False, kind = None):
    """
    A generator that yields values along individual rays

    At this time, it has 2 implementations. 
      1. 'yt' - This version directly uses yt-rays. Historically this approach
         was quite slow at scale
      2. 'gasimage' - This version uses the machinery provided within the
         gasimage library.
    """

    if (kind is None) or (kind == 'yt'):
        for e in _yt_ray_values_startend(
                ds = ds, start_end_pairs = start_end_pairs,
                fields = fields, find_indices = find_indices):
            yield e
    elif kind == 'gasimage':
        for e in _gasimage_ray_values_startend(
                ds = ds, start_end_pairs = start_end_pairs,
                fields = fields, find_indices = find_indices):
            yield e
    else:
        raise ValueError(f"Unknown value passed to kind kwarg: {kind!r}")


def _gasimage_ray_values_startend(ds, start_end_pairs, fields,
                                  find_indices = False):
    # the fact that this implementation goes ray-by-ray is fairly inefficient.
    # But, this shouldn't really matter for the sake of testing code... Plus
    # it's more likely to be correct!
    #
    # rather than manually using all of the internal machinery, to implement
    # this function... it may be cleaner to replace this with a call to
    # generate_image, with a custom accumulator!

    # These are effectively constants right now
    length_unit_name, rescale_length_factor = 'code_length', 1.0
    # precompute some stuff!
    cm_per_length_unit = float(ds.quan(1.0, length_unit_name).to('cm').v)

    # convert start & stop into a ray_list!
    start, stop = list(zip(*start_end_pairs))
    print(start)
    ray_list = ConcreteRayList.from_start_stop(
        ray_start_codeLen = np.array(start),
        ray_stop_codeLen = np.array(stop)
    )

    # define helper function that collects & consolidates "field_vals" for a
    # single ray given
    # -> the ray's properties
    # -> the sequence of subgrid-ids the ray intersects
    def _fetch_field_vals(ray_start, ray_uvec, subgrid_seq):
        n_subgrids = subgrid_seq.size
        if n_subgrids == 0: return {}

        parts = {}
        # iterate over each subgrid intersected by the ray
        for grid_id in subgrid_seq:
            grid, spatial_grid_props = fetch_grid_and_spatial_props(
                grid_index = grid_id, ds = ds,
                cm_per_length_unit = cm_per_length_unit,
                length_unit_name = length_unit_name,
                rescale_length_factor = rescale_length_factor)

            indices, dl = traverse_grid(
                line_uvec = ray_uvec, line_start = ray_start,
                grid_left_edge = spatial_grid_props.left_edge,
                cell_width = spatial_grid_props.cell_width,
                grid_shape = spatial_grid_props.grid_shape
            )

            # note: dl is already a numpy array with units of 'code_length'

            idx = (indices[0], indices[1], indices[2])
            if len(parts):
                for field in fields:
                    parts[field].append(grid[field][idx])
                parts['dl'].append(dl)
                if find_indices: parts['indices'].append(indices)
            else:
                for field in fields:
                    parts[field] = [grid[field][idx]]
                parts['dl'] = [dl]
                if find_indices: parts['indices'] = [indices]

        # unyt.uhstack internally confirms if arrays are unyt_arrays, that
        # they have identical units. It also returns ordinary np array if all
        # args are ordinary np-arrays
        if n_subgrids == 1:
            return dict((k, vlist[0]) for k,vlist in parts.items())
        return dict((k, unyt.uhstack(vlist)) for k,vlist in parts.items())

    itr = itr_traverse_top_level_grids(
        ds = ds, ray_list = ray_list, units = length_unit_name,
        rescale_length_factor = rescale_length_factor)

    start = ray_list.ray_start_codeLen
    uvec = ray_list.get_ray_uvec()

    for ray_id, (subgrid_seq, _) in enumerate(itr):
        cur_field_vals = _fetch_field_vals(
            ray_start = start[ray_id], ray_uvec = uvec[ray_id],
            subgrid_seq = subgrid_seq)
        yield cur_field_vals

def _yt_ray_values_startend(ds, start_end_pairs, fields = [],
                            find_indices = False):

    # even if fields is empty, it always contains an entry call dl, which is
    # the length through a cell

    max_level = np.amax(ds.index.grid_levels)

    if find_indices:
        grid_left_edge = ds.domain_left_edge.to('code_length').ndview
        cell_width = (
            ds.domain_width/ds.domain_dimensions).to('code_length').ndview
    else:
        grid_left_edge, cell_width = None, None

    for start,end in start_end_pairs:
        start, end = np.asanyarray(start), np.asanyarray(end)
        total_length = np.linalg.norm(end - start)

        num_matching_components = np.count_nonzero(start == end)

        # set vals for sorted_idx & dl (for the case without intersection)
        sorted_idx, dl = slice(None), np.array([], dtype = 'f8')

        # now actually construct the ray object
        if (num_matching_components == 3) or (total_length == 0.0):
            raise AssertionError("START AND END POINTS ARE COLOCATED")
        elif False and (num_matching_components == 2):
            if start[0] != end[0]:
                cast_ax, plane_coord = 0, (start[1],start[2])
            elif start[1] != end[1]:
                cast_ax, plane_coord = 1, (start[2],start[0])
            else:
                cast_ax, plane_coord = 2, (start[0],start[1])
            ray = ds.ortho_ray(axis = cast_ax, coords = plane_coord)

            l,r = ds.domain_left_edge[cast_ax], ds.domain_right_edge[cast_ax]
            if isinstance(l, unyt.unyt_array):
                l = l.to('code_length').v
            if isinstance(l, unyt.unyt_array):
                r = r.to('code_length').v
            assert l >= min(start[cast_ax], end[cast_ax])
            assert r <= max(start[cast_ax], end[cast_ax])

            if max_level != 0:
                raise RuntimeError("Problems may arise with AMR")

            pos_fields = ['x','y','z']

            _pos_input = ray[pos_fields[cast_ax]]
            if _pos_input.size > 0:
                sorted_idx = np.argsort(_pos_input)
                dl = ray[['dx','dy','dz'][cast_ax]]
        else:
            ray = ds.ray(start, end)

            pos_fields = [('index', ax) for ax in 'xyz']
            if ray["t"].size > 0:
                sorted_idx = np.argsort(ray["t"])
                dl = ray['dts'][sorted_idx] * total_length

        # now actually fill in cur_fields
        cur_fields = {}
        if dl.size > 0:

            for field in fields:
                cur_fields[field] = ray[field][sorted_idx]
            cur_fields['dl'] = coerce_unyt_to_npy(
                dl, 'code_length', passthru_dimensionless = True,
                passthru_npy_array = True, can_return_view = True)

            if find_indices:
                indices = np.empty(shape = (3, sorted_idx.size), dtype = int)
                for i, pos_field in enumerate(pos_fields):
                    left_offset = (
                        ray[pos_field][sorted_idx].to('code_length').ndview -
                        grid_left_edge[i] - 0.5 * cell_width[i])
                    indices[i,:] = np.trunc(
                        left_offset/cell_width[i] + 0.5).astype(int)
                cur_fields['indices'] = indices
        yield cur_fields

def _zip_vectors(a,b):
    arg_shapes = []
    for arg in [a,b]:
        arg_shape = np.shape(arg)
        arg_ndim = len(arg_shape) # == np.ndim(arg)
        if arg_ndim != 1 and arg_ndim != 2:
            raise ValueError("each argument must be 1d or 2d")
        elif arg_shape[-1] != 3:
            raise ValueError("the last axis of each argument must have a "
                             "length of 3")
        arg_shapes.append(arg_shape)

    if np.ndim(a) == 1 and np.ndim(b) == 1:
        yield a,b
    elif np.ndim(a) == 1:
        for b_elem in b:
            yield a,b_elem
    elif np.ndim(b) == 1:
        for a_elem in a:
            yield a_elem, b
    elif arg_shapes[0] != arg_shapes[1]:
        raise ValueError("when both arguments are 2d, they must have "
                         "identical shapes") 
    else:
        yield from zip(a, b)

def _to_uvec(vec):
    vec = np.asanyarray(vec)
    if (vec == 0).all():
        raise ValueError()
    return vec / np.linalg.norm(vec)

def ray_values(ds, ray_start, ray_vec, fields = [],
               find_indices = False, include_uvec = False,
               kind = None):

    # even if fields is empty, it always contains an entry call dl, which is
    # the length through a cell

    left_edge, right_edge = ds.domain_left_edge, ds.domain_right_edge

    pairs = []
    uvec_l = []
    for cur_ray_start, cur_ray_vec in _zip_vectors(ray_start, ray_vec):
        uvec = _to_uvec(cur_ray_vec)

        intersect_dists = ray_box_intersections(cur_ray_start, uvec,
                                                left_edge = left_edge,
                                                right_edge = right_edge)
        assert len(intersect_dists) > 0

        # intentionally overshoot the back edge of the grid!
        cur_ray_stop = cur_ray_start + uvec * 1.2 * np.amax(intersect_dists)
        pairs.append((cur_ray_start, cur_ray_stop))

    itr = ray_values_startend(ds, start_end_pairs = pairs, fields = fields,
                              find_indices = find_indices, kind = kind)

    yield from itr
