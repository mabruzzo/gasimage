import numpy as np
import unyt

from .._ray_intersections_cy import ray_box_intersections

def ray_start_stop(ray_collection, 
                   code_length = None, **kw):
    """
    Returns the start and end points of all rays in the collection 
    that pass through the specified domain.

    Kwargs
    ------
    This can accept the `'left_edge'`, `'right_edge'` kwargs, which are 
    `(3,)` `unyt.unyt_array` instances. Alternatively  the user can specify a
    `ds` object (a yt-dataset returned by `yt.load`).

    TODO: address this hacky solution

    Returns
    -------
    ray_start, ray_stop: `unyt.unyt_array
        Each object has the shape given by `ray_collection.shape + (3,)`,
        where `arr[...,0]`, `arr[...,1]`, and `arr[...,2]` specify x,y,z
        vals respectively
    w_no_intersection: np.ndarray
        An array of shape ray_start.shape[:-1] that holds a value of True for
        all rays that don't intersect with the grid. The only use case for
        ray_stop in these cases is for making pretty pictures.
    """
    if ((len(kw) == 2) and 
        all(e in kw for e in ['left_edge','right_edge'])):
        left_edge, right_edge = kw['left_edge'], kw['right_edge']
    elif (len(kw) == 1) and ('ds' in kw):
        left_edge = kw['ds'].domain_left_edge
        right_edge = kw['ds'].domain_right_edge
    else:
        raise RuntimeError('This requires "left_edge" and "right_edge" kwargs '
                           'or just the "ds" kwarg')
    
    if code_length is None:
        assert left_edge.units == right_edge.units
        code_length = left_edge.units
        left_edge = left_edge.v
        right_edge = right_edge.v
    else:
        left_edge = left_edge.to(code_length).v
        right_edge = right_edge.to(code_length).v
    assert (right_edge > left_edge).all()

    assert hasattr(ray_collection, 'as_concrete_ray_list')
    ray_list = ray_collection.as_concrete_ray_list(
        unyt.unyt_quantity(1.0, code_length).to('cm')
    )

    ray_start = ray_list.ray_start_codeLen
    ray_uvec = ray_list.get_ray_uvec()
    ray_stop = np.empty_like(ray_start)
    has_intersection = np.zeros(dtype = bool, shape = ray_start.shape[:-1])

    for i in range(ray_start.shape[0]):
        distances = ray_box_intersections(
            line_start = ray_start[i,:], line_uvec = ray_uvec[i,:],
            left_edge = left_edge, right_edge = right_edge
        )
        if len(distances) == 0:
            has_intersection[i] = False
            # come up with a ray_stop that will look nice in plots
            d = max(np.nanmax((left_edge - ray_start)/ray_uvec),
                    np.nanmax((right_edge - ray_start)/ray_uvec))
            if d <= 0:
                d = np.nan
        else:
            has_intersection[i] = True
            d = max(distances)
        ray_stop[i,:] = ray_start[i,:] + ray_uvec[i,:] * d

    ray_start = unyt.unyt_array(ray_start, code_length)
    ray_stop = unyt.unyt_array(ray_stop, code_length)
    if len(ray_collection.shape) > 1:
        new_shape = ray_collection.shape + (3,)
        ray_start.shape = new_shape
        ray_stop.shape = new_shape
        has_intersection.shape = new_shape[:-1]
    return ray_start, ray_stop, has_intersection
