import numpy as np
import unyt

from .._ray_intersections_cy import ray_box_intersections

def ray_start_stop(ray_collection, omit_intersectionless = False,
                   code_length = None, **kw):
    """
    Returns the start and end points of all rays in the collection 
    that pass through the specified domain.

    Kwargs
    ------
    ray_collection:
        A collection of rays. This should be an instance of one of the classes
        defined in this package
    omit_intersectionless: bool, Optional
        When False (the default) information for all rays is returned and a
        mask is provided to specify which rays never intersect the grid. 
        Otherwise
    code_length
        The code-length of then simulation. All intersection calculations are
        performed in these units.
    left_edge, right_edge: `unyt.unyt_array`, Optional
        `(3,)`  instances specifying the left and right edges of the simulation
        domain. Both MUST be specified when the `ds` kwarg is specified. If the
        `ds` argument is specified, both must be None
    ds: `yt.data_objects.static_output.Dataset`, Optional
        A yt-dataset returned by `yt.load`. If this is None, then neither
        `left_edge` nor `right_edge` must be specified.

    TODO: address this hacky solution

    Returns
    -------
    ray_start, ray_stop: `unyt.unyt_array
        The length of both arrays along the last axis is 3, In each case,
        `arr[...,0]`, `arr[...,1]`, and `arr[...,2]` specify x,y,z vals 
        respectively. When `omit_intersectionless` is True, these are 2D 
        arrays. Otherwise, the shape is given by `ray_collection.shape + (3,)`.
    has_intersection: np.ndarray
        An array of shape ray_start.shape[:-1] that holds a value of False for
        all rays that don't intersect with the grid. The only use case for
        ray_stop in these cases is for making pretty pictures. Only returned
        when `omit_intersectionless` is False.
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

    if omit_intersectionless:
        return ray_start[has_intersection], ray_stop[has_intersection]
    else:
        out_shape = ray_collection.shape + (3,)
        ray_start.shape = out_shape
        ray_stop.shape = out_shape
        has_intersection.shape = out_shape[:-1]
        return ray_start, ray_stop, has_intersection
