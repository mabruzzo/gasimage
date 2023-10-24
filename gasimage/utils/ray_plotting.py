# some undocumented routines used for plotting ray-collections

import numpy as np
import yt

from .ray_utils import ray_start_stop

def _determine_axes_bounds(prj_axis, ray_start, axes_unit, ds,
                           pad_frac = 0.01):
    # come up with bounds for the image
    
    assert ray_start.shape[-1] == 3 and ray_start.ndim in [1,2]
    ray_start.shape = (-1,3)

    axes_bounds = {}
    for axis_ind,axis_name in enumerate('xyz'):
        tmp_arr = ray_start[:, axis_ind].to(axes_unit)
        if tmp_arr.min() <= ds.domain_left_edge[axis_ind]:
            tmp = tmp_arr.min()
            # calculate width between tmp and 
            # ds.domain_right_edge[axis_ind]
            width = ds.domain_right_edge[axis_ind] - tmp
            # add some padding to the left of tmp
            low = tmp - width*pad_frac
            high = ds.domain_right_edge[axis_ind]
        elif tmp_arr.max() >= ds.domain_right_edge[axis_ind]:
            tmp = tmp_arr.max()
            # calculate width between tmp and 
            # ds.domain_width_edge[axis_ind]
            width = tmp - ds.domain_left_edge[axis_ind]
             # add some padding to the right of tmp
            high = tmp + width*pad_frac
            low = ds.domain_left_edge[axis_ind]
        else:
            low = ds.domain_left_edge[axis_ind]
            high = ds.domain_right_edge[axis_ind]
        
        
        # in the case, where we are including rays that never
        # intersect with the domain:
        if tmp_arr.min() < low:
            low = low - (1 + pad_frac) * (low-tmp_arr.min())
        if tmp_arr.max() > high:
            high = high + (1 + pad_frac) * (tmp_arr.max()-high)

        axes_bounds[axis_name] = (low.to(axes_unit).v,
                                  high.to(axes_unit).v)

    if prj_axis == 'z':
        return axes_bounds['x'], axes_bounds['y']
    elif prj_axis == 'y':
        return axes_bounds['z'], axes_bounds['x']
    else:
        return axes_bounds['y'], axes_bounds['z']

def _axis_label(prj_axis, ax, axes_unit):
    if prj_axis == 'x':
        imx_key, imy_key = 1, 2 # y, z
    elif prj_axis == 'y':
        imx_key, imy_key = 2, 0 # z, x
    else:
        imx_key, imy_key = 0, 1 # x, y
    ax.set_xlabel('xyz'[imx_key] + f' ({axes_unit!s})')
    ax.set_ylabel('xyz'[imy_key] + f' ({axes_unit!s})')

def _visualize_bbox(prj_axis, ax, left_edge, right_edge, axes_unit = None,
                    label_axis = True):
    if prj_axis == 'x':
        imx_key, imy_key = 1, 2 # y, z
    elif prj_axis == 'y':
        imx_key, imy_key = 2, 0 # z, x
    else:
        imx_key, imy_key = 0, 1 # x, y

    if axes_unit is None:
        axes_unit = left_edge.units
    l,r = left_edge.to(axes_unit).v, right_edge.to(axes_unit).v

    ax.fill_between(x = [l[imx_key], r[imx_key]],
                    y1 = l[imy_key], y2 = r[imy_key],
                    color = 'C0')
    
    ax.set_aspect('equal')
    if label_axis:
        _axis_label(prj_axis, ax, axes_unit)

def _plot_projected_ray_path(ax, prj_axis, ray_start, ray_stop,
                             axes_unit, plot_start_fmt = 'ro',
                             plot_stop_fmt = 'C4s'):
    assert prj_axis in ('x','y','z')

    ray_stop_view = ray_stop.view()
    if len(ray_stop.shape) == 0:
        raise RuntimeError()
    if len(ray_stop.shape) == 1:
        iterations = 1
        ray_stop_view.shape = (1,3)
    else:
        assert ray_stop.flags['C_CONTIGUOUS']
        iterations = int(np.prod(ray_stop.shape[:-1]))
        ray_stop_view.shape = (iterations,3)

    for i in range(iterations):
        coord_pairs = {}
        for axis_ind,axis_name in enumerate('xyz'):
            vals = (ray_start[i, axis_ind].to(axes_unit).v,
                    ray_stop_view[i, axis_ind].to(axes_unit).v)
            coord_pairs[axis_name] = vals
        if prj_axis == 'z':
            ax.plot(coord_pairs['x'],coord_pairs['y'], 'k-')
        elif prj_axis == 'y':
            ax.plot(coord_pairs['z'],coord_pairs['x'], 'k-')
        else:
            ax.plot(coord_pairs['y'],coord_pairs['z'], 'k-')

    ind_imx, ind_imy = {'z': (0,1), 'y' : (2,0), 'x' : (1,0)}[prj_axis]

    if (plot_start_fmt is not None) and len(plot_start_fmt) > 0:
        ax.plot(ray_start[:,ind_imx].to(axes_unit).v,
                ray_start[:,ind_imy].to(axes_unit).v, plot_start_fmt)

    if (plot_stop_fmt is not None) and len(plot_stop_fmt) > 0:
        ax.plot(ray_stop[:,ind_imx].to(axes_unit).v,
                ray_stop[:,ind_imy].to(axes_unit).v, plot_stop_fmt)

def _visualize_simple_rays(ds,fig,ax,prj_axis, ray_start, ray_stop,
                           axes_unit = None, full_proj = False):
    assert prj_axis in 'xyz'
    if full_proj:
        prj = yt.ProjectionPlot(
            ds, prj_axis, ("gas", "density"),
        )
        if axes_unit is None:
            axes_unit = prj.get_axes_unit()
        else:
            prj.set_axes_unit(str(axes_unit))
        plot = prj.plots[list(prj.plots)[0]]
        plot.axes = ax
        plot.fig = fig

        # redraw the projection
        prj._setup_plots()
    else:
        # just plot a bounding rectangle
        _visualize_bbox(prj_axis, ax, ds.domain_left_edge, ds.domain_right_edge,
                        axes_unit = axes_unit)

    _plot_projected_ray_path(ax, prj_axis, ray_start, ray_stop,
                             axes_unit, plot_start_fmt = 'ro',
                             plot_stop_fmt = '')

    # determine the axes bounds
    tmp_ray_start = ray_start[(np.isfinite(ray_stop).sum(axis=1)>0),:]

    xlim,ylim = _determine_axes_bounds(prj_axis, tmp_ray_start, axes_unit, ds)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def plot_projected_ray_path(ax, prj_axis, ray_collection, axes_unit,
                            plot_start_fmt = 'ro', plot_stop_fmt = 'C4s',
                            label_axis = False, exclude_intersectionless = True,
                            left_edge = None, right_edge = None, ds = None):
    """
    Plot the projected rays on the provided axes

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes object where the data is plotted
    prj_axis : {'x', 'y', 'z'}
        Axes to project along.
    ray_collection
        A collection of rays. This should be an instance of one of the classes
        defined in this package
    axes_unit
        The unit to convert values to before plotting
    label_axis : bool
        When True, automatically overwrite the axis labels
    exclude_intersectionless : bool
        When True, only plot rays that intersect with the domain. When False,
        all rays are plotted (the ones without intersections extend over an
        arbitrary distance).
    plot_start_fmt, plot_stop_fmt: Optional
        These specify the format argument for plotting the start and stop of 
        each ray. When one of these is `None` or an empty string, the 
        corresponding point is not drawn.
    left_edge, right_edge: `unyt.unyt_array`, Optional
        `(3,)`  instances specifying the left and right edges of the simulation
        domain. When specified, they must currently be in units of
        'code_length'. Both MUST be specified when the `ds` kwarg is specified.
        If the `ds` argument is specified, both must be None.
    ds: `yt.data_objects.static_output.Dataset`, Optional
        A yt-dataset returned by `yt.load`. If this is None, then neither
        `left_edge` nor `right_edge` must be specified.
    """

    if ds is None:
        if (left_edge is not None) and (right_edge is not None):
            kw = {'left_edge' : left_edge, 'right_edge' : right_edge}
        else:
            raise ValueError("when ds is None, left_edge and right_edge must "
                             "not be None")
    else:
        if (left_edge is None) and (right_edge is None):
            kw = {'left_edge' : left_edge, 'right_edge' : right_edge}
        else:
            raise ValueError("when ds is not None, left_edge and right_edge "
                             "must both be None")

    ray_start, ray_stop = ray_start_stop(
        ray_collection, code_length = None, ds = ds,
        omit_intersectionless = exclude_intersectionless
    )[:2]

    _plot_projected_ray_path(ax, prj_axis, ray_start, ray_stop,
                             axes_unit = axes_unit,
                             plot_start_fmt = plot_start_fmt,
                             plot_stop_fmt = plot_stop_fmt)

    my_ray_start,my_ray_stop = ray_start.reshape(-1,3), ray_stop.reshape(-1,3)

    if label_axis:
        _axis_label(prj_axis, ax, axes_unit)

def plot_ray_and_projection(ds, fig, ax, prj_axis, ray_collection,
                            axes_unit = None, full_proj = False,
                            exclude_intersectionless = True):
    ray_start, ray_stop = ray_start_stop(
        ray_collection, code_length = None, ds = ds,
        omit_intersectionless = exclude_intersectionless
    )[:2]

    my_ray_start,my_ray_stop = ray_start.reshape(-1,3), ray_stop.reshape(-1,3)

    _visualize_simple_rays(ds,fig,ax,prj_axis, my_ray_start, my_ray_stop,
                           axes_unit = axes_unit, full_proj = full_proj)
