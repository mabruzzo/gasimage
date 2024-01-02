from typing import Tuple,Union
import numpy as np
import unyt

def alt_build_spatial_grid_props(cm_per_length_unit: float,
                                 grid_shape: Union[Tuple[int,int,int],
                                                   np.ndarray],
                                 grid_left_edge: np.ndarray,
                                 cell_width: np.ndarray):

    # this is a dummy function that was created to help write tests since the
    # __init__ method of SpatialGridProps probably did a little too much

    grid_shape = np.array(grid_shape)

    spatial_props = SpatialGridProps(
        cm_per_length_unit = cm_per_length_unit,
        grid_shape = grid_shape,
        grid_left_edge = grid_left_edge,
        grid_right_edge = grid_left_edge + cell_width * grid_shape)

    # this is a little bit of a hack
    spatial_props.cell_width = cell_width.astype(dtype = 'f8', casting = 'safe',
                                                 copy = True)
    return spatial_props

def _coerce_grid_shape(grid_shape: Union[Tuple[int,int,int], np.ndarray]):
    _TYPE_ERR = (
        "grid_shape must be a 3-tuple of integers OR a numpy array of integers"
    )

    if isinstance(grid_shape, tuple): 
        if ((len(grid_shape) != 3) or
            not all(isinstance(e,int) for e in grid_shape)):
            raise TypeError(_TYPE_ERR)
        out = np.array(grid_shape, dtype = 'i8')
    elif isinstance(grid_shape, np.ndarray):
        if not issubclass(grid_shape.dtype.type, np.integer):
            raise TypeError(
                f"{_TYPE_ERR}; it's a numpy array with a different dtype")
        out = grid_shape
    else:
        raise TypeError(_TYPE_ERR)

    if not (out>=0).all():
        raise ValueError("All elements of grid_shape must be positive")
    return out

def _inspect_edge_args(grid_left_edge, grid_right_edge, allow_unyt = False):

    def _check(arg, arg_name):
        if not isinstance(arg, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        elif isinstance(arg, unyt.unyt_array) and not allow_unyt:
            raise TypeError(f"{name} must NOT be a unyt_array")
        elif arg.shape != (3,):
            raise ValueError(f"{name} must have a shape of (3,)")
        elif not np.isfinite(arg).all():
            raise ValueError(f"all elements in {name} must be finite")
    _check(grid_left_edge, "grid_left_edge")
    _check(grid_right_edge, "grid_right_edge")

    if (grid_left_edge >= grid_right_edge).any():
        raise ValueError("each element in grid_right_edge should exceed the "
                         "corresponding element in grid_left_edge")

class SpatialGridProps:
    """
    This collects spatial properties of the current block (or grid) of the
    dataset

    Notes
    -----
    The original motivation for doing this was to allow rescaling of the grid
    in adiabatic simulations. It's unclear whether that functionality still
    works properly (it definitely hasn't been tested in all contexts).
    """
    cm_per_length_unit : float
    grid_shape: np.ndarray
    left_edge: np.ndarray
    right_edge: np.ndarray
    cell_width: np.ndarray

    def __init__(self, *, cm_per_length_unit: float,
                 grid_shape: Union[Tuple[int,int,int], np.ndarray],
                 grid_left_edge: np.ndarray,
                 grid_right_edge: np.ndarray,
                 rescale_factor: float = 1.0):

        if cm_per_length_unit <= 0 or not np.isfinite(cm_per_length_unit):
            raise ValueError("cm_per_length_unit must be positive & finite")
        elif rescale_factor <= 0 or not np.isfinite(rescale_factor):
            raise ValueError("rescale factor must be positive & finite")
        self.cm_per_length_unit = cm_per_length_unit

        # make a copy, so it owns the data
        self.grid_shape = _coerce_grid_shape(grid_shape).copy(order = 'C')

        _inspect_edge_args(grid_left_edge, grid_right_edge, allow_unyt = False)
        # in each of the following cases, we pass copy = False, since we know
        # that the input array is newly created!
        _kw = dict(order = 'C', dtype = 'f8', casting = 'safe', copy = False)
        self.left_edge = (grid_left_edge * rescale_factor).astype(**_kw)
        self.right_edge = (grid_right_edge * rescale_factor).astype(**_kw)
        self.cell_width = (
            (self.right_edge - self.left_edge) / grid_shape
        ).astype(**_kw)

        for attr in ['grid_shape', 'left_edge', 'right_edge', 'cell_width']:
            getattr(self,attr).flags['WRITEABLE'] = False
            assert getattr(self,attr).flags['OWNDATA'] == True # sanity check!

    @classmethod
    def build_from_unyt_arrays(cls, *, cm_per_length_unit: float,
                               grid_shape: Union[Tuple[int, int, int],
                                                 np.ndarray],
                               grid_left_edge: unyt.unyt_array,
                               grid_right_edge: unyt.unyt_array,
                               length_unit: Union[str,unyt.Unit],
                               rescale_factor: float = 1.0):
        """
        This factory method is the preferred way to build a SpatialGridProps
        instance (it ensures that we are being self-consistent with units

        Notes
        -----
        This behavior used to be used by our constructor, but we found that to
        be a little limiting...

        Unclear whether this should be a separate function detached from
        SpatialGridProps
        """

        return cls(
            cm_per_length_unit = cm_per_length_unit,
            grid_shape = grid_shape,
            grid_left_edge = grid_left_edge.to(length_unit).ndview,
            grid_right_edge = grid_right_edge.to(length_unit).ndview,
            rescale_factor = rescale_factor)

    def __repr__(self):
        # Since we don't manually define a __str__ method, this will be called
        # by the built-in implementation
        #
        # at this moment, because the __init__ statement does a little too
        # much, this description won't be super useful
        return (
            "SpatialGridProps(\n"
            f"    cm_per_length_unit = {self.cm_per_length_unit!r},\n"
            f"    grid_shape = {np.array2string(self.grid_shape)},\n"
            "    grid_left_edge = "
            f"{np.array2string(self.left_edge, floatmode = 'unique')},\n"
            "    grid_right_edge = "
            f"{np.array2string(self.right_edge, floatmode = 'unique')},\n"
            ")"
        )

# these are here mostly just to make sure we perform these calculations
# consistently (plus, if we ever decide to change up this strategy, based on an
# internal change to the attributes, we just need to modify these

def get_grid_center(spatial_props):
    return 0.5 * (spatial_props.left_edge + spatial_props.right_edge)

def get_grid_width(spatial_props):
    return spatial_props.right_edge - spatial_props.left_edge
