from typing import Tuple,Union
import numpy as np
import unyt

def alt_build_spatial_grid_props(cm_per_length_unit: float,
                                 grid_shape: np.ndarray,
                                 grid_left_edge: unyt.unyt_array,
                                 cell_width):
    # this is a dummy function mostly intended for testing purposes
    # -> this is all a little bit of a hack! we should fix this!
    # -> this is because the __init__ statement of SpatialGridProps probably
    #    does a little too much!

    grid_shape = np.array(grid_shape)

    spatial_props = SpatialGridProps(
        cm_per_length_unit = cm_per_length_unit,
        grid_shape = grid_shape,
        grid_left_edge = unyt.unyt_array(grid_left_edge, 'cm'),
        grid_right_edge = unyt.unyt_array(
            grid_left_edge + cell_width * grid_shape, 'cm'),
        length_unit = 'cm', rescale_factor = 1.0)

    spatial_props.cell_width = cell_width.astype(dtype = 'f8', casting = 'safe',
                                                 copy = True)
    return spatial_props 

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
                 grid_shape: np.ndarray,
                 grid_left_edge: unyt.unyt_array,
                 grid_right_edge: unyt.unyt_array,
                 length_unit: str,
                 rescale_factor: float = 1.0):

        assert cm_per_length_unit > 0
        self.cm_per_length_unit = cm_per_length_unit

        assert grid_shape.shape == (3,) and (grid_shape > 0).all()
        assert issubclass(grid_shape.dtype.type, np.integer)
        # make a copy, so it owns the data
        self.grid_shape = grid_shape.copy(order = 'C')

        assert grid_left_edge.shape == grid_right_edge.shape == (3,)

        # in each of the following cases, we pass copy = False, since we know
        # that the input array is newly created!
        _kw = dict(order = 'C', dtype = 'f8', casting = 'safe', copy = False)
        self.left_edge = (
            grid_left_edge.to(length_unit).ndview * rescale_factor
        ).astype(**_kw)
        self.right_edge = (
            grid_right_edge.to(length_unit).v * rescale_factor
        ).astype(**_kw)
        self.cell_width = (
            (self.right_edge - self.left_edge) / np.array(grid_shape)
        ).astype(**_kw)

        for attr in ['grid_shape', 'left_edge', 'right_edge', 'cell_width']:
            getattr(self,attr).flags['WRITEABLE'] = False
            assert getattr(self,attr).flags['OWNDATA'] == True # sanity check!

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
