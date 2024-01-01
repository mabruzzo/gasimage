from typing import Tuple,Union
import numpy as np
import unyt

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


# TODO: unify this with SpatialGridProps.
#
# At the moment, the following just exists to provide a short-term similar
# interface to SpatialGridProps that can be constructed within some functions...
# -> then later in the future, we can just pass an instance of SpatialGridProps
#    to those same functions as an argument
#
# In the longer term, it may make sense to stop tracking cm_per_length_unit as
# an attribute of spatial_grid_props
class InternalSpatialGridProps:
    grid_shape: np.ndarray
    left_edge: np.ndarray
    right_edge: np.ndarray
    cell_width: np.ndarray

    def __init__(self, *, grid_shape : np.ndarray, left_edge : np.ndarray,
                 right_edge : np.ndarray, cell_width : np.ndarray):
        assert grid_shape.shape == (3,) and (grid_shape > 0).all()
        assert issubclass(grid_shape.dtype.type, np.integer)

        # make copies, so it owns the data
        self.grid_shape = grid_shape.copy(order = 'C')
        self.left_edge = left_edge.astype(dtype = 'f8', casting = 'safe',
                                          copy = True)
        self.right_edge = right_edge.astype(dtype = 'f8', casting = 'safe',
                                            copy = True)
        self.cell_width = cell_width.astype(dtype = 'f8', casting = 'safe',
                                            copy = True)

        for attr in ['grid_shape', 'left_edge', 'right_edge', 'cell_width']:
            getattr(self,attr).flags['WRITEABLE'] = False
            assert getattr(self,attr).flags['OWNDATA'] == True # sanity check!
            assert getattr(self,attr).shape == (3,)

    @classmethod
    def build(cls, grid_shape : Union[Tuple[int,int,int], np.ndarray],
              left_edge : np.ndarray, cell_width : np.ndarray):
        if isinstance(grid_shape, tuple):
            grid_shape = np.array(grid_shape, dtype = int)
        return cls(grid_shape = grid_shape, left_edge = left_edge,
                   right_edge = left_edge + cell_width * grid_shape,
                   cell_width = cell_width)

# these are here mostly just to make sure we perform these calculations
# consistently (plus, if we ever decide to change up this strategy, based on an
# internal change to the attributes, we just need to modify these

def get_grid_center(spatial_props):
    return 0.5 * (spatial_props.left_edge + spatial_props.right_edge)

def get_grid_width(spatial_props):
    return spatial_props.right_edge - spatial_props.left_edge
