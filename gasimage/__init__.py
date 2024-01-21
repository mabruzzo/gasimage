__version__ = "0.1.0"

# do some setup so that the user doesn't need to know how modules and
# datastructures are organized within this package

from gasimage.optically_thin_ppv import (
    convert_intensity_to_Tb,
    convert_Tb_to_intensity,
    optically_thin_ppv,
)

from gasimage.snapdsinit import SnapDatasetInitializer

from gasimage.ray_collection import parallel_ray_grid, perspective_ray_grid

from gasimage.rt_config import default_spin_flip_props

from gasimage.ray_traversal._misc_cy import spatial_props_from_ds
