from ._ray_intersections_cy import (
    ray_box_intersections,
    traverse_grid as _legacy_traverse_grid,
    max_num_intersections
)

from ._yt_grid_traversal_cy import traverse_grid as _new_traverse_grid

#traverse_grid = _legacy_traverse_grid
traverse_grid = _new_traverse_grid
