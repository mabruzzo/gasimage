cdef struct SpatialGridPropsStruct:
    double cm_per_length_unit
    int[3] grid_shape
    double[3] left_edge
    double[3] right_edge
    double[3] cell_width
    double[3] inv_cell_width

cdef class SpatialGridProps:
    cdef SpatialGridPropsStruct pack
