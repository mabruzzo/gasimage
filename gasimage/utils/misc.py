import unyt

def _has_consistent_dims(quantity, dims):
    """
    check if ``quantity`` has units with consistent dimensionality to the 
    ``unyt.dimensions`` object passed via ``dims``.

    If ``quantity`` doesn't have associated units, then this function assumes 
    its dimensionless
    """
    _dimensionless = unyt.dimensions.dimensionless
    if isinstance(quantity, unyt.unyt_array):
        if (quantity.units.is_dimensionless) and (dims == _dimensionless):
            return True
        return quantity.units.dimensions == dims
    return dims == _dimensionless

def check_consistent_arg_dims(quantity, dims, arg_name):
    if not _has_consistent_dims(quantity, dims):
        raise ValueError(f"the {arg_name} argument, {quantity} doesn't match "
                         f"the expected dimensionality of {dims}")
