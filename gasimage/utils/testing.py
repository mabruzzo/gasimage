import numpy as np
import unyt

def assert_allclose_units(actual, desired, rtol=1e-07, atol=0, **kwargs):
    """
    A wrapper for `numpy.testing.assert_allclose` that properly handles 
    `unyt.unyt_array objects`.

    Note
    ----
    despite its documentation, `unyt.testing.assert_allclose_units` is not a
    perfect wrapper
    """

    act_is_uarr = isinstance(actual, unyt.unyt_array)
    des_is_uarr = isinstance(desired, unyt.unyt_array)

    def _u_mismatch_err_msg(name_pair = ('x','y')):
        act_unit = str(actual.units) if act_is_uarr else '(dimensionless)'
        des_unit = str(desired.units) if des_is_uarr else '(dimensionless)'

        msg = ['', #first line is empty
               f'Not equal to tolerance rtol={rtol!r}, atol={atol!r}',
               kwargs.get('err_msg',''),
               f"(units {act_unit}, {des_unit} mismatch)"
        ]
        
        for name, arr in zip(name_pair, (actual,desired)):
            # just like in numpy, we limit the array representation to 3 lines
            array_repr_lines = repr(arr).splitlines()
            if len(array_repr_lines) > 3:
                array_str = "\n".join(array_repr_lines[:3]) + '...'
            else:
                array_str = "\n".join(array_repr_lines)
            msg.append(f'{name}: {array_str}')
        return "\n".join(msg)

    if (not act_is_uarr) and (not des_is_uarr):
        np.testing.assert_allclose(actual, desired.v, rtol=rtol,
                                   atol=atol, **kwargs)
    elif act_is_uarr and des_is_uarr:
        try:
            act = actual.to(desired.units) # makes more sense to convert actual
        except unyt.exceptions.UnitConversionError:
            raise AssertionError(_u_mismatch_err_msg())
        np.testing.assert_allclose(act.v, desired.v, rtol=rtol, atol=atol,
                                   **kwargs)
    elif act_is_uarr and actual.units.is_dimensionless:
        np.testing.assert_allclose(actual.v, desired, rtol=rtol, atol=atol,
                                   **kwargs)
    elif des_is_uarr and desired.units.is_dimensionless:
        np.testing.assert_allclose(actual, desired.v, rtol=rtol, atol=atol,
                                   **kwargs)
    else:
        raise AssertionError(_u_mismatch_err_msg())
