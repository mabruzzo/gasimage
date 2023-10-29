from astropy.io import fits
import numpy as np
import unyt

from .misc import check_consistent_arg_dims

def _write_wcs_header(sky_longitude_arr_deg, sky_delta_latitude_arr_deg,
                      sky_latitude_ref_deg, v_channels, rest_freq,
                      v_linearly_scales_with = 'frequency',
                      hdr = None):
    """
    This assumes that the entries of sky_longitude_arr_deg,
    sky_delta_latitude_arr_deg, and v_channels are all evenly spaced.

    Possible Improvements:
    - We should really package the latitude, longitude, and v_channel
      information in data-structures (that explicitly define the constant
      spacing)
    - It would be nice to log some additional simulation information to 
      the headers (just so that we have it for reference)

    Parameters
    ----------
    sky_longitude_arr_deg: np.ndarray
        1D array of sky-longitude locations (in degrees) corresponding to the
        centers of pixels
    sky_delta_latitude_arr_deg: np.ndarray
        1D array of sky-latitude locations, measured with respect to
        `sky_latitude_ref_deg` (in degrees) corresponding to the centers of
        pixels
    sky_latitude_ref_deg : float
        This is added to `sky_delta_latitude_arr_deg` to specify the
        sky-latitudes at the center of each pixel
    v_channels: 1D unyt.unyt_array
        A monotonically increasing array of velocity channels.
    rest_freq: `unyt.unyt_quantity`
        Specifies the rest-frame frequency
    v_scales_lineraly_with: str
        This must be either 'frequency' or 'wavelength'.
    """
    if hdr is None:
        hdr = fits.Header()
    coord_pairs = [('RA---CAR', sky_longitude_arr_deg),
                   ('DEC--CAR', (sky_delta_latitude_arr_deg + 
                                 sky_latitude_ref_deg))]

    for i, (name, coord_vals) in enumerate(coord_pairs):
        i += 1
        hdr['CTYPE' + str(i)] = name
        hdr['CRVAL' + str(i)] = coord_vals[0]
        hdr['CRPIX' + str(i)] = 1
        hdr['CDELT' + str(i)] = np.diff(coord_vals)[0]
        hdr['CROTA' + str(i)] = 0.0
        hdr['CUNIT' + str(i)] = 'deg'

    # handle velocity info
    if v_linearly_scales_with == 'frequency':
        ctype_vel = 'VRAD' # stands for 'radio velocity'
    elif v_linearly_scales_with == 'wavelength':
        ctype_vel = 'VOPT' # stands for optical velocity

    hdr['CTYPE3'] = ctype_vel
    hdr['CRVAL3'] = float(v_channels[0].to('m/s').v)
    hdr['CRPIX3'] = 1
    hdr['CDELT3'] = float(np.diff(v_channels)[0].to('m/s').v)
    hdr['CROTA3'] = 0.0000
    hdr['CUNIT3'] = 'm/s'

    # This was present in the reference file
    hdr['EQUINOX'] = 2000.00

    # This was specified in the reference file
    hdr['OBSFREQ'] = float(rest_freq.to('Hz').v)

    return hdr

def write_to_fits(fname, ppv_Tb, sky_longitude_arr_deg,
                  sky_delta_latitude_arr_deg,
                  sky_latitude_ref_deg,
                  v_channels, rest_freq,
                  v_linearly_scales_with = 'frequency',
                  writeto_kwargs = {}):
    """
    Writes a ppv image to disk as a FITS file. The intensity should be in terms
    of brightness temperature

    Parameters
    ----------
    fname: str
        Specifies where to write the fits file.
    ppv_Tb: `unyt.unyt_array`
        3D array or specifying the brightness temperature
    sky_longitude_arr_deg: np.ndarray
        1D array of sky-longitude locations (in degrees) corresponding to the
        centers of pixels
    sky_delta_latitude_arr_deg: np.ndarray
        1D array of sky-latitude locations, measured with respect to
        `sky_latitude_ref_deg` (in degrees) corresponding to the centers of
        pixels
    sky_latitude_ref_deg : float
        This is added to `sky_delta_latitude_arr_deg` to specify the
        sky-latitudes at the center of each pixel
    v_channels: 1D unyt.unyt_array
        A monotonically increasing array of velocity channels.
    rest_freq: `unyt.unyt_quantity`
        Specifies the rest-frame frequency
    v_scales_lineraly_with: str
        This must be either 'frequency' or 'wavelength'.
    writeto_kwargs : dict, Optional
        Keyword arguments to pass on to the `astropy.io.fits.writeto` function.
    """

    hdr = _write_wcs_header(
        sky_longitude_arr_deg = sky_longitude_arr_deg,
        sky_delta_latitude_arr_deg = sky_delta_latitude_arr_deg,
        sky_latitude_ref_deg = sky_latitude_ref_deg,
        v_channels = v_channels, rest_freq = rest_freq
        v_linearly_scales_with = v_linearly_scales_with,
    )

    expected_ppv_shape = (v_channels.size, sky_delta_latitude_arr_deg.size,
                          sky_longitude_arr_deg.size)
    if not isinstance(ppv_Tb, unyt.unyt_array):
        raise TypeError("ppv_Tb must be a unyt.unyt_array instance")
    elif not _has_consistent_dims(ppv_Tb, dims = unyt.dimensions.temperature):
        raise ValueError("ppv_Tb must specify brightness temperature")
    elif ppv_Tb.ndim != 3:
        raise ValueError("ppv_Tb must be 3D")
    elif ppv_Tb.shape != expected_ppv_shape
        raise ValueError(f"expected ppv_Tb.shape to be {expected_shape}, "
                         f"not {ppv_Tb.shape}")

    # Specify data units
    hdr['BUNIT'] = 'K (Tb)'
    hdr['BSCALE'] = 1.0
    hdr['BZERO'] = 0.0

    if str(ppv_Tb.units) == str(unyt.Kelvin.units):
        data = ppv_Tb.ndarray_view()
    else:
        data = ppv_Tb.to('K').ndarray_view()

    return fits.writeto(fname,
                        data = data.astype(np.float32),
                        header = hdr, **writeto_kwargs)
