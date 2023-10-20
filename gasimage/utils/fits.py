import numpy as np
from astropy.io import fits

def _write_wcs_header(sky_longitude_arr_deg, sky_delta_latitude_arr_deg,
                      sky_latitude_ref_deg, v_channels, rest_freq,
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
    hdr['CTYPE3'] = 'VELO'
    hdr['CRVAL3'] = float(v_channels[0].to('m/s').v)
    hdr['CRPIX3'] = 1
    hdr['CDELT3'] = float(np.diff(v_channels)[0].to('m/s').v)
    hdr['CROTA3'] = 0.0000
    hdr['CUNIT3'] = 'm/s'

    # This was present in the reference file
    hdr['EQUINOX'] = 2000.00

    # Specify data units
    hdr['BUNIT'] = 'K (Tb)'
    hdr['BSCALE'] = 1.0
    hdr['BZERO'] = 0.0

    # This was specified in the reference file
    hdr['OBSFREQ'] = rest_freq

    return hdr

def write_to_fits(fname, ppv_Tb, sky_longitude_arr_deg,
                  sky_delta_latitude_arr_deg,
                  sky_latitude_ref_deg,
                  v_channels, rest_freq,
                  writeto_kwargs = {}):
    """
    Writes a ppv image to disk as a FITS file. The intensity should be in terms
    of brightness temperature
    """
    hdr = _write_wcs_header(sky_longitude_arr_deg,
                            sky_delta_latitude_arr_deg,
                            sky_latitude_ref_deg,
                            v_channels, rest_freq)
    return fits.writeto(fname,
                        data = ppv_Tb.to('K').v.astype(np.float32), 
                        header = hdr, **writeto_kwargs)
