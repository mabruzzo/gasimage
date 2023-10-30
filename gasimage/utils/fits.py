import dataclasses
from typing import Optional

from astropy.io import fits
import astropy.wcs
import numpy as np
import unyt

from .misc import check_consistent_arg_dims, _has_consistent_dims

@dataclasses.dataclass(frozen = True)
class _CoordInfo:
    ref_val: float
    delta: float
    unit: str
    fits_refpix: float # a value of 1 indicates the center of the first pixel 
                       # along the given axis in a FITS file (a value of 0.5
                       # denotes the left edge of that same pixel)

def _compute_coord_vals(size, cinfo):
    # NOTE: a FITS-file reference pixel of 1 corresponds to the CENTER of the
    # leftmost pixel along a given axis (a value of 0.5 corresponds to the left
    # edge of that same pixel)
    return cinfo.ref_val + cinfo.delta * ( np.arange(float(size)) -
                                           (cinfo.fits_refpix - 1))

@dataclasses.dataclass(frozen = True)
class SimpleCubeInfo:
    """
    Stores basic spectral cube information for a cube of where
    latitudinal, longitudinal, and velocity information are perfectly
    aligned with axes of the cube

    Primarily exist to simplify process of serialization/deserialization
    (While you may lose some information when you first construct an 
    instance, serialization and desrialization should be lossless).

    This only supports "Plate carr´ee" projections (aka the Equirectangular 
    projection)
    """
    _long_data: _CoordInfo
    _lat_data: _CoordInfo
    _vel_data: _CoordInfo
    _obs_freq_Hz: float

    # unclear if the next one is meaningful, but tracking it for now...
    # (I think it's only relevant for relativistic velocity)
    _v_linearly_scales_with: Optional[str] # must be None (unknown), #
                                            # "wavelength" or "frequency"
    _equinox: float = 2000.0 # only carried around because of ref file

    @property
    def observed_freq(self):
        return self._obs_freq_Hz * unyt.Hz

    def get_ra_dec(self, cube_shape, units = 'deg', ndim = 1,
                   unyt_arr = False):
        # cube_shape is the shape of the numpy array (AFTER loading from FITS)
        assert len(cube_shape) == 3

        ra = unyt.unyt_array(
            _compute_coord_vals(size = cube_shape[2], cinfo = self._long_data),
            self._long_data.unit
        ).to(units).ndarray_view()
        dec = unyt.unyt_array(
            _compute_coord_vals(size = cube_shape[1], cinfo = self._lat_data),
            self._lat_data.unit
        ).to(units).ndarray_view()

        if ndim == 2:
            ra  = np.broadcast_to( ra[None,    :], cube_shape[1:])
            dec = np.broadcast_to(dec[   :, None], cube_shape[1:])
        elif ndim == 3:
            ra  = np.broadcast_to( ra[None, None,    :], cube_shape)
            dec = np.broadcast_to(dec[None,    :, None], cube_shape)
        elif ndim != 1:
            raise ValueError("ndim must be 1, 2, or 3")

        if unyt_arr:
            # create a view of the underlying data:
            return unyt.unyt_array(ra, units), unyt.unyt_arr(dec, units)
        return ra,dec

    def get_v_channels(self, cube_shape, units = 'm/s', ndim = 1,
                       unyt_arr = False):
        # cube_shape is the shape of the numpy array (AFTER loading from FITS)
        assert len(cube_shape) == 3

        v = unyt.unyt_array(
            _compute_coord_vals(size = cube_shape[0], cinfo = self._vel_data),
            self._vel_data.unit
        ).to(units).ndarray_view()

        if ndim == 3:
            v = np.broadcast_to(v[:, None, None], cube_shape)
        elif ndim != 1:
            raise ValueError("ndim must be 1 or 3")

        if unyt_arr:
            # create a view of the underlying data:
            return unyt.unyt_array(v, units)
        return v

    @classmethod
    def new(cls, sky_longitude_arr_deg, sky_delta_latitude_arr_deg,
            sky_latitude_ref_deg, v_channels, rest_freq,
            v_linearly_scales_with):
        """
        This is the most convenient way to build a SimpleCubeInfo object

        Parameters
        ----------
        sky_longitude_arr_deg: np.ndarray
            1D array of sky-longitude locations (in degrees) corresponding to 
            the centers of pixels. This is assumed to have constant spacing.
        sky_delta_latitude_arr_deg: np.ndarray
            1D array of sky-latitude locations, measured with respect to
            `sky_latitude_ref_deg` (in degrees) corresponding to the centers of
            pixels. This is assumed to have constant spacing.
        sky_latitude_ref_deg : float
            This is added to `sky_delta_latitude_arr_deg` to specify the
            sky-latitudes at the center of each pixel
        v_channels: 1D unyt.unyt_array
            A monotonically increasing array of velocity channels.
            This is assumed to have constant spacing.
        rest_freq: `unyt.unyt_quantity`
            Specifies the rest-frame frequency
        v_linearly_scales_with: str, optional
            This must be either None, 'frequency' or 'wavelength'.
        """
        longitude_arr_deg = sky_longitude_arr_deg
        latitude_arr_deg = sky_delta_latitude_arr_deg + sky_latitude_ref_deg
        assert ( (not isinstance(longitude_arr_deg, unyt.unyt_array)) and
                 (not isinstance(latitude_arr_deg, unyt.unyt_array)) )
        lat_data = _CoordInfo(ref_val = float(latitude_arr_deg[0]),
                              delta = float(np.diff(latitude_arr_deg)[0]),
                              fits_refpix = 1.0, unit = 'deg')
        long_data = _CoordInfo(ref_val = float(longitude_arr_deg[0]),
                               delta = float(np.diff(longitude_arr_deg)[0]),
                               fits_refpix = 1.0, unit = 'deg')
        vel_data = _CoordInfo(ref_val = float(v_channels[0].to('m/s').v),
                              delta = float(np.diff(v_channels)[0].to('m/s').v),
                              fits_refpix = 1.0, unit = 'm/s')

        assert ((v_linearly_scales_with is None) or 
                (v_linearly_scales_with in ['frequency', 'wavelength']))
        return cls(_long_data = long_data, _lat_data = lat_data,
                   _vel_data = vel_data,
                   _obs_freq_Hz = float(rest_freq.to('Hz')),
                   _v_linearly_scales_with = v_linearly_scales_with)
    
    def write_to_hdr(self, hdr = None):
        if hdr is None:
            hdr = fits.Header()
            
        if self._v_linearly_scales_with is None:
            vel_ctype = 'VELO'
        elif self._v_linearly_scales_with == 'frequency':
            vel_ctype = 'VRAD' # stands for 'radio velocity'
        elif self._v_linearly_scales_with == 'wavelength':
            vel_ctype = 'VOPT' # stands for optical velocity
        else:
            raise RuntimeError("somehow v_lineraly_scales_with attribute "
                               "is an invalid value")

        for i, ctype, props in [(1, 'RA---CAR', self._long_data),
                                (2, 'DEC--CAR', self._lat_data),
                                (3, vel_ctype, self._vel_data)]:
            hdr[f'CTYPE{i}'] = ctype
            hdr[f'CRVAL{i}'] = props.ref_val
            
            
            hdr[f'CRPIX{i}'] = props.fits_refpix
            hdr[f'CDELT{i}'] = props.delta
            hdr[f'CROTA{i}'] = 0.0
            hdr[f'CUNIT{i}'] = props.unit

        # This was present in the reference file
        hdr['EQUINOX'] = self._equinox

        # This was specified in the reference file
        hdr['OBSFREQ'] = self._obs_freq_Hz
        return hdr

    @classmethod
    def from_hdr(cls, hdr, bypass_wcs_check = False,
                 default_v_unit = 'm/s'):
        """
        Construct a SimpleCubeInfo instance from the header
        """
        def _get(key, *, expected_val = None):
            if key not in hdr:
                raise ValueError(
                    f"{cls.__name__} can't be constructed from the fits header "
                    f"because the '{key}' header-card is missing.")
            
            val = hdr[key]
            if (expected_val is not None) and (val != expected_val):
                raise ValueError(
                    f"{cls.__name__} can't be constructed from the fits header: "
                    f"The '{key}' header-card must have a value of "
                    f"{expected_val!r}, not {val!r}")
            return val

        def _get_coord_info(i, expected_name = None, fallback_unit = None):
            name = _get(f'CTYPE{i}', expected_val = expected_name)

            # confirm that the following card exists & has expected value
            _get(f'CROTA{i}', expected_val = 0.0)
            
            if fallback_unit is None:
                unit = _get(f'CUNIT{i}')
            else:
                unit = hdr.get(f'CUNIT{i}', fallback_unit)

            return name, _CoordInfo(ref_val = _get(f'CRVAL{i}'),
                                    delta = _get(f'CDELT{i}'),
                                    unit = unit,
                                    fits_refpix = _get(f'CRPIX{i}'))

        _, long_data = _get_coord_info(1, expected_name = 'RA---CAR',
                                       fallback_unit = 'deg')
        _, lat_data = _get_coord_info(2, expected_name = 'DEC--CAR',
                                      fallback_unit = 'deg')
        v_name, vel_data = _get_coord_info(3, 
                                           fallback_unit = default_v_unit)

        v_name = v_name.strip()

        if v_name in ['VELO', 'VELO-LSR']:
            v_linearly_scales_with = None
        elif v_name == 'VRAD':
            v_linearly_scales_with = 'frequency'
        elif v_name == 'VOPT':
            v_linearly_scales_with = 'wavelength'
        else:
            raise ValueError("the 'CTYPE3' header-card has an unexpected "
                             f"value: {v_name}")
                
        obs_freq_Hz = _get('OBSFREQ')
        equinox = _get('EQUINOX')

        out = cls(_long_data = long_data, _lat_data = lat_data,
                  _vel_data = vel_data, _obs_freq_Hz = obs_freq_Hz,
                  _v_linearly_scales_with = v_linearly_scales_with,
                  _equinox = equinox)

        skip_keys = []
        if v_name in ['VELO', 'VELO-LSR']:
            skip_keys = ['CTYPE3', 'SPECSYS']
        
        if bypass_wcs_check:
            return out
        elif _wcs_hdr_equivalence_check(out, ref_hdr = hdr, 
                                        skip_keys = skip_keys):
            return out
        else:
            raise RuntimeError("the header seemed to have some extra wcs keywords "
                               f"that were unexpected by {cls.__name__}")

def _wcs_hdr_equivalence_check(simple_cube_info, ref_hdr, skip_keys = []):
    """
    Essentially, there are a TON of WCS parameters, which makes it hard to know
    if there are extra keywords that somehow invalidate the description of the
    wcs assumed by SimpleCubeInfo.
    
    This performs a crude check of the equivalence of astropy.wcs.WCS objects
    constructed from data in simple_cube_info and data in the reference header
    """
    hdr = simple_cube_info.write_to_hdr()
    new_wcs = astropy.wcs.WCS(hdr, fix = False)
    ref_wcs = astropy.wcs.WCS(ref_hdr, fix = False)
    
    other = new_wcs.to_header()
    from_ref = ref_wcs.to_header()
    
    all_keys = set(other.keys())
    all_keys.update(from_ref.keys())
    for key in filter(lambda k: k not in skip_keys, all_keys):
        if ( (key not in other) or (key not in from_ref) or
              (other[key] != from_ref[key]) ):
            return False
    return True

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
    v_linearly_scales_with: str
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
    v_linearly_scales_with: str
        This must be either 'frequency' or 'wavelength'.
    writeto_kwargs : dict, Optional
        Keyword arguments to pass on to the `astropy.io.fits.writeto` function.
    """

    cube_info = SimpleCubeInfo.new(
        sky_longitude_arr_deg = sky_longitude_arr_deg,
        sky_delta_latitude_arr_deg = sky_delta_latitude_arr_deg,
        sky_latitude_ref_deg = sky_latitude_ref_deg,
        v_channels = v_channels, rest_freq = rest_freq,
        v_linearly_scales_with = v_linearly_scales_with,
    )

    if ppv_Tb.shape != (v_channels.size, sky_delta_latitude_arr_deg.size,
                        sky_longitude_arr_deg.size):
        raise ValueError(f"expected ppv_Tb.shape to be {expected_shape}, "
                         f"not {ppv_Tb.shape}")

    return write_cube_to_fits(fname, ppv_Tb, cube_info,
                              writeto_kwargs = writeto_kwargs)

def write_cube_to_fits(fname, ppv_Tb, cube_info, writeto_kwargs = {}):
    """
    Writes a ppv image to disk as a FITS file. The intensity should be in terms
    of brightness temperature

    Parameters
    ----------
    fname: str
        Specifies where to write the fits file.
    ppv_Tb: `unyt.unyt_array`
        3D array or specifying the brightness temperature
    cube_info: SimpleCubeInfo
        This is generally constructed when you read in another file
    writeto_kwargs : dict, Optional
        Keyword arguments to pass on to the `astropy.io.fits.writeto` function.
    """

    hdr = cube_info.write_to_hdr()

    if not isinstance(ppv_Tb, unyt.unyt_array):
        raise TypeError("ppv_Tb must be a unyt.unyt_array instance")
    elif not _has_consistent_dims(ppv_Tb, dims = unyt.dimensions.temperature):
        raise ValueError("ppv_Tb must specify brightness temperature")
    elif ppv_Tb.ndim != 3:
        raise ValueError("ppv_Tb must be 3D")

    # Specify data units
    hdr['BUNIT'] = 'K (Tb)'
    hdr['BSCALE'] = 1.0
    hdr['BZERO'] = 0.0

    if str(ppv_Tb.units) == str(unyt.Kelvin.units):
        data = ppv_Tb.ndarray_view()
    else:
        data = ppv_Tb.to('K').ndarray_view()

    return fits.writeto(fname, data = data.astype(np.float32),
                        header = hdr, **writeto_kwargs)

def read_cube_from_fits(fname, ignore_data_unit = False,
                        bypass_wcs_check = False,
                        default_v_units = 'm/s'):
    """
    Try to read a ppv cube from disk. On success, returns the spectral-cube and
    and an instance of SimpleCubeInfo.

    Parameters
    ----------
    fname
        Path to the fits file
    ignore_data_unit : bool, Optional
        When True, return a regular numpy array without any units
    bypass_wcs_check : bool, Optional
        Whether to bypass a check about the world-coordinate system when
        reading header info
    default_v_units: str, Optional
        Optional velocity-units to assume when they aren't specified in the
        FITS file

    Returns
    -------
    data
        3D ppv cube. Axis 0 varies in frequency.
    cube_info : SimpleCubeInfo
        Specifies information about the datacube. Can be used to write a new
        cube.
    """
    with fits.open(fname, mode = 'readonly') as hdul:
        assert len(hdul) == 1
        hdr = hdul[0].header
        cube_info = SimpleCubeInfo.from_hdr(
            hdr, bypass_wcs_check = bypass_wcs_check,
            default_v_unit = default_v_units)

        if not ignore_data_unit:
            if 'BUNIT' not in hdr:
                raise RuntimeError("Don't know what unit to assign to data")
            elif hdr['BUNIT'].strip().split() != ['K','(Tb)']:
                raise RuntimeError(f"hdr['BUNIT'] is unknown: {hdr['BUNIT']!r}")
            data = unyt.unyt_array(hdul[0].data, 'K')
        else:
            data = hdul[0].data
    return data, cube_info
        
