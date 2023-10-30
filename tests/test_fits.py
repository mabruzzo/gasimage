import astropy.wcs
import numpy as np

from gasimage.utils.fits import SimpleCubeInfo


_sample_hdr = dict([('SIMPLE', True), ('BITPIX', -32), ('NAXIS', 3),
                    ('NAXIS1', 721), ('NAXIS2', 181), ('NAXIS3', 476),
                    ('CTYPE1', 'RA---CAR'), ('CRVAL1', 172.5), ('CRPIX1', 1),
                    ('CDELT1', 0.0166667), ('CROTA1', 0.0), ('CUNIT1', 'deg'),
                    ('CTYPE2', 'DEC--CAR'), ('CRVAL2', -1.5), ('CRPIX2', 1),
                    ('CDELT2', 0.0166667), ('CROTA2', 0.0), ('CUNIT2', 'deg'),
                    ('CTYPE3', 'VELO'), ('CRVAL3', -170000.0), ('CRPIX3', 1),
                    ('CDELT3', 736.124999999987), ('CROTA3', 0.0),
                    ('CUNIT3', 'm/s'), ('EQUINOX', 2000.0),
                    ('BUNIT', 'K (Tb)'), ('OBSFREQ', 1420405800.0)])

def test_cube_info():
    hdr = _sample_hdr
    cube_info = SimpleCubeInfo.from_hdr(hdr, bypass_wcs_check = False,
                                        default_v_unit = 'm/s')
    wcs = astropy.wcs.WCS(hdr, fix = False)

    def get_pos(ax_name):
        _ind_map = {'NAXIS1' : 0, 'NAXIS2' : 1, 'NAXIS3' : 2}

        ax_len = hdr[ax_name]
        
        pos = np.zeros(shape = (ax_len,3), dtype = 'f8')
        pos[:,_ind_map[ax_name]] = np.arange(float(ax_len))

        origin = 0 # we are telling the function wcs_pix2world that 0 denotes
                   # the center of the first cell
        return wcs.wcs_pix2world(pos, origin, ra_dec_order = False)

    # you'll notice that when you evaluate get_pos('NAXIS1')[:,1], that the
    # values are not constant. As I understand it, this is a bug for a Plate
    # carr´ee projection (I'm just going to assume that the values stay
    # constant along that axis)
    ref_ra_vals = get_pos('NAXIS1')[:,0]
    ref_dec_vals = get_pos('NAXIS2')[:,1]
    ref_v_vals = get_pos('NAXIS3')[:,2]

    # this is the shape after you read it into a numpy array
    data_shape = (hdr['NAXIS3'],hdr['NAXIS2'],hdr['NAXIS1'])
    ra, dec = cube_info.get_ra_dec(data_shape, units = 'deg', ndim=1)
    v_channels = cube_info.get_v_channels(data_shape, units = 'm/s', ndim = 1)

    np.testing.assert_allclose(actual = ra, desired = ref_ra_vals,
                               rtol = 3e-5)
    np.testing.assert_allclose(actual = dec, desired = ref_dec_vals,
                               rtol = 5e-9)
    np.testing.assert_allclose(actual = v_channels, desired = ref_v_vals,
                               rtol = 0)

    # now check the round-trip of cube_info
    dumped_hdr = cube_info.write_to_hdr(hdr = {})
    reloaded_cube_info = SimpleCubeInfo.from_hdr(dumped_hdr,
                                                 bypass_wcs_check = False,
                                                 default_v_unit = 'm/s')
    assert cube_info == reloaded_cube_info
    second_dumped_hdr = reloaded_cube_info.write_to_hdr(hdr = {})
    assert len(dumped_hdr) == len(second_dumped_hdr)
    for k in dumped_hdr:
        assert dumped_hdr[k] == second_dumped_hdr[k]
