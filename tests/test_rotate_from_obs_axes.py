import numpy as np
import yt
from unyt.testing import assert_allclose_units

from gasimage.ray_creation import rotate_from_obs_axes
from gasimage.utils.testing import assert_allclose_units


def test_rotate_from_obs_axes():
    ref_point = yt.YTArray([[3., 5., 10.]], 'kpc')

    # no transformation
    null_transform = rotate_from_obs_axes(
        points = ref_point, 
        observer_latitude_rad = 0.0, 
        domain_theta_rad = np.pi/2,
        domain_phi_rad = np.pi
    )
    assert_allclose_units(null_transform, ref_point,
                          atol = 0.0, rtol = 2e-15)

    # observer x-axis originally anti-aligned with domain x-axis
    """
                    ---------> y_domain
                    |
                    |
                    |
                    V x_domain

                                           (-3, -5, 10)
              x_obs ^     rotate axes       |
                    |     180 degrees       V
            *       |     clockwise         *
                    |       ====>
    y_obs <----------                           ----------> y_obs
                                                |
                                                |
                                                |
                                                V x_obs
    """
    x180 = rotate_from_obs_axes(
        points = ref_point, 
        observer_latitude_rad = 0.0, 
        domain_theta_rad = np.pi/2,
        domain_phi_rad = 0.0)
    assert_allclose_units(
        actual = x180,
        desired = yt.YTArray([[-3., -5., 10.]], 'kpc'),
        atol = 0.0, rtol = 2e-15)

    # observer x-axis originally anti-aligned with domain y-axis
    """
          <----------
     x_domain       |
                    |
                    |
                    V y_domain
                                           (5, -3, 10)
              x_obs ^     rotate axes       |
                    |     270 degrees       V
            *       |     clockwise         * 
                    |       ====>
    y_obs <----------               x_obs <----------
                                                    |
                                                    |
                                                    |
                                                    V y_obs
    """
    x_aligned_with_neg_y = rotate_from_obs_axes(
        points = ref_point, 
        observer_latitude_rad = 0.0, 
        domain_theta_rad = np.pi/2,
        domain_phi_rad = np.pi/2)
    assert_allclose_units(
        actual = x_aligned_with_neg_y,
        desired = yt.YTArray([[5., -3., 10.]], 'kpc'),
        atol = 0.0, rtol = 2e-15)
    
    # observer's x-axis originally aligned with domain y-axis
    """
           y_domain ^
                    |
                    |
                    |
                    ----------> x_domain
                                       (-5, 3, 10)
              x_obs ^     rotate axes       |       ^ y_obs
                    |     270 degrees       V       |
            *       |     clockwise         *       |
                    |       ====>                   |
    y_obs <----------                               ----------> x_obs
    """
    x_aligned_with_pos_y = rotate_from_obs_axes(
        points = ref_point, 
        observer_latitude_rad = 0.0, 
        domain_theta_rad = np.pi/2,
        domain_phi_rad = 3*np.pi/2)
    assert_allclose_units(
        actual = x_aligned_with_pos_y,
        desired = yt.YTArray([[-5., 3., 10.]], 'kpc'),
        atol = 0.0, rtol = 2e-15)

if __name__ == '__main__':
    test_rotate_from_obs_axes()
    print('All tests passed')
