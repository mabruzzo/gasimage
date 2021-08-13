import numpy as np
import yt

_inv_sqrt_pi = 1.0/np.sqrt(np.pi)

def line_profile(obs_freq, doppler_v_width, rest_freq,
                 velocity_offset = None):
    # freq_arr is a 1D array
    # doppler_v_width is a 1D array (it may have a different
    # shape from freq_arr)
    # rest_freq is a scalar

    # temp is equivalent to 1./(sqrt(2) * sigma)
    temp = (
        yt.units.c_cgs/(rest_freq*doppler_v_width)
    )

    norm = _inv_sqrt_pi * temp
    half_div_sigma2 = temp*temp
    
    # need to check this correction!
    if velocity_offset is None:
        emit_freq = obs_freq[:,np.newaxis]
    else:
        assert velocity_offset.shape == doppler_v_width.shape
        v_div_c_plus_1 = 1 + velocity_offset/yt.units.c_cgs
        emit_freq = obs_freq[:,np.newaxis]/(v_div_c_plus_1.to('dimensionless').v)

    delta_nu = (emit_freq - rest_freq)
    delta_nu_sq = delta_nu*delta_nu
    exponent = (-1*delta_nu_sq*half_div_sigma2)

    return norm*np.exp(exponent.to('dimensionless').v)

def _generate_spectrum(obs_freq, velocities, ndens_HI,
                       doppler_v_width, rest_freq, dz):
    _A10 = 2.85e-15*yt.units.Hz
    n1 = 0.75*ndens_HI # need spin temperature to be more exact
    profiles = line_profile(obs_freq = obs_freq,
                            doppler_v_width = doppler_v_width,
                            rest_freq = rest_freq,
                            velocity_offset = velocities)
    j_nu = yt.units.h_cgs * rest_freq *n1* _A10 * profiles/(4*np.pi)
    integrated = (j_nu*dz).sum(axis=1)

    if True:
        # need to think more about the units
        # there may be an implicit dependence on the solid angle
        return integrated.to('erg/cm**2').v
    if False:
        n0 = 0.25*ndens_HI
        g1_div_g0 = 3
        rest_wave = yt.units.c_cgs/rest_freq
        Tspin = 100.0*yt.units.K
        stim_correct = 1.0-np.exp(-0.0682/Tspin.to('K').v)
        alpha_nu = n0*g1_div_g0*_A10*rest_wave**2 * stim_correct * profiles/(8*np.pi)
        optical_depth = (alpha_nu*dz).sum(axis=1)
        return integrated,optical_depth

def _calc_doppler_v_width(ray):
    return np.sqrt(2*yt.units.kb * ray['temperature']/
                  (ray['mean_molecular_weight']*yt.units.mh))

def generate_ray_spectrum(ray, rest_freq, obs_freq, doppler_v_width = None,
                          ndens_HI_field = ('gas', 'H_p0_number_density'),
                          out = None):
    vx_field,vy_field,vz_field = (
        ('gas','velocity_x'), ('gas','velocity_y'), ('gas','velocity_z')
    )

    if out is not None:
        assert out.shape == obs_freq.shape
    else:
        out = np.empty_like(rest_freq)

    problem = False
    try:
        ray['dts']
    except:
        problem = True


    if problem:
        out[:] = np.nan
    else:
        if len(ray[vx_field]) == 0:
            raise RuntimeError()
            
        unit_vec = (ray.vec/np.sqrt((ray.vec*ray.vec).sum())).to('dimensionless').v
        # compute the velocity component. We should probably confirm
        # correctness of the velocity sign
        vlos = (unit_vec[0] * ray[vx_field] +
                unit_vec[1] * ray[vy_field] +
                unit_vec[2] * ray[vz_field])

        # compute the distance through a given cell
        # I think this approach should work, but I'm not completely sure
        _tmp = ray.end_point - ray.start_point
        total_length = np.sqrt((_tmp*_tmp).sum())
        dz = (total_length * ray['dts']).to('cm')

        if doppler_v_width is None:
            # it would probably be more sensible to make doppler_v_width
            # into a field
            cur_doppler_v_width = _calc_doppler_v_width(ray)
        else:
            cur_doppler_v_width = doppler_v_width

        # we should come back to this and handle it properly in the future
        out[:] = _generate_spectrum(obs_freq = obs_freq,
                                    velocities = vlos, 
                                    ndens_HI = ray[ndens_HI_field],
                                    doppler_v_width = cur_doppler_v_width, 
                                    rest_freq = rest_freq, 
                                    dz = dz)
    return out


class LazyRayIterator:
    # creates rays as we iterate
    # this iterator can only be used once (it get's consumed)

    def __init__(self,start_points, end_points,ds):
        assert (start_points.shape == (3,) or 
                start_points.shape == end_points.shape)
        assert np.ndim(end_points) > 1
        assert end_points.shape[-1] == 3
        self._start_points = start_points
        assert end_points.flags['C_CONTIGUOUS']
        self._end_points = end_points
        self._2D_end_points_view = end_points.view()
        self._2D_end_points_view.shape = (-1,3)
        self._ds = ds
        
        self._index = 0
        self._size = self._2D_end_points_view.shape[0]

    def get_shape(self):
        # return the ray shape (callers of this method 
        # don't care about the fact that there are 3 
        # position components of each end-point)
        return self._end_points.shape[:-1]

    def __iter__(self):
        return self

    def __next__(self):
        if self._index <= self._size:
            new_ray = self._ds.ray(
                self._start_points,
                self._2D_end_points_view[self._index,:]
            )
            self._index += 1
            return new_ray
        else:
            raise StopIteration

def optically_thin_ppv(v_channels, rays,
                       ndens_HI_field = ('gas', 'H_p0_number_density'),
                       doppler_v_width = None):
    # this sort of works. I think we are pushing the limits of yt's Ray object

    if isinstance(rays, LazyRayIterator):
        destroy_used_rays = True
        rays_shape = rays.get_shape()
        ndim_rays = len(rays_shape)
        ray_iter_1D = rays

    else:
        destroy_used_rays = False
        ndim_rays = np.ndim(rays)
        if ndim_rays == 0:
            rays = np.atleast_1d(rays)
        else:
            rays = np.asanyarray(rays)
            assert rays.flags['C_CONTIGUOUS']
        rays_shape = rays.shape
        ray_iter_1D = rays.view()
        ray_iter_1D.shape = (rays.size,)

    # create the output array
    out_shape = (v_channels.size,) + rays_shape
    out = np.empty(shape = out_shape)

    # flatten the out array down to 2-dimensions:
    out_2D = out.view()
    out_2D.shape = (v_channels.size, np.prod(rays_shape))

    rest_freq = 1.4204058E+09*yt.units.Hz
    obs_freq = (rest_freq*(1+v_channels/yt.units.c_cgs)).to('Hz')

    # this is a contiguous array where the spectrum for a single
    # ray gets stored
    _temp_buffer = np.empty(shape = (v_channels.size,),
                            dtype = np.float64)
    for i, cur_ray in enumerate(ray_iter_1D):
        print(i)
        # generate the spectrum
        generate_ray_spectrum(ray = cur_ray,
                              rest_freq = rest_freq,
                              obs_freq = obs_freq,
                              doppler_v_width = doppler_v_width,
                              ndens_HI_field = ndens_HI_field,
                              out = _temp_buffer)
        out_2D[:,i] = _temp_buffer
        
        if destroy_used_rays:
            cur_ray.clear_data()
            del cur_ray

    # need to think more about the about output units (specifically,
    # think about dependence on solid angle)
    out_units = 'erg/cm**2'
    if ndim_rays == 0:
        assert out_2D.shape == (v_channels.size, 1)
        return yt.YTArray(out_2D[:,0], out_units)
    else:
        return yt.YTArray(out, out_units)
