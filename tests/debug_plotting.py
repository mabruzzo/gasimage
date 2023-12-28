# this exists for debugging purposes!
# we explicitly import matplotlib inside of functions so that matplotlib is not
# a dependency of the test-suite

import unyt

def plot_ray_spectrum(obs_freq, dz_vals, integrated_source, total_tau,
                      debug_info = None, plot_debug = False,
                      rest_freq = None):
    import matplotlib.pyplot as plt

    I_unit = (r'$\left[\frac{{\rm erg}}{{\rm cm}^{2}\ {\rm s}\ '
              r'{\rm ster}\ {\rm Hz}}\right]$')

    wave = (unyt.c_cgs/obs_freq).to('nm')

    if not plot_debug:
        fig,ax_arr = plt.subplots(2,1, figsize = (4,8), sharex = True)
        if integrated_source.ndim > 1:
            for i in range(integrated_source.shape[1]):
                ax_arr[0].plot(wave, integrated_source[:,i])
                ax_arr[1].plot(wave, total_tau[:,i])
        else:
            ax_arr[0].plot(wave, integrated_source)
            ax_arr[1].plot(wave, total_tau)
        ax_arr[1].set_xlabel('wavelength (nm)')
        ax_arr[0].set_ylabel(r'$I_\nu(\lambda)$ ' + I_unit)
        ax_arr[1].set_ylabel(r'$\tau_\nu(\lambda)$')

        if rest_freq is not None:
            rest_wavelength = (unyt.c_cgs/rest_freq).to('nm')
            ax_arr[0].axvline(rest_wavelength, ls = ':', color = 'k')
            ax_arr[1].axvline(rest_wavelength, ls = ':', color = 'k')
    else:
        tau, source_func = debug_info

        tau_midpoint = 0.5*(tau[:,:-1] + tau[:,1:])
        fig, ax_arr = plt.subplots(3,1, figsize = (4,8))
        ax_arr[0].plot(source_func[0,:])

        center_f = obs_freq.size//2
        for freq_ind in [0,center_f//4, center_f//2, 3*center_f//4, center_f]:
            ax_arr[1].plot(tau[freq_ind, :])
            label = (r'$\lambda$ = ' +
                     f'{float(wave[freq_ind].to("nm").v):.3f} nm')
            ax_arr[2].plot(tau_midpoint[freq_ind, :], source_func[freq_ind,:],
                           label = label)

        ax_arr[0].set_ylabel(r'$S_\nu(z)$ ' + I_unit)
        ax_arr[1].set_ylabel(r'$\tau_\nu(z)$')
        for i in range(2):
            ax_arr[i].set_xlabel('depth')
        ax_arr[2].set_ylabel(r'$S_\nu(\tau_\nu)$ ' + I_unit)
        ax_arr[2].set_xlabel(r'$\tau_\nu$')

        ax_arr[2].set_xlim(0,2)
        ax_arr[2].legend()
    fig.tight_layout()
    plt.show()


def plot_rel_err(obs_freq, actual, other):
    import matplotlib.pyplot as plt

    
    wave = (unyt.c_cgs/obs_freq).to('nm')

    fig,ax_arr = plt.subplots(2,1, figsize = (4,8), sharex = True)

    def rel_err(key, idx = slice(None)):
        return (actual[key] - other[key]) / other[key]

    
    if actual['integrated_source'].ndim > 1:
        for i in range(actual['integrated_source'].shape[1]):
            ax_arr[0].plot(wave,
                           rel_err('integrated_source', idx = (slice(None),i))
            )
            ax_arr[1].plot(wave,
                           rel_err('total_tau', idx = (slice(None),i))
            )
    else:
        ax_arr[0].plot(wave, rel_err('integrated_source'))
        ax_arr[1].plot(wave, rel_err('total_tau'))
    ax_arr[1].set_xlabel('wavelength (nm)')
    ax_arr[0].set_ylabel(r'$I_\nu(\lambda)$')
    ax_arr[1].set_ylabel(r'$\tau_\nu(\lambda)$')
    fig.tight_layout()
    plt.show()
