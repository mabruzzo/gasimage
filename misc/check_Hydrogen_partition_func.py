import numpy as np
import unyt
import matplotlib.pyplot as plt

# taken from Korg (they give higher precision, this is fine for our purposes)
_lnT_table = np.linspace(0.0, 5.0, num = 201)[152:] * np.log(10)
# partition function is 2 at lower values
_HI_partition_table = np.array([
    2.       , 2.0000002, 2.0000005, 2.0000017, 2.0000043, 2.0000105, 2.0000255,
    2.0000591, 2.0001318, 2.0002835, 2.0005887, 2.0011811, 2.0022924, 2.0043073,
    2.0078447, 2.013863 , 2.0237992, 2.0397365, 2.0646014, 2.1023827, 2.1583595,
    2.2393253, 2.353785 , 2.5121121, 2.7266388, 3.0116663, 3.3833802, 3.8596663,
    4.4598236, 5.204184 , 6.113654 , 7.209194 , 8.511263 ,10.039249 ,11.810925 ,
   13.84193  ,16.145329 ,18.73124  ,21.606548 ,24.774748 ,28.235838 ,31.986353 ,
   36.01947  ,40.32519  ,44.8906   ,49.700172 ,54.73613  ,59.9788   ,65.40701
], dtype=np.float32)
def frac_ground_state_HI(T):
    ln_T = np.log(T.to('K'))
    partition_func = np.interp(ln_T, xp = _lnT_table, fp = _HI_partition_table)
    if (ln_T.max() > _lnT_table.max()):
        print("warning: underestimating partition function at above 1e5 K")

    def g(n): return 2*n*n # statistical weight
    def E(n): return 13.6 * (1-1/(n*n)) * unyt.eV
    frac = g(n=1) * np.exp(-E(n=1) / (unyt.kboltz * T)) / partition_func

    return frac

if __name__ == '__main__':
    T_vals = np.geomspace(1e3, 1e7) * unyt.K

    fig,ax = plt.subplots(1,1)
    ax.plot(T_vals, frac_ground_state_HI(T_vals),'k-')
    ax.set_xlabel(r'$T$ [K]')
    ax.set_ylabel(r'$n_{\rm HI}(n=1)/n_{{\rm HI},tot}$')
    ax.set_xscale('log')
    plt.show()
