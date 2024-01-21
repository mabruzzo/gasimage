import argparse
import os.path

import h5py
import numpy as np
import unyt

import matplotlib
matplotlib.use('Agg') # this is necessary before loading yt (on clusters)
import yt

from gasimage import SnapDatasetInitializer
from gasimage.accumulators import configure_single_line_rt
from gasimage.estimate_h_ionization import EnzoEUsefulFieldAdder
from gasimage.generate_image import generate_image
from gasimage.ray_collection import aligned_parallel_ray_grid
from gasimage.rt_config import builtin_halpha_props, crude_H_partition_func

# there are number of configurable options here...
# For now, I am going to hardcode these choices, but if it becomes desirable,
# we will make it selectable at runtime

def prepare_accum_strat():

    return configure_single_line_rt(
        line_props = builtin_halpha_props(),
        species_mass_g = float(unyt.mh_cgs.v),
        ndens_field = ('gas', 'H_p0_number_density'),
        kind = 'noscatter',
        observed_freq = None,
        #v_channels = np.arange(-170,180,0.736125)* unyt.km/unyt.s,
        v_channels = np.arange(-75,75,0.75)* unyt.km/unyt.s,
        partition_func = crude_H_partition_func(electronic_state_max = 20)
    )

class TridentFieldAdder:
    def __init__(self, species_name, trident_fname):
        self.species_name = species_name
        self.trident_fname = os.path.expanduser(trident_fname)
    def __call__(self, ds):
        # create ('gas','temperature') field (with correct units!)
        def _temperature(field, data):
            field = ('enzoe', 'temperature')
            # ('enzoe', 'temperature') may be labelled as dimensionless
            if data[field].units.is_dimensionless:
                return data[field] * unyt.K
            return data[field]

        ds.add_field(('gas', 'temperature'), function = _temperature,
                     sampling_type = 'local', take_log = True,
                     units = 'K')

        import trident
        trident.add_ion_fields(
            ds, ions=[self.species_name],ionization_table = self.trident_fname
        )

def get_derived_field_adder(builtin_ionization_case = None,
                            trident_fname = None):
    if (builtin_ionization_case is None) == (trident_fname is None):
        raise ValueError("either builtin_ionization_case or trident_fname "
                         "must be specified. But not both")

    if builtin_ionization_case is not None:
        define_derived_fields = EnzoEUsefulFieldAdder(
            ndens_HI_field_name = ('gas', 'H_p0_number_density'),
            globally_fixed_metallicity = None, 
            metal_density_field = ('enzoe', 'metal_density'),
            ionization_case = builtin_ionization_case,
            ndens_HI_add_field_kwargs = {}
        )
        return define_derived_fields
    else:
        return TridentFieldAdder('H I', trident_fname = trident_fname)

def get_ray_collection(ds):
    # there's a lot we can do here in order to improve things!
    ray_collection = aligned_parallel_ray_grid(axis = 'z', ds = ds)
    return ray_collection

def save_h5_data(out_fname, rslt, obs_freq):

    def _create_dset(group, name, data, **kwargs):
        if isinstance(data, unyt.unyt_array):
            dset = group.create_dataset(name, data = data.ndview, **kwargs)
            dset.attrs['units'] = str(data.units)
        else:
            dset = group.create_dataset(name, data = data, **kwargs)
        return dset

    with h5py.File(out_fname, 'w') as f:
        # in the future, we may want to group multiple images, so lets save the
        # first one as im0
        cur_group = f.create_group("0")
        assert 'observed_frequency' not in rslt
        _create_dset(cur_group, 'observed_frequency', obs_freq)
        for k, v in rslt.items():
            _create_dset(cur_group, name = k, data = v)
            

def main_helper(sim_fname, out_dir, pool = None):

    accum_strat = prepare_accum_strat()
    define_derived_fields = get_derived_field_adder(
        builtin_ionization_case = 'max_H_ionization',
        trident_fname = None
    )

    # this kinda sucks, but we need to load ds right now in order to construct
    # the ray-collection (we should probably fix this...)
    ds = yt.load(sim_fname)
    ray_collection = get_ray_collection(ds)

    sim_fname_l = [sim_fname]
    for sim_path in sim_fname_l:
        rslt = generate_image(accum_strat, ray_collection,
                              ds = SnapDatasetInitializer(
                                  fname = sim_path,
                                  setup_func = define_derived_fields),
                              pool = pool)

        obs_freq = unyt.unyt_array(accum_strat.obs_freq_Hz, 'Hz')
        out_fname = os.path.join(out_dir, 'test-result.h5')

        save_h5_data(out_fname, rslt, obs_freq)

def main():
    sim_fname = ("../test_data/X100_M1.5_HD_CDdftCstr_R56.38_logP3_Res16/"
                 "cloud_07.5000/cloud_07.5000.block_list")
    from schwimmbad import MultiPool
    with MultiPool(processes = 4) as pool:
        main_helper(sim_fname, out_dir = './', pool = pool)

if __name__ == '__main__':
    main()
