import yt

from gasimage.estimate_h_ionization import create_nHI_field_gracklemmw

def _add_fields(ds, prefix = '',
                globally_fixed_metallicity = None, 
                metal_density_field = None):

    field_names = [('gas', prefix + 'ndens_H_mark1'),
                   ('gas', prefix + 'ndens_H_mark2'),
                   ('gas', prefix + 'ndens_H_mark3')]

    create_nHI_field_gracklemmw(
        ds, field_name = field_names[0],
        globally_fixed_metallicity = globally_fixed_metallicity, 
        metal_density_field = metal_density_field,
        ionization_case = 'max_H_ionization',
        add_field_kwargs = {})

    create_nHI_field_gracklemmw(
        ds, field_name = field_names[1],
        globally_fixed_metallicity = globally_fixed_metallicity, 
        metal_density_field = metal_density_field,
        ionization_case = 'equal-single-primordial-ionization',
        add_field_kwargs = {})

    create_nHI_field_gracklemmw(
        ds, field_name = field_names[2],
        globally_fixed_metallicity = globally_fixed_metallicity, 
        metal_density_field = metal_density_field,
        ionization_case = 'min_H_ionization',
        add_field_kwargs = {})
    return field_names

def test_compare_HI_defs(indata_dir):
    ds = yt.load(f'{indata_dir}/X100_M1.5_HD_CDdftCstr_R56.38_logP3_Res8/'
                 'cloud_07.5000/cloud_07.5000.block_list')
    ad = ds.all_data()

    for i in [0,1]:
        globally_fixed_metallicity, metal_density_field = None, None
        if i == 0:
            field_prefix = 'i0_'
            globally_fixed_metallicity = 1.0
        else:
            field_prefix = 'i1_'
            metal_density_field = ('enzoe', 'metal_density')

        fields = _add_fields(
            ds, prefix = field_prefix,
            globally_fixed_metallicity = globally_fixed_metallicity,
            metal_density_field = metal_density_field)

        assert (ad[fields[0]] <= ad[fields[1]]).all()
        assert (ad[fields[1]] <= ad[fields[2]]).all()
