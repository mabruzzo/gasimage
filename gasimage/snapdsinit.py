import os

import libconf
import numpy as np
import unyt
import yt

def _load_units(p):
    units = {}

    units = {
        'length' : unyt.unyt_quantity(p["Units"]["length"], 'cm'),
        'time' : unyt.unyt_quantity(p["Units"]["time"], "s")
    }

    mass = p["Units"].get("mass")
    if mass is None:
        density = p["Units"].get("density")
        if density is not None:
            mass = density * units["length"]**3
        else:
            mass = 1
    units["mass"] = unyt.unyt_quantity(mass, "g")
    units["velocity_units"] = units["length"] / units["time"]
    return units

class AdiabaticCloudEnzoELoader:
    """
    Functor that is used to rescale the units of the adiabatic cloud
    """
    def __init__(self, target_rcl_cm, target_eint_cl_cm2_per_s2,
                 target_p_div_gm1_dyne_per_cm2):
        self.target_rcl_cm = target_rcl_cm
        self.target_eint_cl_cm2_per_s2 = target_eint_cl_cm2_per_s2
        self.target_p_div_gm1_cgs = target_p_div_gm1_dyne_per_cm2

    def __call__(self, fn, *args, **kwargs):
        """
        Load a Dataset

        Parameters
        ----------
        fn : str, os.Pathlike[str]
            A path to the data location. This can be a file name.
        """
        assert 'units_override' not in kwargs
        assert 'unit_system' not in kwargs

        _suffix = ".block_list"
        assert fn.endswith(_suffix)
        parameter_path = fn[: -len(_suffix)] + '.libconfig'
        assert os.path.exists(parameter_path)
        with open(parameter_path) as f:
            p = libconf.load(f)
        units = _load_units(p)


        initial_rcl_cgs = (p["Initial"]["cloud"]["cloud_radius"] *
                           units["length"].to("cm").v)
        chi = (p["Initial"]["cloud"]["cloud_density"] /
               p["Initial"]["cloud"]["wind_density"])
        initial_eint_w_cgs = (
            p["Initial"]["cloud"]["wind_internal_energy"] * 
            (units["length"].to("cm").v)**2 /
            (units["time"].to("s").v)**2
        )
        initial_eint_cl_cgs = (initial_eint_w_cgs / chi)
        initial_density_cl_cgs = (
            p["Initial"]["cloud"]["cloud_density"] * units["mass"].to("g").v
            / (units["length"].to("cm").v)**3
        )
        initial_density_w_cgs = (
            p["Initial"]["cloud"]["wind_density"] * units["mass"].to("g").v
            / (units["length"].to("cm").v)**3
        )
        initial_pdivgm1_cgs = initial_eint_w_cgs * initial_density_w_cgs

        new_length_u = ((self.target_rcl_cm / initial_rcl_cgs)
                        * units["length"].to("cm").v)
        new_time_u = (
            np.sqrt(initial_eint_cl_cgs / self.target_eint_cl_cm2_per_s2 ) *
            (new_length_u / units["length"].to("cm").v) *
            units["time"].to("s").v)
        if True:
            target_density_cl_cgs = (self.target_p_div_gm1_cgs /
                                     self.target_eint_cl_cm2_per_s2)
            new_mass_u = (
                (target_density_cl_cgs / initial_density_cl_cgs) *
                (new_length_u / units["length"].to("cm").v)**3 *
                units["mass"].to("g").v)

        units_override = {
            'length_unit' : (new_length_u, 'cm'),
            'time_unit' : (new_time_u, 's'),
            'mass_unit' : (new_mass_u, 'g'),
        }

        return yt.load(fn, *args, units_override = units_override,
                       unit_system = 'cgs', **kwargs)

# this is pretty hacky!
_CACHED_DATA = None

class SnapDatasetInitializer:
    """
    This is supposed to manage the creation of a yt-dataset (to help
    facillitate pickling).

    Parameters
    ----------
    fname: str subclass instance of 
        File name of a loadable simulation dataset.
    setup_func: callable, optional
        This is a callable that accepts a dataset object as an argument. This
        Should only be specified if the filename was specified as the first 
        argument. This should be picklable (e.g. it can't be a lambda function
        or a function defined in a function).
    loader: callable, optional
        This is a callable that can wrap yt.load
    """

    def __init__(self, fname, setup_func = None, loader = None):
        self._fname = fname
        assert (setup_func is None) or callable(setup_func)
        self._setup_func = setup_func
        assert (loader is None) or callable(loader)
        self._loader = loader

    def __call__(self):
        """
        Initializes a snapshot dataset object.

        Returns
        -------
        out: instance of a subclass of `yt.data_objects.static_output.Dataset`
        """
        if self._loader is None:
            load_fn = yt.load
        else:
            load_fn = self._loader

        # the way we cache data is super hacky! But it's the only way to do it
        # without pickling the cached data.
        global _CACHED_DATA
        cached_instance = (('_CACHED_DATA' in globals()) and
                           (_CACHED_DATA is not None) and
                           (_CACHED_DATA[0] == self._fname))
        if cached_instance:
            ds = _CACHED_DATA[1]
        else:
            if _CACHED_DATA is not None:
                _CACHED_DATA[1].index.clear_all_data() # keep memory usage down!
            ds = load_fn(self._fname)
            if self._setup_func is not None:
                func = self._setup_func
                func(ds)
            _CACHED_DATA = (self._fname,ds)
        return ds
