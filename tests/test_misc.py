import numpy as np
from gasimage.rt_config import crude_H_partition_func
        


def test_partition_func():
    partition_fn = crude_H_partition_func()

    log10_T_points = partition_fn.log10_T_vals
    log10_partition_points = partition_fn.log10_partition_arr

    def ref_implementation(T_vals):
        xp, fp = log10_T_points, log10_partition_points
        return 10.0**np.interp(x = np.log10(T_vals),
                               xp = xp, fp = fp, left = fp[0], right = fp[-1])


    T_vals = np.geomspace(1, 1e10, num = 5001)

    np.testing.assert_allclose(
        actual = partition_fn(T_vals),
        desired = ref_implementation(T_vals),
        rtol = 2.e-15, atol = 0.0
    )
