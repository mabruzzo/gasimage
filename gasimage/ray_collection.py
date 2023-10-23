import numpy as np

def _is_np_ndarray(obj): return isinstance(obj, np.ndarray)

def _calc_uvec(ray_start, ray_stop):
    # note ray_start & ray_stop are not guaranteed to be contiguous
    assert (ray_start.shape == ray_stop.shape) and (ray_start.ndim == 2)
    assert (ray_start.shape[0] >= 1) and  (ray_start.shape[1] == 3)

    ray_vec = ray_stop - ray_start
    mag_square = (ray_vec*ray_vec).sum(axis=1)
    if (mag_square == 0.0).any():
        raise RuntimeError("there's a case where ray_start is equal to ray_end")
    return ray_vec/np.sqrt(mag_square[np.newaxis].T)

class ConcreteRayCollection:
    def __init__(self, ray_start_codeLen, ray_stop_codeLen):
        if not (_is_np_ndarray(ray_start_codeLen) and
                _is_np_ndarray(ray_stop_codeLen)):
            raise TypeError(
                "ray_start_codeLen and ray_stop_codeLen must be np.ndarrays: "
                f"{type(ray_start_codeLen)}, {type(ray_stop_codeLen)}")
        elif ray_start_codeLen.ndim != 2:
            raise ValueError("ray_start_codeLen must be 2D")
        elif ray_start_codeLen.shape != ray_stop_codeLen.shape:
            raise ValueError(
                "ray_start_codeLen and ray_stop_codeLen have shape mismatch:"
                f"{ray_start_codeLen.shape}, {ray_stop_codeLen.shape}")

        assert ray_start_codeLen.shape[0] >= 1
        assert ray_start_codeLen.shape[1] == 3
        self.ray_start_codeLen = ray_start_codeLen
        self.ray_stop_codeLen = ray_stop_codeLen

        ray_uvec = np.empty_like(self.ray_start_codeLen)
        ray_vec = self.ray_stop_codeLen - self.ray_start_codeLen
        if (ray_vec == 0.0).any():
            raise RuntimeError("there's a case where ray_start_codeLen is "
                               "equal to ray_stop_codeLen")

    def __len__(self):
        return self.ray_stop_codeLen.shape[0]

    def get_ray_uvec(self):
        return _calc_uvec(self.ray_start_codeLen, self.ray_stop_codeLen)


class PerspectiveRayCollection:
    def __init__(self, ray_start_codeLen, ray_stop_codeLen):
        if not (_is_np_ndarray(ray_start_codeLen) and
                _is_np_ndarray(ray_stop_codeLen)):
            raise TypeError(
                "ray_start_codeLen and ray_stop_codeLen must be np.ndarrays: "
                f"{type(ray_start_codeLen)}, {type(ray_stop_codeLen)}"
            )

        assert ray_start_codeLen.shape == (3,)
        assert ray_stop_codeLen.shape[0] > 1
        assert ray_stop_codeLen.shape[1:] == (3,)

        self.ray_start_codeLen = ray_start_codeLen
        self.ray_stop_codeLen = ray_stop_codeLen

    def __len__(self):
        return self.ray_stop_codeLen.shape[0]

    def to_concrete_ray_collection(self):
        # use np.tile to make sure that ray_start_codeLen is contiguous
        return ConcreteRayCollection(
            np.tile(self.ray_start_codeLen, (len(self), 1)),
            self.ray_stop_codeLen
        )

    def get_ray_uvec(self):
        shape = (len(self), 3)
        return _calc_uvec(np.broadcast_to(self.ray_start_codeLen, shape),
                          self.ray_stop_codeLen)


"""
class ParallelRayCollection:
    def __init__(self, ray_start_codeLen, ray_uvec):
        assert (not _is_unyt_arr(ray_start_codeLen)) and (not _is_unyt_arr(ray_uvec))

        assert ray_uvec.shape == (3,)
        assert ray_start_codeLen.shape[0] > 1
        assert ray_start_codeLen.shape[1:] == (3,)
        self.ray_start_codeLen = ray_start_codeLen
        self.ray_uvec = ray_uvec

    def get_ray_uvec(self):
        # use np.tile over np.broadcast to make a copy (and ensure contiguous)
        return np.tile(self.ray_uvec, (self.ray_start_codeLen.shape[0], 3))
"""
