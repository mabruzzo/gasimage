import numpy as np

def _is_np_ndarray(obj): # confirm its an instance, but not a subclass
    return obj.__class__ is np.ndarray

def _convert_vec_l_to_uvec_l(vec_l):
    assert (vec_l.shape[0] >= 1) and  (vec_l.shape[1:] == (3,))
    mag_square = (vec_l*vec_l).sum(axis=1)
    if (mag_square == 0.0).any():
        raise RuntimeError("there's a case where the vector has 0 magnitude")
    return vec_l/np.sqrt(mag_square[np.newaxis].T)

def _check_ray_args(arg0, arg1, name_pair, expect_1D_arg1 = False):
    arg0_name, arg1_name = name_pair
    if not _is_np_ndarray(arg0):
        raise TypeError(
            f"{arg0_name} must be a np.ndarray, not '{type(arg0).__name__}'")
    elif not _is_np_ndarray(arg1):
        raise TypeError(
            f"{arg1_name} must be a np.ndarray, not '{type(arg1).__name__}'")
    elif arg0.ndim != 2:
        raise ValueError(f"{arg0_name} must have 2 dims, not {arg0.ndim}")
    elif arg0.shape[0] < 1:
        raise RuntimeError("not clear how this situation could occur")
    elif arg0.shape[1] != 3:
        raise ValueError(f"{arg0_name}'s shape, {arg0.shape}, is invalid. "
                         "The second element must be 3")
    elif expect_1D_arg1 and (arg1.shape != (3,)):
        raise ValueError(f"{arg1_name}'s shape must be (3,), not {arg1.shape}")
    elif (not expect_1D_arg1) and (arg1.shape != arg0.shape):
        raise ValueError(f"{arg0_name} and {arg1_name} have a shape mismatch, "
                         f"{arg0.shape} and {arg1.shape}")

    # might want to enforce the following in the future
    #if arg0.strides[-1] != arg0.dtype.itemsize:
    #    raise ValueError(f"{arg0_name} must be contiguous along axis -1")
    #elif arg1.strides[-1] != arg1.dtype.itemsize:
    #    raise ValueError(f"{arg1_name} must be contiguous along axis -1")


class ConcreteRayCollection:
    """
    Collection of rays

    Note
    ----
    We choose to store ray_vec instead of ray_uvec for the following reasons:

    - It's difficult to check that a floating-point unit-vector is actually a
      unit vector.

    - usually, you're better off just doing the normalization when you use the
      unit vector. But if you do the normalization on the unit vector, then you
      can wind up with slightly different values. Functionally, this shouldn't
      change things, but it could complicate testing...
    """

    def __init__(self, ray_start_codeLen, ray_vec):

        _check_ray_args(ray_start_codeLen, ray_vec,
                        ("ray_start_codeLen", "ray_vec"),
                        expect_1D_arg1 = False)

        self.ray_start_codeLen = ray_start_codeLen
        self._ray_vec = ray_vec

        problems = (np.square(ray_vec).sum(axis = 1) == 0)
        if problems.any():
            raise RuntimeError(f"ray_vec at indices {np.where(tmp)[0]} "
                               "specify 0-vectors")

    @classmethod
    def from_start_stop(cls, ray_start_codeLen, ray_stop_codeLen):
        _check_ray_args(ray_start_codeLen, ray_stop_codeLen,
                        ("ray_start_codeLen", "ray_stop_codeLen"),
                        expect_1D_arg1 = False)
        return cls(ray_start_codeLen = ray_start_codeLen,
                   ray_vec = ray_stop_codeLen - ray_start_codeLen)

    def __repr__(self):
        return (f'{type(self).__name__}({self.ray_start_codeLen!r}, '
                f'{self._ray_vec!r})')

    def __len__(self):
        return self.ray_start_codeLen.shape[0]

    def as_concrete_ray_collection(self):
        return self

    def get_ray_uvec(self):
        return _convert_vec_l_to_uvec_l(self._ray_vec)


class PerspectiveRayCollection:
    def __init__(self, ray_start_codeLen, ray_stop_codeLen):
        _check_ray_args(ray_stop_codeLen, ray_start_codeLen,
                        ("ray_stop_codeLen", "ray_start_codeLen"),
                        expect_1D_arg1 = True)
        self.ray_start_codeLen = ray_start_codeLen
        self.ray_stop_codeLen = ray_stop_codeLen

    def __len__(self):
        return self.ray_stop_codeLen.shape[0]

    def as_concrete_ray_collection(self):
        # TODO: consider trying out np.broadcast_to to attempt to reduce space
        #       (while the result won't be contiguous along axis 0, it should
        #       still be contiguous along axis 1)
        return ConcreteRayCollection.from_start_stop(
            np.tile(self.ray_start_codeLen, (len(self), 1)),
            self.ray_stop_codeLen
        )

"""
class ParallelRayCollection:
    def __init__(self, ray_start_codeLen, ray_vec):
        _check_ray_args(ray_start_codeLen, ray_vec
                        ("ray_start_codeLen", "ray_vec"),
                        expect_1D_arg1 = True)
        self.ray_start_codeLen = ray_start_codeLen
        self._ray_vec = ray_vec

    def __len__(self):
        return self.ray_stop_codeLen.shape[0]

    def get_ray_uvec(self):
        # use np.tile over np.broadcast to make a copy (and ensure contiguous)
        return np.tile(self.ray_uvec, (self.ray_start_codeLen.shape[0], 3))

    def as_concrete_ray_collection(self):
        # TODO: consider trying out np.broadcast to attempt to reduce space
        #       (while the result won't be contiguous along axis 0, it should
        #       still be contiguous along axis 1)
        return ConcreteRayCollection(self.ray_start_codeLen,
                                     np.tile(self._ray_vec, (len(self), 1)))
"""
