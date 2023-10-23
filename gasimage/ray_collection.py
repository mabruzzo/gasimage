import numpy as np

def _is_np_ndarray(obj): # confirm its an instance, but not a subclass
    return obj.__class__ is np.ndarray

def _convert_vec_l_to_uvec_l(vec_l):
    assert (vec_l.shape[0] >= 1) and  (vec_l.shape[1:] == (3,))
    mag_square = (vec_l*vec_l).sum(axis=1)
    if (mag_square == 0.0).any():
        raise RuntimeError("there's a case where the vector has 0 magnitude")
    return vec_l/np.sqrt(mag_square[np.newaxis].T)

def _check_ray_args(arg0, arg1, name_pair,
                    expect_3D_arg0 = False,
                    expect_1D_arg1 = False):
    arg0_name, arg1_name = name_pair
    if not _is_np_ndarray(arg0):
        raise TypeError(
            f"{arg0_name} must be a np.ndarray, not '{type(arg0).__name__}'")
    elif not _is_np_ndarray(arg1):
        raise TypeError(
            f"{arg1_name} must be a np.ndarray, not '{type(arg1).__name__}'")
    elif arg0.shape[-1] != 3:
        raise ValueError(f"{arg0_name}'s shape, {arg0.shape}, is invalid. "
                         "The last element must be 3")
    elif arg1.shape[-1] != 3:
        raise ValueError(f"{arg1_name}'s shape, {arg1.shape}, is invalid. "
                         "The last element must be 3")
    elif (any(e < 1 for e in arg0.shape[:-1]) or
          any(e < 1 for e in arg1.shape[:-1])):
        raise RuntimeError("not clear how this situation could occur")

    if expect_3D_arg0 and arg0.ndim != 3:
        raise ValueError(f"{arg0_name} must have 3 dims, not {arg0.ndim}")
    elif (not expect_3D_arg0) and (arg0.ndim != 2):
        raise ValueError(f"{arg0_name} must have 2 dims, not {arg0.ndim}")

    if expect_1D_arg1 and (arg1.ndim != 1):
        raise ValueError(f"{arg1_name} must have 1 dim, not {arg1.ndim}")
    elif (not expect_1D_arg1) and (arg1.ndim !=2):
        raise ValueError(f"{arg1_name} must have 2 dims, not {arg1.ndim}")

    if ((not expect_3D_arg0) and (not expect_1D_arg1) and
        (arg1.shape != arg0.shape)):
        raise ValueError(f"{arg0_name} and {arg1_name} have a shape mismatch, "
                         f"{arg0.shape} and {arg1.shape}")

    # might want to enforce the following in the future
    #if arg0.strides[-1] != arg0.dtype.itemsize:
    #    raise ValueError(f"{arg0_name} must be contiguous along axis -1")
    #elif arg1.strides[-1] != arg1.dtype.itemsize:
    #    raise ValueError(f"{arg1_name} must be contiguous along axis -1")


class ConcreteRayList:
    """
    Immutable list of rays.

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
                        ("ray_start_codeLen", "ray_vec"))

        self.ray_start_codeLen = ray_start_codeLen
        self._ray_vec = ray_vec

        problems = (np.square(ray_vec).sum(axis = 1) == 0)
        if problems.any():
            raise RuntimeError(f"ray_vec at indices {np.where(tmp)[0]} "
                               "specify 0-vectors")

    @classmethod
    def from_start_stop(cls, ray_start_codeLen, ray_stop_codeLen):
        _check_ray_args(ray_start_codeLen, ray_stop_codeLen,
                        ("ray_start_codeLen", "ray_stop_codeLen"))
        return cls(ray_start_codeLen = ray_start_codeLen,
                   ray_vec = ray_stop_codeLen - ray_start_codeLen)

    def __repr__(self):
        return (f'{type(self).__name__}({self.ray_start_codeLen!r}, '
                f'{self._ray_vec!r})')

    def __len__(self):
        return self.ray_start_codeLen.shape[0]

    def shape(self):
        return (len(self.shape),)

    def get_ray_uvec(self):
        return _convert_vec_l_to_uvec_l(self._ray_vec)

    def get_selected_raystart_rayuvec(self, idx):
        ray_start = self.ray_start_codeLen[idx, :]
        ray_start.flags['WRITEABLE'] = False

        ray_uvec = self.get_ray_uvec()[idx, :]
        ray_uvec.flags['WRITEABLE'] = False

        # if self.ray_start_codeLen.strides[0] OR self._ray_vec.strides[0] is
        # equal to 0, we may be able to get clever (and reduce the size of the
        # returned arrays... - this could be useful when they are dispatched as
        # messages)
        return ray_start, ray_uvec


class PerspectiveRayGrid2D:
    def __init__(self, ray_start_codeLen, ray_stop_codeLen):
        _check_ray_args(ray_stop_codeLen, ray_start_codeLen,
                        ("ray_stop_codeLen", "ray_start_codeLen"),
                        expect_3D_arg0 = True,
                        expect_1D_arg1 = True)
        self.ray_start_codeLen = ray_start_codeLen
        self.ray_stop_codeLen = ray_stop_codeLen

        self._ray_stop_2D = self.ray_stop_codeLen.view()
        self._ray_stop_2D.shape = (-1,3)
        assert self._ray_stop_2D.flags['C_CONTIGUOUS']

    def shape(self):
        return self.ray_stop_codeLen.shape[:-1]

    def as_concrete_ray_list(self):
        # TODO: consider trying out np.broadcast_to to attempt to reduce space
        #       (while the result won't be contiguous along axis 0, it should
        #       still be contiguous along axis 1)

        num_list_entries = self._ray_stop_2D.shape[0]
        
        return ConcreteRayList.from_start_stop(
            np.tile(self.ray_start_codeLen, (num_list_entries, 1)),
            self._ray_stop_2D
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
        return ConcreteRayList(self.ray_start_codeLen,
                               np.tile(self._ray_vec, (len(self), 1)))
"""
