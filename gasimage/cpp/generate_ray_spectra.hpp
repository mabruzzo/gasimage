#pragma once

#include "err.hpp"

#define DEBUG_INDICES 1

/// Represents a Collection of 1 or more mathematical vectors. In this case,
/// a mathematical vector holds 3 elements
///
/// @note
/// We currently assume that the wrapped data is contiguous and that iterating
/// over the elements of a single vector is equivalent to iterating along the
/// fast axis,
class MathVecCollecView{

public: // interface

  MathVecCollecView()
    : ptr_(nullptr), length_(0)
  {}

  MathVecCollecView(const double* ptr, long length)
    : ptr_(ptr), length_(length)
  {}

  const double* operator[](long i) const noexcept {

#if DEBUG_INDICES
    if (i < 0) {
      ERROR("Can't access a negative index");
    } else if (i >= length_) {
      ERROR("index must be smaller than the length, %ld", length_);
    }
#endif
    return ptr_ + (3 * i);

  }

  long length() const noexcept { return this->length_; }

private: // attributes

  const double* ptr_;
  long length_;

};

/// Compute the spectrum along each specified ray!
///
/// @param ray_aligned_prop_buffer struct of preallocated buffers that are used
///     internally to temporarily store data along each ray
/// @param ray_data_extractor This should be a RayDataExtractor python
///     extension type instance that was casted to a void pointer
///
/// @returns 0 on success
///
/// @note
/// make partition_fn_pack into a std::optional
int generate_ray_spectra(int legacy_optically_thin_spin_flip,
                         const MathVecCollecView ray_start_list,
                         const MathVecCollecView ray_uvec_list,
                         double particle_mass_in_grams,
                         LinInterpPartitionFn partition_fn_pack,
                         long nfreq, const double* obs_freq_Hz,
                         int using_precalculated_doppler,
                         RayAlignedProps& ray_aligned_prop_buffer,
                         void* ray_data_extractor,
                         double* out_integrated_source,
                         double* out_tau) noexcept;
