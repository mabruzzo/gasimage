#pragma once

#include "err.hpp"
#include <optional>

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

#include "stuff.hpp"

struct ArgPack {
  MathVecCollecView ray_start_list;
  MathVecCollecView ray_uvec_list;
  C_LineProps line_props;
  double inv_particle_mass_cgs;
  const std::optional<LinInterpPartitionFn>& partition_fn_pack;
  long nfreq;
  const double* obs_freq_Hz;
  void* ray_data_extractor;
  double* out_integrated_source;
  double* out_tau;
};


// We implement a hacky workaround here.
// -> we want to include "../_generate_spec_cy.h" here so that we can get the
//    declaration for get_ray_data
// -> this header is automatically generated from _generate_spec_cy.pyx
// -> in princple this should be done in a source file. Doing it in a header
//    file (as we're doing right now), leads to some issues with includsion
//    order.
// -> unfortunately, I ran into some linking issues when I compiled a source
//    file...
// -> thus for now, we manually forward declare the function interface (this is
//    very fragile!)
#include "../_generate_spec_cy.h"
extern "C" void get_ray_data(struct RayAlignedProps *, double const *,
                             double const *, void *);

template<bool legacy_optically_thin_spin_flip,
         bool using_precalculated_doppler,
         NdensStrat ndens_strat>
int generate_ray_spectra_(const ArgPack pack,
                          RayAlignedProps ray_data_buffer) noexcept
{
  if ((! legacy_optically_thin_spin_flip) && (! bool(pack.partition_fn_pack))) {
    ERROR("an empty partition_fn_pack was specified. A valid selection is "
          "currently required for non-optically thin rt");
  }

  NdensFetcher<ndens_strat> ndens_fetcher(pack.partition_fn_pack);

  const long nrays = pack.ray_start_list.length();
  if (nrays != pack.ray_uvec_list.length()) {
    ERROR("There is a mismatch in the lists of rays.");
  }

  for (long i = 0; i < nrays; i++) {
    // both of the following pointers are 3 element arrays
    const double* ray_start = pack.ray_start_list[i];
    const double* ray_uvec = pack.ray_uvec_list[i];

    // the following function stores grid-aligned values within the
    // ray_data_buffer object
    //
    // note: this function actually involves some python operations, so it's a
    //       little slow! In the future, we can definitely refactor!
    get_ray_data(&ray_data_buffer, ray_start, ray_uvec,
                 pack.ray_data_extractor);

    if constexpr (legacy_optically_thin_spin_flip) {
      optically_thin_21cm_ray_spectrum_impl<using_precalculated_doppler,
                                            ndens_strat>
        (pack.line_props, pack.inv_particle_mass_cgs,
         pack.nfreq, pack.obs_freq_Hz,
         ray_data_buffer, ndens_fetcher,
         pack.out_integrated_source + i * pack.nfreq);
    } else if constexpr (!legacy_optically_thin_spin_flip) {
      generate_noscatter_spectrum_impl<using_precalculated_doppler,
                                       ndens_strat>
        (pack.line_props, pack.inv_particle_mass_cgs,
         pack.nfreq, pack.obs_freq_Hz, ray_data_buffer,
         ndens_fetcher,
         pack.out_integrated_source + i * pack.nfreq,
         pack.out_tau + i * pack.nfreq);
    }

  }
  return 0; // no issues
}

namespace details{

inline void check_args_(int legacy_optically_thin_spin_flip,
                        int using_precalculated_doppler,
                        const std::optional<LinInterpPartitionFn>& partition_fn_pack,
                        double * out_integrated_source,
                        double * out_tau)
{
  if (legacy_optically_thin_spin_flip < 0) {
    ERROR("legacy_optically_thin_spin_flip must be non-negative");
  } else if (using_precalculated_doppler < 0) {
    ERROR("using_precalculated_doppler must be non-negative");
  } else if ((!legacy_optically_thin_spin_flip) && (out_tau == nullptr)) {
    ERROR("out_tau can only be null when using legacy optically_thin strat");
  } else if (out_integrated_source == nullptr) {
    ERROR("out_integrated_source can't be NULL");
  } else if ((! legacy_optically_thin_spin_flip) &&
             (! bool(partition_fn_pack))) {
    ERROR("an empty partition_fn_pack was specified. A valid selection is "
          "currently required for non-optically thin rt");
  }

}

} // namespace details

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
inline int generate_ray_spectra(
  int legacy_optically_thin_spin_flip,
  const MathVecCollecView ray_start_list,
  const MathVecCollecView ray_uvec_list,
  C_LineProps line_props,
  double particle_mass_in_grams,
  const std::optional<LinInterpPartitionFn>& partition_fn_pack,
  long nfreq, const double* obs_freq_Hz,
  int using_precalculated_doppler,
  RayAlignedProps& ray_aligned_prop_buffer,
  void* ray_data_extractor,
  double* out_integrated_source,
  double* out_tau)
{
  details::check_args_(legacy_optically_thin_spin_flip,
                       using_precalculated_doppler,
                       partition_fn_pack,
                       out_integrated_source, out_tau);

  if (particle_mass_in_grams <= 0){
    ERROR("particle_mass_in_grams must be positive");
  } 

  ArgPack pack =
    {ray_start_list, ray_uvec_list,
     line_props, 1.0 / particle_mass_in_grams, partition_fn_pack,
     nfreq, obs_freq_Hz, ray_data_extractor, out_integrated_source, out_tau};

  if (legacy_optically_thin_spin_flip) {
    if (using_precalculated_doppler) {
      return generate_ray_spectra_<true, true,
                                   NdensStrat::SpecialLegacySpinFlipCase>
        (pack, ray_aligned_prop_buffer);
    } else {
      return generate_ray_spectra_<true, false,
                                   NdensStrat::SpecialLegacySpinFlipCase>
        (pack, ray_aligned_prop_buffer);
    }
  } else {
    if (using_precalculated_doppler) {
      return generate_ray_spectra_<false, true,
                                   NdensStrat::IonNDensGrid_LTERatio>
        (pack, ray_aligned_prop_buffer);
    } else {
      return generate_ray_spectra_<false, false,
                                   NdensStrat::IonNDensGrid_LTERatio>
        (pack, ray_aligned_prop_buffer);
    }
  }


}




                       

// function that will get called externally to test rt functionality!
inline int external_single_ray_rt_wrapper
(int legacy_optically_thin_spin_flip, C_LineProps line_props,
 const long nfreq, const double* obs_freq,
 const RayAlignedProps ray_aligned_data,
 const std::optional<LinInterpPartitionFn>& partition_fn_pack,
 double* out_integrated_source,
 double* out_tau)
{
  const int using_precalculated_doppler = 1;

  details::check_args_(legacy_optically_thin_spin_flip,
                       using_precalculated_doppler,
                       partition_fn_pack,
                       out_integrated_source, out_tau);

  const double inv_particle_mass_cgs = 0.0; // pick a dummy default value!

  if (legacy_optically_thin_spin_flip) {
    NdensFetcher<NdensStrat::SpecialLegacySpinFlipCase> ndens_fetcher
      (partition_fn_pack);

    optically_thin_21cm_ray_spectrum_impl<true,
                                          NdensStrat::SpecialLegacySpinFlipCase>
      (line_props, inv_particle_mass_cgs, nfreq, obs_freq, ray_aligned_data,
       ndens_fetcher,
       out_integrated_source);
  } else {
    NdensFetcher<NdensStrat::IonNDensGrid_LTERatio> ndens_fetcher
      (partition_fn_pack);

    generate_noscatter_spectrum_impl<true,
                                     NdensStrat::IonNDensGrid_LTERatio>
      (line_props, inv_particle_mass_cgs, nfreq, obs_freq, ray_aligned_data,
       ndens_fetcher,
       out_integrated_source, out_tau);
  }

  return 0;
}
