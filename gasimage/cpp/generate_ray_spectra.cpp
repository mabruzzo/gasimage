#include "stuff.hpp"
#include "generate_ray_spectra.hpp"

// the following header is automatically generated from _generate_spec_cy.pyx
// -> we need to include it so that we have access to the get_ray_data function
#include "../_generate_spec_cy.h"

struct ArgPack {
  MathVecCollecView ray_start_list;
  MathVecCollecView ray_uvec_list;
  C_LineProps line_props;
  double inv_particle_mass_cgs;
  LinInterpPartitionFn partition_fn_pack;
  long nfreq;
  const double* obs_freq_Hz;
  void* ray_data_extractor;
  double* out_integrated_source;
  double* out_tau;
};

template<bool legacy_optically_thin_spin_flip,
         bool using_precalculated_doppler>
int generate_ray_spectra_(const ArgPack pack,
                          RayAlignedProps ray_data_buffer) noexcept
{
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

    if (!using_precalculated_doppler) {
      // todo: move the following functionality into the actual rt functions
      for(int j = 0; j < ray_data_buffer.num_segments; j++) {
        ray_data_buffer.precomputed_doppler_parameter_b[j] =
          doppler_parameter_b_from_temperature(ray_data_buffer.kinetic_T[j],
                                               pack.inv_particle_mass_cgs);
      }
    }

    if (legacy_optically_thin_spin_flip) {
      optically_thin_21cm_ray_spectrum_impl
        (pack.line_props, pack.nfreq, pack.obs_freq_Hz,
         ray_data_buffer,
         pack.out_integrated_source + i * pack.nfreq);
    } else {
      generate_noscatter_spectrum_impl
        (pack.line_props, pack.nfreq, pack.obs_freq_Hz, ray_data_buffer,
         pack.partition_fn_pack,
         pack.out_integrated_source + i * pack.nfreq,
         pack.out_tau + i * pack.nfreq);
    }

  }
  return 0; // no issues
}

int generate_ray_spectra(int legacy_optically_thin_spin_flip,
                         const MathVecCollecView ray_start_list,
                         const MathVecCollecView ray_uvec_list,
                         C_LineProps line_props,
                         double particle_mass_in_grams,
                         LinInterpPartitionFn partition_fn_pack,
                         long nfreq, const double* obs_freq_Hz,
                         int using_precalculated_doppler,
                         RayAlignedProps& ray_aligned_prop_buffer,
                         void* ray_data_extractor,
                         double* out_integrated_source,
                         double* out_tau) noexcept
{
  if ((!legacy_optically_thin_spin_flip) && (out_tau == nullptr)) {
    ERROR("out_tau can only be null when using legacy optically_thin strat");
  } else if (out_integrated_source == nullptr) {
    ERROR("out_integrated_source can't be NULL");
  } else if (particle_mass_in_grams <= 0){
    ERROR("particle_mass_in_grams must be positive");
  }

  ArgPack pack =
    {ray_start_list, ray_uvec_list,
     line_props, 1.0 / particle_mass_in_grams, partition_fn_pack,
     nfreq, obs_freq_Hz, ray_data_extractor, out_integrated_source, out_tau};

  if (legacy_optically_thin_spin_flip) {
    if (using_precalculated_doppler) {
      return generate_ray_spectra_<true, true>(pack,
                                               ray_aligned_prop_buffer);
    } else {
      return generate_ray_spectra_<true, false>(pack,
                                                ray_aligned_prop_buffer);
    }
  } else {
    if (using_precalculated_doppler) {
      return generate_ray_spectra_<false, true>(pack,
                                                ray_aligned_prop_buffer);
    } else {
      return generate_ray_spectra_<false, false>(pack,
                                                 ray_aligned_prop_buffer);
    }
  }

}
