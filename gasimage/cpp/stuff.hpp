#pragma once

#include "err.hpp"
#include <algorithm>   // std::distance, std::lower_bound
#include <cmath>       // std::exp, std::isnan, std::log10, std::sqrt
#include <optional>    // std::is_same_v


// define some constants!
#define INV_SQRT_PI 0.5641895835477563
#define QUARTER_DIV_PI 0.07957747154594767
// define some Macros equal to some yt-constants
#define C_CGS 29979245800.0            /* yt.units.c_cgs */
#define HPLANCK_CGS 6.62606957e-27     /* yt.units.h_cgs */
#define MH_CGS 1.6737352238051868e-24  /* yt.units.mh_cgs */
#define KBOLTZ_CGS 1.3806488e-16       /* yt.units.kboltz_cgs */

struct C_LineProps{
  int g_lo;
  int g_hi;
  double freq_Hz; // rest frequency
  double energy_lo_erg; // energy of the lower state

  // 2 of the 3 Einstein coefficients. Technically, we just need to know one of
  // them, but for now, we store both for backwards compatability!
  // -> going forward, it's my intention to stop tracking B12_cgs
  double A21_Hz;  // The einstein coefficient for spontaneous emission (in Hz)
  double B12_cgs; // when multiplied by average intensity, gives the rate of
                  // absorption
};

typedef double flt_t;

// linearly interpolated partition function
//
// This is a short-lived object that doesn't own the wrapped data...
// -> this isn't optimal, but it will make pickling the cython wrapper
//    extension class far easier
struct LinInterpPartitionFn {
  const flt_t* log10_T;
  const flt_t* log10_partition;
  long len;
};


/// an aggregate-ish struct used to group together arguments for radiative
/// transfer functions
///
/// At present, the idea is to aggregate the values extracted from the grid
/// into this function.
///
/// @note
/// In the future, it would be more flexible to instead store the 3D grid
/// values inside a struct and access them as they are needed. This would:
/// 1. facillitate higher order integration (e.g. trilinear of the LOS velocity
///    interpolation).
/// 2. allow us to move all of the ray-tracing code entirely to C++ (this
///    should provide some performance improvements, and will clean up the
///    usage of templates...)
/// This transition will require us to track the strides of each array
struct RayAlignedProps{
  /// the number of segments that the ray has been divided into (i.e. the number
  /// of cell intersections). It's also the length of the arrays directly
  /// stored as members of this struct
  long num_segments;

  /// @defgroup RayAlignedValues
  ///  Each of these entries has a length given by `this->num_segments`
  ///@{

  /// The distance travelled by the ray through each cell (in cm)
  double* dz;
  /// bulk velocity of the gas in cells along the ray (in cm/s).
  double* vLOS;
  /// number density of the given species in cells along the ray (in cm**-3).
  /// The exact meaning of this argument is context dependent
  double* ndens;
  /// The kinetic temperature of the gas in cells along the ray (in K)
  double* kinetic_T;
  /// The precomputed doppler parameter (aka Doppler broadening parameter) of
  /// the gas in cells along the ray (in cm/s)
  ///
  /// @note
  /// The doppler broadening parameter is often abbreviated with the variable
  /// ``b``. ``b/sqrt(2)`` specifies the standard deviation of the line-of-sight
  /// velocity component (due to thermal motions). When this quantity is
  /// multiplied by ``rest_freq/c_cgs`` (i.e. the inverse of a transition's
  /// rest-frame wavelength), you get what Rybicki and Lightman call the
  /// "Doppler width". This alternative quantity is a factor of ``sqrt(2)``
  /// larger than the standard deviation of the frequency profile.
  double* precomputed_doppler_parameter_b;
  ///@}

};


/// evaluate the partition function at the specified value
///
/// we assume units of Kelvin! If the user specifies a value outside of the
/// table then we clamp the value
inline flt_t eval_partition_fn(const LinInterpPartitionFn& pack,
                               flt_t T_val) noexcept
{
  const long last_i = pack.len-1;

  const flt_t x = (std::isnan(T_val) | (T_val <= 0))
    ? pack.log10_T[0]
    : std::log10(T_val);

  if (x <= pack.log10_T[0]) {
    return std::pow(10.0, pack.log10_partition[0]);
  } else if (x >= pack.log10_T[last_i]) {
    return std::pow(10.0, pack.log10_partition[last_i]);
  } else {

    // find left_i & right_i, where log10_T[left_i] < x <= log10_T[right_i]
    // -> invariant from branches:   1 <= right_i <= (log10_T.size() - 1)
    // -> std::lower_bound finds first element, e, in log10_T where e >= x
    const flt_t* rslt = std::lower_bound(pack.log10_T,
                                         pack.log10_T + pack.len,
                                         x);
    const std::size_t right_i = std::distance(pack.log10_T, rslt);
    const std::size_t left_i = right_i - 1;

    // calculate slope:
    const flt_t m =
      ((pack.log10_partition[right_i] - pack.log10_partition[left_i]) /
       (pack.log10_T[right_i] - pack.log10_T[left_i]));
    // formula of the line: y = m * (x - x0) + y0
    // -> we adopt (x0,y0) = (log10__[right_i], log10_partition[right_i]),
    //    since log10_T_[right_i] may equal x
    flt_t y = m * (x - pack.log10_T[right_i]) + pack.log10_partition[right_i];

    return std::pow(10.0, y);
  }
}

/// Compute the doppler parameter (in cm/s)
///
/// @param kinetic_T kinetic temperature in Kelvin
/// @param inv_particle_mass_cgs is 1.0 divided by the particle mass
///
/// @notes
/// The Doppler parameter had units consistent with velocity and is ofte
/// represented by the variable ``b``. ``b/sqrt(2)`` specifies the standard
/// deviation of the line-of-sight velocity component. For a given line
/// transition with a rest-frame frequency ``rest_freq``,
/// ``b * rest_freq/unyt.c_cgs`` specifies what Rybicki and Lightman call the
/// "Doppler width". The "Doppler width" divided by ``sqrt(2)`` is the standard
/// deviation of the line-profile for the given transition.
inline double doppler_parameter_b_from_temperature(double kinetic_T,
                                                   double inv_particle_mass_cgs)
  noexcept
{
  // this assumes that the temperature is already in units of kelvin
  return std::sqrt(2.0 * KBOLTZ_CGS * kinetic_T * inv_particle_mass_cgs);
}

namespace { //anonymous namespace

// When considering a transition we construct a new LineProfileStruct for each
// gas element.
// -> An instance is configured based on the doppler_parameter_b value and los
//    component of that element. It also depends on the transition's rest-frame
//    frequency
// -> An instance holds some precalcuated quantities for evaluating the
//    normalized gaussian:
//        norm * exp( neg_half_div_sigma2 *
//                    (obs_freq*emit_freq_factor - rest_freq)**2 )
struct LineProfileStruct {
  // norm is the normalization factor of the line profile (in frequency
  // space). Multiplying this quantity by the exponential term will normalize
  // the gaussian. (This has units of Hz**-1)
  double norm;

  // half_div_sigma2 is used in the exponential. (This has units of Hz**-1)
  double neg_half_div_sigma2;

  // emit_freq_factor is ``1.0 / (1 + bulk_vlos / c)``. Multiplying the
  // observed frequency by this factor gives the emission-frequency (i.e. the
  // frequency in the reference frame where the los-component of the bulk
  // velocity is zero)
  double emit_freq_factor;
};

inline LineProfileStruct prep_LineProfHelper(double rest_freq,
                                             double doppler_parameter_b,
                                             double velocity_offset) noexcept
{
  // compute the INVERSE of what Rybicki and Lightman call the "Doppler width"
  // -> NOTE: the input doppler_parameter_b argument specifies a value that is
  //    the standard deviation of the los VELOCITY profile times sqrt(2)
  // -> the quantity Rybicki and Lightman call the "Doppler width" is the
  //    standard-deviation of the FREQUENCY profile times sqrt(2)
  double temp = C_CGS / (rest_freq * doppler_parameter_b);

  LineProfileStruct out;
  out.norm = INV_SQRT_PI * temp;
  out.neg_half_div_sigma2 = -1.0 * temp * temp;
  out.emit_freq_factor = 1.0 / (1.0 + velocity_offset/C_CGS);
  return out;
}

inline double eval_line_profile(double obs_freq, double rest_freq,
                                LineProfileStruct prof) noexcept
{
  // convert from the observed frequency, obs_freq, to emit_freq, the frequency
  // emitted/absorbed (in the frame where the bulk gas velocity is zero)
  //
  // obs_freq = emit_freq * (1 + bulk_vlos / c)
  // emit_freq = obs_freq / (1 + bulk_vlos / c)
  double emit_freq = obs_freq * prof.emit_freq_factor;

  // compute the exponential term:
  //     exp(-(emit_freq - rest_freq)**2 / (2 * sigma**2))
  double delta_freq = (emit_freq - rest_freq);
  double exp_term = std::exp(delta_freq * delta_freq *
                             prof.neg_half_div_sigma2);

  // finally, multiply the exponential term by the normalization
  return prof.norm * exp_term;
}

struct Ndens_And_Ratio{
  double ndens_1;
  double n2g1_div_n1g2;
};

} // anonymous namespace

/// specifies the "strategy" when it comes to number density.
///
/// To be a little more clear:
///   - this specifies the interpretation of the ndens field
///   - it also specifies how we compute the ratio of the number densities
///     of the absorbers and emitters
enum class NdensStrat{
  // the ndens field gives the total number density of the ion species.
  // -> the user must provide a partition function; it is used to compute the
  //    number density of the absorber (assuming LTE)
  // -> the number-density ratio is assumed to match LTE
  IonNDensGrid_LTERatio,

  AbsorberNDensGrid_LTERatio,

  // It might be nice to support the following case in the future
  //AbsorberNDensGrid_ArbitraryRatio

  // this is what we have traditionally used in the optically-thin case
  // -> we effectively assume that the number density field gives the
  //    number-density of Hydrogen in the electronic ground state (n = 1).
  //    This is the sum of the number densities in the spin 0 and spin 1 states
  // -> we assume that the ratio of number densities is directly given by the
  //    statistical weights...
  SpecialLegacySpinFlipCase
};

template<NdensStrat Strat>
struct NdensFetcher{
  const std::optional<LinInterpPartitionFn> partition_func;

  NdensFetcher(const std::optional<LinInterpPartitionFn>& partition_func)
    : partition_func(partition_func)
  {
    if ((Strat == NdensStrat::IonNDensGrid_LTERatio) &&
        (! bool(this->partition_func))) {
      ERROR("partition_func must not be empty for "
            "NdensStrat::IonNDensGrid_LTERatio");
    }
  }

  [[gnu::always_inline]] inline Ndens_And_Ratio
  get_pair(long pos_i, const RayAlignedProps ray_data,
           const C_LineProps line_props) const noexcept
  {
    if constexpr (Strat == NdensStrat::IonNDensGrid_LTERatio) {

      // in this case:
      // -> treat level_pops_arg as partition function
      // -> treat ndens as number density of particles described by
      //    partition function
      // -> assume LTE

      double restframe_energy_photon_erg = line_props.freq_Hz * HPLANCK_CGS;
      double beta_cgs = 1.0 / (ray_data.kinetic_T[pos_i] * KBOLTZ_CGS);

      double ndens_1 =
        ( ray_data.ndens[pos_i] * line_props.g_lo *
          std::exp(-line_props.energy_lo_erg * beta_cgs) /
          eval_partition_fn(*this->partition_func,
                            ray_data.kinetic_T[pos_i]) );

      // n1/n2 = (g1/g2) * exp(restframe_energy_photon_cgs * beta_cgs)
      // n2/n1 = (g2/g1) * exp(-restframe_energy_photon_cgs * beta_cgs)
      // (n2*g1)/(n1*g2) = exp(-restframe_energy_photon_cgs * beta_cgs)
      double n2g1_div_n1g2 = std::exp(-1 * restframe_energy_photon_erg *
                                      beta_cgs);
      return {ndens_1, n2g1_div_n1g2};

    } else if constexpr (Strat == NdensStrat::AbsorberNDensGrid_LTERatio) {

      // not rigorously tested!
      double restframe_energy_photon_erg = line_props.freq_Hz * HPLANCK_CGS;
      double beta_cgs = 1.0 / (ray_data.kinetic_T[pos_i] * KBOLTZ_CGS);

      double ndens_1 = ray_data.ndens[pos_i];
      double n2g1_div_n1g2 = std::exp(-1 * restframe_energy_photon_erg *
                                      beta_cgs);
      return {ndens_1, n2g1_div_n1g2};

    } else if constexpr (Strat == NdensStrat::SpecialLegacySpinFlipCase) {

      // in this case, assume that ray_data.ndens gives the total number
      // density of neutral hydrogen in the n = 1 state (the electronic ground
      // state) 
      //
      // compute ndens in state 2 (the higher energy state), by combining info
      // from following points:
      // - this fn ASSUMES that `exp(-h_cgs*rest_freq / (kboltz_cgs * T_spin))`
      //   is approximately 1 (this is a fairly decent assumption)
      // - consequently:                                n_2 / n_1 == g_2 / g_1
      // - for 21cm transition, g_2 = 3 and g_1 = 1
      // - since n_1 and n_2 are the only states:         n1 + n2 == ndens
      // -> Putting these together:                 n_2 / 3 + n_2 == ndens
      // -> Thus: n_1 = 0.25 * ndens & n_2 = 0.75 * ndens

      double ndens_1 = 0.25 * ray_data.ndens[pos_i];
      double n2g1_div_n1g2 = 1.0;
      return {ndens_1, n2g1_div_n1g2};

    } else {
      // maybe use static_assert(false) here!
      ERROR("THERE IS AN ERROR");
    }
  }

  // the following exists for backwards compatibility
  [[gnu::always_inline]] inline double
  get_ndens_emitter(long pos_i, const RayAlignedProps ray_data,
                    const C_LineProps line_props) const noexcept
  {
    if constexpr (Strat == NdensStrat::SpecialLegacySpinFlipCase) {
      return 0.75 * ray_data.ndens[pos_i];
    } else {
      Ndens_And_Ratio pair = get_pair(pos_i, ray_data, line_props);
      return (line_props.g_hi * pair.ndens_1 * pair.n2g1_div_n1g2 /
              double(line_props.g_lo)); 
    }
  }

};


template<bool using_precalculated_doppler,
         NdensStrat ndens_strat = NdensStrat::SpecialLegacySpinFlipCase>
inline void optically_thin_21cm_ray_spectrum_impl
(const C_LineProps line_props, 
 const double inv_particle_mass_cgs,
 const long nfreq,
 const double* obs_freq,
 const RayAlignedProps ray_aligned_data,
 const NdensFetcher<ndens_strat> ndens_fetcher,
 double* out) noexcept
{
  static_assert(ndens_strat == NdensStrat::SpecialLegacySpinFlipCase,
                "we haven't tested other ndens strategies");

  // set all values in the output array to 0.0
  for(long freq_i = 0; freq_i < nfreq; freq_i++) { out[freq_i] = 0.0; }

  // there are 2 sets of equivalent notation that are commonly used to describe
  // the Hydrogen spin-flip transition.
  // -> specific notation (tailored to this particular transition):
  //      - the number density in the lower-energy spin-0 state is $n_0$
  //      - the number density in the higher-energy spin-1 state is $n_1$
  //      - the rate of spontaneous emission is $A_{10}$
  // -> generic notation (that can describe any bound-bound transition):
  //      - the states are enumerated as 1 and 2
  //      - the number density in the lower-energy state is $n_1$
  //      - the number density in the higher-energy state is $n_2$
  //      - the rate of spontaneous emission is $A_{21}$
  // We adopt the latter notation

  const double rest_freq = line_props.freq_Hz;

  // iterate over gas-elements
  for(long pos_i = 0; pos_i < ray_aligned_data.num_segments; pos_i++) {
    const double n_2 = ndens_fetcher.get_ndens_emitter(pos_i, ray_aligned_data,
                                                       line_props);

    // fetch the length of the path through the current gas-element
    const double cur_dz = ray_aligned_data.dz[pos_i];

    // construct the profile for the current gas-element
    double doppler_parameter_b = (using_precalculated_doppler)
      ? ray_aligned_data.precomputed_doppler_parameter_b[pos_i]
      : doppler_parameter_b_from_temperature(ray_aligned_data.kinetic_T[pos_i],
                                             inv_particle_mass_cgs);

    const LineProfileStruct prof = prep_LineProfHelper
      (rest_freq, doppler_parameter_b, ray_aligned_data.vLOS[pos_i]);

    for (long freq_i = 0; freq_i < nfreq; freq_i++) { // iterate over obs_freq
      double profile_val = eval_line_profile(obs_freq[freq_i], rest_freq, prof);
      double cur_jnu = (HPLANCK_CGS * rest_freq * n_2 * line_props.A21_Hz *
                        profile_val * QUARTER_DIV_PI);
      out[freq_i] += cur_jnu * cur_dz;
    }
  }

}


/// The following struct is used to solve for the optical depth and the
/// integrated source_function diminished by absorption.
///
/// Our definition of optical depth, differs from Rybicki and Lightman. They
/// would define the maximum optical depth at the observer's location. Our
/// choice of definition is a little more consistent with the choice used in
/// the context of stars.
///
/// We are essentially solving the following 2 equations:
///
/// \f[
///   \tau_\nu (s) = \int_{s}^{s_0} \alpha_\nu(s^\prime)\, ds^\prime
/// \f]
///
/// \f[
///   I_\nu (\tau_\nu=0) =  I_\nu(\tau_\nu)\, e^{-\tau_\nu} + f,
/// \f]
///
/// where `f`, the integral term, is given by:
///
/// \f[
///   f = -\int_{\tau_\nu}^0  S_\nu(\tau_\nu^\prime)\, e^{-\tau_\nu^\prime}\, d\tau_\nu^\prime.
/// \f]
struct IntegralStructNoScatterRT {
  /// this specifies the length of all pointers held by this struct
  const long nfreq;

  // the next 2 variables are accumulator variables that serve as outputs
  // -> they each accumulate values separately for each considered frequency
  // -> their lifetimes are externally managed
  double* total_tau;
  double* integrated_source;

  // this last variable is managed by the struct (we use it to cache
  // expensive exponential evaluations)
  double* segStart_expNegTau;

  IntegralStructNoScatterRT() = delete;

  IntegralStructNoScatterRT(long nfreq, double* total_tau,
                            double* integrated_source) noexcept
    : nfreq(nfreq),
      total_tau(total_tau),
      integrated_source(integrated_source),
      segStart_expNegTau(nullptr)
  {
    if (nfreq <= 0) ERROR("nfreq must be positive!");
    this->segStart_expNegTau = new double[nfreq];

    // finally, lets initialize total_tau & segStart_expNegTau so that they have
    // the correct values for the start of the integral
    // -> essentially, we need to initialize the value at the end of the ray
    //    closest to the observer
    // -> by convention, we define the tau at this location to have an optical
    //    depth of zeros at all frequencies (we can always increase the optical
    //    depth used here after we finish the integral)
    // we also set integrated_source to have values of 0.0
    for (long i = 0; i < nfreq; i++){
      this->total_tau[i] = 0.0;
      this->segStart_expNegTau[i] = 1.0; // = std::exp(-this->total_tau[i])
      this->integrated_source[i] = 0.0;
    }
  }

  ~IntegralStructNoScatterRT() noexcept {
    if (this->segStart_expNegTau != nullptr) {
      delete[] this->segStart_expNegTau;
    }
  }

  /// Updates the accumulated values for a single frequency, given values for
  /// the current ray-segment
  ///
  /// @note
  /// This function should be called repeatedly for all frequencies at a
  /// current segment of the ray. Then it should be called again repeatedly for
  /// sections of the ray progressively further from the observer.
  ///
  /// @note
  /// It's okay for this be declared ``const`` since we are updating values
  /// within the pointers
  [[gnu::always_inline]] inline void update(long freq_i,
                                            double absorption_coef,
                                            double source_function,
                                            double dz) const noexcept;
};

/* the following function effectively updates the tracked integral (at all
 * frequencies, for a single segment of the ray)
 *
 * In general, we are effectively solving the following integral (dependence on
 * frequency is dropped to simplify notation)
 *               /\ tau = 0
 *               |
 *     f = -1 *  | S(tau) * exp(-tau) dtau
 *               |
 *              \/ tau = tau
 * We are integrating the light as it moves from the far end of the ray
 * towards the observer.
 *
 * We can reverse this integral so we are integrating along the ray from
 * near to far
 *           /\ tau = tau
 *           |
 *     f =   |   S(tau) * exp(-tau) dtau
 *           |
 *          \/  tau = 0
 *
 * Now let's break this integral up into N segments:
 *
 *          ----- (N - 1)   /\ tau = tau_(i+1)
 *          \               |
 *    f =    \              |    S(tau) * exp(-tau) dtau
 *           /              |
 *          /               |
 *          ----- (i = 0)  \/  tau = tau_i
 * -> each integral integrates between tau_i and tau_(i+1) (which correspond to
 *    the tau values at the edges of each segment).
 *
 * Now if we assume that S(tau) has a constant value S_i we can pull S_i out
 * of the integral and solve the integral analytically.
 * ->  ∫ exp(-tau) dtau from tau_i to tau_(i+1) is

 *     /\ tau = tau_(i+1)
 *     |
 *     |           exp(-tau) dtau   =    -exp(-tau_(i+1)) - (-exp(-tau_i))
 *     |
 *    \/  tau = tau_i
 *                                  =    exp(-tau_i) - exp(-tau_(i+1))
 *
 *
 * Putting this togeter, we find that:
 *          --- (N - 1)
 *    f =   \             S_i * ( exp(-tau_i) - exp(-tau_(i+1)) )
 *          /
 *          --- (i = 0)
 *
 * Coming back the following function:
 * -> the function considers a single section of the above summation and
 *    evaluates the integral over tau AND the integrated source-term
 *
 * This function should be repeatedly called moving progressively further from
 * the observer
 */
[[gnu::always_inline]] inline void
IntegralStructNoScatterRT::update(long freq_i,
                                  double absorption_coef,
                                  double source_function,
                                  double dz) const noexcept
{
  // implicit assumption: 0 <= freq_i < this->nfreq

  // part 0: precompute -exp(this->total_tau[freq_i])
  // -> we need to know the exponential of negative optical depth at start
  //    of current segment
  //
  // this is done implicitly. The value is equivalent to
  // this->segStart_expNegTau[freq_i]

  // part 1: update this->total_tau[freq_i] so that it holds the
  //         optical-depth at the end of the current segment
  // -> this is equivalent to saying do the integral over tau in the
  //    current segment
  //
  // recall: we defined tau so that it is increasing as we move away from
  //         the observer
  this->total_tau[freq_i] += (absorption_coef * dz);

  // part 2: perform the integral over the source term in current segment
  // first, compute the value of exp(-tau) at end of the current segment
  double cur_segEnd_expNegTau = std::exp(-this->total_tau[freq_i]);
  // next, update the integrated source term
  double diff = this->segStart_expNegTau[freq_i] - cur_segEnd_expNegTau;
  this->integrated_source[freq_i] += (source_function * diff);

  // part 3: prepare for next segment (the value of expNegTau at the end of
  // the current segment is the value at the start of the next segment)
  this->segStart_expNegTau[freq_i] = cur_segEnd_expNegTau;
}

template<bool using_precalculated_doppler,
         NdensStrat ndens_strat = NdensStrat::IonNDensGrid_LTERatio>
inline int generate_noscatter_spectrum_impl
(C_LineProps line_props,
 const double inv_particle_mass_cgs,
 const long nfreq,
 const double* obs_freq,
 const RayAlignedProps ray_aligned_data,
 const NdensFetcher<ndens_strat> ndens_fetcher,
 double* out_integrated_source,
 double* out_tau)
{
  static_assert(ndens_strat == NdensStrat::IonNDensGrid_LTERatio,
                "we haven't tested other ndens strategies");

  // load transition properties:
  const double rest_freq_Hz = line_props.freq_Hz;

  // consider 2 states: states 1 and 2.
  // - State2 is the upper level and State1 is the lower level
  // - B12 multiplied by average intensity gives the rate of absorption
  // - A21 gives the rate of spontaneous emission
  const double B12_cgs = line_props.B12_cgs;

  IntegralStructNoScatterRT accumulator(nfreq, out_tau, out_integrated_source);

  if (accumulator.nfreq < 1) return 1; // prob with initializing accumulator!

  const double rest_freq3 = rest_freq_Hz*rest_freq_Hz*rest_freq_Hz;
  const double source_func_numerator = (2*HPLANCK_CGS * rest_freq3 /
                                        (C_CGS * C_CGS));

  for (long pos_i = 0; pos_i < ray_aligned_data.num_segments; pos_i++) {

    Ndens_And_Ratio tmp_pair = ndens_fetcher.get_pair(pos_i, ray_aligned_data,
                                                      line_props);

    // Using equations 1.78 and 1.79 of Rybicki and Lighman to get
    // - absorption coefficient (with units of cm**-1), including the
    //   correction for stimulated-emission
    // - the source function
    // - NOTE: there are some weird ambiguities in the frequency dependence in
    //   these equations. These are discussed below.

    // first compute alpha_norm (normalization of linear absoprtion coef)
    // -> the absorption at a given frequency is this quantity multiplied by
    //    the line profile
    double stim_emission_correction = (1.0 - tmp_pair.n2g1_div_n1g2);
    double alpha_norm = (HPLANCK_CGS * rest_freq_Hz * tmp_pair.ndens_1 *
                         B12_cgs * stim_emission_correction) / (4* M_PI);

    // construct the profile for the current gas-element
    double doppler_parameter_b = (using_precalculated_doppler)
      ? ray_aligned_data.precomputed_doppler_parameter_b[pos_i]
      : doppler_parameter_b_from_temperature(ray_aligned_data.kinetic_T[pos_i],
                                             inv_particle_mass_cgs);

    const LineProfileStruct prof = prep_LineProfHelper
      (rest_freq_Hz, doppler_parameter_b, ray_aligned_data.vLOS[pos_i]);

    double source_func_cgs =
      source_func_numerator / ((1.0/tmp_pair.n2g1_div_n1g2) - 1.0);

    // FREQUENCY AMBIGUITIES:
    // - in Rybicki and Lighman, the notation used in equations 1.78 and 1.79
    //   suggest that all frequencies used in computing linear_absorption and
    //   source_function should use the observed frequency (in the
    //   reference-frame where gas-parcel has no bulk motion)
    // - currently, we use the line's central rest-frame frequency everywhere
    //   other than in the calculation of the line profile.
    //
    // In the absorption-coefficient, I think our choice is well-motivated!
    // -> if you look back at the derivation the equation 1.74, it seems
    //    that the leftmost frequency should be the rest_freq (it seems like
    //    they dropped this along the way and carried it through to 1.78)
    // -> in the LTE case, they don't use rest-freq in the correction for
    //    stimulated emission (eqn 1.80). I think what we do in the LTE case
    //    is more correct.
    //
    // Overall, I suspect the reason that Rybicki & Lightman play a little
    // fast and loose with frequency dependence is the sort of fuzzy
    // assumptions made when deriving the Einstein relations. (It also helps
    // them arive at result that source-function is a black-body in LTE)
    //
    // At the end of the day, it probably doesn't matter much which
    // frequencies we use (the advantage of our approach is we can put all
    // considerations of LOS velocities into the calculation of the line
    // profile)
    //
    // now that we know the source-function and the absorption coefficient,
    // let's compute the integral(s) for the current section of the array

    for (long freq_i = 0; freq_i < nfreq; freq_i++){
      // get the absorption coefficient at current freq
      double profile_val = eval_line_profile(obs_freq[freq_i], rest_freq_Hz,
                                             prof);
      double alpha_nu_cgs = alpha_norm * profile_val;

      // actually update the integrated quantities
      accumulator.update(freq_i, alpha_nu_cgs, source_func_cgs,
                         ray_aligned_data.dz[pos_i]);
    }
  }

  return 0;
}
