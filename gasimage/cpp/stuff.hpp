#include "err.hpp"
#include <algorithm> // std::distance, std::lower_bound
#include <cmath> // std::isnan, std::log10


// define some constants!
#define INV_SQRT_PI 0.5641895835477563
#define QUARTER_DIV_PI 0.07957747154594767
// define some Macros equal to some yt-constants
#define C_CGS 29979245800.0            /* yt.units.c_cgs */
#define HPLANCK_CGS 6.62606957e-27     /* yt.units.h_cgs */
#define MH_CGS 1.6737352238051868e-24  /* yt.units.mh_cgs */
#define KBOLTZ_CGS 1.3806488e-16       /* yt.units.kboltz_cgs */

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

namespace { //anonymous namespace

struct C_LineProps{
  int g_lo;
  int g_hi;
  double freq_Hz; // rest frequency
  double energy_lo_erg; // energy of the lower state
  double B12_cgs; // when multiplied by average intensity, gives the rate of
                  // absorption
};

struct Ndens_And_Ratio{
  double ndens_1;
  double n2g1_div_n1g2;
};

inline Ndens_And_Ratio ndens_and_ratio_from_partition(
  LinInterpPartitionFn partition_fn_pack, double kinetic_T,
  double ndens_ion_species, C_LineProps line_props) noexcept
{

  double restframe_energy_photon_erg = line_props.freq_Hz * HPLANCK_CGS;
  double beta_cgs = 1.0 / (kinetic_T * KBOLTZ_CGS);

  double ndens_1 = (
    ndens_ion_species * line_props.g_lo * std::exp(-line_props.energy_lo_erg
                                                   * beta_cgs)
    / eval_partition_fn(partition_fn_pack, kinetic_T) 
  );

  // n1/n2 = (g1/g2) * exp(restframe_energy_photon_cgs * beta_cgs)
  // n2/n1 = (g2/g1) * exp(-restframe_energy_photon_cgs * beta_cgs)
  // (n2*g1)/(n1*g2) = exp(-restframe_energy_photon_cgs * beta_cgs)
  double n2g1_div_n1g2 = std::exp(-1 * restframe_energy_photon_erg * beta_cgs);

  return {ndens_1, n2g1_div_n1g2};
}

} // anonymous namespace

///
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
  long nfreq;

  // the next 2 variables are accumulator variables that serve as outputs
  // -> they each accumulate values separately for each considered frequency
  // -> their lifetimes are externally managed
  double* total_tau;
  double* integrated_source;

  // this last variable is managed by the struct (we use it to cache
  // expensive exponential evaluations)
  double* segStart_expNegTau;
};

IntegralStructNoScatterRT prep_IntegralStructNoScatterRT(
  long nfreq, double* total_tau, double* integrated_source
)
{
  // the nfreq field of the resulting struct is set to 0 if this function
  // encounters issues

  if (nfreq <= 0) return {0, nullptr, nullptr, nullptr};

  IntegralStructNoScatterRT out{nfreq, total_tau, integrated_source, nullptr};
  out.segStart_expNegTau = new double[nfreq];

  // finally, lets initialize total_tau & segStart_expNegTau so that they have
  // the correct values for the start of the integral
  // -> essentially, we need to initialize the value at the end of the ray
  //    closest to the observer
  // -> by convention, we define the tau at this location to have an optical
  //    depth of zeros at all frequencies (we can always increase the optical
  //    depth used here after we finish the integral)
  // we also set integrated_source to have values of 0.0
  for (long freq_i = 0; freq_i < nfreq; freq_i++){
    out.total_tau[freq_i] = 0.0;
    out.segStart_expNegTau[freq_i] = 1.0; // = std::exp(-out.total_tau[freq_i])
    out.integrated_source[freq_i] = 0.0;
  }
  return out;
}

/*
 * the following function effectively updates the tracked integral (at all
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
void update_IntegralStructNoScatterRT(const IntegralStructNoScatterRT& obj,
                                      const double* absorption_coef,
                                      double source_function, double dz)
{
  // NOTE: its ok to pass obj by value since the only thing being updated are
  //       pointers pointer held by obj
  // implicit assumption: absorption_coef has obj.nfreq entries

  for (long freq_i = 0; freq_i < obj.nfreq; freq_i++) {
    // part 0: precompute -exp(obj.total_tau[freq_i])
    // -> we need to know the exponential of negative optical depth at start
    //    of current segment
    //
    // this is done implicitly. The value is equivalent to
    // obj.segStart_expNegTau[freq_i]

    // part 1: update obj.total_tau[freq_i] so that it holds the
    //         optical-depth at the end of the current segment
    // -> this is equivalent to saying do the integral over tau in the
    //    current segment
    //
    // recall: we defined tau so that it is increasing as we move away from
    //         the observer
    obj.total_tau[freq_i] += (absorption_coef[freq_i] * dz);

    // part 2: perform the integral over the source term in current segment
    // first, compute the value of exp(-tau) at end of the current segment
    double cur_segEnd_expNegTau = std::exp(-obj.total_tau[freq_i]);
    // next, update the integrated source term
    double diff = obj.segStart_expNegTau[freq_i] - cur_segEnd_expNegTau;
    obj.integrated_source[freq_i] += (source_function * diff);

    // part 3: prepare for next segment (the value of expNegTau at the end of
    // the current segment is the value at the start of the next segment)
    obj.segStart_expNegTau[freq_i] = cur_segEnd_expNegTau;
  }
}

void clean_IntegralStructNoScatterRT(const IntegralStructNoScatterRT obj)
{
  if ((obj.nfreq > 0) && (obj.segStart_expNegTau != nullptr)) {
      delete[] obj.segStart_expNegTau;
  }
}
