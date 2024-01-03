//#include "err.hpp"
#include <algorithm> // std::distance, std::lower_bound
#include <cmath> // std::isnan, std::log10

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
