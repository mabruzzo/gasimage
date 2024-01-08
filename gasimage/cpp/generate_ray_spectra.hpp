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
