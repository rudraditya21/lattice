#include "runtime/decimal.h"

#include <cmath>

namespace lattice::runtime {

namespace {
int& PrecisionRef() {
  static int precision = 18;
  return precision;
}
}  // namespace

int GetDecimalPrecision() {
  return PrecisionRef();
}

void SetDecimalPrecision(int precision) {
  if (precision < 0) precision = 0;
  if (precision > 30) precision = 30;  // clamp to a sane range
  PrecisionRef() = precision;
}

long double RoundDecimal(long double value) {
  int p = PrecisionRef();
  if (p <= 0) return std::round(value);
  long double scale = std::pow(10.0L, static_cast<long double>(p));
  return std::round(value * scale) / scale;
}

}  // namespace lattice::runtime
