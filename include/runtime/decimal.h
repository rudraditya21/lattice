#ifndef LATTICE_RUNTIME_DECIMAL_H_
#define LATTICE_RUNTIME_DECIMAL_H_

namespace lattice::runtime {

// Configurable decimal precision (digits after the decimal point).
int GetDecimalPrecision();
void SetDecimalPrecision(int precision);
long double RoundDecimal(long double value);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_DECIMAL_H_
