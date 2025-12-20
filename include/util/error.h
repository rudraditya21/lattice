#ifndef LATTICE_UTIL_ERROR_H_
#define LATTICE_UTIL_ERROR_H_

#include <stdexcept>
#include <string>

namespace lattice::util {

class Error : public std::runtime_error {
 public:
  /// Creates an error with message and source location (1-based line/column).
  Error(const std::string& message, int line, int column)
      : std::runtime_error(message), line_(line), column_(column) {}

  /// Line where the error was detected.
  int line() const { return line_; }
  /// Column where the error was detected.
  int column() const { return column_; }

  /// Returns a human-readable string with location context.
  std::string formatted() const {
    return "Error at " + std::to_string(line_) + ":" + std::to_string(column_) + " - " + what();
  }

 private:
  int line_;
  int column_;
};

}  // namespace lattice::util

#endif  // LATTICE_UTIL_ERROR_H_
