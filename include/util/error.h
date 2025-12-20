#ifndef LATTICE_UTIL_ERROR_H_
#define LATTICE_UTIL_ERROR_H_

#include <stdexcept>
#include <string>

namespace lattice::util {

class Error : public std::runtime_error {
 public:
  Error(const std::string& message, int line, int column)
      : std::runtime_error(message), line_(line), column_(column) {}

  int line() const { return line_; }
  int column() const { return column_; }

  std::string formatted() const {
    return "Error at " + std::to_string(line_) + ":" + std::to_string(column_) + " - " + what();
  }

 private:
  int line_;
  int column_;
};

}  // namespace lattice::util

#endif  // LATTICE_UTIL_ERROR_H_
