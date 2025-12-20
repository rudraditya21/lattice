#ifndef LATTICE_UTIL_STRING_H_
#define LATTICE_UTIL_STRING_H_

#include <string>
#include <string_view>
#include <vector>

namespace lattice::util {

inline std::string Trim(std::string_view input) {
  size_t start = 0;
  while (start < input.size() && (input[start] == ' ' || input[start] == '\t' ||
                                  input[start] == '\n' || input[start] == '\r')) {
    ++start;
  }
  size_t end = input.size();
  while (end > start && (input[end - 1] == ' ' || input[end - 1] == '\t' ||
                         input[end - 1] == '\n' || input[end - 1] == '\r')) {
    --end;
  }
  return std::string(input.substr(start, end - start));
}

inline std::vector<std::string> SplitLines(const std::string& input) {
  std::vector<std::string> lines;
  std::string current;
  for (char ch : input) {
    if (ch == '\n') {
      lines.push_back(current);
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  lines.push_back(current);
  return lines;
}

}  // namespace lattice::util

#endif  // LATTICE_UTIL_STRING_H_
