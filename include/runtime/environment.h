#ifndef LATTICE_RUNTIME_ENVIRONMENT_H_
#define LATTICE_RUNTIME_ENVIRONMENT_H_

#include <optional>
#include <string>
#include <unordered_map>

#include "runtime/value.h"

namespace lattice::runtime {

class Environment {
 public:
  void Define(const std::string& name, const Value& value) { values_[name] = value; }

  std::optional<Value> Get(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

 private:
  std::unordered_map<std::string, Value> values_;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_ENVIRONMENT_H_
