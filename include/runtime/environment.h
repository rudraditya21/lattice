#ifndef LATTICE_RUNTIME_ENVIRONMENT_H_
#define LATTICE_RUNTIME_ENVIRONMENT_H_

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "runtime/value.h"

namespace lattice::runtime {

class Environment {
 public:
  explicit Environment(std::shared_ptr<Environment> parent = nullptr) : parent_(parent) {}

  /// Stores or replaces a named value.
  void Define(const std::string& name, const Value& value) { values_[name] = value; }

  /// Updates an existing binding in the nearest scope; returns false if not found.
  bool Assign(const std::string& name, const Value& value) {
    auto it = values_.find(name);
    if (it != values_.end()) {
      it->second = value;
      return true;
    }
    if (parent_) {
      return parent_->Assign(name, value);
    }
    return false;
  }

  /// Looks up a name, returning std::nullopt if it is undefined.
  std::optional<Value> Get(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
      if (!parent_) {
        return std::nullopt;
      }
      return parent_->Get(name);
    }
    return it->second;
  }

 private:
  std::unordered_map<std::string, Value> values_;
  std::shared_ptr<Environment> parent_;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_ENVIRONMENT_H_
