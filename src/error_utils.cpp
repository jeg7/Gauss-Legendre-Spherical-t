// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "error_utils.hpp"

#include <stdexcept>
#include <string>

[[noreturn]] void utl::throw_error(const std::string_view function_name,
                                   const std::string_view message) {
  constexpr std::string_view prefix = "FATAL ERROR: ";
  constexpr std::string_view separator = ": ";

  std::string error;
  error.reserve(prefix.size() + function_name.size() + separator.size() +
                message.size());

  error.append(prefix.data(), prefix.size());
  error.append(function_name.data(), function_name.size());
  error.append(separator.data(), separator.size());
  error.append(message.data(), message.size());

  throw std::runtime_error(error);
}
