// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __GLST_ERROR_UTILS__
#define __GLST_ERROR_UTILS__

#include <string_view>

namespace utl {

[[noreturn]] void throw_error(const std::string_view function_name,
                              const std::string_view message);

inline void require(const bool condition, const std::string_view function_name,
                    const std::string_view message) {
  if (!condition)
    utl::throw_error(function_name, message);
  return;
}

} // namespace utl

#endif
