// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __GLST_STRING_UTILS__
#define __GLST_STRING_UTILS__

#include <cstddef>
#include <string>
#include <vector>

namespace utl {

void ltrim_ip(std::string &str);

void rtrim_ip(std::string &str);

void trim_ip(std::string &str);

std::string ltrim(const std::string &str);

std::string rtrim(const std::string &str);

std::string trim(const std::string &str);

void to_lower_ip(std::string &str);

void to_upper_ip(std::string &str);

std::string to_lower(const std::string &str);

std::string to_upper(const std::string &str);

std::vector<std::string> split(const std::string &str,
                               const std::string &delimiter = " ");

} // namespace utl

#endif
