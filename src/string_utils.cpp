// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "string_utils.hpp"
#include <algorithm>

void utl::ltrim_ip(std::string &str) {
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char c) { return !std::isspace(c); }));
  return;
}

void utl::rtrim_ip(std::string &str) {
  str.erase(std::find_if(str.rbegin(), str.rend(),
                         [](unsigned char c) { return !std::isspace(c); })
                .base(),
            str.end());
  return;
}

void utl::trim_ip(std::string &str) {
  utl::rtrim_ip(str);
  utl::ltrim_ip(str);
  return;
}

std::string utl::ltrim(const std::string &str) {
  std::string s = str;
  utl::ltrim_ip(s);
  return s;
}

std::string utl::rtrim(const std::string &str) {
  std::string s = str;
  utl::rtrim_ip(s);
  return s;
}

std::string utl::trim(const std::string &str) { return ltrim(rtrim(str)); }

void utl::to_lower_ip(std::string &str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return;
}

void utl::to_upper_ip(std::string &str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return;
}

std::string utl::to_lower(const std::string &str) {
  std::string s = str;
  utl::to_lower_ip(s);
  return s;
}

std::string utl::to_upper(const std::string &str) {
  std::string s = str;
  utl::to_upper_ip(s);
  return s;
}

std::vector<std::string> utl::split(const std::string &str,
                                    const std::string &delimiter) {
  std::string s = utl::trim(str);
  std::vector<std::string> tokens;
  std::size_t pos = 0;

  while ((pos = s.find(delimiter)) != std::string::npos) {
    tokens.push_back(s.substr(0, pos));
    s.erase(0, pos + delimiter.length());
    utl::ltrim_ip(s);
  }
  tokens.push_back(s);

  return tokens;
}
