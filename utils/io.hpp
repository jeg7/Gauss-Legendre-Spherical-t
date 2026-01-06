// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __UTILS_IO__
#define __UTILS_IO__

#include <cstddef>
#include <string>
#include <vector>

inline void get_line(std::string &line, std::size_t &pos,
                     const std::string &file_data) {
  const std::size_t pos1 = file_data.find_first_of('\n', pos);
  line = file_data.substr(pos, pos1 - pos);
  pos = pos1 + 1;
  return;
}

void read_file_into_string(std::string &file_data, const std::string &fname);

std::size_t get_natom_psf(const std::string &fname);

void read_charmm_cor(std::vector<double> &rx, std::vector<double> &ry,
                     std::vector<double> &rz, const std::size_t natom,
                     const std::string &fname);

void read_charmm_psf(std::vector<double> &qc, const std::size_t natom,
                     const std::string &fname);

#endif
