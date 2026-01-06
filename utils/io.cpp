// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "io.hpp"

#include <fstream>
#include <stdexcept>

void read_file_into_string(std::string &file_data, const std::string &fname) {
  std::ifstream ifs(fname, std::ios::in | std::ios::binary | std::ios::ate);
  if (ifs.is_open() == false)
    throw std::runtime_error("Failed to open file \"" + fname + "\"");

  // Store the size of the file
  const std::size_t fsize = ifs.tellg();

  // Initialize a std::string with length fsize, filled with null characters
  file_data = std::string(fsize, '\0');

  // Go back to the beginning of the file and store the contents in the string
  ifs.seekg(0, std::ios::beg);
  ifs.read(&file_data[0], fsize);
  ifs.close();

  return;
}

std::size_t get_natom_psf(const std::string &fname) {
  std::string file_data;
  read_file_into_string(file_data, fname);

  std::size_t natom = 0, pos = 0;
  while (pos < file_data.length()) {
    std::string line = "";
    get_line(line, pos, file_data);
    if (line.length() < 11)
      continue;
    if (line.substr(11, 6) == "!NATOM") {
      natom = std::stoull(line.substr(0, 11));
      break;
    }
  }

  return natom;
}

void read_charmm_cor(std::vector<double> &rx, std::vector<double> &ry,
                     std::vector<double> &rz, const std::size_t natom,
                     const std::string &fname) {
  std::string file_data;
  read_file_into_string(file_data, fname);

  std::size_t pos = 0, natom_chk = 0;

  // Read past header
  while (pos < file_data.length()) {
    std::string line = "";
    get_line(line, pos, file_data);
    if ((line.length() > 0) && (line[0] != '*')) {
      natom_chk = std::stoull(line.substr(0, 10));
      break;
    }
  }

  if (natom != natom_chk)
    throw std::runtime_error("Number of atoms differs from size of array");

  // Read coordinate data
  for (std::size_t i = 0; i < natom; i++) {
    std::string line = "";
    get_line(line, pos, file_data);
    rx[i] = std::stod(line.substr(40, 20));
    ry[i] = std::stod(line.substr(60, 20));
    rz[i] = std::stod(line.substr(80, 20));
  }

  return;
}

void read_charmm_psf(std::vector<double> &qc, const std::size_t natom,
                     const std::string &fname) {
  std::string file_data;
  read_file_into_string(file_data, fname);

  std::size_t pos = 0, natom_chk = 0;

  // Read header
  while (pos < file_data.length()) {
    std::string line = "";
    get_line(line, pos, file_data);
    if (line.length() < 11)
      continue;
    if (line.substr(11, 6) == "!NATOM") {
      natom_chk = std::stoull(line.substr(0, 10));
      break;
    }
  }

  if (natom != natom_chk)
    throw std::runtime_error("Number of atoms differs from size of array");

  // Read charge data
  for (std::size_t i = 0; i < natom; i++) {
    std::string line = "";
    get_line(line, pos, file_data);
    qc[i] = std::stod(line.substr(50, 14));
  }

  return;
}
