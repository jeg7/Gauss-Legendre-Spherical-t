// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#ifndef __UTILS_UTILS__
#define __UTILS_UTILS__

#include <cstddef>
#include <vector>

double avg(const std::vector<double> &x);

double var(const std::vector<double> &x);

double stdev(const std::vector<double> &x);

void recenter(std::vector<double> &rx, std::vector<double> &ry,
              std::vector<double> &rz, const std::size_t natom);

void print_error_report(
    const std::vector<double> &fx_glst, const std::vector<double> &fy_glst,
    const std::vector<double> &fz_glst, const std::vector<double> &en_glst,
    const std::vector<double> &fx_coul, const std::vector<double> &fy_coul,
    const std::vector<double> &fz_coul, const std::vector<double> &en_coul,
    const std::size_t natom, const double tol);

#endif
