// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "utils.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

double avg(const std::vector<double> &x) {
  double sum = 0.0;
  for (const double xi : x)
    sum += xi;
  return sum / static_cast<double>(x.size());
}

double var(const std::vector<double> &x) {
  const double xa = avg(x);
  double sum = 0.0;
  for (const double xi : x)
    sum += ((xi - xa) * (xi - xa));
  return sum / static_cast<double>(x.size());
}

double stdev(const std::vector<double> &x) { return std::sqrt(var(x)); }

void recenter(std::vector<double> &rx, std::vector<double> &ry,
              std::vector<double> &rz, const std::size_t natom) {
  // Ensure "bottom left corner" of box is at (0, 0, 0). i.e. The atomic
  // coordinates are centered at (Lx/2. Ly/2. Lz/2).

  // Find minimum coordinate to shift to (0, 0, 0)
  double x_min = std::numeric_limits<double>::max();
  double y_min = std::numeric_limits<double>::max();
  double z_min = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < natom; i++) {
    x_min = (rx[i] < x_min) ? rx[i] : x_min;
    y_min = (ry[i] < y_min) ? ry[i] : y_min;
    z_min = (rz[i] < z_min) ? rz[i] : z_min;
  }

  // Shift coordinates
  for (std::size_t i = 0; i < natom; i++) {
    rx[i] -= x_min;
    ry[i] -= y_min;
    rz[i] -= z_min;
  }

  return;
}

void print_error_report(
    const std::vector<double> &fx_glst, const std::vector<double> &fy_glst,
    const std::vector<double> &fz_glst, const std::vector<double> &en_glst,
    const std::vector<double> &fx_coul, const std::vector<double> &fy_coul,
    const std::vector<double> &fz_coul, const std::vector<double> &en_coul,
    const std::size_t natom, const double tol) {
  double fx_err_min = std::numeric_limits<double>::max();
  double fy_err_min = std::numeric_limits<double>::max();
  double fz_err_min = std::numeric_limits<double>::max();
  double en_err_min = std::numeric_limits<double>::max();
  double fx_err_max = std::numeric_limits<double>::lowest();
  double fy_err_max = std::numeric_limits<double>::lowest();
  double fz_err_max = std::numeric_limits<double>::lowest();
  double en_err_max = std::numeric_limits<double>::lowest();
  double fx_err_avg = 0.0, fy_err_avg = 0.0, fz_err_avg = 0.0, en_err_avg = 0.0;
  double fx_err_rms = 0.0, fy_err_rms = 0.0, fz_err_rms = 0.0, en_err_rms = 0.0;
  double fx_rms = 0.0, fy_rms = 0.0, fz_rms = 0.0, en_rms = 0.0;

  for (std::size_t i = 0; i < natom; i++) {
    double fx_err = fx_coul[i] - fx_glst[i];
    double fy_err = fy_coul[i] - fy_glst[i];
    double fz_err = fz_coul[i] - fz_glst[i];
    double en_err = en_coul[i] - en_glst[i];
    double fx_abs_err = std::abs(fx_err);
    double fy_abs_err = std::abs(fy_err);
    double fz_abs_err = std::abs(fz_err);
    double en_abs_err = std::abs(en_err);
    fx_err_min = (fx_abs_err < fx_err_min) ? fx_abs_err : fx_err_min;
    fy_err_min = (fy_abs_err < fy_err_min) ? fy_abs_err : fy_err_min;
    fz_err_min = (fz_abs_err < fz_err_min) ? fz_abs_err : fz_err_min;
    en_err_min = (en_abs_err < en_err_min) ? en_abs_err : en_err_min;
    fx_err_max = (fx_abs_err > fx_err_max) ? fx_abs_err : fx_err_max;
    fy_err_max = (fy_abs_err > fy_err_max) ? fy_abs_err : fy_err_max;
    fz_err_max = (fz_abs_err > fz_err_max) ? fz_abs_err : fz_err_max;
    en_err_max = (en_abs_err > en_err_max) ? en_abs_err : en_err_max;
    fx_err_avg += fx_abs_err;
    fy_err_avg += fy_abs_err;
    fz_err_avg += fz_abs_err;
    en_err_avg += en_abs_err;
    fx_err_rms += (fx_err * fx_err);
    fy_err_rms += (fy_err * fy_err);
    fz_err_rms += (fz_err * fz_err);
    en_err_rms += (en_err * en_err);
    fx_rms += (fx_glst[i] * fx_glst[i]);
    fy_rms += (fy_glst[i] * fy_glst[i]);
    fz_rms += (fz_glst[i] * fz_glst[i]);
    en_rms += (en_glst[i] * en_glst[i]);
  }
  fx_err_avg /= static_cast<double>(natom);
  fy_err_avg /= static_cast<double>(natom);
  fz_err_avg /= static_cast<double>(natom);
  en_err_avg /= static_cast<double>(natom);
  fx_err_rms = std::sqrt(fx_err_rms / static_cast<double>(natom));
  fy_err_rms = std::sqrt(fy_err_rms / static_cast<double>(natom));
  fz_err_rms = std::sqrt(fz_err_rms / static_cast<double>(natom));
  en_err_rms = std::sqrt(en_err_rms / static_cast<double>(natom));
  fx_rms = std::sqrt(fx_rms / static_cast<double>(natom));
  fy_rms = std::sqrt(fy_rms / static_cast<double>(natom));
  fz_rms = std::sqrt(fz_rms / static_cast<double>(natom));
  en_rms = std::sqrt(en_rms / static_cast<double>(natom));

  std::ios fmt(nullptr);
  fmt.copyfmt(std::cout);
  std::cout << std::endl;
  std::cout << "Target error: " << std::scientific << tol << std::endl;
  std::cout << std::endl;
  std::cout << "  X Force Error: " << std::endl;
  std::cout << "         Absolute Min: " << std::showpos << fx_err_min
            << " (Norm: " << fx_err_min / fx_rms << ")" << std::endl;
  std::cout << "         Absolute Max: " << std::showpos << fx_err_max
            << " (Norm: " << fx_err_max / fx_rms << ")" << std::endl;
  std::cout << "              Average: " << std::showpos << fx_err_avg
            << " (Norm: " << fx_err_avg / fx_rms << ")" << std::endl;
  std::cout << "    Root Mean Squared: " << fx_err_rms
            << " (Norm: " << fx_err_rms / fx_rms << ")" << std::endl;
  std::cout << std::endl;
  std::cout << "  Y Force Error: " << std::endl;
  std::cout << "         Absolute Min: " << std::showpos << fy_err_min
            << " (Norm: " << fy_err_min / fy_rms << ")" << std::endl;
  std::cout << "         Absolute Max: " << std::showpos << fy_err_max
            << " (Norm: " << fy_err_max / fy_rms << ")" << std::endl;
  std::cout << "              Average: " << std::showpos << fy_err_avg
            << " (Norm: " << fy_err_avg / fy_rms << ")" << std::endl;
  std::cout << "    Root Mean Squared: " << fy_err_rms
            << " (Norm: " << fy_err_rms / fy_rms << ")" << std::endl;
  std::cout << std::endl;
  std::cout << "  Z Force Error: " << std::endl;
  std::cout << "         Absolute Min: " << std::showpos << fz_err_min
            << " (Norm: " << fz_err_min / fz_rms << ")" << std::endl;
  std::cout << "         Absolute Max: " << std::showpos << fz_err_max
            << " (Norm: " << fz_err_max / fz_rms << ")" << std::endl;
  std::cout << "              Average: " << std::showpos << fz_err_avg
            << " (Norm: " << fz_err_avg / fz_rms << ")" << std::endl;
  std::cout << "    Root Mean Squared: " << fz_err_rms
            << " (Norm: " << fz_err_rms / fz_rms << ")" << std::endl;
  std::cout << std::endl;
  std::cout << "   Energy Error: " << std::endl;
  std::cout << "         Absolute Min: " << std::showpos << en_err_min
            << " (Norm: " << en_err_min / en_rms << ")" << std::endl;
  std::cout << "         Absolute Max: " << std::showpos << en_err_max
            << " (Norm: " << en_err_max / en_rms << ")" << std::endl;
  std::cout << "              Average: " << std::showpos << en_err_avg
            << " (Norm: " << en_err_avg / en_rms << ")" << std::endl;
  std::cout << "    Root Mean Squared: " << en_err_rms
            << " (Norm: " << en_err_rms / en_rms << ")" << std::endl;
  std::cout << std::endl;

  return;
}
