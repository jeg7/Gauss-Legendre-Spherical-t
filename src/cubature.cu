// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II, Wonmuk Hwang
//
// ENDLICENSE

#include "cubature.hcu"

#include "cuda_utils.hcu"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

cubature::cubature(void)
    : num_cubatures_(0), tot_num_nodes_(0), points_(), num_nodes_(), xyzw_(),
      cuda_count_(-1) {
  cudaCheck(cudaGetDeviceCount(&this->cuda_count_));
  if (this->cuda_count_ < 1) {
    throw std::runtime_error(
        "glst_force::init: Could not find any CUDA capable devices");
  }
  this->points_.resize(this->cuda_count_);
  this->num_nodes_.resize(this->cuda_count_);
  this->xyzw_.resize(this->cuda_count_);
}

cubature::cubature(const double tol, const unsigned int nalpha,
                   const std::vector<double> &rmax,
                   const std::vector<double> &alpha,
                   const std::vector<double> &zcut)
    : cubature() {
  this->initialize(tol, nalpha, rmax, alpha, zcut);
}

unsigned int cubature::num_cubatures(void) const {
  return this->num_cubatures_;
}

unsigned int cubature::tot_num_nodes(void) const {
  return this->tot_num_nodes_;
}

const std::vector<cuda_container<unsigned int>> &cubature::points(void) const {
  return this->points_;
}

const std::vector<cuda_container<unsigned int>> &
cubature::num_nodes(void) const {
  return this->num_nodes_;
}

const std::vector<cuda_container<double4>> &cubature::xyzw(void) const {
  return this->xyzw_;
}

void cubature::initialize(const double tol, const unsigned int nalpha,
                          const std::vector<double> &rmax,
                          const std::vector<double> &alpha,
                          const std::vector<double> &zcut) {
  this->num_cubatures_ = nalpha;
  this->tot_num_nodes_ = 0;

  std::vector<unsigned int> ngl0;
  std::vector<std::vector<double>> xgl, wgl;
  this->assign_gauss_legendre_quadratures(ngl0, xgl, wgl, tol, nalpha, rmax,
                                          alpha, zcut);

  std::vector<std::vector<std::array<double, 3>>> sm;
  this->assign_spherical_tdesigns(sm, ngl0, xgl, wgl, tol, nalpha, rmax, alpha,
                                  zcut);

  std::vector<unsigned int> points(nalpha, 0);
  std::vector<unsigned int> num_nodes(nalpha, 0);
  std::vector<double4> xyzw;
  for (unsigned int grp = 0; grp < nalpha; grp++) {
    num_nodes[grp] =
        static_cast<unsigned int>(xgl[grp].size() * sm[grp].size());
    this->tot_num_nodes_ += num_nodes[grp];
    if (grp > 0)
      points[grp] = points[grp - 1] + num_nodes[grp - 1];
    const double zcut2 = zcut[grp] * zcut[grp];
    for (std::size_t i = 0; i < xgl[grp].size(); i++) {
      for (std::size_t j = 0; j < sm[grp].size(); j++) {
        const double x =
            2.0 * alpha[grp] * zcut[grp] * xgl[grp][i] * sm[grp][j][0];
        const double y =
            2.0 * alpha[grp] * zcut[grp] * xgl[grp][i] * sm[grp][j][1];
        const double z =
            2.0 * alpha[grp] * zcut[grp] * xgl[grp][i] * sm[grp][j][2];
        const double w = 4.0 * alpha[grp] * zcut[grp] * wgl[grp][i] *
                         std::exp(-zcut2 * xgl[grp][i] * xgl[grp][i]) /
                         (M_PI * static_cast<double>(sm[grp].size()));
        xyzw.push_back(make_double4(x, y, z, w));
      }
    }
  }

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    this->points_[dev] = points;
    this->num_nodes_[dev] = num_nodes;
    this->xyzw_[dev] = xyzw;
  }

  return;
}

void cubature::eval_legendre_poly(double &p, double &dpdx, const double x,
                                  const unsigned int n) {
  // Evaluate Legendre polynomial Pl_n(x) and its derivative at x
  if (n == 0) {
    p = 1.0;
    dpdx = 0.0;
    return;
  }

  if (n == 1) {
    p = x;
    dpdx = 1.0;
    return;
  }

  double p0 = 1.0, p1 = x, p2 = 0.0;
  double dpdx1 = 1.0, dpdx2 = 0.0;
  for (unsigned int i = 1; i < n; i++) { // Use recurrence relation
    double a1 = static_cast<double>(i) + 1.0;
    double a2 = 2.0 * static_cast<double>(i) + 1.0;
    p2 = (a2 * x * p1 - static_cast<double>(i) * p0) / a1;
    dpdx2 = a1 * p1 + x * dpdx1;
    p0 = p1;
    p1 = p2;
    dpdx1 = dpdx2;
  }

  p = p2;
  dpdx = dpdx2;

  return;
}

void cubature::get_wgl(std::vector<double> &wgl, const std::vector<double> &xgl,
                       const unsigned int n) {
  // Get Gauss-Legendre quadrature weights (Golub 1969, w_j)
  //
  // w_j = 2 / (1 - x_j^2)[Pl_N'(x_j)^2] (x_j: j-th root of Pl.)
  //
  // This requires Xgl already be calculated.

  wgl.clear();
  for (std::size_t i = 0; i < xgl.size(); i++) {
    double x = xgl[i];
    double p = 0.0, dpdx = 0.0;
    this->eval_legendre_poly(p, dpdx, x, n);
    wgl.push_back(2.0 / ((1.0 - x * x) * dpdx * dpdx));
  }

  return;
}

double cubature::eval_legendre_root(const double x0, const unsigned int n) {
  // Newton's method for finding first root of Pl_n(x) arouund x0
  double p = 1.0, dpdx = 0.0;
  double dx1 = 0.0, dx2 = -1.0;
  double x = x0;

  for (int i = 0; i < 10; i++) {
    this->eval_legendre_poly(p, dpdx, x, n);
    dx2 = dx1;
    dx1 = p / dpdx;
    if (std::isnan(dx1))
      dx1 = 0.0;
    x -= dx1;
    // Break if either dx1 is very small or if dx1 doesn't change
    if (std::abs(dx1) < std::numeric_limits<double>::epsilon())
      break;
    if (std::abs(dx1 - dx2) < std::numeric_limits<double>::epsilon())
      break;
  }

  return x;
}

void cubature::get_xgl(std::vector<double> &xgl, const unsigned int n) {
  // Find positive roots of Pl_n(x). Strategy:
  //
  // For odd n, put x = 0 as the first root. There are m = (n + 1) / 2 positive
  // and zero roots.
  // For even n, there are m = n / 2 positive roots.
  // Set dx = 0.1*(1 / (m + 1)) (divide the [0,1] by m intervals, then chop it
  // into 10).
  //
  // Even n: Increase x by dx, find x where |P_n(x)| passes a minimum (closest
  // to 0). Then find root around x.
  //
  // Odd n: Increase x by dx. When |P_n(x)| passes through a max, keep advancing
  // and find x where |P_n(x)| is closest to zero. Then find the root around x.
  //
  // After finding the first nonzero root r0, set dx = 0.2*r0 (even n) or
  // 0.1*r0 (odd n), then follow procedure described for Odd n above (pass
  // extremum -> closest approach to zero -> find root).
  //
  // Stop after finding all m roots.

  xgl.clear();

  if (n == 0)
    throw std::runtime_error("No roots for n == 0");

  if (n == 1) {
    xgl.push_back(0.0);
    return;
  }

  if (n == 2) {
    xgl.push_back(1.0 / std::sqrt(3.0));
    return;
  }

  unsigned int m = (n % 2 == 0) ? (n / 2) : ((n + 1) / 2);
  double dx = 0.1 / static_cast<double>(m + 1);
  double x = dx;
  double p0 = 0.0, p1 = 0.0, dp0 = 0.0, dp1 = 0.0;

  // Find first nonzero root
  bool flag = (n % 2 == 0) ? true : false;
  if (!flag) {
    xgl.push_back(0.0);
    for (int i = 0; i < 10; i++) {
      this->eval_legendre_poly(p0, dp0, x, n);
      x += dx;
      this->eval_legendre_poly(p1, dp1, x, n);
      if (dp0 * dp1 < 0.0) { // Passed extremum
        flag = true;
        break;
      }
    }
  }
  assert(flag);
  flag = false;
  for (int i = 0; i < 10; i++) {
    this->eval_legendre_poly(p0, dp0, x, n);
    x += dx;
    this->eval_legendre_poly(p1, dp1, x, n);
    if (std::abs(p0) < std::abs(p1)) { // Closest to zero
      flag = true;
      x -= dx;
      break;
    }
  }
  assert(flag);

  double r0 = this->eval_legendre_root(x, n); // First root
  xgl.push_back(r0);

  unsigned int m0 = (n % 2 == 0) ? 1 : 2; // Number of roots found
  x = r0;
  dx = (n % 2 == 0) ? 0.2 * r0 : 0.1 * r0; // Initial dx after first root
  while (m0 < m) {
    flag = false;
    for (int i = 0; i < 10; i++) {
      this->eval_legendre_poly(p0, dp0, x, n);
      x += dx;
      this->eval_legendre_poly(p1, dp1, x, n);
      if (dp0 * dp1 < 0.0) { // Passed extremum
        flag = true;
        break;
      }
    }
    assert(flag);
    flag = false;
    for (int i = 0; i < 10; i++) {
      this->eval_legendre_poly(p0, dp0, x, n);
      x += dx;
      this->eval_legendre_poly(p1, dp1, x, n);
      if (std::abs(p0) < std::abs(p1)) { // Closest to zero
        flag = true;
        x -= dx;
        break;
      }
    }
    assert(flag);
    double r1 = this->eval_legendre_root(x, n);
    xgl.push_back(r1);
    x = r1;
    dx = 0.1 * (r1 - r0);
    r0 = r1;
    m0++;
    if (m0 == m)
      break;
  }

  return;
}

void cubature::get_gauss_legendre(unsigned int &ngl, unsigned int &ngl0,
                                  std::vector<double> &xgl,
                                  std::vector<double> &wgl, const double tol,
                                  const double dist, const double alpha,
                                  const double zcut) {
  constexpr unsigned int MIN_DEGREE = 2;
  constexpr unsigned int MAX_DEGREE = 324;

  const double erf0 = 0.5 * M_PI * std::erf(alpha * dist);
  const double tol0 = 0.5 * M_PI * dist * tol;

  bool flag = false; // Check if quadrature meets accuracy requirements
  // for (ngl = MIN_DEGREE; ngl <= MAX_DEGREE; ngl++) {
  for (ngl = MIN_DEGREE; ngl <= MAX_DEGREE; ngl += 2) {
    std::vector<double> xgl0, wgl0;
    this->get_xgl(xgl0, ngl);
    this->get_wgl(wgl0, xgl0, ngl);

    unsigned int i0 = (ngl % 2 == 0) ? 0 : 1;
    double s0 = (ngl % 2 == 0) ? 0.0 : alpha * dist * zcut * wgl0[0];
    for (std::size_t i = i0; i < xgl0.size(); i++) {
      double val = std::exp(-zcut * zcut * xgl0[i] * xgl0[i]) *
                   std::sin(2.0 * alpha * dist * zcut * xgl0[i]) / xgl0[i];
      s0 += wgl0[i] * val;
    }

    if (std::abs(s0 - erf0) < tol0) {
      for (std::size_t i = 0; i < xgl0.size(); i++) {
        xgl.push_back(xgl0[i]);
        wgl.push_back(wgl0[i]);
      }
      flag = true;
      break;
    }

    if (flag)
      break;
  }

  if (!flag) {
    throw std::runtime_error(
        "Suitable Gauss-Legendre quadrature could not be found");
  }

  ngl0 = ngl;
  ngl = (ngl0 % 2 == 0) ? (ngl0 / 2) : ((ngl0 + 1) / 2);

  return;
}

void cubature::assign_gauss_legendre_quadratures(
    std::vector<unsigned int> &ngl0, std::vector<std::vector<double>> &xgl,
    std::vector<std::vector<double>> &wgl, const double tol,
    const unsigned int nalpha, const std::vector<double> &rmax,
    const std::vector<double> &alpha, const std::vector<double> &zcut) {
  for (unsigned int grp = 0; grp < nalpha; grp++) {
    unsigned int deg = 0, deg0 = 0;
    std::vector<double> roots, weights;
    this->get_gauss_legendre(deg, deg0, roots, weights, tol, rmax[grp],
                             alpha[grp], zcut[grp]);
    ngl0.push_back(deg0);
    xgl.push_back(roots);
    wgl.push_back(weights);
  }
  return;
}

void cubature::read_tdesign(std::vector<std::array<double, 3>> &sm,
                            const std::string &tfname) {
  std::ifstream fs(tfname);

  if (!fs.is_open())
    throw std::runtime_error("Failed to open file \"" + tfname + "\"");

  for (std::size_t i = 0; i < sm.size(); i++) {
    std::string line = "";
    std::getline(fs, line);
    sm[i][0] = std::stod(line.substr(0, 25));
    sm[i][1] = std::stod(line.substr(25, 25));
    sm[i][2] = std::stod(line.substr(50, 25));
  }

  return;
}

void cubature::get_tdesign_props(std::string &tfname, unsigned int &m,
                                 const unsigned int tdeg,
                                 const std::string &dir) {
  for (const auto &file : std::filesystem::directory_iterator(dir)) {
    std::string path = file.path();
    std::size_t pos0 = path.length() - 9;
    unsigned int tdeg_cmp = std::stoull(path.substr(pos0, 3));
    if (tdeg_cmp == tdeg) {
      std::size_t pos1 = path.length() - 5;
      // Divide by 2 to only take the upper hemisphere
      m = std::stoull(path.substr(pos1, 5)) / 2;
      tfname = path;
      break;
    }
  }

  return;
}

void cubature::get_spherical_tdesign(std::vector<std::array<double, 3>> &sm,
                                     const unsigned int ngl0,
                                     const std::vector<double> &xgl,
                                     const std::vector<double> &wgl,
                                     const double tol, const double dist,
                                     const double alpha, const double zcut) {
  // Given Gauss-Legendre quadrature Xgl (roots) and Wgl (weights), determine
  // the tdeg of t-design and save the corresponding unit vectors \hat t_m to
  // Sm. For this, increment t-design tdeg, read {\hat t_m}, and calculate
  // ../doc/legendre.pdf:Eq7.

  constexpr unsigned int MAX_DEG = 162;

  const double zcut2 = zcut * zcut;
  const double erf0 = std::erf(alpha * dist) / dist;

  unsigned int n0 =
      static_cast<unsigned int>(xgl.size()) / 2; // Lower bound of t-degree
  int nmin = -1;
  double dmin = std::numeric_limits<double>::max();
  unsigned int i0 = (ngl0 % 2 == 0) ? 0 : 1;
  unsigned int n = 0;

  for (int j = 0; j < 3; j++) { // Put r in 3 dirs and satisfy the condition
    for (n = n0; n < MAX_DEG; n++) { // t-degree > ngl (nonneg: GL roots)
      std::string tfname = "";
      unsigned int m = 0;
      this->get_tdesign_props(tfname, m, 2 * n + 1, "SS31-Mar-2016/");
      sm.assign(m, {0.0, 0.0, 0.0});
      this->read_tdesign(sm, tfname);

      double s0 = 0.0;
      for (std::size_t i = i0; i < xgl.size(); i++) { // G-L quadrature
        const double xgl2 = xgl[i] * xgl[i];
        double coef = 2.0 * alpha * zcut * dist * xgl[i];
        double val = 0.0;
        for (unsigned int k = 0; k < m; k++)
          val += std::cos(coef * sm[k][j]);
        s0 += wgl[i] * std::exp(-zcut2 * xgl2) * val;
      }
      s0 *= (4.0 * alpha * zcut / (M_PI * static_cast<double>(m)));
      if (ngl0 % 2 == 1)
        s0 += 2.0 * alpha * zcut * wgl[0] / M_PI;
      double err = std::abs(s0 - erf0);
      if (err < dmin) {
        dmin = err;
        nmin = static_cast<int>(n);
      }
      if (err < tol) {
        n0 = n;
        nmin = -2;
        break;
      }
    }
  }

  if (nmin > 0)
    throw std::runtime_error("Suitable t-design could not be found");

  return;
}

void cubature::assign_spherical_tdesigns(
    std::vector<std::vector<std::array<double, 3>>> &sm,
    const std::vector<unsigned int> &ngl0,
    const std::vector<std::vector<double>> &xgl,
    const std::vector<std::vector<double>> &wgl, const double tol,
    const unsigned int nalpha, const std::vector<double> &rmax,
    const std::vector<double> &alpha, const std::vector<double> &zcut) {
  for (unsigned int grp = 0; grp < nalpha; grp++) {
    std::vector<std::array<double, 3>> tsvec;
    this->get_spherical_tdesign(tsvec, ngl0[grp], xgl[grp], wgl[grp], tol,
                                rmax[grp], alpha[grp], zcut[grp]);
    sm.push_back(tsvec);
  }

  return;
}
