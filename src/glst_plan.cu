// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENS

#include "glst_plan.hcu"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

static double erfc_inv(const double y, const double tol) {
  constexpr unsigned long long int MAX_IT = 100000000;
  const double rdum = 2.0 / std::sqrt(M_PI);

  double x = 0.0; // Initial guess for x
  double err = std::numeric_limits<double>::max();

  for (unsigned long long int i = 0; i < MAX_IT; i++) {
    err = 1.0 - std::erf(x) - y;
    if (std::abs(err) < tol)
      break;

    // Calculate the derivative of erfc
    double deriv = -std::exp(-x * x) * rdum;
    x -= err / deriv; // Newton-Raphson formula: x = x - f(x) / f'(x)
  }

  if (std::abs(err) >= tol) {
    throw std::runtime_error("FATAL ERROR: erfc_inv(const double, const "
                             "double): Inverse of erfc not found after " +
                             std::to_string(MAX_IT) + " iterations");
  }

  return x;
}

glst_plan::glst_plan(void)
    : natom_(0), box_dim_x_(0.0), box_dim_y_(0.0), box_dim_z_(0.0), ncell_x_(0),
      ncell_y_(0), ncell_z_(0), ncell_(0), cell_dim_x_(0.0), cell_dim_y_(0.0),
      cell_dim_z_(0.0), ngroup_(0), grp_r_in_(), grp_r_out_(),
      ncell_alpha_group_(), rmax_(), alpha_(), zcut_() {}

glst_plan::~glst_plan(void) {}

unsigned int glst_plan::natom(void) const { return this->natom_; }

double glst_plan::box_dim_x(void) const { return this->box_dim_x_; }

double glst_plan::box_dim_y(void) const { return this->box_dim_y_; }

double glst_plan::box_dim_z(void) const { return this->box_dim_z_; }

unsigned int glst_plan::ncell_x(void) const { return this->ncell_x_; }

unsigned int glst_plan::ncell_y(void) const { return this->ncell_y_; }

unsigned int glst_plan::ncell_z(void) const { return this->ncell_z_; }

unsigned int glst_plan::ncell(void) const { return this->ncell_; }

double glst_plan::cell_dim_x(void) const { return this->cell_dim_x_; }

double glst_plan::cell_dim_y(void) const { return this->cell_dim_y_; }

double glst_plan::cell_dim_z(void) const { return this->cell_dim_z_; }

unsigned int glst_plan::ngroup(void) const { return this->ngroup_; }

const std::vector<unsigned int> &glst_plan::grp_r_in(void) const {
  return this->grp_r_in_;
}

const std::vector<unsigned int> &glst_plan::grp_r_out(void) const {
  return this->grp_r_out_;
}

const std::vector<unsigned int> &glst_plan::ncell_alpha_group(void) const {
  return this->ncell_alpha_group_;
}

const std::vector<double> &glst_plan::rmax(void) const { return this->rmax_; }

const std::vector<double> &glst_plan::alpha(void) const { return this->alpha_; }

const std::vector<double> &glst_plan::zcut(void) const { return this->zcut_; }

void glst_plan::init_cells(const unsigned int natom, const double box_dim_x,
                           const double box_dim_y, const double box_dim_z,
                           const double rcut) {
  this->natom_ = natom;

  this->box_dim_x_ = box_dim_x;
  this->box_dim_y_ = box_dim_y;
  this->box_dim_z_ = box_dim_z;

  this->cell_dim_x_ = rcut;
  this->cell_dim_y_ = rcut;
  this->cell_dim_z_ = rcut;

  this->ncell_x_ = static_cast<unsigned int>(box_dim_x / this->cell_dim_x_);
  this->ncell_y_ = static_cast<unsigned int>(box_dim_y / this->cell_dim_y_);
  this->ncell_z_ = static_cast<unsigned int>(box_dim_z / this->cell_dim_z_);

  if (static_cast<double>(this->ncell_x_) * this->cell_dim_x_ < box_dim_x)
    this->ncell_x_++;
  if (static_cast<double>(this->ncell_y_) * this->cell_dim_y_ < box_dim_y)
    this->ncell_y_++;
  if (static_cast<double>(this->ncell_z_) * this->cell_dim_z_ < box_dim_z)
    this->ncell_z_++;

  this->ncell_ = this->ncell_x_ * this->ncell_y_ * this->ncell_z_;

  return;
}

void glst_plan::init_alpha_groups(const double tol) {
  this->ncell_alpha_group_.clear();
  this->rmax_.clear();
  this->alpha_.clear();
  this->zcut_.clear();
  this->grp_r_in_.clear();
  this->grp_r_out_.clear();

  int width = 1;
  int tot_width = 0;
  int ncell_remain = this->ncell_x_;
  ncell_remain = (static_cast<int>(this->ncell_y_) > ncell_remain)
                     ? static_cast<int>(this->ncell_y_)
                     : ncell_remain;
  ncell_remain = (static_cast<int>(this->ncell_z_) > ncell_remain)
                     ? static_cast<int>(this->ncell_z_)
                     : ncell_remain;
  ncell_remain -= 2;

  this->ngroup_ = 0;

  while (ncell_remain >= 0) {
    this->ncell_alpha_group_.push_back(static_cast<unsigned int>(width));

    double lmin = this->cell_dim_x_;
    lmin = (this->cell_dim_y_ < lmin) ? this->cell_dim_y_ : lmin;
    lmin = (this->cell_dim_z_ < lmin) ? this->cell_dim_z_ : lmin;

    double lmax = this->cell_dim_x_;
    lmax = (this->cell_dim_y_ > lmax) ? this->cell_dim_y_ : lmax;
    lmax = (this->cell_dim_z_ > lmax) ? this->cell_dim_z_ : lmax;

    const double rmin0 = lmin * static_cast<double>(tot_width + 1);
    const double rmax0 =
        std::sqrt(3.0) * (rmin0 + lmax * static_cast<double>(width + 1));
    this->rmax_.push_back(rmax0);

    const double alpha0 =
        erfc_inv(tol * rmin0, std::numeric_limits<double>::epsilon()) / rmin0;
    this->alpha_.push_back(alpha0);

    const double zcut0 = erfc_inv(0.5 * std::sqrt(M_PI) / alpha0 * tol,
                                  std::numeric_limits<double>::epsilon());
    this->zcut_.push_back(zcut0);

    ncell_remain -= width;
    tot_width += width;
    if (ncell_remain < 1)
      break;

    // Determine the next alpha group width
    int width1 = 2 * width; // Width of next complete alpha group
    int width2 =
        6 * width; // Sum of the next two alpha group widths (2 + 4) * width
    if (ncell_remain <= width1)
      width = ncell_remain; // The last group
    else if ((width1 < ncell_remain) && (ncell_remain < width2))
      width = ncell_remain / 2; // Two groups left
    else
      width = width1; // Keep going
  }

  this->ngroup_ = static_cast<unsigned int>(this->ncell_alpha_group_.size());

  this->grp_r_in_.resize((this->ngroup_ > 0) ? this->ngroup_ : 1);
  this->grp_r_out_.resize((this->ngroup_ > 0) ? this->ngroup_ : 1);

  this->grp_r_in_[0] = 1;
  for (unsigned int group = 0; group < this->ngroup_; group++) {
    if (group > 0)
      this->grp_r_in_[group] = this->grp_r_out_[group - 1];
    this->grp_r_out_[group] =
        this->grp_r_in_[group] + this->ncell_alpha_group_[group];
  }

  return;
}
