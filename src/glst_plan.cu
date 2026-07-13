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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
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
      ncell_alpha_group_(), rmax_(), alpha_(), zcut_(), cubature_(nullptr),
      max_tile_nodes_(2048), tile_count_(0), tile_group_(), tile_node_point_(),
      tile_node_count_() {}

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

unsigned int glst_plan::num_cubatures(void) const {
  return this->cubature_data().num_cubatures();
}

unsigned int glst_plan::tot_num_nodes(void) const {
  return this->cubature_data().tot_num_nodes();
}

const std::vector<cuda_container<unsigned int>> &glst_plan::points(void) const {
  return this->cubature_data().points();
}

const std::vector<cuda_container<unsigned int>> &
glst_plan::num_nodes(void) const {
  return this->cubature_data().num_nodes();
}

const std::vector<cuda_container<double>> &glst_plan::x(void) const {
  return this->cubature_data().x();
}

const std::vector<cuda_container<double>> &glst_plan::y(void) const {
  return this->cubature_data().y();
}

const std::vector<cuda_container<double>> &glst_plan::z(void) const {
  return this->cubature_data().z();
}

const std::vector<cuda_container<double>> &glst_plan::w(void) const {
  return this->cubature_data().w();
}

const std::vector<cuda_container<unsigned int>> &glst_plan::group(void) const {
  return this->cubature_data().group();
}

const cubature<double> &glst_plan::cubature_data(void) const {
  if (this->cubature_ == nullptr) {
    throw std::runtime_error("FATAL ERROR: glst_plan::cubature_data: Cubature "
                             "has not been initialized");
  }
  return *(this->cubature_);
}

unsigned int glst_plan::max_tile_nodes(void) const {
  return this->max_tile_nodes_;
}

unsigned int glst_plan::tile_count(void) const { return this->tile_count_; }

const std::vector<unsigned int> &glst_plan::tile_group(void) const {
  return this->tile_group_;
}

const std::vector<unsigned int> &glst_plan::tile_node_point(void) const {
  return this->tile_node_point_;
}

const std::vector<unsigned int> &glst_plan::tile_node_count(void) const {
  return this->tile_node_count_;
}

unsigned int glst_plan::tile_group(const unsigned int tile) const {
  if (tile >= this->tile_count_) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::tile_group: Tile index out of range");
  }
  return this->tile_group_[tile];
}

unsigned int glst_plan::tile_node_point(const unsigned int tile) const {
  if (tile >= this->tile_count_) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::tile_node_point: Tile index out of range");
  }
  return this->tile_node_point_[tile];
}

unsigned int glst_plan::tile_node_count(const unsigned int tile) const {
  if (tile >= this->tile_count_) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::tile_node_count: Tile index out of range");
  }
  return this->tile_node_count_[tile];
}

unsigned int glst_plan::tiles_in_group(const unsigned int group) const {
  return static_cast<unsigned int>(
      std::count(this->tile_group_.begin(), this->tile_group_.end(), group));
}

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

void glst_plan::init_cubature(const double tol) {
  if (this->ngroup_ == 0) {
    throw std::runtime_error("FATAL ERROR: glst_plan::init_cubature: Alpha "
                             "groups have not been initialized");
  }

  this->cubature_ = std::make_unique<cubature<double>>(
      tol, this->ngroup_, this->rmax_, this->alpha_, this->zcut_);

  return;
}

void glst_plan::init_tile_schedule(const unsigned int max_tile_nodes) {
  if (max_tile_nodes == 0) {
    throw std::runtime_error("FATAL ERROR: glst_plan::init_tile_schedule: "
                             "max_tile_nodes must be > 0");
  }

  if (this->cubature_ == nullptr) {
    throw std::runtime_error("FATAL ERROR: glst_plan::init_tile_schedule: "
                             "Cubature has not been initialized");
  }

  this->max_tile_nodes_ = max_tile_nodes;
  this->tile_count_ = 0;

  this->tile_group_.clear();
  this->tile_node_point_.clear();
  this->tile_node_count_.clear();

  const unsigned int total_nodes =
      static_cast<unsigned int>(this->cubature_->tot_num_nodes());

  unsigned int scheduled_nodes = 0;

  for (unsigned int group = 0; group < this->ngroup_; group++) {
    const unsigned int group_point =
        static_cast<unsigned int>(this->cubature_->points()[0][group]);

    const unsigned int group_node_count =
        static_cast<unsigned int>(this->cubature_->num_nodes()[0][group]);

    if ((group_point > total_nodes) ||
        (group_node_count > total_nodes - group_point)) {
      throw std::runtime_error("FATAL ERROR: glst_plan::init_tile_schedule: "
                               "Cubature group node range is out of bounds");
    }

    unsigned int group_offset = 0;

    while (group_offset < group_node_count) {
      const unsigned int remaining = group_node_count - group_offset;
      const unsigned int this_tile_count =
          std::min(this->max_tile_nodes_, remaining);

      this->tile_group_.push_back(group);
      this->tile_node_point_.push_back(group_point + group_offset);
      this->tile_node_count_.push_back(this_tile_count);

      group_offset += this_tile_count;
      scheduled_nodes += this_tile_count;
    }
  }

  this->tile_count_ = static_cast<unsigned int>(this->tile_node_count_.size());

  if (scheduled_nodes != total_nodes) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::init_tile_schedule: Scheduled tile nodes do "
        "not match total cubature nodes");
  }

  for (unsigned int tile = 0; tile < this->tile_count_; tile++) {
    if (this->tile_node_count_[tile] == 0) {
      throw std::runtime_error("FATAL ERROR: glst_plan::init_tile_schedule: "
                               "Zero-sized tile encountered");
    }

    if (this->tile_node_count_[tile] > this->max_tile_nodes_) {
      throw std::runtime_error("FATAL ERROR: glst_plan::init_tile_schedule: "
                               "Tile exceeds max_tile_nodes");
    }

    const unsigned int group = this->tile_group_[tile];

    if (group >= this->ngroup_) {
      throw std::runtime_error("FATAL ERROR: glst_plan::init_tile_schedule: "
                               "Tile has invalid cubature group");
    }

    const unsigned int group_point =
        static_cast<unsigned int>(this->cubature_->points()[0][group]);

    const unsigned int group_node_count =
        static_cast<unsigned int>(this->cubature_->num_nodes()[0][group]);

    const unsigned int tile_begin = this->tile_node_point_[tile];
    const unsigned int tile_end = tile_begin + this->tile_node_count_[tile];
    const unsigned int group_end = group_point + group_node_count;

    if ((tile_begin < group_point) || (tile_end > group_end)) {
      throw std::runtime_error("FATAL ERROR: glst_plan::init_tile_schedule: "
                               "Tile crosses a cubature group boundary");
    }

    if (tile > 0) {
      const unsigned int previous_group = this->tile_group_[tile - 1];

      if (group < previous_group) {
        throw std::runtime_error("FATAL ERROR: glst_plan::init_tile_schedule: "
                                 "Tile groups are not monotonically ordered");
      }

      if (group == previous_group) {
        const unsigned int previous_end =
            this->tile_node_point_[tile - 1] + this->tile_node_count_[tile - 1];

        if (tile_begin != previous_end) {
          throw std::runtime_error(
              "FATAL ERROR: glst_plan::init_tile_schedule: Non-contiguous tile "
              "inside cubature group");
        }
      }
    }
  }

  return;
}

void glst_plan::print_tile_diagnostics(std::ostream &os) const {
  os << "              Number of GLST tiles: " << tile_count_ << '\n';
  os << "   Maximum cubature nodes per tile: " << max_tile_nodes_ << '\n';
  for (unsigned int group = 0; group < this->ngroup_; group++) {
    os << "      Number of tiles in group " << group << ": "
       << tiles_in_group(group) << '\n';
  }
  return;
}
