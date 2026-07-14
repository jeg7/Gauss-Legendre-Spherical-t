// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "glst_plan.hcu"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

const std::vector<cuda_container<unsigned int>> &
glst_plan::grp_r_in(void) const {
  return this->grp_r_in_;
}

const std::vector<cuda_container<unsigned int>> &
glst_plan::grp_r_out(void) const {
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

const cubature &glst_plan::cubature_data(void) const {
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

  this->grp_r_in_.resize(1);
  this->grp_r_out_.resize(1);

  this->grp_r_in_[0].resize((this->ngroup_ > 0) ? this->ngroup_ : 1);
  this->grp_r_out_[0].resize((this->ngroup_ > 0) ? this->ngroup_ : 1);

  this->grp_r_in_[0][0] = 1;
  for (unsigned int group = 0; group < this->ngroup_; group++) {
    if (group > 0)
      this->grp_r_in_[0][group] = this->grp_r_out_[0][group - 1];
    this->grp_r_out_[0][group] =
        this->grp_r_in_[0][group] + this->ncell_alpha_group_[group];
  }

  this->grp_r_in_[0].transfer_to_device();
  this->grp_r_out_[0].transfer_to_device();

  return;
}

void glst_plan::init_cubature(const double tol) {
  if (this->ngroup_ == 0) {
    throw std::runtime_error("FATAL ERROR: glst_plan::init_cubature: Alpha "
                             "groups have not been initialized");
  }

  this->cubature_ = std::make_unique<cubature>(tol, this->ngroup_, this->rmax_,
                                               this->alpha_, this->zcut_);

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
    }
  }

  this->tile_count_ = static_cast<unsigned int>(this->tile_node_count_.size());

  this->validate();

  return;
}

void glst_plan::validate(void) const {
  const std::size_t expected_ncell = static_cast<std::size_t>(this->ncell_x_) *
                                     static_cast<std::size_t>(this->ncell_y_) *
                                     static_cast<std::size_t>(this->ncell_z_);

  if (expected_ncell >
      static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw std::runtime_error("FATAL ERROR: glst_plan::validate: ncell product "
                             "exceeds unsigned int range");
  }

  if (static_cast<std::size_t>(this->ncell_) != expected_ncell) {
    throw std::runtime_error("FATAL ERROR: glst_plan::validate: ncell does not "
                             "match ncell_x * ncell_y * ncell_z");
  }

  if (this->cubature_ == nullptr) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::validate: Cubature has not been initialized");
  }

  if (this->ngroup_ != this->cubature_->num_cubatures()) {
    throw std::runtime_error("FATAL ERROR: glst_plan::validate: ngroup does "
                             "not match cubature num_cubatures");
  }

  if ((this->grp_r_in_[0].size() != static_cast<std::size_t>(this->ngroup_)) ||
      (this->grp_r_out_[0].size() != static_cast<std::size_t>(this->ngroup_)) ||
      (this->ncell_alpha_group_.size() !=
       static_cast<std::size_t>(this->ngroup_)) ||
      (this->rmax_.size() != static_cast<std::size_t>(this->ngroup_)) ||
      (this->alpha_.size() != static_cast<std::size_t>(this->ngroup_)) ||
      (this->zcut_.size() != static_cast<std::size_t>(this->ngroup_))) {
    throw std::runtime_error("FATAL ERROR: glst_plan::validate: Alpha-group "
                             "metadata sizes do not match ngroup");
  }

  const auto &points = this->cubature_->points();
  const auto &num_nodes = this->cubature_->num_nodes();
  const auto &x = this->cubature_->x();
  const auto &y = this->cubature_->y();
  const auto &z = this->cubature_->z();
  const auto &w = this->cubature_->w();
  const auto &group_array = this->cubature_->group();

  if (points.empty() || num_nodes.empty() || x.empty() || y.empty() ||
      z.empty() || w.empty() || group_array.empty()) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::validate: Cubature storage is empty");
  }

  if ((points[0].size() < static_cast<std::size_t>(this->ngroup_)) ||
      (num_nodes[0].size() < static_cast<std::size_t>(this->ngroup_))) {
    throw std::runtime_error("FATAL ERROR: glst_plan::validate: Cubature group "
                             "metadata is smaller than ngroup");
  }

  const std::size_t total_nodes =
      static_cast<std::size_t>(this->cubature_->tot_num_nodes());

  if (total_nodes == 0) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::validate: Cubature has zero nodes");
  }

  if ((x[0].size() != total_nodes) || (y[0].size() != total_nodes) ||
      (z[0].size() != total_nodes) || (w[0].size() != total_nodes) ||
      (group_array[0].size() != total_nodes)) {
    throw std::runtime_error("FATAL ERROR: glst_plan::validate: Cubature node "
                             "arrays do not match total node count");
  }

  std::size_t group_node_sum = 0;
  std::vector<std::size_t> group_begin(this->ngroup_);
  std::vector<std::size_t> group_end(this->ngroup_);

  for (unsigned int group = 0; group < this->ngroup_; group++) {
    const std::size_t group_point = static_cast<std::size_t>(points[0][group]);
    const std::size_t group_count =
        static_cast<std::size_t>(num_nodes[0][group]);

    if ((group_point > total_nodes) ||
        (group_count > total_nodes - group_point)) {
      throw std::runtime_error("FATAL ERROR: glst_plan::validate: Cubature "
                               "group node range is out of bounds");
    }

    group_begin[group] = group_point;
    group_end[group] = group_point + group_count;
    group_node_sum += group_count;
  }

  if (group_node_sum != total_nodes) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::validate: Sum of cubature group node counts "
        "does not match total cubature node count");
  }

  if (this->max_tile_nodes_ == 0) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::validate: max_tile_nodes is 0");
  }

  if ((static_cast<std::size_t>(this->tile_count_) !=
       this->tile_group_.size()) ||
      (static_cast<std::size_t>(this->tile_count_) !=
       this->tile_node_point_.size()) ||
      (static_cast<std::size_t>(this->tile_count_) !=
       this->tile_node_count_.size())) {
    throw std::runtime_error("FATAL ERROR: glst_plan::validate: Tile metadata "
                             "sizes do not match tile_count");
  }

  std::size_t tile_node_sum = 0;
  std::vector<std::size_t> next_node_point(this->ngroup_);
  for (unsigned int group = 0; group < this->ngroup_; group++)
    next_node_point[group] = group_begin[group];

  for (unsigned int tile = 0; tile < this->tile_count_; tile++) {
    const unsigned int tile_group = this->tile_group_[tile];

    if (this->tile_node_count_[tile] == 0) {
      throw std::runtime_error(
          "FATAL ERROR: glst_plan::validate: Zero-sized tile encountered");
    }

    if (this->tile_node_count_[tile] > this->max_tile_nodes_) {
      throw std::runtime_error(
          "FATAL ERROR: glst_plan::validate: Tile exceeds max_tile_nodes");
    }

    if (tile_group >= this->ngroup_) {
      throw std::runtime_error(
          "FATAL ERROR: glst_plan::validate: Tile has invalid cubature group");
    }

    if ((tile > 0) && (tile_group < this->tile_group_[tile - 1])) {
      throw std::runtime_error("FATAL ERROR: glst_plan::validate: Tile groups "
                               "are not monotonically ordered");
    }

    const std::size_t tile_begin =
        static_cast<std::size_t>(this->tile_node_point_[tile]);
    const std::size_t tile_count =
        static_cast<std::size_t>(this->tile_node_count_[tile]);

    if ((tile_begin > total_nodes) || (tile_count > total_nodes - tile_begin)) {
      throw std::runtime_error(
          "FATAL ERROR: glst_plan::validate: Tile node range is out of bounds");
    }

    const std::size_t tile_end = tile_begin + tile_count;

    if ((tile_begin < group_begin[tile_group]) ||
        (tile_end > group_end[tile_group])) {
      throw std::runtime_error("FATAL ERROR: glst_plan::validate: Tile crosses "
                               "a cubature group boundary");
    }

    if (tile_begin != next_node_point[tile_group]) {
      throw std::runtime_error(
          "FATAL ERROR: glst_plan::validate: Non-contiguous tile coverage "
          "inside cubature group");
    }

    next_node_point[tile_group] = tile_end;
    tile_node_sum += tile_count;
  }

  if (tile_node_sum != total_nodes) {
    throw std::runtime_error(
        "FATAL ERROR: glst_plan::validate: Sum of tile node counts does not "
        "match total cubature node count");
  }

  for (unsigned int group = 0; group < this->ngroup_; group++) {
    if (next_node_point[group] != group_end[group]) {
      throw std::runtime_error(
          "FATAL ERROR: glst_plan::validate: Tile schedule does not cover "
          "every cubature node in group " +
          std::to_string(group));
    }
  }

  return;
}

void glst_plan::print_tile_diagnostics(std::ostream &os) const {
  const std::size_t tile_buffer_count =
      static_cast<std::size_t>(this->ncell_) *
      static_cast<std::size_t>(this->max_tile_nodes_);

  // Initial tiled workspace stores four tile-sized double buffers: sf_re,
  // sf_im, rmt_sum_re, rmt_sum_im
  const std::size_t tile_buffer_bytes =
      static_cast<std::size_t>(4) * tile_buffer_count * sizeof(double);

  const double tile_buffer_mib =
      static_cast<double>(tile_buffer_bytes) / (1024.0 * 1024.0);

  os << "              Number of GLST tiles: " << tile_count_ << '\n';
  os << "   Maximum cubature nodes per tile: " << max_tile_nodes_ << '\n';
  os << "       Tile-buffer memory estimate: " << tile_buffer_mib << " MiB ("
     << tile_buffer_bytes << " bytes)" << '\n';
  for (unsigned int group = 0; group < this->ngroup_; group++) {
    os << "        Number of tiles in group " << group << ": "
       << tiles_in_group(group) << '\n';
  }

  return;
}
