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

#include "cuda_utils.hcu"
#include "error_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>

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
    utl::throw_error("erfc_inv", "Inverse of erfc not found after " +
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
      tile_node_count_(), tile_partition_count_(1), tile_partition_idx_(),
      partition_tile_idx_(), partition_tile_node_count_(),
      cell_partition_count_(1), cell_partition_idx_(),
      cell_partition_x_point_(), cell_partition_x_count_(),
      partition_cell_idx_(), partition_left_halo_cell_idx_(),
      partition_right_halo_cell_idx_(), partition_halo_cell_idx_(),
      partition_sr_source_cell_idx_() {}

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
  utl::require(this->cubature_ != nullptr, "glst_plan::cubature_data",
               "Cubature has not been initialized");

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
  utl::require(tile < this->tile_count_, "glst_plan::tile_group",
               "Tile index out of range");

  return this->tile_group_[tile];
}

unsigned int glst_plan::tile_node_point(const unsigned int tile) const {
  utl::require(tile < this->tile_count_, "glst_plan::tile_node_point",
               "Tile index out of range");

  return this->tile_node_point_[tile];
}

unsigned int glst_plan::tile_node_count(const unsigned int tile) const {
  utl::require(tile < this->tile_count_, "glst_plan::tile_node_count",
               "Tile index out of range");

  return this->tile_node_count_[tile];
}

unsigned int glst_plan::tiles_in_group(const unsigned int group) const {
  return static_cast<unsigned int>(
      std::count(this->tile_group_.begin(), this->tile_group_.end(), group));
}

unsigned int glst_plan::tile_partition_count(void) const {
  return this->tile_partition_count_;
}

const std::vector<unsigned int> &glst_plan::tile_partition_idx(void) const {
  return this->tile_partition_idx_;
}

const std::vector<std::vector<unsigned int>> &
glst_plan::partition_tile_idx(void) const {
  return this->partition_tile_idx_;
}

const std::vector<unsigned int> &
glst_plan::partition_tile_node_count(void) const {
  return this->partition_tile_node_count_;
}

unsigned int glst_plan::tile_partition_idx(const unsigned int tile) const {
  utl::require(tile < this->tile_count_, "glst_plan::tile_partition_idx",
               "Tile index out of range");

  return this->tile_partition_idx_[tile];
}

const std::vector<unsigned int> &
glst_plan::partition_tile_idx(const unsigned int tile_partition) const {
  utl::require(tile_partition < this->tile_partition_count_,
               "glst_plan::partition_tile_idx",
               "Tile partition index out of range");

  return this->partition_tile_idx_[tile_partition];
}

unsigned int
glst_plan::partition_tile_node_count(const unsigned int tile_partition) const {
  utl::require(tile_partition < this->tile_partition_count_,
               "glst_plan::partition_tile_node_count",
               "Tile partition index out of range");

  return this->partition_tile_node_count_[tile_partition];
}

unsigned int glst_plan::cell_partition_count(void) const {
  return this->cell_partition_count_;
}

const std::vector<unsigned int> &glst_plan::cell_partition_idx(void) const {
  return this->cell_partition_idx_;
}

const std::vector<unsigned int> &glst_plan::cell_partition_x_point(void) const {
  return this->cell_partition_x_point_;
}

const std::vector<unsigned int> &glst_plan::cell_partition_x_count(void) const {
  return this->cell_partition_x_count_;
}

const std::vector<std::vector<unsigned int>> &
glst_plan::partition_cell_idx(void) const {
  return this->partition_cell_idx_;
}

unsigned int glst_plan::cell_partition_idx(const unsigned int cell) const {
  utl::require(cell < this->ncell_, "glst_plan::cell_partition_idx",
               "Cell index out of range");

  return this->cell_partition_idx_[cell];
}

const std::vector<unsigned int> &
glst_plan::partition_cell_idx(const unsigned int cell_partition) const {
  utl::require(cell_partition < this->cell_partition_count_,
               "glst_plan::partition_cell_idx",
               "Cell partition index out of range");

  return this->partition_cell_idx_[cell_partition];
}

unsigned int
glst_plan::local_cell_count(const unsigned int cell_partition) const {
  return static_cast<unsigned int>(
      this->partition_cell_idx(cell_partition).size());
}

const std::vector<std::vector<unsigned int>> &
glst_plan::partition_left_halo_cell_idx(void) const {
  return this->partition_left_halo_cell_idx_;
}

const std::vector<std::vector<unsigned int>> &
glst_plan::partition_right_halo_cell_idx(void) const {
  return this->partition_right_halo_cell_idx_;
}

const std::vector<std::vector<unsigned int>> &
glst_plan::partition_halo_cell_idx(void) const {
  return this->partition_halo_cell_idx_;
}

const std::vector<std::vector<unsigned int>> &
glst_plan::partition_sr_source_cell_idx(void) const {
  return this->partition_sr_source_cell_idx_;
}

const std::vector<unsigned int> &glst_plan::partition_left_halo_cell_idx(
    const unsigned int cell_partition) const {
  utl::require(cell_partition < this->cell_partition_count_,
               "glst_plan::partition_left_halo_cell_idx",
               "Cell partition index out of range");

  return this->partition_left_halo_cell_idx_[cell_partition];
}

const std::vector<unsigned int> &glst_plan::partition_right_halo_cell_idx(
    const unsigned int cell_partition) const {
  utl::require(cell_partition < this->cell_partition_count_,
               "glst_plan::partition_right_halo_cell_idx",
               "Cell partition index out of range");

  return this->partition_right_halo_cell_idx_[cell_partition];
}

const std::vector<unsigned int> &
glst_plan::partition_halo_cell_idx(const unsigned int cell_partition) const {
  utl::require(cell_partition < this->cell_partition_count_,
               "glst_plan::partition_halo_cell_idx",
               "Cell partition index out of range");

  return this->partition_halo_cell_idx_[cell_partition];
}

const std::vector<unsigned int> &glst_plan::partition_sr_source_cell_idx(
    const unsigned int cell_partition) const {
  utl::require(cell_partition < this->cell_partition_count_,
               "glst_plan::partition_sr_source_cell_idx",
               "Cell partition index out of range");

  return this->partition_sr_source_cell_idx_[cell_partition];
}

unsigned int
glst_plan::first_global_cell(const unsigned int cell_partition) const {
  utl::require(cell_partition < this->cell_partition_count_,
               "glst_plan::first_global_cell",
               "Cell partition index out of range");

  return this->cell_partition_x_point_[cell_partition] * this->ncell_y_ *
         this->ncell_z_;
}

unsigned int
glst_plan::local_cell_from_global_cell(const unsigned int cell_partition,
                                       const unsigned int global_cell) const {
  constexpr std::string_view function_name =
      "glst_plan::local_cell_from_global_cell";

  utl::require(cell_partition < this->cell_partition_count_, function_name,
               "Cell partition index out of range");

  utl::require(global_cell < this->ncell_, function_name,
               "Global cell index out of range");

  const unsigned int yz_count = this->ncell_y_ * this->ncell_z_;
  const unsigned int x = global_cell / yz_count;
  const unsigned int x_point = this->cell_partition_x_point_[cell_partition];
  const unsigned int x_count = this->cell_partition_x_count_[cell_partition];

  utl::require((x >= x_point) && (x < x_point + x_count), function_name,
               "Global cell is outside cell partition");

  return global_cell - this->first_global_cell(cell_partition);
}

unsigned int
glst_plan::global_cell_from_local_cell(const unsigned int cell_partition,
                                       const unsigned int local_cell) const {
  constexpr std::string_view function_name =
      "glst_plan::global_cell_from_local_cell";

  utl::require(cell_partition < this->cell_partition_count_, function_name,
               "Cell partition index out of range");

  utl::require(local_cell < this->local_cell_count(cell_partition),
               function_name, "Local cell index out of range");

  return this->first_global_cell(cell_partition) + local_cell;
}

void glst_plan::global_cell_coords(unsigned int &x, unsigned int &y,
                                   unsigned int &z,
                                   const unsigned int global_cell) const {
  utl::require(global_cell < this->ncell_, "glst_plan::global_cell_coords",
               "Global cell index out of range");

  const unsigned int yz_count = this->ncell_y_ * this->ncell_z_;

  x = global_cell / yz_count;
  y = (global_cell / this->ncell_z_) % this->ncell_y_;
  z = global_cell % this->ncell_z_;

  return;
}

void glst_plan::local_cell_coords(unsigned int &x, unsigned int &y,
                                  unsigned int &z,
                                  const unsigned int cell_partition,
                                  const unsigned int local_cell) const {
  const unsigned int global_cell =
      this->global_cell_from_local_cell(cell_partition, local_cell);

  this->global_cell_coords(x, y, z, global_cell);

  return;
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

  this->init_cell_partitions(1);

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

  int device_count = 0;
  cudaCheck(cudaGetDeviceCount(&device_count));
  utl::require(device_count >= 1, "glst_plan::init_alpha_groups",
               "Could not find and CUDA-capable devices");

  std::vector<unsigned int> grp_r_in((this->ngroup_ > 0) ? this->ngroup_ : 1);
  std::vector<unsigned int> grp_r_out((this->ngroup_ > 0) ? this->ngroup_ : 1);

  grp_r_in[0] = 1;
  for (unsigned int group = 0; group < this->ngroup_; group++) {
    if (group > 0)
      grp_r_in[group] = grp_r_out[group - 1];

    grp_r_out[group] = grp_r_in[group] + this->ncell_alpha_group_[group];
  }

  this->grp_r_in_.resize(device_count);
  this->grp_r_out_.resize(device_count);

  for (int dev = 0; dev < device_count; dev++) {
    cudaCheck(cudaSetDevice(dev));
    this->grp_r_in_[dev] = grp_r_in;
    this->grp_r_out_[dev] = grp_r_out;
  }

  return;
}

void glst_plan::init_cubature(const double tol) {
  utl::require(this->ngroup_ > 0, "glst_plan::init_cubature",
               "Alpha groups have not been initialized");

  this->cubature_ = std::make_unique<cubature>(tol, this->ngroup_, this->rmax_,
                                               this->alpha_, this->zcut_);

  return;
}

void glst_plan::init_tile_schedule(const unsigned int max_tile_nodes) {
  constexpr std::string_view function_name = "glst_plan::init_tile_schedule";

  utl::require(max_tile_nodes > 0, function_name, "max_tile_nodes must be > 0");

  utl::require(this->cubature_ != nullptr, function_name,
               "Cubature has not been initialized");

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

    utl::require((group_point <= total_nodes) &&
                     (group_node_count <= total_nodes - group_point),
                 function_name, "Cubature group node range is out of bounds");

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

  this->init_tile_partitions(1);

  return;
}

void glst_plan::init_tile_partitions(const unsigned int tile_partition_count) {
  constexpr std::string_view function_name = "glst_plan::init_tile_partitions";

  utl::require(tile_partition_count > 0, function_name,
               "tile_partition_count must be > 0");

  utl::require(this->tile_count_ > 0, function_name,
               "Tile schedule has not been initialized");

  this->tile_partition_count_ = tile_partition_count;

  this->tile_partition_idx_.assign(this->tile_count_, 0);
  this->partition_tile_idx_.clear();
  this->partition_tile_idx_.resize(tile_partition_count);
  this->partition_tile_node_count_.assign(tile_partition_count, 0);

  for (unsigned int tile = 0; tile < this->tile_count_; tile++) {
    const unsigned int tile_partition = tile % tile_partition_count;

    this->tile_partition_idx_[tile] = tile_partition;
    this->partition_tile_idx_[tile_partition].push_back(tile);
    this->partition_tile_node_count_[tile_partition] +=
        this->tile_node_count_[tile];
  }

  this->validate();

  return;
}

void glst_plan::init_cell_partitions(const unsigned int cell_partition_count) {
  constexpr std::string_view function_name = "glst_plan::init_cell_partitions";

  utl::require(cell_partition_count > 0, function_name,
               "cell_partition_count must be > 0");

  const unsigned int ncell_x = this->ncell_x_;
  const unsigned int ncell_y = this->ncell_y_;
  const unsigned int ncell_z = this->ncell_z_;
  const unsigned int ncell = this->ncell_;

  utl::require((ncell_x > 0) && (ncell_y > 0) && (ncell_z > 0), function_name,
               "Cell dimensions are invalid");

  this->cell_partition_count_ = cell_partition_count;
  this->cell_partition_idx_.assign(ncell, 0);
  this->cell_partition_x_point_.assign(cell_partition_count, 0);
  this->cell_partition_x_count_.assign(cell_partition_count, 0);

  this->partition_cell_idx_.clear();
  this->partition_cell_idx_.resize(cell_partition_count);

  std::vector<unsigned int> cell_visit_count(ncell, 0);

  const unsigned int base_x_count = ncell_x / cell_partition_count;
  const unsigned int rem_x_count = ncell_x % cell_partition_count;

  for (unsigned int part = 0; part < cell_partition_count; part++) {
    const unsigned int x_count =
        base_x_count + ((part < rem_x_count) ? 1u : 0u);
    const unsigned int x_point =
        part * base_x_count + ((part < rem_x_count) ? part : rem_x_count);

    this->cell_partition_x_point_[part] = x_point;
    this->cell_partition_x_count_[part] = x_count;

    std::vector<unsigned int> &cells = this->partition_cell_idx_[part];

    const std::size_t reserve_count = static_cast<std::size_t>(x_count) *
                                      static_cast<std::size_t>(ncell_y) *
                                      static_cast<std::size_t>(ncell_z);
    cells.reserve(reserve_count);

    for (unsigned int x0 = 0; x0 < x_count; x0++) {
      const unsigned int x = x_point + x0;

      for (unsigned int y = 0; y < ncell_y; y++) {
        for (unsigned int z = 0; z < ncell_z; z++) {
          const unsigned int cell = (x * ncell_y + y) * ncell_z + z;

          utl::require(cell < ncell, function_name,
                       "Computed cell index is out of range");

          this->cell_partition_idx_[cell] = part;
          cell_visit_count[cell]++;
          cells.push_back(cell);
        }
      }
    }
  }

  for (unsigned int cell = 0; cell < ncell; cell++) {
    utl::require(
        cell_visit_count[cell] == 1, function_name,
        "Every global cell must be assigned to exactly one cell partition");
  }

  this->init_short_range_halo_plan();

  return;
}

void glst_plan::validate(void) const {
  constexpr std::string_view function_name = "glst_plan::validate";

  const std::size_t expected_ncell = static_cast<std::size_t>(this->ncell_x_) *
                                     static_cast<std::size_t>(this->ncell_y_) *
                                     static_cast<std::size_t>(this->ncell_z_);

  utl::require(expected_ncell <= static_cast<std::size_t>(
                                     std::numeric_limits<unsigned int>::max()),
               function_name, "ncell product exceeds unsigned int range");

  utl::require(static_cast<std::size_t>(this->ncell_) == expected_ncell,
               function_name,
               "ncell does not match ncell_x * ncell_y * ncell_z");

  utl::require(this->cubature_ != nullptr, function_name,
               "Cubature has not been initialized");

  utl::require(this->ngroup_ == this->cubature_->num_cubatures(), function_name,
               "ngroup does not match cubature num_cubatures");

  const auto &points = this->cubature_->points();
  const auto &num_nodes = this->cubature_->num_nodes();
  const auto &x = this->cubature_->x();
  const auto &y = this->cubature_->y();
  const auto &z = this->cubature_->z();
  const auto &w = this->cubature_->w();
  const auto &group_array = this->cubature_->group();

  utl::require(!this->grp_r_in_.empty() && !this->grp_r_out_.empty(),
               function_name, "Alpha-group radius metadata is empty");

  utl::require((this->grp_r_in_.size() == points.size()) &&
                   (this->grp_r_out_.size() == points.size()),
               function_name,
               "Alpha-group radius metadata device count does not match "
               "cubature device count");

  for (std::size_t dev = 0; dev < this->grp_r_in_.size(); dev++) {
    utl::require((this->grp_r_in_[dev].size() ==
                  static_cast<std::size_t>(this->ngroup_)) &&
                     (this->grp_r_out_[dev].size() ==
                      static_cast<std::size_t>(this->ngroup_)),
                 function_name,
                 "Alpha-group radius metadata sizes do not match ngroup");
  }

  utl::require(
      (this->ncell_alpha_group_.size() ==
       static_cast<std::size_t>(this->ngroup_)) &&
          (this->rmax_.size() == static_cast<std::size_t>(this->ngroup_)) &&
          (this->alpha_.size() == static_cast<std::size_t>(this->ngroup_)) &&
          (this->zcut_.size() == static_cast<std::size_t>(this->ngroup_)),
      function_name, "Alpha-group host metadata sizes do not match ngroup");

  utl::require(!points.empty() && !num_nodes.empty() && !x.empty() &&
                   !y.empty() && !z.empty() && !w.empty() &&
                   !group_array.empty(),
               function_name, "Cubature storage is empty");

  utl::require(
      (points[0].size() >= static_cast<std::size_t>(this->ngroup_)) &&
          (num_nodes[0].size() >= static_cast<std::size_t>(this->ngroup_)),
      function_name, "Cubature group metadata is smaller than ngroup");

  const std::size_t total_nodes =
      static_cast<std::size_t>(this->cubature_->tot_num_nodes());

  utl::require(total_nodes > 0, function_name, "Cubature has zero nodes");

  utl::require(
      (x[0].size() == total_nodes) && (y[0].size() == total_nodes) &&
          (z[0].size() == total_nodes) && (w[0].size() == total_nodes) &&
          (group_array[0].size() == total_nodes),
      function_name, "Cubature node arrays do not match total node count");

  std::size_t group_node_sum = 0;
  std::vector<std::size_t> group_begin(this->ngroup_);
  std::vector<std::size_t> group_end(this->ngroup_);

  for (unsigned int group = 0; group < this->ngroup_; group++) {
    const std::size_t group_point = static_cast<std::size_t>(points[0][group]);
    const std::size_t group_count =
        static_cast<std::size_t>(num_nodes[0][group]);

    utl::require((group_point <= total_nodes) &&
                     (group_count <= total_nodes - group_point),
                 function_name, "Cubature group node range is out of bounds");

    group_begin[group] = group_point;
    group_end[group] = group_point + group_count;
    group_node_sum += group_count;
  }

  utl::require(group_node_sum == total_nodes, function_name,
               "Sum of cubature group node counts does not match total "
               "cubature node count");

  utl::require(this->max_tile_nodes_ > 0, function_name, "max_tile_nodes is 0");

  utl::require((static_cast<std::size_t>(this->tile_count_) ==
                this->tile_group_.size()) &&
                   (static_cast<std::size_t>(this->tile_count_) ==
                    this->tile_node_point_.size()) &&
                   (static_cast<std::size_t>(this->tile_count_) ==
                    this->tile_node_count_.size()),
               function_name, "Tile metadata sizes do not match tile_count");

  std::size_t tile_node_sum = 0;
  std::vector<std::size_t> next_node_point(this->ngroup_);
  for (unsigned int group = 0; group < this->ngroup_; group++)
    next_node_point[group] = group_begin[group];

  for (unsigned int tile = 0; tile < this->tile_count_; tile++) {
    const unsigned int tile_group = this->tile_group_[tile];

    utl::require(this->tile_node_count_[tile] > 0, function_name,
                 "Zero-sized tile encountered");

    utl::require(this->tile_node_count_[tile] <= this->max_tile_nodes_,
                 function_name, "Tile exceeds max_tile_nodes");

    utl::require(tile_group < this->ngroup_, function_name,
                 "Tile has invalid cubature group");

    utl::require((tile == 0) || (tile_group >= this->tile_group_[tile - 1]),
                 function_name, "Tile groups are not monotonically ordered");

    const std::size_t tile_begin =
        static_cast<std::size_t>(this->tile_node_point_[tile]);
    const std::size_t tile_count =
        static_cast<std::size_t>(this->tile_node_count_[tile]);

    utl::require((tile_begin <= total_nodes) &&
                     (tile_count <= total_nodes - tile_begin),
                 function_name, "Tile node range is out of bounds");

    const std::size_t tile_end = tile_begin + tile_count;

    utl::require((tile_begin >= group_begin[tile_group]) &&
                     (tile_end <= group_end[tile_group]),
                 function_name, "Tile crosses a cubature group boundary");

    utl::require(tile_begin == next_node_point[tile_group], function_name,
                 "Non-contiguous tile coverage inside cubature group");

    next_node_point[tile_group] = tile_end;
    tile_node_sum += tile_count;
  }

  utl::require(
      tile_node_sum == total_nodes, function_name,
      "Sum of tile node counts does not match total cubature node count");

  for (unsigned int group = 0; group < this->ngroup_; group++) {
    if (next_node_point[group] != group_end[group]) {
      utl::throw_error(
          function_name,
          "Tile schedule does not cover every cubature node in group " +
              std::to_string(group));
    }
  }

  utl::require(this->tile_partition_count_ > 0, function_name,
               "tile_partition_count == 0");

  utl::require(this->tile_partition_idx_.size() ==
                   static_cast<std::size_t>(this->tile_count_),
               function_name,
               "tile_partition_idx size does not match tile_count");

  utl::require(this->partition_tile_idx_.size() ==
                   static_cast<std::size_t>(this->tile_partition_count_),
               function_name,
               "partition_tile_idx size does not match tile_partition_count");

  utl::require(
      this->partition_tile_node_count_.size() ==
          static_cast<std::size_t>(this->tile_partition_count_),
      function_name,
      "partition_tile_node_count size does not match tile_partition_count");

  std::vector<unsigned int> tile_visit_count(this->tile_count_, 0);
  std::vector<unsigned int> observed_partition_node_count(
      this->tile_partition_count_, 0);

  for (unsigned int tile = 0; tile < this->tile_count_; tile++) {
    const unsigned int partition = this->tile_partition_idx_[tile];

    utl::require(partition < this->tile_partition_count_, function_name,
                 "Tile has invalid tile partition");

    utl::require(partition == tile % this->tile_partition_count_, function_name,
                 "Tile partition assignment is not deterministic");
  }

  for (unsigned int partition = 0; partition < this->tile_partition_count_;
       partition++) {
    const std::vector<unsigned int> &tiles =
        this->partition_tile_idx_[partition];

    for (std::size_t i = 0; i < tiles.size(); i++) {
      const unsigned int tile = tiles[i];

      utl::require(tile < this->tile_count_, function_name,
                   "Partition tile list contains out-of-range tile");

      utl::require(this->tile_partition_idx_[tile] == partition, function_name,
                   "Partition tile list disagrees with tile_partition_idx");

      utl::require((i == 0) || (tiles[i] > tiles[i - 1]), function_name,
                   "Partition tile list is not in deterministic order");

      tile_visit_count[tile]++;
      observed_partition_node_count[partition] += this->tile_node_count_[tile];
    }

    utl::require(observed_partition_node_count[partition] ==
                     this->partition_tile_node_count_[partition],
                 function_name, "Partition node count is inconsistent");
  }

  std::size_t partition_node_sum = 0;
  for (unsigned int partition = 0; partition < this->tile_partition_count_;
       partition++) {
    partition_node_sum +=
        static_cast<std::size_t>(this->partition_tile_node_count_[partition]);
  }

  utl::require(partition_node_sum == total_nodes, function_name,
               "Sum of tile partition node counts does not match total "
               "cubature node count");

  for (unsigned int tile = 0; tile < this->tile_count_; tile++) {
    utl::require(tile_visit_count[tile] == 1, function_name,
                 "Every global tile must belong to exactly one tile partition");
  }

  utl::require(this->cell_partition_count_ > 0, function_name,
               "cell_partition_count_ == 0");

  utl::require(this->cell_partition_idx_.size() ==
                   static_cast<std::size_t>(this->ncell_),
               function_name, "cell_partition_idx size does not match ncell");

  utl::require(
      this->cell_partition_x_point_.size() ==
          static_cast<std::size_t>(this->cell_partition_count_),
      function_name,
      "cell_partition_x_point size does not match cell_partition_count");

  utl::require(
      this->cell_partition_x_count_.size() ==
          static_cast<std::size_t>(this->cell_partition_count_),
      function_name,
      "cell_partition_x_count size does not match cell_partition_count");

  utl::require(this->partition_cell_idx_.size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name,
               "partition_cell_idx size does not match cell_partition_count");

  std::vector<unsigned int> cell_visit_count(this->ncell_, 0);

  for (unsigned int part = 0; part < this->cell_partition_count_; part++) {
    const unsigned int x_point = this->cell_partition_x_point_[part];
    const unsigned int x_count = this->cell_partition_x_count_[part];

    const unsigned int yz_count = this->ncell_y_ * this->ncell_z_;
    const std::size_t expected_cell_count =
        static_cast<std::size_t>(x_count) * static_cast<std::size_t>(yz_count);

    utl::require(this->partition_cell_idx_[part].size() == expected_cell_count,
                 function_name,
                 "Partition cell list size does not match x-range size");

    utl::require((x_point <= this->ncell_x_) &&
                     (x_count <= this->ncell_x_ - x_point),
                 function_name, "Cell partition x-range is out of bounds");

    const std::vector<unsigned int> &cells = this->partition_cell_idx_[part];

    for (std::size_t i = 0; i < cells.size(); i++) {
      const unsigned int cell = cells[i];

      utl::require(cell < this->ncell_, function_name,
                   "Partition cell list contains out-of-range cell");

      utl::require(this->cell_partition_idx_[cell] == part, function_name,
                   "partition_cell_idx disagrees with cell_partition_idx");

      const unsigned int x = cell / (this->ncell_y_ * this->ncell_z_);
      utl::require((x >= x_point) && (x < x_point + x_count), function_name,
                   "Partition cell is outside its x-range");

      const unsigned int local_cell =
          this->local_cell_from_global_cell(part, cell);

      utl::require(local_cell == static_cast<unsigned int>(i), function_name,
                   "Local cell index is not deterministic");

      const unsigned int round_trip_cell =
          this->global_cell_from_local_cell(part, local_cell);

      utl::require(round_trip_cell == cell, function_name,
                   "Local/global cell round-trip failed");

      unsigned int cx = 0, cy = 0, cz = 0;
      this->global_cell_coords(cx, cy, cz, cell);

      const unsigned int rebuilt_cell =
          (cx * this->ncell_y_ + cy) * this->ncell_z_ + cz;

      utl::require(rebuilt_cell == cell, function_name,
                   "Global cell coordinate conversion failed");

      cell_visit_count[cell]++;
    }
  }

  for (unsigned int cell = 0; cell < this->ncell_; cell++) {
    utl::require(cell_visit_count[cell] == 1, function_name,
                 "Every global cell must belong to exactly one cell partition");
  }

  utl::require(this->partition_left_halo_cell_idx_.size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name,
               "Left halo metadata size does not match cell partitions");

  utl::require(this->partition_right_halo_cell_idx_.size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name,
               "Right halo metadata size does not match cell partitions");

  utl::require(this->partition_halo_cell_idx_.size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name,
               "Halo metadata size does not match cell partitions");

  utl::require(
      this->partition_sr_source_cell_idx_.size() ==
          static_cast<std::size_t>(this->cell_partition_count_),
      function_name,
      "Short-range source metadata size does not match cell partitions");

  const unsigned int yz_count = this->ncell_y_ * this->ncell_z_;

  for (unsigned int part = 0; part < this->cell_partition_count_; part++) {
    const unsigned int x_point = this->cell_partition_x_point_[part];
    const unsigned int x_count = this->cell_partition_x_count_[part];
    const unsigned int x_end = x_point + x_count;

    const std::vector<unsigned int> &owned_cells =
        this->partition_cell_idx_[part];
    const std::vector<unsigned int> &left_cells =
        this->partition_left_halo_cell_idx_[part];
    const std::vector<unsigned int> &right_cells =
        this->partition_right_halo_cell_idx_[part];
    const std::vector<unsigned int> &halo_cells =
        this->partition_halo_cell_idx_[part];
    const std::vector<unsigned int> &source_cells =
        this->partition_sr_source_cell_idx_[part];

    const std::size_t expected_left_count =
        ((x_count > 0) && (x_point > 0)) ? static_cast<std::size_t>(yz_count)
                                         : 0;
    const std::size_t expected_right_count =
        ((x_count > 0) && (x_end < this->ncell_x_))
            ? static_cast<std::size_t>(yz_count)
            : 0;

    utl::require(left_cells.size() == expected_left_count, function_name,
                 "Left halo plane size is incorrect");

    utl::require(right_cells.size() == expected_right_count, function_name,
                 "Right halo plane size is incorrect");

    utl::require(halo_cells.size() == left_cells.size() + right_cells.size(),
                 function_name, "Combined halo size is inconsistent");

    utl::require(source_cells.size() == owned_cells.size() + halo_cells.size(),
                 function_name, "Short-range source size is inconsistent");

    for (std::size_t i = 0; i < owned_cells.size(); i++) {
      utl::require(source_cells[i] == owned_cells[i], function_name,
                   "Short-range source list does not start with owned cells");
    }

    for (std::size_t i = 0; i < halo_cells.size(); i++) {
      utl::require(source_cells[owned_cells.size() + i] == halo_cells[i],
                   function_name,
                   "Short-range source list does not append halo cells");
    }

    for (std::size_t i = 0; i < left_cells.size(); i++) {
      unsigned int x = 0, y = 0, z = 0;
      this->global_cell_coords(x, y, z, left_cells[i]);

      utl::require(x == x_point - 1, function_name,
                   "Left halo cell is not on the adjacent x plane");

      utl::require(this->cell_partition_idx_[left_cells[i]] != part,
                   function_name,
                   "Left halo cell is owned by the target partition");
    }

    for (std::size_t i = 0; i < right_cells.size(); i++) {
      unsigned int x = 0, y = 0, z = 0;
      this->global_cell_coords(x, y, z, right_cells[i]);

      utl::require(x == x_end, function_name,
                   "Right halo cell is not on the adjacent x plane");

      utl::require(this->cell_partition_idx_[right_cells[i]] != part,
                   function_name,
                   "Right halo cell is owned by the target partition");
    }

    if (this->cell_partition_count_ == 1) {
      utl::require(halo_cells.empty(), function_name,
                   "Single-GPU partition has halo cells");
    }
  }

  return;
}

void glst_plan::print_tile_diagnostics(std::ostream &os) const {
  os << "              Number of GLST tiles: " << this->tile_count_
     << std::endl;
  os << "   Maximum cubature nodes per tile: " << this->max_tile_nodes_
     << std::endl;

  for (unsigned int group = 0; group < this->ngroup_; group++) {
    os << "        Number of tiles in group " << group << ": "
       << this->tiles_in_group(group) << std::endl;
  }

  os << "        Number of tile partitions: " << this->tile_partition_count_
     << std::endl;

  for (unsigned int partition = 0; partition < this->tile_partition_count_;
       partition++) {
    os << "        Number of tiles in tile partition " << partition << ": "
       << this->partition_tile_idx_[partition].size() << " ("
       << this->partition_tile_node_count_[partition] << " nodes)" << std::endl;
  }

  return;
}

void glst_plan::init_short_range_halo_plan(void) {
  const unsigned int yz_count = this->ncell_y_ * this->ncell_z_;

  this->partition_left_halo_cell_idx_.clear();
  this->partition_right_halo_cell_idx_.clear();
  this->partition_halo_cell_idx_.clear();
  this->partition_sr_source_cell_idx_.clear();

  this->partition_left_halo_cell_idx_.resize(this->cell_partition_count_);
  this->partition_right_halo_cell_idx_.resize(this->cell_partition_count_);
  this->partition_halo_cell_idx_.resize(this->cell_partition_count_);
  this->partition_sr_source_cell_idx_.resize(this->cell_partition_count_);

  for (unsigned int partition = 0; partition < this->cell_partition_count_;
       partition++) {
    const unsigned int x_point = this->cell_partition_x_point_[partition];
    const unsigned int x_count = this->cell_partition_x_count_[partition];
    const unsigned int x_end = x_point + x_count;

    std::vector<unsigned int> &left_cells =
        this->partition_left_halo_cell_idx_[partition];
    std::vector<unsigned int> &right_cells =
        this->partition_right_halo_cell_idx_[partition];
    std::vector<unsigned int> &halo_cells =
        this->partition_halo_cell_idx_[partition];
    std::vector<unsigned int> &source_cells =
        this->partition_sr_source_cell_idx_[partition];

    if (x_count == 0)
      continue;

    left_cells.reserve(yz_count);
    right_cells.reserve(yz_count);

    if (x_point > 0) {
      const unsigned int x = x_point - 1;
      for (unsigned int y = 0; y < this->ncell_y_; y++) {
        for (unsigned int z = 0; z < this->ncell_z_; z++)
          left_cells.push_back((x * this->ncell_y_ + y) * this->ncell_z_ + z);
      }
    }

    if (x_end < this->ncell_x_) {
      const unsigned int x = x_end;
      for (unsigned int y = 0; y < this->ncell_y_; y++) {
        for (unsigned int z = 0; z < this->ncell_z_; z++)
          right_cells.push_back((x * this->ncell_y_ + y) * this->ncell_z_ + z);
      }
    }

    halo_cells.reserve(left_cells.size() + right_cells.size());
    halo_cells.insert(halo_cells.end(), left_cells.begin(), left_cells.end());
    halo_cells.insert(halo_cells.end(), right_cells.begin(), right_cells.end());

    const std::vector<unsigned int> &owned_cells =
        this->partition_cell_idx_[partition];

    source_cells.reserve(owned_cells.size() + halo_cells.size());
    source_cells.insert(source_cells.end(), owned_cells.begin(),
                        owned_cells.end());
    source_cells.insert(source_cells.end(), halo_cells.begin(),
                        halo_cells.end());
  }

  return;
}
