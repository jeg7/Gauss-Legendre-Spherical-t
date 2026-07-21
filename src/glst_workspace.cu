// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "glst_workspace.hcu"

#include "cuda_utils.hcu"
#include "error_utils.hpp"

#include <cub/cub.cuh>
#include <limits>
#include <string>
#include <string_view>

inline static std::size_t checked_mul(const std::size_t a, const std::size_t b,
                                      const std::string_view label) {
  if ((a != 0) && (b > std::numeric_limits<std::size_t>::max() / a)) {
    utl::throw_error("checked_mul",
                     "Overflow while computing " + std::string(label));
  }
  return (a * b);
}

glst_workspace::glst_workspace(void)
    : max_atom_capacity_(0), max_cell_capacity_(0), tile_node_capacity_(0),
      max_sf_tile_buffer_capacity_(0), max_sf_exchange_tile_buffer_capacity_(0),
      max_rmt_tile_buffer_capacity_(0),
      max_prefix_partition_total_buffer_capacity_(0),
      max_prefix_base_buffer_capacity_(0), atom_capacity_(), cell_capacity_(),
      sf_tile_buffer_capacity_(), sf_exchange_tile_buffer_capacity_(),
      rmt_tile_buffer_capacity_(), prefix_partition_total_buffer_capacity_(),
      prefix_base_buffer_capacity_(), prefix_plane_slot_capacity_(),
      owned_atom_count_(), source_atom_count_(), sr_source_cell_capacity_(),
      idx_(), sorted_idx_(), rx_(), ry_(), rz_(), qc_(), packets_(),
      sorted_packets_(), atom_cell_idx_(), atom_cell_sorted_idx_(), fx_(),
      fy_(), fz_(), en_(), cell_atom_point_(), cell_atom_count_(),
      max_atoms_cell_(), sr_source_cell_atom_point_(),
      sr_source_cell_atom_count_(), sf_re_(), sf_im_(), sf_exchange_re_(),
      sf_exchange_im_(), prefix_partition_total_re_(),
      prefix_partition_total_im_(), prefix_base_re_(), prefix_base_im_(),
      prefix_plane_slot_(), rmt_sum_re_(), rmt_sum_im_(), cub_work_buffer_(),
      cub_work_buffer_size_() {}

glst_workspace::glst_workspace(const glst_plan &plan, const int device_count)
    : glst_workspace() {
  this->init(plan, device_count);
}

glst_workspace::glst_workspace(
    const glst_plan &plan, const std::vector<unsigned int> &dev_cell_partition,
    const int device_count, const bool use_full_sf_buffer,
    const bool use_distributed_prefix,
    const unsigned int sf_exchange_chunk_x_count)
    : glst_workspace() {
  this->init(plan, dev_cell_partition, device_count, use_full_sf_buffer,
             use_distributed_prefix, sf_exchange_chunk_x_count);
}

glst_workspace::~glst_workspace(void) { this->deallocate_cub(); }

std::size_t glst_workspace::max_atom_capacity(void) const {
  return this->max_atom_capacity_;
}

std::size_t glst_workspace::max_cell_capacity(void) const {
  return this->max_cell_capacity_;
}

std::size_t glst_workspace::tile_node_capacity(void) const {
  return this->tile_node_capacity_;
}

std::size_t glst_workspace::max_sf_tile_buffer_capacity(void) const {
  return this->max_sf_tile_buffer_capacity_;
}

std::size_t glst_workspace::max_sf_exchange_tile_buffer_capacity(void) const {
  return this->max_sf_exchange_tile_buffer_capacity_;
}

std::size_t glst_workspace::max_rmt_tile_buffer_capacity(void) const {
  return this->max_rmt_tile_buffer_capacity_;
}

std::size_t
glst_workspace::max_prefix_partition_total_buffer_capacity(void) const {
  return this->max_prefix_partition_total_buffer_capacity_;
}

std::size_t glst_workspace::max_prefix_base_buffer_capacity(void) const {
  return this->max_prefix_base_buffer_capacity_;
}

std::size_t glst_workspace::atom_capacity(const int dev) const {
  utl::require(static_cast<std::size_t>(dev) < this->atom_capacity_.size(),
               "glst_workspace::atom_capacity", "Device index out of range");

  return this->atom_capacity_[dev];
}

std::size_t glst_workspace::cell_capacity(const int dev) const {
  utl::require(static_cast<std::size_t>(dev) < this->cell_capacity_.size(),
               "glst_workspace::cell_capacity", "Device index out of range");

  return this->cell_capacity_[dev];
}

std::size_t glst_workspace::sf_tile_buffer_capacity(const int dev) const {
  utl::require(
      static_cast<std::size_t>(dev) < this->sf_tile_buffer_capacity_.size(),
      "glst_workspace::sf_tile_buffer_capacity", "Device index out of range");

  return this->sf_tile_buffer_capacity_[dev];
}

std::size_t
glst_workspace::sf_exchange_tile_buffer_capacity(const int dev) const {
  utl::require(static_cast<std::size_t>(dev) <
                   this->sf_exchange_tile_buffer_capacity_.size(),
               "glst_workspace::sf_exchange_tile_buffer_capacity",
               "Device index out of range");

  return this->sf_exchange_tile_buffer_capacity_[dev];
}

std::size_t glst_workspace::rmt_tile_buffer_capacity(const int dev) const {
  utl::require(
      static_cast<std::size_t>(dev) < this->rmt_tile_buffer_capacity_.size(),
      "glst_workspace::rmt_tile_buffer_capacity", "Device index out of range");

  return this->rmt_tile_buffer_capacity_[dev];
}

std::size_t
glst_workspace::prefix_partition_total_buffer_capacity(const int dev) const {
  utl::require(static_cast<std::size_t>(dev) <
                   this->prefix_partition_total_buffer_capacity_.size(),
               "glst_workspace::prefix_partition_total_buffer_capacity",
               "Device index out of range");

  return this->prefix_partition_total_buffer_capacity_[dev];
}

std::size_t glst_workspace::prefix_base_buffer_capacity(const int dev) const {
  utl::require(static_cast<std::size_t>(dev) <
                   this->prefix_base_buffer_capacity_.size(),
               "glst_workspace::prefix_base_buffer_capacity",
               "Device index out of range");

  return this->prefix_base_buffer_capacity_[dev];
}

std::size_t glst_workspace::prefix_plane_slot_capacity(const int dev) const {
  utl::require(static_cast<std::size_t>(dev) <
                   this->prefix_plane_slot_capacity_.size(),
               "glst_workspace::prefix_plane_slot_capacity",
               "Device index out of range");

  return this->prefix_plane_slot_capacity_[dev];
}

std::size_t glst_workspace::owned_atom_count(const int dev) const {
  utl::require(static_cast<std::size_t>(dev) < this->owned_atom_count_.size(),
               "glst_workspace::owned_atom_count", "Device index out of range");

  return this->owned_atom_count_[dev];
}

std::size_t glst_workspace::source_atom_count(const int dev) const {
  utl::require(static_cast<std::size_t>(dev) < this->source_atom_count_.size(),
               "glst_workspace::source_atom_count",
               "Device index out of range");

  return this->source_atom_count_[dev];
}

std::size_t glst_workspace::sr_source_cell_capacity(const int dev) const {
  utl::require(
      static_cast<std::size_t>(dev) < this->sr_source_cell_capacity_.size(),
      "glst_workspace::sr_source_cell_capacity", "Device index out of range");

  return this->sr_source_cell_capacity_[dev];
}

const std::vector<std::size_t> &glst_workspace::owned_atom_count(void) const {
  return this->owned_atom_count_;
}

const std::vector<std::size_t> &glst_workspace::source_atom_count(void) const {
  return this->source_atom_count_;
}

const std::vector<std::size_t> &
glst_workspace::sr_source_cell_capacity(void) const {
  return this->sr_source_cell_capacity_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::idx(void) const {
  return this->idx_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::sorted_idx(void) const {
  return this->sorted_idx_;
}

const std::vector<cuda_container<double>> &glst_workspace::rx(void) const {
  return this->rx_;
}

const std::vector<cuda_container<double>> &glst_workspace::ry(void) const {
  return this->ry_;
}

const std::vector<cuda_container<double>> &glst_workspace::rz(void) const {
  return this->rz_;
}

const std::vector<cuda_container<double>> &glst_workspace::qc(void) const {
  return this->qc_;
}

const std::vector<cuda_container<atom_packet>> &
glst_workspace::packets(void) const {
  return this->packets_;
}

const std::vector<cuda_container<atom_packet>> &
glst_workspace::sorted_packets(void) const {
  return this->sorted_packets_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::atom_cell_idx(void) const {
  return this->atom_cell_idx_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::atom_cell_sorted_idx(void) const {
  return this->atom_cell_sorted_idx_;
}

const std::vector<cuda_container<double>> &glst_workspace::fx(void) const {
  return this->fx_;
}

const std::vector<cuda_container<double>> &glst_workspace::fy(void) const {
  return this->fy_;
}

const std::vector<cuda_container<double>> &glst_workspace::fz(void) const {
  return this->fz_;
}

const std::vector<cuda_container<double>> &glst_workspace::en(void) const {
  return this->en_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::cell_atom_point(void) const {
  return this->cell_atom_point_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::cell_atom_count(void) const {
  return this->cell_atom_count_;
}

const std::vector<unsigned int> &glst_workspace::max_atoms_cell(void) const {
  return this->max_atoms_cell_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::sr_source_cell_atom_point(void) const {
  return this->sr_source_cell_atom_point_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::sr_source_cell_atom_count(void) const {
  return this->sr_source_cell_atom_count_;
}

const std::vector<cuda_container<double>> &glst_workspace::sf_re(void) const {
  return this->sf_re_;
}

const std::vector<cuda_container<double>> &glst_workspace::sf_im(void) const {
  return this->sf_im_;
}

const std::vector<cuda_container<double>> &
glst_workspace::sf_exchange_re(void) const {
  return this->sf_exchange_re_;
}

const std::vector<cuda_container<double>> &
glst_workspace::sf_exchange_im(void) const {
  return this->sf_exchange_im_;
}

const std::vector<cuda_container<double>> &
glst_workspace::prefix_partition_total_re(void) const {
  return this->prefix_partition_total_re_;
}

const std::vector<cuda_container<double>> &
glst_workspace::prefix_partition_total_im(void) const {
  return this->prefix_partition_total_im_;
}

const std::vector<cuda_container<double>> &
glst_workspace::prefix_base_re(void) const {
  return this->prefix_base_re_;
}

const std::vector<cuda_container<double>> &
glst_workspace::prefix_base_im(void) const {
  return this->prefix_base_im_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::prefix_plane_slot(void) const {
  return this->prefix_plane_slot_;
}

const std::vector<cuda_container<double>> &
glst_workspace::rmt_sum_re(void) const {
  return this->rmt_sum_re_;
}

const std::vector<cuda_container<double>> &
glst_workspace::rmt_sum_im(void) const {
  return this->rmt_sum_im_;
}

const std::vector<void *> &glst_workspace::cub_work_buffer(void) const {
  return this->cub_work_buffer_;
}

const std::vector<std::size_t> &
glst_workspace::cub_work_buffer_size(void) const {
  return this->cub_work_buffer_size_;
}

std::vector<cuda_container<unsigned int>> &glst_workspace::idx(void) {
  return this->idx_;
}

std::vector<cuda_container<unsigned int>> &glst_workspace::sorted_idx(void) {
  return this->sorted_idx_;
}

std::vector<cuda_container<double>> &glst_workspace::rx(void) {
  return this->rx_;
}

std::vector<cuda_container<double>> &glst_workspace::ry(void) {
  return this->ry_;
}

std::vector<cuda_container<double>> &glst_workspace::rz(void) {
  return this->rz_;
}

std::vector<cuda_container<double>> &glst_workspace::qc(void) {
  return this->qc_;
}

std::vector<cuda_container<atom_packet>> &glst_workspace::packets(void) {
  return this->packets_;
}

std::vector<cuda_container<atom_packet>> &glst_workspace::sorted_packets(void) {
  return this->sorted_packets_;
}

std::vector<cuda_container<unsigned int>> &glst_workspace::atom_cell_idx(void) {
  return this->atom_cell_idx_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::atom_cell_sorted_idx(void) {
  return this->atom_cell_sorted_idx_;
}

std::vector<cuda_container<double>> &glst_workspace::fx(void) {
  return this->fx_;
}

std::vector<cuda_container<double>> &glst_workspace::fy(void) {
  return this->fy_;
}

std::vector<cuda_container<double>> &glst_workspace::fz(void) {
  return this->fz_;
}

std::vector<cuda_container<double>> &glst_workspace::en(void) {
  return this->en_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::cell_atom_point(void) {
  return this->cell_atom_point_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::cell_atom_count(void) {
  return this->cell_atom_count_;
}

std::vector<unsigned int> &glst_workspace::max_atoms_cell(void) {
  return this->max_atoms_cell_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::sr_source_cell_atom_point(void) {
  return this->sr_source_cell_atom_point_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::sr_source_cell_atom_count(void) {
  return this->sr_source_cell_atom_count_;
}

std::vector<cuda_container<double>> &glst_workspace::sf_re(void) {
  return this->sf_re_;
}

std::vector<cuda_container<double>> &glst_workspace::sf_im(void) {
  return this->sf_im_;
}

std::vector<cuda_container<double>> &glst_workspace::sf_exchange_re(void) {
  return this->sf_exchange_re_;
}

std::vector<cuda_container<double>> &glst_workspace::sf_exchange_im(void) {
  return this->sf_exchange_im_;
}

std::vector<cuda_container<double>> &
glst_workspace::prefix_partition_total_re(void) {
  return this->prefix_partition_total_re_;
}

std::vector<cuda_container<double>> &
glst_workspace::prefix_partition_total_im(void) {
  return this->prefix_partition_total_im_;
}

std::vector<cuda_container<double>> &glst_workspace::prefix_base_re(void) {
  return this->prefix_base_re_;
}

std::vector<cuda_container<double>> &glst_workspace::prefix_base_im(void) {
  return this->prefix_base_im_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::prefix_plane_slot(void) {
  return this->prefix_plane_slot_;
}

std::vector<cuda_container<double>> &glst_workspace::rmt_sum_re(void) {
  return this->rmt_sum_re_;
}

std::vector<cuda_container<double>> &glst_workspace::rmt_sum_im(void) {
  return this->rmt_sum_im_;
}

std::vector<void *> &glst_workspace::cub_work_buffer(void) {
  return this->cub_work_buffer_;
}

std::vector<std::size_t> &glst_workspace::cub_work_buffer_size(void) {
  return this->cub_work_buffer_size_;
}

void glst_workspace::init(const glst_plan &plan, const int device_count) {
  utl::require(device_count >= 1, "glst_workspace::init", "device_count < 1");

  std::vector<unsigned int> dev_cell_partition(device_count, 0);
  for (int dev = 0; dev < device_count; dev++) {
    if (static_cast<unsigned int>(dev) < plan.cell_partition_count())
      dev_cell_partition[dev] = static_cast<unsigned int>(dev);
  }

  this->init(plan, dev_cell_partition, device_count, true, false, 0);

  return;
}

void glst_workspace::init(const glst_plan &plan,
                          const std::vector<unsigned int> &dev_cell_partition,
                          const int device_count, const bool use_full_sf_buffer,
                          const bool use_distributed_prefix,
                          const unsigned int sf_exchange_chunk_x_count) {
  constexpr std::string_view function_name = "glst_workspace::init";

  this->clear();

  utl::require(device_count >= 1, function_name, "device_count < 1");

  utl::require(
      dev_cell_partition.size() == static_cast<std::size_t>(device_count),
      function_name, "dev_cell_partition size does not match device_count");

  const std::size_t global_natom = static_cast<std::size_t>(plan.natom());
  const std::size_t global_ncell = static_cast<std::size_t>(plan.ncell());
  const std::size_t max_tile_nodes =
      static_cast<std::size_t>(plan.max_tile_nodes());

  utl::require(global_natom > 0, function_name, "natom is 0");

  utl::require(global_ncell > 0, function_name, "ncell is 0");

  utl::require(max_tile_nodes > 0, function_name, "max_tile_nodes is 0");

  utl::require(
      !(use_full_sf_buffer && use_distributed_prefix), function_name,
      "Full SF storage and distributed prefix storage cannot both be selected");

  const std::size_t yz_cell_count = checked_mul(
      static_cast<std::size_t>(plan.ncell_y()),
      static_cast<std::size_t>(plan.ncell_z()), "Prefix plane yz-cell count");

  const std::size_t prefix_plane_capacity =
      checked_mul(yz_cell_count, max_tile_nodes, "Prefix plane entry capacity");

  std::size_t chunk_sf_exchange_capacity = 0;

  if ((!use_full_sf_buffer) && (!use_distributed_prefix) &&
      (plan.cell_partition_count() > 1)) {
    utl::require(
        sf_exchange_chunk_x_count > 0, function_name,
        "Local S_tile exchange requires a positive x-plane chunk count");

    utl::require(sf_exchange_chunk_x_count <= plan.ncell_x(), function_name,
                 "S_tile exchange chunk exceeds ncell_x");

    const std::size_t sf_exchange_cell_count =
        checked_mul(static_cast<std::size_t>(sf_exchange_chunk_x_count),
                    yz_cell_count, "S_tile exchange buffer capacity");

    chunk_sf_exchange_capacity =
        checked_mul(sf_exchange_cell_count, max_tile_nodes,
                    "S_tile exchange tile buffer capacity");
  }

  this->max_atom_capacity_ = 0;
  this->max_cell_capacity_ = 0;
  this->tile_node_capacity_ = max_tile_nodes;
  this->max_sf_tile_buffer_capacity_ = 0;
  this->max_sf_exchange_tile_buffer_capacity_ = 0;
  this->max_rmt_tile_buffer_capacity_ = 0;
  this->max_prefix_partition_total_buffer_capacity_ = 0;
  this->max_prefix_base_buffer_capacity_ = 0;

  this->atom_capacity_.assign(device_count, 0);
  this->cell_capacity_.assign(device_count, 0);
  this->sf_tile_buffer_capacity_.assign(device_count, 0);
  this->sf_exchange_tile_buffer_capacity_.assign(device_count, 0);
  this->rmt_tile_buffer_capacity_.assign(device_count, 0);
  this->prefix_partition_total_buffer_capacity_.assign(device_count, 0);
  this->prefix_base_buffer_capacity_.assign(device_count, 0);
  this->prefix_plane_slot_capacity_.assign(device_count, 0);
  this->owned_atom_count_.assign(device_count, 0);
  this->source_atom_count_.assign(device_count, 0);
  this->sr_source_cell_capacity_.assign(device_count, 0);

  for (int dev = 0; dev < device_count; dev++) {
    const unsigned int cell_partition = dev_cell_partition[dev];

    utl::require(cell_partition < plan.cell_partition_count(), function_name,
                 "Device cell partition is out of range");

    const std::size_t local_cell_count =
        (device_count == 1)
            ? global_ncell
            : static_cast<std::size_t>(plan.local_cell_count(cell_partition));

    const std::size_t halo_cell_count =
        (device_count == 1)
            ? 0
            : plan.partition_halo_cell_idx(cell_partition).size();

    const std::size_t sr_source_cell_count = local_cell_count + halo_cell_count;

    std::size_t local_atom_capacity = global_natom;

    if (device_count > 1) {
      const std::size_t atom_cell_product = checked_mul(
          global_natom, local_cell_count, "local atom capacity numerator");

      utl::require(
          (global_ncell <= 1) ||
              (atom_cell_product <=
               std::numeric_limits<std::size_t>::max() - (global_ncell - 1)),
          function_name, "Overflow while computing local atom capacity");

      local_atom_capacity =
          (atom_cell_product + global_ncell - 1) / global_ncell;
    }

    const std::size_t sf_cell_count =
        (use_full_sf_buffer) ? global_ncell : local_cell_count;

    const std::size_t sf_capacity =
        checked_mul(sf_cell_count, max_tile_nodes, "sf tile buffer capacity");

    const std::size_t rmt_capacity = checked_mul(
        local_cell_count, max_tile_nodes, "rmt_sum tile buffer capacity");

    std::size_t sf_exchange_capacity = chunk_sf_exchange_capacity;

    std::size_t prefix_partition_total_capacity = 0;
    std::size_t prefix_base_capacity = 0;
    std::size_t prefix_plane_slot_capacity = 0;

    if (use_distributed_prefix && (plan.cell_partition_count() > 1)) {
      const std::size_t imported_plane_capacity = static_cast<std::size_t>(
          plan.partition_max_prefix_plane_count(cell_partition));

      sf_exchange_capacity =
          checked_mul(imported_plane_capacity, prefix_plane_capacity,
                      "Imported prefix-plane buffer capacity");

      prefix_partition_total_capacity = checked_mul(
          static_cast<std::size_t>(plan.cell_partition_count()),
          prefix_plane_capacity, "Partition-total prefix buffer capacity");

      prefix_base_capacity = prefix_plane_capacity;

      prefix_plane_slot_capacity =
          checked_mul(static_cast<std::size_t>(plan.ngroup()),
                      static_cast<std::size_t>(plan.ncell_x()),
                      "Prefix-plane slot-map capacity");
    }

    this->atom_capacity_[dev] = local_atom_capacity;
    this->cell_capacity_[dev] = local_cell_count;
    this->sf_tile_buffer_capacity_[dev] = sf_capacity;
    this->sf_exchange_tile_buffer_capacity_[dev] = sf_exchange_capacity;
    this->rmt_tile_buffer_capacity_[dev] = rmt_capacity;
    this->prefix_partition_total_buffer_capacity_[dev] =
        prefix_partition_total_capacity;
    this->prefix_base_buffer_capacity_[dev] = prefix_base_capacity;
    this->prefix_plane_slot_capacity_[dev] = prefix_plane_slot_capacity;
    this->owned_atom_count_[dev] = local_atom_capacity;
    this->source_atom_count_[dev] = local_atom_capacity;
    this->sr_source_cell_capacity_[dev] = sr_source_cell_count;

    if (local_atom_capacity > this->max_atom_capacity_)
      this->max_atom_capacity_ = local_atom_capacity;

    if (local_cell_count > this->max_cell_capacity_)
      this->max_cell_capacity_ = local_cell_count;

    if (sf_capacity > this->max_sf_tile_buffer_capacity_)
      this->max_sf_tile_buffer_capacity_ = sf_capacity;

    if (sf_exchange_capacity > this->max_sf_exchange_tile_buffer_capacity_)
      this->max_sf_exchange_tile_buffer_capacity_ = sf_exchange_capacity;

    if (rmt_capacity > this->max_rmt_tile_buffer_capacity_)
      this->max_rmt_tile_buffer_capacity_ = rmt_capacity;

    if (prefix_partition_total_capacity >
        this->max_prefix_partition_total_buffer_capacity_) {
      this->max_prefix_partition_total_buffer_capacity_ =
          prefix_partition_total_capacity;
    }

    if (prefix_base_capacity > this->max_prefix_base_buffer_capacity_)
      this->max_prefix_base_buffer_capacity_ = prefix_base_capacity;
  }

  this->idx_.resize(device_count);
  this->sorted_idx_.resize(device_count);
  this->rx_.resize(device_count);
  this->ry_.resize(device_count);
  this->rz_.resize(device_count);
  this->qc_.resize(device_count);
  this->packets_.resize(device_count);
  this->sorted_packets_.resize(device_count);
  this->atom_cell_idx_.resize(device_count);
  this->atom_cell_sorted_idx_.resize(device_count);

  this->fx_.resize(device_count);
  this->fy_.resize(device_count);
  this->fz_.resize(device_count);
  this->en_.resize(device_count);

  this->cell_atom_point_.resize(device_count);
  this->cell_atom_count_.resize(device_count);
  this->max_atoms_cell_.resize(device_count);
  this->sr_source_cell_atom_point_.resize(device_count);
  this->sr_source_cell_atom_count_.resize(device_count);

  this->sf_re_.resize(device_count);
  this->sf_im_.resize(device_count);
  this->sf_exchange_re_.resize(device_count);
  this->sf_exchange_im_.resize(device_count);
  this->prefix_partition_total_re_.resize(device_count);
  this->prefix_partition_total_im_.resize(device_count);
  this->prefix_base_re_.resize(device_count);
  this->prefix_base_im_.resize(device_count);
  this->prefix_plane_slot_.resize(device_count);
  this->rmt_sum_re_.resize(device_count);
  this->rmt_sum_im_.resize(device_count);

  for (int dev = 0; dev < device_count; dev++) {
    cudaCheck(cudaSetDevice(dev));

    this->idx_[dev].resize(this->atom_capacity_[dev]);
    this->sorted_idx_[dev].resize(this->atom_capacity_[dev]);
    this->rx_[dev].resize(this->atom_capacity_[dev]);
    this->ry_[dev].resize(this->atom_capacity_[dev]);
    this->rz_[dev].resize(this->atom_capacity_[dev]);
    this->qc_[dev].resize(this->atom_capacity_[dev]);
    this->packets_[dev].resize(this->atom_capacity_[dev]);
    this->sorted_packets_[dev].resize(this->atom_capacity_[dev]);
    this->atom_cell_idx_[dev].resize(this->atom_capacity_[dev]);
    this->atom_cell_sorted_idx_[dev].resize(this->atom_capacity_[dev]);

    this->fx_[dev].resize(this->atom_capacity_[dev]);
    this->fy_[dev].resize(this->atom_capacity_[dev]);
    this->fz_[dev].resize(this->atom_capacity_[dev]);
    this->en_[dev].resize(this->atom_capacity_[dev]);

    this->cell_atom_point_[dev].resize(this->cell_capacity_[dev]);
    this->cell_atom_count_[dev].resize(this->cell_capacity_[dev]);
    this->sr_source_cell_atom_point_[dev].resize(
        this->sr_source_cell_capacity_[dev]);
    this->sr_source_cell_atom_count_[dev].resize(
        this->sr_source_cell_capacity_[dev]);
    this->max_atoms_cell_[dev] = 0;

    this->sf_re_[dev].resize(this->sf_tile_buffer_capacity_[dev]);
    this->sf_im_[dev].resize(this->sf_tile_buffer_capacity_[dev]);

    this->sf_exchange_re_[dev].resize(
        this->sf_exchange_tile_buffer_capacity_[dev]);
    this->sf_exchange_im_[dev].resize(
        this->sf_exchange_tile_buffer_capacity_[dev]);

    this->prefix_partition_total_re_[dev].resize(
        this->prefix_partition_total_buffer_capacity_[dev]);
    this->prefix_partition_total_im_[dev].resize(
        this->prefix_partition_total_buffer_capacity_[dev]);

    this->prefix_base_re_[dev].resize(this->prefix_base_buffer_capacity_[dev]);
    this->prefix_base_im_[dev].resize(this->prefix_base_buffer_capacity_[dev]);

    if (this->prefix_plane_slot_capacity_[dev] > 0) {
      const unsigned int cell_partition = dev_cell_partition[dev];
      this->prefix_plane_slot_[dev] =
          plan.partition_prefix_slot_by_group_x(cell_partition);
    } else
      this->prefix_plane_slot_[dev].clear();

    this->rmt_sum_re_[dev].resize(this->rmt_tile_buffer_capacity_[dev]);
    this->rmt_sum_im_[dev].resize(this->rmt_tile_buffer_capacity_[dev]);
  }

  this->allocate_cub(device_count);

  return;
}

void glst_workspace::resize_atom_storage(const int dev,
                                         const std::size_t atom_capacity) {
  utl::require(static_cast<std::size_t>(dev) < this->atom_capacity_.size(),
               "glst_workspace::resize_atom_storage",
               "Device index out of range");

  cudaCheck(cudaSetDevice(dev));

  this->deallocate_cub_for_device(dev);

  this->atom_capacity_[dev] = atom_capacity;

  this->owned_atom_count_[dev] = atom_capacity;
  this->source_atom_count_[dev] = atom_capacity;

  this->idx_[dev].resize(atom_capacity);
  this->sorted_idx_[dev].resize(atom_capacity);
  this->rx_[dev].resize(atom_capacity);
  this->ry_[dev].resize(atom_capacity);
  this->rz_[dev].resize(atom_capacity);
  this->qc_[dev].resize(atom_capacity);
  this->packets_[dev].resize(atom_capacity);
  this->sorted_packets_[dev].resize(atom_capacity);
  this->atom_cell_idx_[dev].resize(atom_capacity);
  this->atom_cell_sorted_idx_[dev].resize(atom_capacity);

  this->fx_[dev].resize(atom_capacity);
  this->fy_[dev].resize(atom_capacity);
  this->fz_[dev].resize(atom_capacity);
  this->en_[dev].resize(atom_capacity);

  this->max_atom_capacity_ = 0;
  for (std::size_t i = 0; i < this->atom_capacity_.size(); i++) {
    if (this->atom_capacity_[i] > this->max_atom_capacity_)
      this->max_atom_capacity_ = this->atom_capacity_[i];
  }

  this->allocate_cub_for_device(dev);

  return;
}

void glst_workspace::set_source_atom_count(
    const int dev, const std::size_t source_atom_count) {
  constexpr std::string_view function_name =
      "glst_workspace::set_source_atom_count";

  utl::require(static_cast<std::size_t>(dev) < this->source_atom_count_.size(),
               function_name, "Device index out of range");

  utl::require(source_atom_count <= this->atom_capacity_[dev], function_name,
               "source_atom_count exceeds atom capacity");

  this->source_atom_count_[dev] = source_atom_count;

  return;
}

void glst_workspace::set_owned_atom_count(const int dev,
                                          const std::size_t owned_atom_count) {
  constexpr std::string_view function_name =
      "glst_workspace::set_owned_atom_count";

  utl::require(static_cast<std::size_t>(dev) < this->owned_atom_count_.size(),
               function_name, "Device index out of range");

  utl::require(owned_atom_count <= this->source_atom_count_[dev], function_name,
               "owned_atom_count exceeds source atom count");

  this->owned_atom_count_[dev] = owned_atom_count;

  return;
}

void glst_workspace::clear(void) {
  this->deallocate_cub();

  this->max_atom_capacity_ = 0;
  this->max_cell_capacity_ = 0;
  this->tile_node_capacity_ = 0;
  this->max_sf_tile_buffer_capacity_ = 0;
  this->max_sf_exchange_tile_buffer_capacity_ = 0;
  this->max_rmt_tile_buffer_capacity_ = 0;
  this->max_prefix_partition_total_buffer_capacity_ = 0;
  this->max_prefix_base_buffer_capacity_ = 0;

  this->atom_capacity_.clear();
  this->cell_capacity_.clear();
  this->sf_tile_buffer_capacity_.clear();
  this->sf_exchange_tile_buffer_capacity_.clear();
  this->rmt_tile_buffer_capacity_.clear();
  this->prefix_partition_total_buffer_capacity_.clear();
  this->prefix_base_buffer_capacity_.clear();
  this->prefix_plane_slot_capacity_.clear();
  this->owned_atom_count_.clear();
  this->source_atom_count_.clear();
  this->sr_source_cell_capacity_.clear();

  this->idx_.clear();
  this->sorted_idx_.clear();
  this->rx_.clear();
  this->ry_.clear();
  this->rz_.clear();
  this->qc_.clear();
  this->packets_.clear();
  this->sorted_packets_.clear();
  this->atom_cell_idx_.clear();
  this->atom_cell_sorted_idx_.clear();

  this->fx_.clear();
  this->fy_.clear();
  this->fz_.clear();
  this->en_.clear();

  this->cell_atom_point_.clear();
  this->cell_atom_count_.clear();
  this->sr_source_cell_atom_point_.clear();
  this->sr_source_cell_atom_count_.clear();
  this->max_atoms_cell_.clear();

  this->sf_re_.clear();
  this->sf_im_.clear();
  this->sf_exchange_re_.clear();
  this->sf_exchange_im_.clear();
  this->prefix_partition_total_re_.clear();
  this->prefix_partition_total_im_.clear();
  this->prefix_base_re_.clear();
  this->prefix_base_im_.clear();
  this->prefix_plane_slot_.clear();
  this->rmt_sum_re_.clear();
  this->rmt_sum_im_.clear();

  this->cub_work_buffer_.clear();
  this->cub_work_buffer_size_.clear();

  return;
}

void glst_workspace::allocate_cub_for_device(const int dev) {
  cudaCheck(cudaSetDevice(dev));

  this->cub_work_buffer_[dev] = nullptr;
  this->cub_work_buffer_size_[dev] = 0;

  const std::size_t atom_capacity = this->atom_capacity_[dev];

  if (atom_capacity == 0)
    return;

  utl::require(atom_capacity <=
                   static_cast<std::size_t>(std::numeric_limits<int>::max()),
               "glst_workspace::allocate_cub",
               "Atom capacity exceeds CUB int range");

  const int num_items = static_cast<int>(atom_capacity);

  // Determine storage requirements for CUB functions
  std::size_t size0 = 0, size1 = 0, size2 = 0;

  cub::DeviceRadixSort::SortPairs(
      this->cub_work_buffer_[dev], size0,
      this->atom_cell_idx_[dev].d_array().data(),
      this->atom_cell_sorted_idx_[dev].d_array().data(),
      this->idx_[dev].d_array().data(), this->sorted_idx_[dev].d_array().data(),
      num_items);

  cub::DeviceRadixSort::SortPairs(
      this->cub_work_buffer_[dev], size1,
      this->atom_cell_idx_[dev].d_array().data(),
      this->atom_cell_sorted_idx_[dev].d_array().data(),
      this->fx_[dev].d_array().data(), this->fx_[dev].d_array().data(),
      num_items);

  cub::DeviceRadixSort::SortPairs(
      this->cub_work_buffer_[dev], size2,
      this->atom_cell_idx_[dev].d_array().data(),
      this->atom_cell_sorted_idx_[dev].d_array().data(),
      this->packets_[dev].d_array().data(),
      this->sorted_packets_[dev].d_array().data(), num_items);

  this->cub_work_buffer_size_[dev] = size0;
  if (size1 > this->cub_work_buffer_size_[dev])
    this->cub_work_buffer_size_[dev] = size1;
  if (size2 > this->cub_work_buffer_size_[dev])
    this->cub_work_buffer_size_[dev] = size2;

  cudaCheck(cudaMalloc(&(this->cub_work_buffer_[dev]),
                       this->cub_work_buffer_size_[dev]));

  return;
}

void glst_workspace::allocate_cub(const int device_count) {
  this->cub_work_buffer_.resize(device_count);
  this->cub_work_buffer_size_.resize(device_count);

  for (int dev = 0; dev < device_count; dev++)
    this->allocate_cub_for_device(dev);

  return;
}

void glst_workspace::deallocate_cub_for_device(const int dev) {
  cudaCheck(cudaSetDevice(dev));
  if (this->cub_work_buffer_[dev] != nullptr) {
    cudaCheck(cudaFree(this->cub_work_buffer_[dev]));
    this->cub_work_buffer_[dev] = nullptr;
  }

  this->cub_work_buffer_size_[dev] = 0;

  return;
}

void glst_workspace::deallocate_cub(void) {
  for (std::size_t dev = 0; dev < this->cub_work_buffer_.size(); dev++)
    this->deallocate_cub_for_device(static_cast<int>(dev));

  return;
}
