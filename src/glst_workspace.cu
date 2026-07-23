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

inline static std::size_t checked_add(const std::size_t a, const std::size_t b,
                                      const std::string_view label) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    utl::throw_error("checked_add",
                     "Overflow while computing " + std::string(label));
  }
  return (a + b);
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
      owned_atom_count_(), source_atom_count_(), atom_storage_growth_count_(),
      cub_work_buffer_growth_count_(), sr_source_cell_capacity_(),
      partition_atom_range_(), idx_(), sorted_idx_(), rx_(), ry_(), rz_(),
      qc_(), packets_(), sorted_packets_(), atom_cell_idx_(),
      atom_cell_sorted_idx_(), global_sort_key_in_(), global_sort_key_out_(),
      global_packet_in_(), global_packet_out_(), global_cell_atom_count_(),
      global_cell_atom_point_(), global_x_plane_atom_point_(),
      global_max_atoms_cell_(), atom_assignment_metadata_(), fx_(), fy_(),
      fz_(), en_(), cell_atom_point_(), cell_atom_count_(), max_atoms_cell_(),
      sr_source_cell_atom_point_(), sr_source_cell_atom_count_(), sf_re_(),
      sf_im_(), sf_exchange_re_(), sf_exchange_im_(),
      prefix_partition_total_re_(), prefix_partition_total_im_(),
      prefix_base_re_(), prefix_base_im_(), prefix_plane_slot_(), rmt_sum_re_(),
      rmt_sum_im_(), cub_work_buffer_(), cub_work_buffer_size_() {}

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

std::size_t glst_workspace::atom_storage_growth_count(const int dev) const {
  utl::require(
      static_cast<std::size_t>(dev) < this->atom_storage_growth_count_.size(),
      "glst_workspace::atom_storage_growth_count", "Device index out of range");

  return this->atom_storage_growth_count_[dev];
}

std::size_t glst_workspace::cub_work_buffer_growth_count(const int dev) const {
  utl::require(static_cast<std::size_t>(dev) <
                   this->cub_work_buffer_growth_count_.size(),
               "glst_workspace::cub_work_buffer_growth_count",
               "Device index out of range");

  return this->cub_work_buffer_growth_count_[dev];
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

const std::vector<atom_partition_range> &
glst_workspace::partition_atom_range(void) const {
  return this->partition_atom_range_;
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

const device_vector<atom_sort_key> &
glst_workspace::global_sort_key_in(void) const {
  return this->global_sort_key_in_;
}

const device_vector<atom_sort_key> &
glst_workspace::global_sort_key_out(void) const {
  return this->global_sort_key_out_;
}

const device_vector<atom_packet> &glst_workspace::global_packet_in(void) const {
  return this->global_packet_in_;
}

const device_vector<atom_packet> &
glst_workspace::global_packet_out(void) const {
  return this->global_packet_out_;
}

const device_vector<unsigned int> &
glst_workspace::global_cell_atom_count(void) const {
  return this->global_cell_atom_count_;
}

const device_vector<unsigned int> &
glst_workspace::global_cell_atom_point(void) const {
  return this->global_cell_atom_point_;
}

const device_vector<unsigned int> &
glst_workspace::global_x_plane_atom_point(void) const {
  return this->global_x_plane_atom_point_;
}

const device_vector<unsigned int> &
glst_workspace::global_max_atoms_cell(void) const {
  return this->global_max_atoms_cell_;
}

const std::vector<cuda_container<unsigned int>> &
glst_workspace::atom_assignment_metadata(void) const {
  return this->atom_assignment_metadata_;
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

std::vector<atom_partition_range> &glst_workspace::partition_atom_range(void) {
  return this->partition_atom_range_;
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

device_vector<atom_sort_key> &glst_workspace::global_sort_key_in(void) {
  return this->global_sort_key_in_;
}

device_vector<atom_sort_key> &glst_workspace::global_sort_key_out(void) {
  return this->global_sort_key_out_;
}

device_vector<atom_packet> &glst_workspace::global_packet_in(void) {
  return this->global_packet_in_;
}

device_vector<atom_packet> &glst_workspace::global_packet_out(void) {
  return this->global_packet_out_;
}

device_vector<unsigned int> &glst_workspace::global_cell_atom_count(void) {
  return this->global_cell_atom_count_;
}

device_vector<unsigned int> &glst_workspace::global_cell_atom_point(void) {
  return this->global_cell_atom_point_;
}

device_vector<unsigned int> &glst_workspace::global_x_plane_atom_point(void) {
  return this->global_x_plane_atom_point_;
}

device_vector<unsigned int> &glst_workspace::global_max_atoms_cell(void) {
  return this->global_max_atoms_cell_;
}

std::vector<cuda_container<unsigned int>> &
glst_workspace::atom_assignment_metadata(void) {
  return this->atom_assignment_metadata_;
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
  const std::size_t global_partition_count =
      static_cast<std::size_t>(plan.cell_partition_count());
  const std::size_t max_tile_nodes =
      static_cast<std::size_t>(plan.max_tile_nodes());

  const bool allocate_global_classification_scratch = (device_count > 1);

  std::size_t global_cell_point_count = 0;
  std::size_t global_x_plane_point_count = 0;

  utl::require(global_natom > 0, function_name, "natom is 0");

  utl::require(global_ncell > 0, function_name, "ncell is 0");

  utl::require(global_partition_count > 0, function_name,
               "cell_partition_count is 0");

  utl::require(max_tile_nodes > 0, function_name, "max_tile_nodes is 0");

  if (allocate_global_classification_scratch) {
    const std::size_t cub_item_limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max());

    utl::require(global_natom <= cub_item_limit, function_name,
                 "Global atom count exceeds CUB int range");

    utl::require(global_ncell <= cub_item_limit, function_name,
                 "Global cell count exceeds CUB int range");

    global_cell_point_count =
        checked_add(global_ncell, static_cast<std::size_t>(1),
                    "Global cell atom-point count");

    global_x_plane_point_count = checked_add(
        static_cast<std::size_t>(plan.ncell_x()), static_cast<std::size_t>(1),
        "Global x-plane atom-point count");

    // device_vector performs count * sizeof(T) internally. Validate every
    // distinct allocation shape before any cudaMalloc can occur.
    static_cast<void>(checked_mul(global_natom, sizeof(atom_sort_key),
                                  "Global sort-key buffer bytes"));

    static_cast<void>(checked_mul(global_natom, sizeof(atom_packet),
                                  "Global atom-packet buffer bytes"));

    static_cast<void>(checked_mul(global_ncell, sizeof(unsigned int),
                                  "Global cell atom-point buffer bytes"));

    static_cast<void>(checked_mul(global_cell_point_count, sizeof(unsigned int),
                                  "Global cell atom-point buffer bytes"));

    static_cast<void>(checked_mul(global_x_plane_point_count,
                                  sizeof(unsigned int),
                                  "Global x-plane atom-point buffer bytes"));

    static_cast<void>(checked_mul(global_partition_count, sizeof(unsigned int),
                                  "Global partition max-atoms buffer size"));
  }

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
  this->atom_storage_growth_count_.assign(device_count, 0);
  this->cub_work_buffer_growth_count_.assign(device_count, 0);
  this->sr_source_cell_capacity_.assign(device_count, 0);

  this->partition_atom_range_.assign(plan.cell_partition_count(),
                                     atom_partition_range());

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
    if (device_count == 1) {
      this->owned_atom_count_[dev] = global_natom;
      this->source_atom_count_[dev] = global_natom;
    }
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

  this->atom_assignment_metadata_.resize(device_count);

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

    this->atom_assignment_metadata_[dev].resize(3);

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

  if (allocate_global_classification_scratch) {
    cudaCheck(cudaSetDevice(0));

    this->global_sort_key_in_.resize(global_natom);
    this->global_sort_key_out_.resize(global_natom);

    this->global_packet_in_.resize(global_natom);
    this->global_packet_out_.resize(global_natom);

    this->global_cell_atom_count_.resize(global_ncell);
    this->global_cell_atom_point_.resize(global_cell_point_count);
    this->global_x_plane_atom_point_.resize(global_x_plane_point_count);

    this->global_max_atoms_cell_.resize(global_partition_count);
  }

  this->allocate_cub(device_count);

  return;
}

void glst_workspace::ensure_atom_capacity(
    const int dev, const std::size_t required_source_atom_count) {
  constexpr std::string_view function_name =
      "glst_workspace::ensure_atom_capacity";

  utl::require((dev >= 0) && (static_cast<std::size_t>(dev) <
                              this->atom_capacity_.size()),
               function_name, "Device index out of range");

  const std::size_t current_capacity = this->atom_capacity_[dev];

  utl::require(this->owned_atom_count_[dev] <= this->source_atom_count_[dev],
               function_name,
               "Existing owned atom count exceeds source atom count");

  utl::require(this->source_atom_count_[dev] <= current_capacity, function_name,
               "Existing source atom count exceeds atom capacity");

  if (required_source_atom_count <= current_capacity)
    return;

  const std::size_t cub_item_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());

  utl::require(required_source_atom_count <= cub_item_limit, function_name,
               "Required source atom count exceeds CUB int range");

  // Use bounded, fixed-quantum headroom rather than proportional growth.
  //
  // Below 1M atoms, round to 4096 atoms. At and above 1M atoms, round to 65536
  // atoms. The maximum unused tail is therefore bounded and does not grow in
  // proportion to a multi-million-atom system.
  //
  // The current workspace has approximately 160 bytes of atom-sized storage per
  // capacity entry, so the large quantum adds at most about 10 MiB per device
  // instead of the potentially hundres of MiB added by 1.5x growth.
  constexpr std::size_t LARGE_CAPACITY_THRESHOLD = 1048576;
  constexpr std::size_t SMALL_GROWTH_QUANTUM = 4096;
  constexpr std::size_t LARGE_GROWTH_QUANTUM = 65536;

  const std::size_t growth_quantum =
      (required_source_atom_count < LARGE_CAPACITY_THRESHOLD)
          ? SMALL_GROWTH_QUANTUM
          : LARGE_GROWTH_QUANTUM;

  const std::size_t growth_padding = growth_quantum - 1;
  std::size_t new_capacity = required_source_atom_count;

  if (required_source_atom_count <= cub_item_limit - growth_padding) {
    new_capacity =
        ((required_source_atom_count + growth_padding) / growth_quantum) *
        growth_quantum;
  }

  cudaCheck(cudaSetDevice(dev));

  this->idx_[dev].resize(new_capacity);
  this->sorted_idx_[dev].resize(new_capacity);
  this->rx_[dev].resize(new_capacity);
  this->ry_[dev].resize(new_capacity);
  this->rz_[dev].resize(new_capacity);
  this->qc_[dev].resize(new_capacity);
  this->packets_[dev].resize(new_capacity);
  this->sorted_packets_[dev].resize(new_capacity);
  this->atom_cell_idx_[dev].resize(new_capacity);
  this->atom_cell_sorted_idx_[dev].resize(new_capacity);

  this->fx_[dev].resize(new_capacity);
  this->fy_[dev].resize(new_capacity);
  this->fz_[dev].resize(new_capacity);
  this->en_[dev].resize(new_capacity);

  this->atom_capacity_[dev] = new_capacity;

  if (new_capacity > this->max_atom_capacity_)
    this->max_atom_capacity_ = new_capacity;

  this->atom_storage_growth_count_[dev]++;

  this->ensure_cub_capacity_for_device(dev, true);

  return;
}

void glst_workspace::set_atom_counts(const int dev,
                                     const std::size_t source_atom_count,
                                     const std::size_t owned_atom_count) {
  constexpr std::string_view function_name = "glst_workspace::set_atom_counts";

  utl::require((dev >= 0) && (static_cast<std::size_t>(dev) <
                              this->atom_capacity_.size()),
               function_name, "Device index out of range");

  utl::require(source_atom_count <= this->atom_capacity_[dev], function_name,
               "Source atom count exceeds atom capacity");

  utl::require(owned_atom_count <= source_atom_count, function_name,
               "Owned atom count exceeds source atom count");

  this->source_atom_count_[dev] = source_atom_count;
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
  this->atom_storage_growth_count_.clear();
  this->cub_work_buffer_growth_count_.clear();
  this->sr_source_cell_capacity_.clear();

  this->partition_atom_range_.clear();

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

  this->global_sort_key_in_.clear();
  this->global_sort_key_out_.clear();

  this->global_packet_in_.clear();
  this->global_packet_out_.clear();

  this->global_cell_atom_count_.clear();
  this->global_cell_atom_point_.clear();
  this->global_x_plane_atom_point_.clear();
  this->global_max_atoms_cell_.clear();

  this->atom_assignment_metadata_.clear();

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

void glst_workspace::ensure_cub_capacity_for_device(
    const int dev, const bool count_growth_event) {
  constexpr std::string_view function_name =
      "glst_workspace::ensure_cub_capacity_for_device";

  utl::require(static_cast<std::size_t>(dev) < this->atom_capacity_.size(),
               function_name, "Device index out of range");

  utl::require(static_cast<std::size_t>(dev) < this->cub_work_buffer_.size(),
               function_name, "CUB work-buffer device index out of range");

  utl::require(static_cast<std::size_t>(dev) <
                   this->cub_work_buffer_size_.size(),
               function_name, "CUB work-buffer-size device index out of range");

  cudaCheck(cudaSetDevice(dev));

  const std::size_t atom_capacity = this->atom_capacity_[dev];

  utl::require(
      static_cast<std::size_t>(dev) < this->sr_source_cell_capacity_.size(),
      function_name, "Short-range source-cell device index out of range");

  const std::size_t source_cell_capacity = this->sr_source_cell_capacity_[dev];

  const bool has_global_classification_scratch =
      ((dev == 0) && (!this->global_sort_key_in_.empty()));

  if ((atom_capacity == 0) && (source_cell_capacity == 0) &&
      (!has_global_classification_scratch))
    return;

  void *tmp = nullptr;
  std::size_t required_size = 0;

  if (atom_capacity > 0) {
    utl::require(atom_capacity <=
                     static_cast<std::size_t>(std::numeric_limits<int>::max()),
                 function_name, "Atom capacity exceeds CUB int range");

    const int num_items = static_cast<int>(atom_capacity);

    // Determine storage requirements for CUB functions
    std::size_t index_sort_size = 0;
    std::size_t value_sort_size = 0;
    std::size_t packet_sort_size = 0;

    cub::DeviceRadixSort::SortPairs(
        tmp, index_sort_size, this->atom_cell_idx_[dev].d_array().data(),
        this->atom_cell_sorted_idx_[dev].d_array().data(),
        this->idx_[dev].d_array().data(),
        this->sorted_idx_[dev].d_array().data(), num_items);

    cub::DeviceRadixSort::SortPairs(
        tmp, value_sort_size, this->atom_cell_idx_[dev].d_array().data(),
        this->atom_cell_sorted_idx_[dev].d_array().data(),
        this->fx_[dev].d_array().data(), this->fy_[dev].d_array().data(),
        num_items);

    cub::DeviceRadixSort::SortPairs(
        tmp, packet_sort_size, this->atom_cell_idx_[dev].d_array().data(),
        this->atom_cell_sorted_idx_[dev].d_array().data(),
        this->packets_[dev].d_array().data(),
        this->sorted_packets_[dev].d_array().data(), num_items);

    required_size = index_sort_size;

    if (value_sort_size > required_size)
      required_size = value_sort_size;

    if (packet_sort_size > required_size)
      required_size = packet_sort_size;
  }

  if (source_cell_capacity > 0) {
    utl::require(source_cell_capacity <=
                     static_cast<std::size_t>(std::numeric_limits<int>::max()),
                 function_name,
                 "Short-range source-cell capacity exceeds CUB int range");

    std::size_t source_cell_scan_size = 0;

    cudaCheck(cub::DeviceScan::ExclusiveSum(
        tmp, source_cell_scan_size,
        this->sr_source_cell_atom_count_[dev].d_array().data(),
        this->sr_source_cell_atom_point_[dev].d_array().data(),
        static_cast<int>(source_cell_capacity)));

    if (source_cell_scan_size > required_size)
      required_size = source_cell_scan_size;
  }

  if (has_global_classification_scratch) {
    const std::size_t global_atom_count = this->global_sort_key_in_.size();
    const std::size_t global_cell_count = this->global_cell_atom_count_.size();

    utl::require(global_atom_count > 0, function_name,
                 "Global atom-classification capacity is zero");

    utl::require(global_cell_count > 0, function_name,
                 "Global cell-count capacity is zero");

    utl::require(this->global_sort_key_out_.size() == global_atom_count,
                 function_name, "Global sort-key capacities differ");

    utl::require(this->global_packet_in_.size() == global_atom_count,
                 function_name,
                 "Global input packet capacity does not match atom count");

    utl::require(this->global_packet_out_.size() == global_atom_count,
                 function_name,
                 "Global output packet capacity does not match atom count");

    utl::require(this->global_cell_atom_point_.size() == global_cell_count + 1,
                 function_name,
                 "Global cell atom-point capacity lacks the final endpoint");

    utl::require(!this->global_x_plane_atom_point_.empty(), function_name,
                 "Global x-plane atom-point capacity is zero");

    utl::require(!this->global_max_atoms_cell_.empty(), function_name,
                 "Global reduction-output capacity is zero");

    const std::size_t cub_item_limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max());

    utl::require(global_atom_count <= cub_item_limit, function_name,
                 "Global atom count exceeds CUB int range");

    utl::require(global_cell_count <= cub_item_limit, function_name,
                 "Global cell count exceeds CUB int range");

    const int global_atom_items = static_cast<int>(global_atom_count);
    const int global_cell_items = static_cast<int>(global_cell_count);

    std::size_t global_sort_size = 0;
    std::size_t global_scan_size = 0;
    std::size_t global_reduce_size = 0;

    cudaCheck(cub::DeviceRadixSort::SortPairs(
        tmp, global_sort_size, this->global_sort_key_in_.data(),
        this->global_sort_key_out_.data(), this->global_packet_in_.data(),
        this->global_packet_out_.data(), global_atom_items, 0,
        static_cast<int>(8 * sizeof(atom_sort_key))));

    cudaCheck(cub::DeviceScan::ExclusiveSum(
        tmp, global_scan_size, this->global_cell_atom_count_.data(),
        this->global_cell_atom_point_.data(), global_cell_items));

    cudaCheck(cub::DeviceReduce::Max(
        tmp, global_reduce_size, this->global_cell_atom_count_.data(),
        this->global_max_atoms_cell_.data(), global_cell_items));

    if (global_sort_size > required_size)
      required_size = global_sort_size;

    if (global_scan_size > required_size)
      required_size = global_scan_size;

    if (global_reduce_size > required_size)
      required_size = global_reduce_size;
  }

  if ((this->cub_work_buffer_[dev] != nullptr) &&
      (required_size <= this->cub_work_buffer_size_[dev])) {
    return;
  }

  // This path is reached only when scratch must grow. Release the old scratch
  // before allocating the larger block to avoid temporarily holding both CUB
  // allocations on memory-constrained devices.
  if (this->cub_work_buffer_[dev] != nullptr) {
    cudaCheck(cudaFree(this->cub_work_buffer_[dev]));
    this->cub_work_buffer_[dev] = nullptr;
    this->cub_work_buffer_size_[dev] = 0;
  }

  if (required_size > 0) {
    cudaCheck(cudaMalloc(&(this->cub_work_buffer_[dev]), required_size));
    this->cub_work_buffer_size_[dev] = required_size;
  }

  if (count_growth_event)
    this->cub_work_buffer_growth_count_[dev]++;

  return;
}

void glst_workspace::allocate_cub(const int device_count) {
  this->cub_work_buffer_.assign(device_count, nullptr);
  this->cub_work_buffer_size_.assign(device_count, 0);

  for (int dev = 0; dev < device_count; dev++)
    this->ensure_cub_capacity_for_device(dev, false);

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
