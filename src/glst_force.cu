// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "glst_force.hcu"

#include "cuda_utils.hcu"
#include "error_utils.hpp"
#include "reduce.hcu"

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

// Make the raw-byte NCCL tranfer an explicit compile-time requirement.
static_assert(std::is_trivially_copyable<atom_packet>::value,
              "atom_packet must remain trivially copyable");

static constexpr double gpu_layout_memory_budget_fraction = 0.80;

static double
profile_elapsed_ms(const std::chrono::steady_clock::time_point &start,
                   const std::chrono::steady_clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

glst_force::glst_force(void)
    : plan_(nullptr), workspace_(nullptr),
      execution_mode_(GLST_EXECUTION_MODE::SINGLE_GPU_TILED),
      sf_exchange_mode_(GLST_SF_EXCHANGE_MODE::LOCAL_CHUNK_BROADCAST),
      sf_exchange_chunk_x_count_(0), cell_partition_count_(1),
      tile_partition_count_(1), dev_cell_partition_(), dev_tile_partition_(),
      cuda_count_(-1), dev_cell_idx_(), comp_streams_(), comm_streams_(),
      comp_events_(), comm_events_(), nccl_devs_(), nccl_comms_(),
      cell_comm_devs_(), tile_comm_devs_(), cell_comms_(), tile_comms_(),
      profiling_enabled_(false), profile_(), gpu_layout_min_free_bytes_(0),
      gpu_layout_memory_budget_bytes_(0), gpu_layout_estimated_peak_bytes_(0),
      gpu_layout_reason_(""), gpu_layout_user_set_(false),
      cuda_initialized_(false) {}

glst_force::glst_force(const unsigned int natom, const double tol,
                       const double box_dim_x, const double box_dim_y,
                       const double box_dim_z, const double rcut)
    : glst_force() {
  this->init(natom, tol, box_dim_x, box_dim_y, box_dim_z, rcut);
}

glst_force::glst_force(const unsigned int natom, const double tol,
                       const double box_dim_x, const double box_dim_y,
                       const double box_dim_z, const unsigned int ncell_x,
                       const unsigned int ncell_y, const unsigned int ncell_z)
    : glst_force() {
  double rcxd = box_dim_x / static_cast<double>(ncell_x);
  double rcyd = box_dim_y / static_cast<double>(ncell_y);
  double rczd = box_dim_z / static_cast<double>(ncell_z);
  double rcut = rcxd;
  rcut = (rcyd < rcut) ? rcyd : rcut;
  rcut = (rczd < rcut) ? rczd : rcut;

  this->init(natom, tol, box_dim_x, box_dim_y, box_dim_z, rcut);
}

glst_force::~glst_force(void) { this->deallocate(); }

const std::vector<cuda_container<double>> &glst_force::fx(void) const {
  return this->workspace_->fx();
}

const std::vector<cuda_container<double>> &glst_force::fy(void) const {
  return this->workspace_->fy();
}

const std::vector<cuda_container<double>> &glst_force::fz(void) const {
  return this->workspace_->fz();
}

const std::vector<cuda_container<double>> &glst_force::en(void) const {
  return this->workspace_->en();
}

std::vector<cuda_container<double>> &glst_force::fx(void) {
  return this->workspace_->fx();
}

std::vector<cuda_container<double>> &glst_force::fy(void) {
  return this->workspace_->fy();
}

std::vector<cuda_container<double>> &glst_force::fz(void) {
  return this->workspace_->fz();
}

std::vector<cuda_container<double>> &glst_force::en(void) {
  return this->workspace_->en();
}

void glst_force::get_ef(cuda_container<double> &fx, cuda_container<double> &fy,
                        cuda_container<double> &fz,
                        cuda_container<double> &en) {
  constexpr std::string_view function_name = "glst_force::get_ef";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  const std::size_t natom = static_cast<std::size_t>(this->plan_->natom());

  cudaCheck(cudaSetDevice(0));

  fx.resize(natom);
  fy.resize(natom);
  fz.resize(natom);
  en.resize(natom);

  // Preserve the existing single-GPU path exactly. Its local atom allocation
  // contains all atoms, so the existing CUB sort remains valid.
  if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
    std::size_t cub_work_buffer_size =
        this->workspace_->cub_work_buffer_size()[0];

    cub::DeviceRadixSort::SortPairs(
        this->workspace_->cub_work_buffer()[0], cub_work_buffer_size,
        this->workspace_->sorted_idx()[0].d_array().data(),
        this->workspace_->idx()[0].d_array().data(),
        this->workspace_->fx()[0].d_array().data(), fx.d_array().data(),
        this->plan_->natom(), 0, static_cast<int>(8 * sizeof(unsigned int)),
        this->comp_streams_[0]);

    cub::DeviceRadixSort::SortPairs(
        this->workspace_->cub_work_buffer()[0], cub_work_buffer_size,
        this->workspace_->sorted_idx()[0].d_array().data(),
        this->workspace_->idx()[0].d_array().data(),
        this->workspace_->fy()[0].d_array().data(), fy.d_array().data(),
        this->plan_->natom(), 0, static_cast<int>(8 * sizeof(unsigned int)),
        this->comp_streams_[0]);

    cub::DeviceRadixSort::SortPairs(
        this->workspace_->cub_work_buffer()[0], cub_work_buffer_size,
        this->workspace_->sorted_idx()[0].d_array().data(),
        this->workspace_->idx()[0].d_array().data(),
        this->workspace_->fz()[0].d_array().data(), fz.d_array().data(),
        this->plan_->natom(), 0, static_cast<int>(8 * sizeof(unsigned int)),
        this->comp_streams_[0]);

    cub::DeviceRadixSort::SortPairs(
        this->workspace_->cub_work_buffer()[0], cub_work_buffer_size,
        this->workspace_->sorted_idx()[0].d_array().data(),
        this->workspace_->idx()[0].d_array().data(),
        this->workspace_->en()[0].d_array().data(), en.d_array().data(),
        this->plan_->natom(), 0, static_cast<int>(8 * sizeof(unsigned int)),
        this->comp_streams_[0]);

    cudaCheck(cudaStreamSynchronize(this->comp_streams_[0]));

    fx.transfer_to_host();
    fy.transfer_to_host();
    fz.transfer_to_host();
    en.transfer_to_host();

    return;
  }

  // Multi-GPU path. There must be one tile communicator per cell partition.
  // Rank 0 of each communicator it tile partition 0 and owns the complete
  // reduced local result.
  utl::require(
      this->tile_comm_devs_.size() ==
          static_cast<std::size_t>(this->cell_partition_count_),
      function_name,
      "Tile communicator topology does not match cell partition count");

  std::size_t total_owned_atom_count = 0;
  std::size_t max_owned_atom_count = 0;

  // Validate the root-device topology and determine the temporary-buffer size
  for (unsigned int cell_partition = 0;
       cell_partition < this->cell_partition_count_; cell_partition++) {
    const std::vector<int> &devs = this->tile_comm_devs_[cell_partition];

    utl::require(
        devs.size() == static_cast<std::size_t>(this->tile_partition_count_),
        function_name,
        "Tile communicator device count does not match tile partition count");

    utl::require(!devs.empty(), function_name,
                 "Tile communicator has no root device");

    const int root_dev = devs[0];

    utl::require((root_dev >= 0) && (root_dev < this->cuda_count_),
                 function_name, "Root device is out of range");

    utl::require(this->dev_cell_partition_[root_dev] == cell_partition,
                 function_name,
                 "Root device belongs to the wrong cell partition");

    const std::size_t owned_atom_count =
        this->workspace_->owned_atom_count(root_dev);

    utl::require(owned_atom_count <= this->workspace_->atom_capacity(root_dev),
                 function_name,
                 "Root owned atom count exceeds its atom capacity");

    if ((total_owned_atom_count > natom) ||
        (owned_atom_count > natom - total_owned_atom_count)) {
      utl::throw_error(function_name,
                       "Sum of root owned atom counts exceeds natom");
    }

    total_owned_atom_count += owned_atom_count;

    if (owned_atom_count > max_owned_atom_count)
      max_owned_atom_count = owned_atom_count;
  }

  utl::require(total_owned_atom_count == natom, function_name,
               "Root owned atom counts do not sum to natom");

  // Allocate one reusable set of host staging arrays. Their size is the largest
  // cell partition, not natom, so the gather does not create another complete
  // global force/energy copy.
  std::vector<unsigned int> local_idx(max_owned_atom_count);
  std::vector<double> local_fx(max_owned_atom_count);
  std::vector<double> local_fy(max_owned_atom_count);
  std::vector<double> local_fz(max_owned_atom_count);
  std::vector<double> local_en(max_owned_atom_count);

  std::vector<unsigned int> atom_seen(natom, 0u);

  std::fill(fx.h_array().begin(), fx.h_array().end(), 0.0);
  std::fill(fy.h_array().begin(), fy.h_array().end(), 0.0);
  std::fill(fz.h_array().begin(), fz.h_array().end(), 0.0);
  std::fill(en.h_array().begin(), en.h_array().end(), 0.0);

  total_owned_atom_count = 0;

  // Gather one root at a time and scatter directly by original atom index.
  for (unsigned int cell_partition = 0;
       cell_partition < this->cell_partition_count_; cell_partition++) {
    const int root_dev = this->tile_comm_devs_[cell_partition][0];
    const std::size_t owned_atom_count =
        this->workspace_->owned_atom_count(root_dev);

    if (owned_atom_count == 0)
      continue;

    cudaCheck(cudaSetDevice(root_dev));

    // comm_ef() already establishes this dependency. Synchronizing here keeps
    // get_ef() independently safe if it is called after the public phased API.
    cudaCheck(cudaStreamSynchronize(this->comp_streams_[root_dev]));

    const std::size_t index_bytes = owned_atom_count * sizeof(unsigned int);
    const std::size_t value_bytes = owned_atom_count * sizeof(double);

    cudaCheck(cudaMemcpy(
        static_cast<void *>(local_idx.data()),
        static_cast<const void *>(
            this->workspace_->sorted_idx()[root_dev].d_array().data()),
        index_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(static_cast<void *>(local_fx.data()),
                         static_cast<const void *>(
                             this->workspace_->fx()[root_dev].d_array().data()),
                         value_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(static_cast<void *>(local_fy.data()),
                         static_cast<const void *>(
                             this->workspace_->fy()[root_dev].d_array().data()),
                         value_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(static_cast<void *>(local_fz.data()),
                         static_cast<const void *>(
                             this->workspace_->fz()[root_dev].d_array().data()),
                         value_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(static_cast<void *>(local_en.data()),
                         static_cast<const void *>(
                             this->workspace_->en()[root_dev].d_array().data()),
                         value_bytes, cudaMemcpyDeviceToHost));

    for (std::size_t local_atom = 0; local_atom < owned_atom_count;
         local_atom++) {
      const unsigned int global_atom = local_idx[local_atom];

      utl::require(static_cast<std::size_t>(global_atom) < natom, function_name,
                   "Original atom index is out of range");

      utl::require(atom_seen[global_atom] == 0u, function_name,
                   "Original atom index was gathered more than once");

      atom_seen[global_atom] = 1u;

      fx[global_atom] = local_fx[local_atom];
      fy[global_atom] = local_fy[local_atom];
      fz[global_atom] = local_fz[local_atom];
      en[global_atom] = local_en[local_atom];
    }

    total_owned_atom_count += owned_atom_count;
  }

  utl::require(total_owned_atom_count == natom, function_name,
               "Gathered atom count does not equal natom");

  for (std::size_t atom = 0; atom < natom; atom++) {
    utl::require(atom_seen[atom] != 0u, function_name,
                 "At least one original atom index was not gathered");
  }

  // The host arrays are already complete. Synchronize the device side of the
  // public output containers on GPU 0 to preserve the existing get_ef behavior.
  cudaCheck(cudaSetDevice(0));

  fx.transfer_to_device();
  fy.transfer_to_device();
  fz.transfer_to_device();
  en.transfer_to_device();

  return;
}

void glst_force::init(const unsigned int natom, const double tol,
                      const double box_dim_x, const double box_dim_y,
                      const double box_dim_z, const double rcut) {
  constexpr std::string_view function_name = "glst_force::init";

  // Manual layouts can still be validated before the more expensive plan
  // construction. Automatic layout selection requires the completed global
  // cell, cubature, and tile metadata.
  if (this->gpu_layout_user_set_)
    this->init_cuda_resources();

  this->plan_ = std::make_unique<glst_plan>();
  this->plan_->init_cells(natom, box_dim_x, box_dim_y, box_dim_z, rcut);

  // init_alpha_groups() constructs distributed-prefix metadata, so it requires
  // a valid cell partition first. Automatic selection begins with one temporary
  // partition and replaces it after the layout has been selected.
  const unsigned int initial_cell_partition_count =
      (this->gpu_layout_user_set_) ? this->cell_partition_count_ : 1;

  this->plan_->init_cell_partitions(initial_cell_partition_count);
  this->plan_->init_alpha_groups(tol);
  this->plan_->init_cubature(tol);
  this->plan_->init_tile_schedule(2048);

  if (!this->gpu_layout_user_set_)
    this->init_cuda_resources();

  if (this->plan_->cell_partition_count() != this->cell_partition_count_)
    this->plan_->init_cell_partitions(this->cell_partition_count_);

  this->plan_->init_tile_partitions(this->tile_partition_count_);

  unsigned int max_partition_x_count = 0;

  for (unsigned int partition = 0;
       partition < this->plan_->cell_partition_count(); partition++) {
    const unsigned int x_count =
        this->plan_->cell_partition_x_count()[partition];

    if (x_count > max_partition_x_count)
      max_partition_x_count = x_count;
  }

  if ((this->sf_exchange_mode_ ==
       GLST_SF_EXCHANGE_MODE::LOCAL_CHUNK_BROADCAST) &&
      (this->cell_partition_count_ > 1)) {
    this->sf_exchange_chunk_x_count_ = (max_partition_x_count + 1u) / 2u;

    utl::require(this->sf_exchange_chunk_x_count_ > 0, function_name,
                 "Could not select a positive S_tile exchange chunk size");
  } else
    this->sf_exchange_chunk_x_count_ = 0;

  const bool use_full_sf_buffer =
      (this->sf_exchange_mode_ == GLST_SF_EXCHANGE_MODE::FULL_GLOBAL_ALLREDUCE);

  const bool use_distributed_prefix =
      (this->sf_exchange_mode_ == GLST_SF_EXCHANGE_MODE::DISTRIBUTED_PREFIX);

  this->workspace_ = std::make_unique<glst_workspace>(
      *(this->plan_), this->dev_cell_partition_, this->cuda_count_,
      use_full_sf_buffer, use_distributed_prefix,
      this->sf_exchange_chunk_x_count_);

  if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
    cudaCheck(cudaSetDevice(0));

    const std::size_t atom_capacity = this->workspace_->atom_capacity(0);

    for (std::size_t i = 0; i < atom_capacity; i++)
      this->workspace_->idx()[0][i] = static_cast<unsigned int>(i);

    if (atom_capacity > 0)
      this->workspace_->idx()[0].transfer_to_device();
  }

  // Layout validation
  utl::require(this->cell_partition_count_ > 0, function_name,
               "cell_partition_count == 0");

  utl::require(this->tile_partition_count_ > 0, function_name,
               "tile_partition_count == 0");

  utl::require(this->dev_cell_partition_.size() ==
                   static_cast<std::size_t>(this->cuda_count_),
               function_name,
               "dev_cell_partition size does not match cuda_count");

  utl::require(this->dev_tile_partition_.size() ==
                   static_cast<std::size_t>(this->cuda_count_),
               function_name,
               "dev_tile_partition size does not match cuda_count");

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    utl::require(this->dev_cell_partition_[dev] < this->cell_partition_count_,
                 function_name, "Device cell partition is out of range");

    utl::require(this->dev_tile_partition_[dev] < this->tile_partition_count_,
                 function_name, "Device tile partition is out of range");
  }

  this->cells2dev();

  std::cout << std::endl;
  std::cout << "          Number of atoms: " << this->plan_->natom()
            << std::endl;
  std::cout << "    System dimensions [A]: " << this->plan_->box_dim_x()
            << " x " << this->plan_->box_dim_y() << " x "
            << this->plan_->box_dim_z() << std::endl;
  std::cout << "          Number of cells: " << this->plan_->ncell_x() << ", "
            << this->plan_->ncell_y() << ", " << this->plan_->ncell_z()
            << std::endl;
  std::cout << "      Cell dimensions [A]: " << this->plan_->cell_dim_x()
            << " x " << this->plan_->cell_dim_y() << " x "
            << this->plan_->cell_dim_z() << std::endl;
  std::cout
      << "  Total space covered [A]: "
      << static_cast<double>(this->plan_->ncell_x()) * this->plan_->cell_dim_x()
      << " x "
      << static_cast<double>(this->plan_->ncell_y()) * this->plan_->cell_dim_y()
      << " x "
      << static_cast<double>(this->plan_->ncell_z()) * this->plan_->cell_dim_z()
      << std::endl;
  std::cout << std::endl;

  std::string_view mode_name = "UNKNOWN";
  switch (this->execution_mode_) {
  case GLST_EXECUTION_MODE::SINGLE_GPU_TILED:
    mode_name = "SINGLE_GPU_TILED";
    break;
  case GLST_EXECUTION_MODE::MULTI_GPU_CELL:
    mode_name = "MULTI_GPU_CELL";
    break;
  case GLST_EXECUTION_MODE::MULTI_GPU_TILE:
    mode_name = "MULTI_GPU_TILE";
    break;
  case GLST_EXECUTION_MODE::MULTI_GPU_CELL_TILE:
    mode_name = "MULTI_GPU_CELL_TILE";
    break;
  }
  std::cout << "           Number of GPUs: " << this->cuda_count_ << std::endl;
  std::cout << "       GPU layout policy: "
            << (this->gpu_layout_user_set_ ? "MANUAL" : "AUTOMATIC")
            << std::endl;

  if (!this->gpu_layout_user_set_) {
    const double min_free_mib =
        static_cast<double>(this->gpu_layout_min_free_bytes_) /
        (1024.0 * 1024.0);

    const double budget_mib =
        static_cast<double>(this->gpu_layout_memory_budget_bytes_) /
        (1024.0 * 1024.0);

    const double estimated_mib =
        static_cast<double>(this->gpu_layout_estimated_peak_bytes_) /
        (1024.0 * 1024.0);

    std::cout << "        Minimum free GPU memory: " << min_free_mib << " MiB"
              << std::endl;

    std::cout << "           Layout memory budget: " << budget_mib << " MiB"
              << std::endl;

    std::cout << "    Estimated added GLST memory: " << estimated_mib << " MiB"
              << std::endl;
  }

  std::cout << "    GPU layout rationale: " << this->gpu_layout_reason_
            << std::endl;
  std::cout << "          Execution mode: " << mode_name << std::endl;
  std::cout << "         Cell partitions: " << this->cell_partition_count_
            << std::endl;
  std::cout << "         Tile partitions: " << this->tile_partition_count_
            << std::endl;
  std::cout << "              GPU layout: " << std::endl;
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    std::cout << "                  GPU " << dev << " -> cell partition "
              << this->dev_cell_partition_[dev] << ", tile partition "
              << this->dev_tile_partition_[dev] << ", owned cells "
              << this->dev_cell_idx_[dev].size() << std::endl;
  }

  std::string_view sf_exchange_mode_name = "UNKNOWN";

  switch (this->sf_exchange_mode_) {
  case GLST_SF_EXCHANGE_MODE::DISTRIBUTED_PREFIX:
    sf_exchange_mode_name = "DISTRIBUTED_PREFIX";
    break;
  case GLST_SF_EXCHANGE_MODE::LOCAL_CHUNK_BROADCAST:
    sf_exchange_mode_name = "LOCAL_CHUNK_BROADCAST";
    break;
  case GLST_SF_EXCHANGE_MODE::FULL_GLOBAL_ALLREDUCE:
    sf_exchange_mode_name = "FULL_GLOBAL_ALLREDUCE";
    break;
  }

  std::cout << "    S_tile exchange mode: " << sf_exchange_mode_name
            << std::endl;

  if (this->sf_exchange_chunk_x_count_ > 0) {
    std::cout << "   S_tile exchange chunk: "
              << this->sf_exchange_chunk_x_count_ << " x-planes" << std::endl;
  }

  std::cout << " Cell partition x-ranges: " << std::endl;
  for (unsigned int partition = 0;
       partition < this->plan_->cell_partition_count(); partition++) {
    const unsigned int x_point =
        this->plan_->cell_partition_x_point()[partition];
    const unsigned int x_count =
        this->plan_->cell_partition_x_count()[partition];
    std::cout << "        cell partition " << partition << ": x[" << x_point
              << ", " << x_point + x_count << "), cells "
              << this->plan_->partition_cell_idx(partition).size() << std::endl;
  }
  std::cout << " Short-range halo planning: " << std::endl;
  for (unsigned int partition = 0;
       partition < this->plan_->cell_partition_count(); partition++) {
    const std::size_t owned_count =
        this->plan_->partition_cell_idx(partition).size();
    const std::size_t left_count =
        this->plan_->partition_left_halo_cell_idx(partition).size();
    const std::size_t right_count =
        this->plan_->partition_right_halo_cell_idx(partition).size();
    const std::size_t halo_count =
        this->plan_->partition_halo_cell_idx(partition).size();
    const std::size_t source_count =
        this->plan_->partition_sr_source_cell_idx(partition).size();

    std::cout << "        cell partition " << partition << ": owned targets "
              << owned_count << ", halo cells " << halo_count << " (left "
              << left_count << ", right " << right_count
              << "), short-range source cells " << source_count << std::endl;
  }
  std::cout << std::endl;

  this->print_nccl_topology(std::cout);

  std::cout << std::endl;
  std::cout << "  Number of alpha groups: " << this->plan_->ngroup()
            << std::endl;

  std::cout << "       Total number of cubature nodes: "
            << this->plan_->tot_num_nodes() << std::endl;
  for (unsigned int group = 0; group < this->plan_->ngroup(); group++) {
    std::cout << "  Number of cubature nodes in group " << group << ": "
              << this->plan_->num_nodes()[0][group] << std::endl;
  }
  std::cout << std::endl;

  this->plan_->print_tile_diagnostics(std::cout);

  std::cout << std::endl;

  std::cout << "  Workspace allocation:" << std::endl;
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    const std::size_t atom_capacity = this->workspace_->atom_capacity(dev);
    const std::size_t cell_capacity = this->workspace_->cell_capacity(dev);
    const std::size_t sf_capacity =
        this->workspace_->sf_tile_buffer_capacity(dev);
    const std::size_t sf_exchange_capacity =
        this->workspace_->sf_exchange_tile_buffer_capacity(dev);
    const std::size_t rmt_capacity =
        this->workspace_->rmt_tile_buffer_capacity(dev);
    const std::size_t prefix_partition_total_capacity =
        this->workspace_->prefix_partition_total_buffer_capacity(dev);
    const std::size_t prefix_base_capacity =
        this->workspace_->prefix_base_buffer_capacity(dev);
    const std::size_t prefix_slot_capacity =
        this->workspace_->prefix_plane_slot_capacity(dev);

    const double sf_mib =
        static_cast<double>(2 * sf_capacity * sizeof(double)) /
        (1024.0 * 1024.0);
    const double sf_exchange_mib =
        static_cast<double>(2 * sf_exchange_capacity * sizeof(double)) /
        (1024.0 * 1024.0);
    const double rmt_mib =
        static_cast<double>(2 * rmt_capacity * sizeof(double)) /
        (1024.0 * 1024.0);
    const double prefix_partition_total_mib =
        static_cast<double>(2 * prefix_partition_total_capacity *
                            sizeof(double)) /
        (1024.0 * 1024.0);
    const double prefix_base_mib =
        static_cast<double>(2 * prefix_base_capacity * sizeof(double)) /
        (1024.0 * 1024.0);
    const double prefix_slot_mib =
        static_cast<double>(prefix_slot_capacity * sizeof(unsigned int)) /
        (1024.0 * 1024.0);

    std::cout << "    GPU " << dev << ": atom capacity " << atom_capacity
              << ", local cells " << cell_capacity << std::endl;
    std::cout << "      local sf entries " << sf_capacity << " (" << sf_mib
              << " MiB)"
              << ", exchange entries " << sf_exchange_capacity << " ("
              << sf_exchange_mib << " MiB)" << std::endl;
    std::cout << "      prefix partition-total entries "
              << prefix_partition_total_capacity << " ("
              << prefix_partition_total_mib << " MiB)"
              << ", prefix base entries " << prefix_base_capacity << " ("
              << prefix_base_mib << " MiB)" << std::endl;
    std::cout << "      prefix slot entries " << prefix_slot_capacity << " ("
              << prefix_slot_mib << " MiB)"
              << ", rmt tile entries " << rmt_capacity << " (" << rmt_mib
              << " MiB)" << std::endl;
  }

  return;
}

void glst_force::set_gpu_layout(const unsigned int cell_partition_count,
                                const unsigned int tile_partition_count) {
  constexpr std::string_view function_name = "glst_force::set_gpu_layout";

  utl::require(!this->cuda_initialized_, function_name,
               "GPU layout must be set before init");

  utl::require((cell_partition_count > 0) && (tile_partition_count > 0),
               function_name, "Partition counts must be positive");

  this->cell_partition_count_ = cell_partition_count;
  this->tile_partition_count_ = tile_partition_count;
  this->gpu_layout_user_set_ = true;

  return;
}

void glst_force::set_sf_exchange_mode(const GLST_SF_EXCHANGE_MODE mode) {
  utl::require(!this->cuda_initialized_, "glst_force::set_sf_exchange_mode",
               "S_tile exchange mode must be set before init");

  this->sf_exchange_mode_ = mode;

  return;
}

void glst_force::assign_atoms(const double *d_rx, const double *d_ry,
                              const double *d_rz, const double *d_qc) {
  if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
    this->assign_atoms_single_gpu(d_rx, d_ry, d_rz, d_qc);
    return;
  }

  this->assign_atoms_multi_gpu(d_rx, d_ry, d_rz, d_qc);

  return;
}

void glst_force::calc_sf(void) {
  this->require_single_gpu_runtime("calc_sf");
  return;
}

void glst_force::sum_rmt_sf(void) {
  this->require_single_gpu_runtime("sum_rmt_sf");
  return;
}

void glst_force::calc_lr_ef(void) {
  utl::require(this->plan_ != nullptr, "glst_force::calc_lr_ef",
               "Plan is not initialized");

  this->require_single_gpu_runtime("calc_lr_ef");

  this->zero_ef();

  for (unsigned int tile = 0; tile < this->plan_->tile_count(); tile++) {
    this->calc_sf_tile(tile);
    this->build_rmt_sum_tile(tile);
    this->calc_lr_ef_tile(tile);
  }

  return;
}

template <unsigned int BLOCK>
__global__ static void calc_sr_ef_intra_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const double *__restrict__ rx,
    const double *__restrict__ ry, const double *__restrict__ rz,
    const double *__restrict__ qc,
    const unsigned int *__restrict__ cell_atom_points,
    const unsigned int *__restrict__ cell_atom_counts,
    const unsigned int *__restrict__ cells, const unsigned int cell_count,
    const unsigned int first_global_cell) {
  __shared__ double s_cache[BLOCK * 4];
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int c = blockIdx.y; c < cell_count; c += gridDim.y) {
    const unsigned int global_cell = cells[c];
    const unsigned int local_cell = global_cell - first_global_cell;
    const unsigned int apnt = cell_atom_points[local_cell];
    const unsigned int acnt = cell_atom_counts[local_cell];
    const bool active = (idx < acnt);

    double xi = 0.0, yi = 0.0, zi = 0.0, qi = 0.0;
    if (active) {
      xi = rx[apnt + idx];
      yi = ry[apnt + idx];
      zi = rz[apnt + idx];
      qi = qc[apnt + idx];
    }

    double fx0 = 0.0, fy0 = 0.0, fz0 = 0.0, en0 = 0.0;
    for (unsigned int j = 0; j < acnt; j += BLOCK) {
      // Read block of atom data into shared memory
      __syncthreads();
      if (j + threadIdx.x < acnt) {
        s_cache[threadIdx.x * 4 + 0] = rx[apnt + j + threadIdx.x];
        s_cache[threadIdx.x * 4 + 1] = ry[apnt + j + threadIdx.x];
        s_cache[threadIdx.x * 4 + 2] = rz[apnt + j + threadIdx.x];
        s_cache[threadIdx.x * 4 + 3] = qc[apnt + j + threadIdx.x];
      }
      __syncthreads();

      if (active) { // Only "active" threads do expensive calculations
        const unsigned int n = min(BLOCK, acnt - j);
        for (unsigned int k = 0; k < n; k++) {
          if (j + k == idx) // Do not interact self
            continue;
          const double xj = s_cache[k * 4 + 0];
          const double yj = s_cache[k * 4 + 1];
          const double zj = s_cache[k * 4 + 2];
          const double qj = s_cache[k * 4 + 3];
          const double qij = qi * qj;
          const double xij = xi - xj;
          const double yij = yi - yj;
          const double zij = zi - zj;
          const double rij2 = xij * xij + yij * yij + zij * zij;
          const double rij = sqrt(rij2);
          const double irij = 1.0 / rij;
          const double dudr = qij / rij2; // u = qij / rij
          const double drdx = xij * irij;
          const double drdy = yij * irij;
          const double drdz = zij * irij;
          fx0 += dudr * drdx;
          fy0 += dudr * drdy;
          fz0 += dudr * drdz;
          en0 += qij * irij;
        }
      }
    }

    if (active) {
      fx[apnt + idx] += fx0;
      fy[apnt + idx] += fy0;
      fz[apnt + idx] += fz0;
      en[apnt + idx] += en0;
    }
  }

  return;
}

template <unsigned int BLOCK>
__global__ static void calc_sr_ef_inter_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const double *__restrict__ rx,
    const double *__restrict__ ry, const double *__restrict__ rz,
    const double *__restrict__ qc,
    const unsigned int *__restrict__ target_cell_atom_points,
    const unsigned int *__restrict__ target_cell_atom_counts,
    const unsigned int *__restrict__ source_cell_atom_points,
    const unsigned int *__restrict__ source_cell_atom_counts,
    const unsigned int *__restrict__ cells, const unsigned int cell_count,
    const unsigned int first_global_cell, const unsigned int owned_cell_count,
    const unsigned int x_point, const unsigned int x_count,
    const unsigned int left_halo_cell_count, const unsigned int ncell_x,
    const unsigned int ncell_y, const unsigned int ncell_z) {
  __shared__ double s_cache[BLOCK * 4];
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int c = blockIdx.y; c < cell_count; c += gridDim.y) {
    const unsigned int global_cell = cells[c];
    const unsigned int xcell = global_cell / (ncell_y * ncell_z);
    const unsigned int ycell = (global_cell / ncell_z) % ncell_y;
    const unsigned int zcell = global_cell % ncell_z;

    const unsigned int target_local_cell = global_cell - first_global_cell;
    const unsigned int apnt = target_cell_atom_points[target_local_cell];
    const unsigned int acnt = target_cell_atom_counts[target_local_cell];
    const bool active = (idx < acnt);

    double xi = 0.0, yi = 0.0, zi = 0.0, qi = 0.0;
    if (active) {
      xi = rx[apnt + idx];
      yi = ry[apnt + idx];
      zi = rz[apnt + idx];
      qi = qc[apnt + idx];
    }

    double fx0 = 0.0, fy0 = 0.0, fz0 = 0.0, en0 = 0.0;
    for (int a = -1; a <= 1; a++) {
      if ((xcell == 0) && (a == -1))
        continue;
      if ((xcell == ncell_x - 1) && (a == 1))
        continue;
      const unsigned int nbrx =
          static_cast<unsigned int>(static_cast<int>(xcell) + a);
      for (int b = -1; b <= 1; b++) {
        if ((ycell == 0) && (b == -1))
          continue;
        if ((ycell == ncell_y - 1) && (b == 1))
          continue;
        const unsigned int nbry =
            static_cast<unsigned int>(static_cast<int>(ycell) + b);
        for (int c = -1; c <= 1; c++) {
          if ((zcell == 0) && (c == -1))
            continue;
          if ((zcell == ncell_z - 1) && (c == 1))
            continue;
          const unsigned int nbrz =
              static_cast<unsigned int>(static_cast<int>(zcell) + c);
          if ((a == 0) && (b == 0) && (c == 0))
            continue;
          const unsigned int nbr = (nbrx * ncell_y + nbry) * ncell_z + nbrz;

          const unsigned int x_end = x_point + x_count;
          unsigned int source_local_cell = 0;

          if ((nbrx >= x_point) && (nbrx < x_end))
            source_local_cell = nbr - first_global_cell;
          else if ((x_point > 0) && (nbrx == x_point - 1))
            source_local_cell = owned_cell_count + nbry * ncell_z + nbrz;
          else if ((x_end < ncell_x) && (nbrx == x_end)) {
            source_local_cell =
                owned_cell_count + left_halo_cell_count + nbry * ncell_z + nbrz;
          } else
            continue;

          const unsigned int bpnt = source_cell_atom_points[source_local_cell];
          const unsigned int bcnt = source_cell_atom_counts[source_local_cell];
          for (unsigned int j = 0; j < bcnt; j += BLOCK) {
            // Read block of atom data into shared memory
            __syncthreads();
            if (j + threadIdx.x < bcnt) {
              s_cache[threadIdx.x * 4 + 0] = rx[bpnt + j + threadIdx.x];
              s_cache[threadIdx.x * 4 + 1] = ry[bpnt + j + threadIdx.x];
              s_cache[threadIdx.x * 4 + 2] = rz[bpnt + j + threadIdx.x];
              s_cache[threadIdx.x * 4 + 3] = qc[bpnt + j + threadIdx.x];
            }
            __syncthreads();

            if (active) { // Only "active" threads do expensive calculations
              const unsigned int n = min(BLOCK, bcnt - j);
              for (unsigned int k = 0; k < n; k++) {
                const double xj = s_cache[k * 4 + 0];
                const double yj = s_cache[k * 4 + 1];
                const double zj = s_cache[k * 4 + 2];
                const double qj = s_cache[k * 4 + 3];
                const double qij = qi * qj;
                const double xij = xi - xj;
                const double yij = yi - yj;
                const double zij = zi - zj;
                const double rij2 = xij * xij + yij * yij + zij * zij;
                const double rij = sqrt(rij2);
                const double irij = 1.0 / rij;
                const double dudr = qij / rij2; // u = qij / rij
                const double drdx = xij * irij;
                const double drdy = yij * irij;
                const double drdz = zij * irij;
                fx0 += dudr * drdx;
                fy0 += dudr * drdy;
                fz0 += dudr * drdz;
                en0 += qij * irij;
              }
            }
          }
        }
      }
    }

    if (active) {
      fx[apnt + idx] += fx0;
      fy[apnt + idx] += fy0;
      fz[apnt + idx] += fz0;
      en[apnt + idx] += en0;
    }
  }

  return;
}

void glst_force::calc_sr_ef(void) {
  constexpr std::string_view function_name = "glst_force::calc_sr_ef";

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    if ((this->execution_mode_ != GLST_EXECUTION_MODE::SINGLE_GPU_TILED) &&
        (this->dev_tile_partition_[dev] != 0)) {
      continue;
    }

    cudaCheck(cudaSetDevice(dev));

    const unsigned int cell_partition = this->dev_cell_partition_[dev];
    const unsigned int first_global_cell =
        this->plan_->first_global_cell(cell_partition);
    const unsigned int owned_cell_count =
        static_cast<unsigned int>(this->dev_cell_idx_[dev].size());

    if (owned_cell_count == 0)
      continue;

    const unsigned int max_atoms_cell = this->workspace_->max_atoms_cell()[dev];

    if (max_atoms_cell == 0)
      continue;

    const unsigned int x_point =
        this->plan_->cell_partition_x_point()[cell_partition];
    const unsigned int x_count =
        this->plan_->cell_partition_x_count()[cell_partition];
    const unsigned int left_halo_cell_count = static_cast<unsigned int>(
        this->plan_->partition_left_halo_cell_idx(cell_partition).size());

    const unsigned int *cell_atom_point = nullptr;
    const unsigned int *cell_atom_count = nullptr;

    if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
      cell_atom_point =
          this->workspace_->cell_atom_point()[dev].d_array().data();
      cell_atom_count =
          this->workspace_->cell_atom_count()[dev].d_array().data();
    } else {
      utl::require(static_cast<std::size_t>(owned_cell_count) <=
                       this->workspace_->sr_source_cell_capacity(dev),
                   function_name,
                   "Owned-cell count exceeds source-cell metadata capacity");
      cell_atom_point =
          this->workspace_->sr_source_cell_atom_point()[dev].data();
      cell_atom_count =
          this->workspace_->sr_source_cell_atom_count()[dev].data();
    }

    constexpr dim3 num_threads(64, 1, 1);
    const dim3 num_blocks((max_atoms_cell + num_threads.x - 1) / num_threads.x,
                          std::min(65535u, owned_cell_count), 1);

    calc_sr_ef_intra_kernel<num_threads.x>
        <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
            this->workspace_->fx()[dev].d_array().data(),
            this->workspace_->fy()[dev].d_array().data(),
            this->workspace_->fz()[dev].d_array().data(),
            this->workspace_->en()[dev].d_array().data(),
            this->workspace_->rx()[dev].d_array().data(),
            this->workspace_->ry()[dev].d_array().data(),
            this->workspace_->rz()[dev].d_array().data(),
            this->workspace_->qc()[dev].d_array().data(), cell_atom_point,
            cell_atom_count, this->dev_cell_idx_[dev].d_array().data(),
            static_cast<unsigned int>(this->dev_cell_idx_[dev].size()),
            first_global_cell);

    cudaCheck(cudaGetLastError());

    calc_sr_ef_inter_kernel<num_threads.x>
        <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
            this->workspace_->fx()[dev].d_array().data(),
            this->workspace_->fy()[dev].d_array().data(),
            this->workspace_->fz()[dev].d_array().data(),
            this->workspace_->en()[dev].d_array().data(),
            this->workspace_->rx()[dev].d_array().data(),
            this->workspace_->ry()[dev].d_array().data(),
            this->workspace_->rz()[dev].d_array().data(),
            this->workspace_->qc()[dev].d_array().data(), cell_atom_point,
            cell_atom_count, cell_atom_point, cell_atom_count,
            this->dev_cell_idx_[dev].d_array().data(),
            static_cast<unsigned int>(this->dev_cell_idx_[dev].size()),
            first_global_cell, owned_cell_count, x_point, x_count,
            left_halo_cell_count, this->plan_->ncell_x(),
            this->plan_->ncell_y(), this->plan_->ncell_z());

    cudaCheck(cudaGetLastError());
  }

  return;
}

void glst_force::comm_ef(void) {
  this->reduce_tile_partition_ef();

  // Preserve the existing synchronous public-method behavior. For
  // tile-decomposed cases, each compute stream waits on its
  // communication-completion event before this synchronization.
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamSynchronize(this->comp_streams_[dev]));
  }

  return;
}

void glst_force::calc_ener_force(const double *d_rx, const double *d_ry,
                                 const double *d_rz, const double *d_qc) {
  constexpr std::string_view function_name = "glst_force::calc_ener_force";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  if (!this->profiling_enabled_) {
    // In multi-GPU mode, assign_atoms_multi_gpu also constructs and distributes
    // the owned-plus-halo short-range source arrays
    this->assign_atoms(d_rx, d_ry, d_rz, d_qc);

    // Assignment may grow local atom storage, but never shrinks it. Zero the
    // active writable target prefix after assignment updates the active counts.
    this->zero_ef();

    // Iterate in the canonical global tile order. Each tile-aware method
    // filters execution to the devices that own the tile's tile partition.
    for (unsigned int tile = 0; tile < this->plan_->tile_count(); tile++) {
      this->calc_sf_tile(tile);
      this->build_rmt_sum_tile(tile);
      this->calc_lr_ef_tile(tile);
    }

    // Only tile partition 0 computes short-range interactions in multi-GPU
    // modes. The subsequent tile-partition reduction adds the long-range
    // contributions from all other tile ranks.
    this->calc_sr_ef();
    this->comm_ef();

    return;
  }

  this->reset_profile();

  std::size_t atom_storage_growth_before = 0;
  std::size_t cub_work_buffer_growth_before = 0;

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    atom_storage_growth_before +=
        this->workspace_->atom_storage_growth_count(dev);

    cub_work_buffer_growth_before +=
        this->workspace_->cub_work_buffer_growth_count(dev);
  }

  const std::chrono::steady_clock::time_point total_start =
      std::chrono::steady_clock::now();

  this->assign_atoms(d_rx, d_ry, d_rz, d_qc);

  std::size_t atom_storage_growth_after = 0;
  std::size_t cub_work_buffer_growth_after = 0;

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    atom_storage_growth_after +=
        this->workspace_->atom_storage_growth_count(dev);

    cub_work_buffer_growth_after +=
        this->workspace_->cub_work_buffer_growth_count(dev);
  }

  utl::require(atom_storage_growth_after >= atom_storage_growth_before,
               function_name, "Atom-storage growth counter decreased");

  utl::require(cub_work_buffer_growth_after >= cub_work_buffer_growth_before,
               function_name, "CUB growth counter decreased");

  this->profile_.atom_storage_growth_events =
      atom_storage_growth_after - atom_storage_growth_before;
  this->profile_.cub_work_buffer_growth_events =
      cub_work_buffer_growth_after - cub_work_buffer_growth_before;

  std::chrono::steady_clock::time_point phase_start =
      std::chrono::steady_clock::now();

  this->zero_ef();
  this->synchronize_compute_streams();

  std::chrono::steady_clock::time_point phase_end =
      std::chrono::steady_clock::now();

  this->profile_.zero_ef_ms = profile_elapsed_ms(phase_start, phase_end);

  for (unsigned int tile = 0; tile < this->plan_->tile_count(); tile++) {
    const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);

    phase_start = std::chrono::steady_clock::now();
    this->calc_sf_tile(tile);
    this->synchronize_tile_partition_compute_streams(tile_partition);
    phase_end = std::chrono::steady_clock::now();
    this->profile_.calc_sf_ms += profile_elapsed_ms(phase_start, phase_end);

    // build_rmt_sum_tile owns:
    // - Full-buffer all-reduce timing and byte accounting
    // - Local chunk-broadcast timing and byte accounting
    // - Distributed-prefix timing and byte accounting
    // - Remote-sum timing and byte accounting
    this->build_rmt_sum_tile(tile);

    phase_start = std::chrono::steady_clock::now();
    this->calc_lr_ef_tile(tile);
    this->synchronize_tile_partition_compute_streams(tile_partition);
    phase_end = std::chrono::steady_clock::now();
    this->profile_.calc_lr_ef_ms += profile_elapsed_ms(phase_start, phase_end);
  }

  phase_start = std::chrono::steady_clock::now();
  this->calc_sr_ef();
  this->synchronize_tile_partition_compute_streams(0);
  phase_end = std::chrono::steady_clock::now();
  this->profile_.calc_sr_ef_ms = profile_elapsed_ms(phase_start, phase_end);

  phase_start = std::chrono::steady_clock::now();
  this->comm_ef();
  phase_end = std::chrono::steady_clock::now();
  this->profile_.reduce_tile_ef_ms = profile_elapsed_ms(phase_start, phase_end);

  const std::chrono::steady_clock::time_point total_end =
      std::chrono::steady_clock::now();

  this->profile_.instrumented_compute_ms =
      profile_elapsed_ms(total_start, total_end);

  return;
}

template <unsigned int ATOM_TILE>
__global__ static void
calc_sf_kernel(double *__restrict__ sf_re, double *__restrict__ sf_im,
               const double *__restrict__ cx, const double *__restrict__ cy,
               const double *__restrict__ cz, const unsigned int nc,
               const double *__restrict__ rx, const double *__restrict__ ry,
               const double *__restrict__ rz, const double *__restrict__ qc,
               const unsigned int *__restrict__ cell_atom_points,
               const unsigned int *__restrict__ cell_atom_counts,
               const unsigned int local_cell_count,
               const unsigned int sf_cell_point) {
  __shared__ double s_cache[ATOM_TILE * 4];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = (idx < nc);

  double xc = 0.0, yc = 0.0, zc = 0.0;
  if (active) {
    xc = cx[idx];
    yc = cy[idx];
    zc = cz[idx];
  }

  for (unsigned int local_cell = blockIdx.y; local_cell < local_cell_count;
       local_cell += gridDim.y) {
    const unsigned int sf_cell = sf_cell_point + local_cell;
    const unsigned int apnt = cell_atom_points[local_cell];
    const unsigned int acnt = cell_atom_counts[local_cell];

    double sf_re0 = 0.0, sf_im0 = 0.0;
    for (unsigned int i = 0; i < acnt; i += ATOM_TILE) {
      __syncthreads();
      if ((threadIdx.x < ATOM_TILE) && (i + threadIdx.x < acnt)) {
        s_cache[threadIdx.x * 4 + 0] = rx[apnt + i + threadIdx.x];
        s_cache[threadIdx.x * 4 + 1] = ry[apnt + i + threadIdx.x];
        s_cache[threadIdx.x * 4 + 2] = rz[apnt + i + threadIdx.x];
        s_cache[threadIdx.x * 4 + 3] = qc[apnt + i + threadIdx.x];
      }
      __syncthreads();

      if (active) { // Only "active" threads do expensive sincos work
        const unsigned int n = min(ATOM_TILE, acnt - i);
        for (unsigned int j = 0; j < n; j++) {
          const double xa = s_cache[j * 4 + 0];
          const double ya = s_cache[j * 4 + 1];
          const double za = s_cache[j * 4 + 2];
          const double qa = s_cache[j * 4 + 3];

          const double theta = xc * xa + yc * ya + zc * za;
          double re = 0.0, im = 0.0;
          sincos(theta, &im, &re);

          sf_re0 += qa * re;
          sf_im0 -= qa * im;
        }
      }
    }

    if (active) {
      sf_re[sf_cell * nc + idx] = sf_re0;
      sf_im[sf_cell * nc + idx] = sf_im0;
    }
  }

  return;
}

__global__ static void
init_cell_atom_count_kernel(unsigned int *__restrict__ cell_atom_count,
                            const unsigned int ncell) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < ncell)
    cell_atom_count[idx] = 0;

  return;
}

__global__ void copy_coords_kernel(
    double *__restrict__ rx, double *__restrict__ ry, double *__restrict__ rz,
    double *__restrict__ qc, const double *__restrict__ d_rx,
    const double *__restrict__ d_ry, const double *__restrict__ d_rz,
    const double *__restrict__ d_qc, const unsigned int natom) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < natom) {
    rx[idx] = d_rx[idx];
    ry[idx] = d_ry[idx];
    rz[idx] = d_rz[idx];
    qc[idx] = d_qc[idx];
  }

  return;
}

__global__ static void calc_cell_atom_count_kernel(
    unsigned int *__restrict__ atom_cell_idx,
    unsigned int *__restrict__ cell_atom_count, const double *__restrict__ rx,
    const double *__restrict__ ry, const double *__restrict__ rz,
    const unsigned int natom, const double cell_dim_x, const double cell_dim_y,
    const double cell_dim_z, const unsigned int ncell_x,
    const unsigned int ncell_y, const unsigned int ncell_z) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const double inv_cell_dim_x = 1.0 / cell_dim_x;
  const double inv_cell_dim_y = 1.0 / cell_dim_y;
  const double inv_cell_dim_z = 1.0 / cell_dim_z;

  if (idx < natom) {
    int cx = static_cast<int>(rx[idx] * inv_cell_dim_x);
    int cy = static_cast<int>(ry[idx] * inv_cell_dim_y);
    int cz = static_cast<int>(rz[idx] * inv_cell_dim_z);

    cx = (cx >= static_cast<int>(ncell_x)) ? static_cast<int>(ncell_x - 1) : cx;
    cy = (cy >= static_cast<int>(ncell_y)) ? static_cast<int>(ncell_y - 1) : cy;
    cz = (cz >= static_cast<int>(ncell_z)) ? static_cast<int>(ncell_z - 1) : cz;

    cx = (cx < 0) ? 0 : cx;
    cy = (cy < 0) ? 0 : cy;
    cz = (cz < 0) ? 0 : cz;

    unsigned int cell = (static_cast<unsigned int>(cx) * ncell_y +
                         static_cast<unsigned int>(cy)) *
                            ncell_z +
                        static_cast<unsigned int>(cz);

    atom_cell_idx[idx] = cell;
    atomicAdd(&cell_atom_count[cell], 1);
  }

  return;
}

__global__ static void
calc_cell_atom_point_kernel(unsigned int *__restrict__ cell_atom_point,
                            const unsigned int *__restrict__ cell_atom_count,
                            const unsigned int ncell) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < ncell) {
    unsigned int point = 0;

    for (unsigned int cell = 0; cell < idx; cell++)
      point += cell_atom_count[cell];

    cell_atom_point[idx] = point;
  }

  return;
}

__global__ static void
pack_kernel(atom_packet *__restrict__ packet, const double *__restrict__ rx,
            const double *__restrict__ ry, const double *__restrict__ rz,
            const double *__restrict__ qc, const unsigned int *__restrict__ id,
            const unsigned int *__restrict__ atom_cell_idx,
            const unsigned int natom) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < natom) {
    packet[idx] = atom_packet(id[idx], atom_cell_idx[idx], rx[idx], ry[idx],
                              rz[idx], qc[idx]);
  }

  return;
}

__global__ static void unpack_kernel(
    double *__restrict__ rx, double *__restrict__ ry, double *__restrict__ rz,
    double *__restrict__ qc, unsigned int *__restrict__ id,
    const atom_packet *__restrict__ packet, const unsigned int natom) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < natom) {
    rx[idx] = packet[idx].x;
    ry[idx] = packet[idx].y;
    rz[idx] = packet[idx].z;
    qc[idx] = packet[idx].q;
    id[idx] = packet[idx].i;
  }

  return;
}

void glst_force::assign_atoms_single_gpu(const double *d_rx, const double *d_ry,
                                         const double *d_rz,
                                         const double *d_qc) {
  std::chrono::steady_clock::time_point assignment_start;
  if (this->profiling_enabled_)
    assignment_start = std::chrono::steady_clock::now();

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    this->workspace_->set_atom_counts(
        dev, static_cast<std::size_t>(this->plan_->natom()),
        static_cast<std::size_t>(this->plan_->natom()));

    { // Fast reset of cell atom count array
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->plan_->ncell() + num_threads - 1) / num_threads;

      init_cell_atom_count_kernel<<<num_blocks, num_threads, 0,
                                    this->comp_streams_[dev]>>>(
          this->workspace_->cell_atom_count()[dev].d_array().data(),
          this->plan_->ncell());

      cudaCheck(cudaGetLastError());
    }

    { // Store input coordinates
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->plan_->natom() + num_threads - 1) / num_threads;

      copy_coords_kernel<<<num_blocks, num_threads, 0,
                           this->comp_streams_[dev]>>>(
          this->workspace_->rx()[dev].d_array().data(),
          this->workspace_->ry()[dev].d_array().data(),
          this->workspace_->rz()[dev].d_array().data(),
          this->workspace_->qc()[dev].d_array().data(), d_rx, d_ry, d_rz, d_qc,
          this->plan_->natom());

      cudaCheck(cudaGetLastError());
    }

    { // Determine which cell each atom is in and count how many atoms are in
      // each cell
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->plan_->natom() + num_threads - 1) / num_threads;

      calc_cell_atom_count_kernel<<<num_blocks, num_threads, 0,
                                    this->comp_streams_[dev]>>>(
          this->workspace_->atom_cell_idx()[dev].d_array().data(),
          this->workspace_->cell_atom_count()[dev].d_array().data(),
          this->workspace_->rx()[dev].d_array().data(),
          this->workspace_->ry()[dev].d_array().data(),
          this->workspace_->rz()[dev].d_array().data(), this->plan_->natom(),
          this->plan_->cell_dim_x(), this->plan_->cell_dim_y(),
          this->plan_->cell_dim_z(), this->plan_->ncell_x(),
          this->plan_->ncell_y(), this->plan_->ncell_z());

      cudaCheck(cudaGetLastError());
    }

    // JEG260127: Find optimial place to do this
    cudaCheck(cudaStreamSynchronize(this->comp_streams_[dev]));

    this->workspace_->cell_atom_count()[dev].transfer_to_host();
    this->workspace_->max_atoms_cell()[dev] = 0;
    for (unsigned int cell = 0; cell < this->plan_->ncell(); cell++) {
      this->workspace_->max_atoms_cell()[dev] =
          (this->workspace_->cell_atom_count()[dev][cell] >
           this->workspace_->max_atoms_cell()[dev])
              ? this->workspace_->cell_atom_count()[dev][cell]
              : this->workspace_->max_atoms_cell()[dev];
    }

    { // Determine where each cell's atom data is stored
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->plan_->ncell() + num_threads - 1) / num_threads;

      calc_cell_atom_point_kernel<<<num_blocks, num_threads, 0,
                                    this->comp_streams_[dev]>>>(
          this->workspace_->cell_atom_point()[dev].d_array().data(),
          this->workspace_->cell_atom_count()[dev].d_array().data(),
          this->plan_->ncell());

      cudaCheck(cudaGetLastError());
    }

    { // Pack atom data
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->plan_->natom() + num_threads - 1) / num_threads;

      pack_kernel<<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
          this->workspace_->packets()[dev].d_array().data(),
          this->workspace_->rx()[dev].d_array().data(),
          this->workspace_->ry()[dev].d_array().data(),
          this->workspace_->rz()[dev].d_array().data(),
          this->workspace_->qc()[dev].d_array().data(),
          this->workspace_->idx()[dev].d_array().data(),
          this->workspace_->atom_cell_idx()[dev].d_array().data(),
          this->plan_->natom());

      cudaCheck(cudaGetLastError());
    }

    { // Sort atoms based on cell indices
      std::size_t cub_work_buffer_size =
          this->workspace_->cub_work_buffer_size()[dev];

      cub::DeviceRadixSort::SortPairs(
          this->workspace_->cub_work_buffer()[dev], cub_work_buffer_size,
          this->workspace_->atom_cell_idx()[dev].d_array().data(),
          this->workspace_->atom_cell_sorted_idx()[dev].d_array().data(),
          this->workspace_->packets()[dev].d_array().data(),
          this->workspace_->sorted_packets()[dev].d_array().data(),
          this->plan_->natom(), 0, static_cast<int>(8 * sizeof(unsigned int)),
          this->comp_streams_[dev]);
    }

    { // Unpack atom data
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->plan_->natom() + num_threads - 1) / num_threads;

      unpack_kernel<<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
          this->workspace_->rx()[dev].d_array().data(),
          this->workspace_->ry()[dev].d_array().data(),
          this->workspace_->rz()[dev].d_array().data(),
          this->workspace_->qc()[dev].d_array().data(),
          this->workspace_->sorted_idx()[dev].d_array().data(),
          this->workspace_->sorted_packets()[dev].d_array().data(),
          this->plan_->natom());

      cudaCheck(cudaGetLastError());
    }
  }

  if (this->profiling_enabled_) {
    this->synchronize_compute_streams();

    const std::chrono::steady_clock::time_point assignment_end =
        std::chrono::steady_clock::now();

    this->profile_.atom_assignment_scatter_ms =
        profile_elapsed_ms(assignment_start, assignment_end);

    this->profile_.owned_halo_source_scatter_ms = 0.0;
    this->profile_.owned_atom_replicas =
        static_cast<std::size_t>(this->plan_->natom());
    this->profile_.halo_atom_replicas = 0;
  }

  return;
}

__global__ static void classify_pack_global_kernel(
    atom_sort_key *__restrict__ sort_key, atom_packet *__restrict__ packet,
    unsigned int *__restrict__ cell_atom_count, const double *__restrict__ rx,
    const double *__restrict__ ry, const double *__restrict__ rz,
    const double *__restrict__ qc, const unsigned int natom,
    const double cell_dim_x, const double cell_dim_y, const double cell_dim_z,
    const unsigned int ncell_x, const unsigned int ncell_y,
    const unsigned int ncell_z) {
  constexpr unsigned int ATOM_INDEX_BITS =
      static_cast<unsigned int>(8u * sizeof(unsigned int));

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = gridDim.x * blockDim.x;

  for (unsigned int atom = idx; atom < natom; atom += stride) {
    int cx = static_cast<int>(rx[atom] / cell_dim_x);
    int cy = static_cast<int>(ry[atom] / cell_dim_y);
    int cz = static_cast<int>(rz[atom] / cell_dim_z);

    // Keep this operation order identical to the existing host and single-GPU
    // classification paths: Upper clamp first, then lower clamp.
    cx = (cx >= static_cast<int>(ncell_x) ? static_cast<int>(ncell_x - 1) : cx);
    cy = (cy >= static_cast<int>(ncell_y) ? static_cast<int>(ncell_y - 1) : cy);
    cz = (cz >= static_cast<int>(ncell_z) ? static_cast<int>(ncell_z - 1) : cz);

    cx = (cx < 0) ? 0 : cx;
    cy = (cy < 0) ? 0 : cy;
    cz = (cz < 0) ? 0 : cz;

    const unsigned int global_cell = (static_cast<unsigned int>(cx) * ncell_y +
                                      static_cast<unsigned int>(cy)) *
                                         ncell_z +
                                     static_cast<unsigned int>(cz);

    const atom_sort_key key =
        (static_cast<atom_sort_key>(global_cell) << ATOM_INDEX_BITS) |
        static_cast<atom_sort_key>(atom);

    sort_key[atom] = key;
    packet[atom] =
        atom_packet(atom, global_cell, rx[atom], ry[atom], rz[atom], qc[atom]);

    atomicAdd(&cell_atom_count[global_cell], 1u);
  }

  return;
}

__global__ static void finalize_global_atom_points_kernel(
    unsigned int *__restrict__ cell_atom_point,
    unsigned int *__restrict__ x_plane_atom_point, const unsigned int natom,
    const unsigned int ncell, const unsigned int ncell_x,
    const unsigned int yz_cell_count) {
  // One block is deliberately used for this kernel. The block-wide barrier
  // guarantees that cell_atom_point[ncell] exists before the terminal x-plane
  // reads it.
  if (threadIdx.x == 0)
    cell_atom_point[ncell] = natom;

  __syncthreads();

  for (unsigned int x = threadIdx.x; x <= ncell_x; x += blockDim.x) {
    const unsigned int first_cell = x * yz_cell_count;
    x_plane_atom_point[x] = cell_atom_point[first_cell];
  }

  return;
}

void glst_force::build_global_atom_reference(const double *d_rx,
                                             const double *d_ry,
                                             const double *d_rz,
                                             const double *d_qc) {
  constexpr std::string_view function_name =
      "glst_force::build_global_atom_reference";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  utl::require(
      this->execution_mode_ != GLST_EXECUTION_MODE::SINGLE_GPU_TILED,
      function_name,
      "Global reference scratch is allocated only for multi-GPU execution");

  utl::require((d_rx != nullptr) && (d_ry != nullptr) && (d_rz != nullptr) &&
                   (d_qc != nullptr),
               function_name, "At least one input atom array is null");

  utl::require(!this->comp_streams_.empty(), function_name,
               "GPU 0 compute stream is not initialized");

  const unsigned int natom = this->plan_->natom();
  const unsigned int ncell = this->plan_->ncell();
  const unsigned int ncell_x = this->plan_->ncell_x();
  const unsigned int ncell_y = this->plan_->ncell_y();
  const unsigned int ncell_z = this->plan_->ncell_z();
  const unsigned int yz_cell_count = ncell_y * ncell_z;

  const std::size_t atom_count = static_cast<std::size_t>(natom);
  const std::size_t cell_count = static_cast<std::size_t>(ncell);
  const std::size_t cell_point_count = cell_count + 1u;
  const std::size_t x_plane_point_count =
      static_cast<std::size_t>(ncell_x) + 1u;

  const std::size_t cub_item_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());

  utl::require(atom_count <= cub_item_limit, function_name,
               "Global atom count exceeds the CUB int item-count range");

  utl::require(cell_count <= cub_item_limit, function_name,
               "Global cell count exceeds the CUB int item-count range");

  utl::require(this->workspace_->global_sort_key_in().size() == atom_count,
               function_name, "Global input-key capacity is incorrect");

  utl::require(this->workspace_->global_sort_key_out().size() == atom_count,
               function_name, "Global output-key capacity is incorrect");

  utl::require(this->workspace_->global_packet_in().size() == atom_count,
               function_name, "Global input-packet capacity is incorrect");

  utl::require(this->workspace_->global_packet_out().size() == atom_count,
               function_name, "Global output-packet capacity is incorrect");

  utl::require(this->workspace_->global_cell_atom_count().size() == cell_count,
               function_name, "Global cell-count capacity is incorrect");

  utl::require(this->workspace_->global_cell_atom_point().size() ==
                   cell_point_count,
               function_name, "Global cell-point capacity is incorrect");

  utl::require(this->workspace_->global_x_plane_atom_point().size() ==
                   x_plane_point_count,
               function_name, "Global x-plane-point capacity is incorrect");

  utl::require(this->workspace_->global_max_atoms_cell().size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name,
               "Global partition max-atoms capacity is incorrect");

  utl::require(!this->workspace_->cub_work_buffer().empty(), function_name,
               "GPU 0 CUB work-buffer array is empty");

  utl::require(!this->workspace_->cub_work_buffer_size().empty(), function_name,
               "GPU 0 CUB work-buffer-size array is empty");

  cudaCheck(cudaSetDevice(0));

  const cudaStream_t stream = this->comp_streams_[0];

  // Step 1: Asynchronously reset global cell counts.
  cudaCheck(cudaMemsetAsync(
      static_cast<void *>(this->workspace_->global_cell_atom_count().data()), 0,
      cell_count * sizeof(unsigned int), stream));

  // Step 2: Classify every atom, construct its packet, construct its complete
  // deterministic sort key, and increment its global cell count.
  {
    constexpr unsigned int num_threads = 512;
    const unsigned int num_blocks = (natom + num_threads - 1u) / num_threads;

    classify_pack_global_kernel<<<num_blocks, num_threads, 0, stream>>>(
        this->workspace_->global_sort_key_in().data(),
        this->workspace_->global_packet_in().data(),
        this->workspace_->global_cell_atom_count().data(), d_rx, d_ry, d_rz,
        d_qc, natom, this->plan_->cell_dim_x(), this->plan_->cell_dim_y(),
        this->plan_->cell_dim_z(), ncell_x, ncell_y, ncell_z);
    cudaCheck(cudaGetLastError());
  }

  void *const cub_work_buffer = this->workspace_->cub_work_buffer()[0];

  const std::size_t cub_work_buffer_capacity =
      this->workspace_->cub_work_buffer_size()[0];

  utl::require(cub_work_buffer != nullptr, function_name,
               "GPU 0 CUB work buffer is null");

  utl::require(cub_work_buffer_capacity > 0, function_name,
               "GPU 0 CUB work-buffer capacity is zero");

  // Step 3: one radix sort over the complete (cell, original-index) key.
  //
  // Reset temp_storage_bytes to the complete capacity before every CUB call.
  // CUB treats this parameter as both input and output.

  std::size_t cub_work_buffer_size = cub_work_buffer_capacity;

  cudaCheck(cub::DeviceRadixSort::SortPairs(
      cub_work_buffer, cub_work_buffer_size,
      this->workspace_->global_sort_key_in().data(),
      this->workspace_->global_sort_key_out().data(),
      this->workspace_->global_packet_in().data(),
      this->workspace_->global_packet_out().data(), static_cast<int>(natom), 0,
      static_cast<int>(8u * sizeof(atom_sort_key)), stream));

  // Step 4: Exclusive scan over global cell counts.
  cub_work_buffer_size = cub_work_buffer_capacity;

  cudaCheck(cub::DeviceScan::ExclusiveSum(
      cub_work_buffer, cub_work_buffer_size,
      this->workspace_->global_cell_atom_count().data(),
      this->workspace_->global_cell_atom_point().data(),
      static_cast<int>(ncell), stream));

  // Step 5: Compute the maximum owned-cell population for each cell partition.
  // X-slab ownership makes every partition's global cell-count range
  // contiguous.
  for (unsigned int partition = 0; partition < this->cell_partition_count_;
       partition++) {
    const unsigned int first_global_cell =
        this->plan_->first_global_cell(partition);
    const unsigned int owned_cell_count =
        this->plan_->local_cell_count(partition);

    utl::require(first_global_cell <= ncell, function_name,
                 "Partition first global cell is out of range");

    utl::require(owned_cell_count <= ncell - first_global_cell, function_name,
                 "Partition owned-cell range is out of bounds");

    unsigned int *partition_max =
        this->workspace_->global_max_atoms_cell().data() + partition;

    if (owned_cell_count == 0u) {
      cudaCheck(cudaMemsetAsync(static_cast<void *>(partition_max), 0,
                                sizeof(unsigned int), stream));
      continue;
    }

    cub_work_buffer_size = cub_work_buffer_capacity;

    cudaCheck(cub::DeviceReduce::Max(
        cub_work_buffer, cub_work_buffer_size,
        this->workspace_->global_cell_atom_count().data() + first_global_cell,
        partition_max, static_cast<int>(owned_cell_count), stream));
  }

  // Step 6 and 7: Append the terminal cell point and derive x-plane points.
  {
    constexpr unsigned int num_threads = 256;

    finalize_global_atom_points_kernel<<<1, num_threads, 0, stream>>>(
        this->workspace_->global_cell_atom_point().data(),
        this->workspace_->global_x_plane_atom_point().data(), natom, ncell,
        ncell_x, yz_cell_count);
    cudaCheck(cudaGetLastError());
  }

  // Deliberately asynchronous. The caller may enqueue later GPU work on the
  // same stream. The targeted test synchronizes once after the complete
  // pipeline before copying reference results to the host.

  return;
}

void glst_force::scatter_sorted_atom_packets(const double *d_rx,
                                             const double *d_ry,
                                             const double *d_rz,
                                             const double *d_qc) {
  constexpr std::string_view function_name =
      "glst_force::scatter_sorted_atom_packets";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  utl::require(this->execution_mode_ != GLST_EXECUTION_MODE::SINGLE_GPU_TILED,
               function_name,
               "Packet scatter requires a multi-GPU execution mode");

  const std::size_t device_count = static_cast<std::size_t>(this->cuda_count_);

  utl::require((this->comp_streams_.size() == device_count) &&
                   (this->comm_streams_.size() == device_count) &&
                   (this->comp_events_.size() == device_count) &&
                   (this->comm_events_.size() == device_count),
               function_name, "CUDA stream/event topology is incomplete");

  utl::require(this->cell_comm_devs_.size() ==
                   static_cast<std::size_t>(this->tile_partition_count_),
               function_name,
               "Cell communicator topology does not match G_tile");

  utl::require(this->cell_comms_.size() ==
                   static_cast<std::size_t>(this->tile_partition_count_),
               function_name, "Cell communicator handles do not match G_tile");

  utl::require(!this->cell_comm_devs_.empty(), function_name,
               "Tile-partition-0 cell communicator is missing");

  const std::vector<int> &root_devs = this->cell_comm_devs_[0];
  const std::vector<ncclComm_t> &root_comms = this->cell_comms_[0];

  utl::require(root_devs.size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name,
               "Cell-root communicator device count does not match G_cell");

  utl::require(root_comms.size() == root_devs.size(), function_name,
               "Cell-root communicator handle count is incorrect");

  utl::require(this->tile_comm_devs_.size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name,
               "Tile communicator topology does not match G_cell");

  utl::require(this->tile_comms_.size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name, "Tile communicator handles do not match G_cell");

  for (unsigned int partition = 0; partition < this->cell_partition_count_;
       partition++) {
    const std::vector<int> &tile_devs = this->tile_comm_devs_[partition];

    const std::vector<ncclComm_t> &tile_comms = this->tile_comms_[partition];

    utl::require(tile_devs.size() ==
                     static_cast<std::size_t>(this->tile_partition_count_),
                 function_name,
                 "Tile communicator device count does not match G_tile");

    utl::require(tile_comms.size() == tile_devs.size(), function_name,
                 "Tile communicator handle count is incorrect");

    const int root_dev = tile_devs[0];

    utl::require((root_dev >= 0) && (root_dev < this->cuda_count_),
                 function_name, "Cell-partition root device is out of range");

    utl::require(root_devs[partition] == root_dev, function_name,
                 "Cell communicator rank does not match tile root");

    for (unsigned int tile_partition = 0;
         tile_partition < this->tile_partition_count_; tile_partition++) {
      const int dev = tile_devs[tile_partition];

      utl::require((dev >= 0) && (dev < this->cuda_count_), function_name,
                   "Tile communicator device is out of range");

      utl::require(this->dev_cell_partition_[dev] == partition, function_name,
                   "Tile communicator device is in the wrong cell partition");

      utl::require(this->dev_tile_partition_[dev] == tile_partition,
                   function_name,
                   "Tile communicator rank does not match tile partition");
    }
  }

  utl::require(root_devs[0] == 0, function_name,
               "GPU 0 must be the root of cell partition 0");

  // Enqueues classify, pack, deterministic radix sort, cell scan, and
  // x-plane-point construction on GPU 0's compute stream.
  this->build_global_atom_reference(d_rx, d_ry, d_rz, d_qc);

  const unsigned int ncell_x = this->plan_->ncell_x();
  const std::size_t x_plane_point_count =
      static_cast<std::size_t>(ncell_x) + 1u;

  std::vector<unsigned int> x_plane_atom_point(x_plane_point_count);

  const std::size_t partition_count =
      static_cast<std::size_t>(this->cell_partition_count_);

  std::vector<unsigned int> partition_max_atoms_cell(partition_count);

  // These are the only device-to-host values in the optimized root path. Their
  // combined size is O(ncell_x + G_cell), not O(natom).
  cudaCheck(cudaSetDevice(0));

  cudaCheck(
      cudaMemcpyAsync(static_cast<void *>(x_plane_atom_point.data()),
                      static_cast<const void *>(
                          this->workspace_->global_x_plane_atom_point().data()),
                      x_plane_point_count * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost, this->comp_streams_[0]));

  cudaCheck(
      cudaMemcpyAsync(static_cast<void *>(partition_max_atoms_cell.data()),
                      static_cast<const void *>(
                          this->workspace_->global_max_atoms_cell().data()),
                      partition_count * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost, this->comp_streams_[0]));

  // One synchronization after the complete GPU 0 classify/sort/scan pipeline
  // and the small descriptor transfer. There are no synchronizations between
  // individual assignment operations.
  cudaCheck(cudaStreamSynchronize(this->comp_streams_[0]));

  utl::require(x_plane_atom_point.front() == 0u, function_name,
               "First x-plane atom point is not zero");

  utl::require(x_plane_atom_point.back() == this->plan_->natom(), function_name,
               "Terminal x-plane atom point does not equal natom");

  for (std::size_t x = 1; x < x_plane_atom_point.size(); x++) {
    utl::require(x_plane_atom_point[x] >= x_plane_atom_point[x - 1u],
                 function_name,
                 "X-plane atom points are not monotonically ordered");
  }

  std::vector<atom_partition_range> &partition_range =
      this->workspace_->partition_atom_range();

  utl::require(partition_range.size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name,
               "Partition atom-range metadata does not match G_cell");

  std::vector<std::array<std::size_t, 3>> cell_source_offset(
      this->cell_partition_count_);
  std::vector<std::array<std::size_t, 3>> cell_destination_offset(
      this->cell_partition_count_);
  std::vector<std::array<std::size_t, 3>> cell_segment_count(
      this->cell_partition_count_);

  // Derive source ranges and establish active capacities/counts before
  // recording communication dependencies. A first assignment may grow; later
  // assignments with the same or smaller source population do not allocate.
  for (unsigned int partition = 0; partition < this->cell_partition_count_;
       partition++) {
    const unsigned int x_point =
        this->plan_->cell_partition_x_point()[partition];
    const unsigned int x_count =
        this->plan_->cell_partition_x_count()[partition];

    utl::require(x_point <= ncell_x, function_name,
                 "Partition x point is out of range");

    utl::require(x_count <= ncell_x - x_point, function_name,
                 "Partition x count is out of range");

    const unsigned int x_end = x_point + x_count;

    atom_partition_range range;

    const std::size_t owned_begin =
        static_cast<std::size_t>(x_plane_atom_point[x_point]);
    const std::size_t owned_end =
        static_cast<std::size_t>(x_plane_atom_point[x_end]);

    utl::require(owned_end >= owned_begin, function_name,
                 "Owned packet range is reversed");

    range.owned_source_offset = owned_begin;
    range.owned_atom_count = owned_end - owned_begin;

    // Keep zero-sized optional ranges anchored to a valid packet boundary.
    range.left_source_offset = owned_begin;
    range.right_source_offset = owned_end;

    // Empty x-slabs have no halo source cells in glst_plan.
    if (x_count > 0u) {
      if (x_point > 0u) {
        const std::size_t left_begin =
            static_cast<std::size_t>(x_plane_atom_point[x_point - 1u]);

        utl::require(owned_begin >= left_begin, function_name,
                     "Left-halo packet range is reversed");

        range.left_source_offset = left_begin;
        range.left_atom_count = owned_begin - left_begin;
      }

      if (x_end < ncell_x) {
        const std::size_t right_end =
            static_cast<std::size_t>(x_plane_atom_point[x_end + 1u]);

        utl::require(right_end >= owned_end, function_name,
                     "Right-halo packet range is reversed");

        range.right_source_offset = owned_end;
        range.right_atom_count = right_end - owned_end;
      }
    }

    utl::require(
        range.left_atom_count <=
            std::numeric_limits<std::size_t>::max() - range.owned_atom_count,
        function_name, "Owned plus left-halo atom count overflows size_t");

    const std::size_t owned_left_count =
        range.owned_atom_count + range.left_atom_count;

    utl::require(range.right_atom_count <=
                     std::numeric_limits<std::size_t>::max() - owned_left_count,
                 function_name, "Total source atom count overflows size_t");

    range.source_atom_count = owned_left_count + range.right_atom_count;

    // Owned and adjacent non-PBC halo planes are disjoint.
    utl::require(range.source_atom_count <=
                     static_cast<std::size_t>(this->plan_->natom()),
                 function_name, "Partition source atom count exceeds natom");

    partition_range[partition] = range;

    const std::vector<unsigned int> &left_cells =
        this->plan_->partition_left_halo_cell_idx(partition);

    const std::vector<unsigned int> &right_cells =
        this->plan_->partition_right_halo_cell_idx(partition);

    const std::size_t owned_cell_count =
        static_cast<std::size_t>(this->plan_->local_cell_count(partition));

    const std::size_t left_cell_count = left_cells.size();
    const std::size_t right_cell_count = right_cells.size();

    const std::size_t first_owned_cell =
        static_cast<std::size_t>(this->plan_->first_global_cell(partition));

    const std::size_t first_left_cell =
        left_cells.empty() ? first_owned_cell
                           : static_cast<std::size_t>(left_cells.front());

    const std::size_t first_right_cell =
        right_cells.empty() ? first_owned_cell
                            : static_cast<std::size_t>(right_cells.front());

    cell_source_offset[partition] = {first_owned_cell, first_left_cell,
                                     first_right_cell};

    cell_destination_offset[partition] = {0u, owned_cell_count,
                                          owned_cell_count + left_cell_count};

    cell_segment_count[partition] = {owned_cell_count, left_cell_count,
                                     right_cell_count};

    const unsigned int max_atoms_cell = partition_max_atoms_cell[partition];

    utl::require(
        range.source_atom_count <=
            static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        function_name,
        "Partition source atom count exceeds unsigned int range");

    utl::require(
        range.owned_atom_count <=
            static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        function_name, "Partition owned atom count exceeds unsigned int range");

    utl::require(static_cast<std::size_t>(max_atoms_cell) <=
                     range.owned_atom_count,
                 function_name,
                 "Partition maximum cell population exceeds owned atom count");

    const std::vector<int> &tile_devs = this->tile_comm_devs_[partition];

    for (std::size_t rank = 0; rank < tile_devs.size(); rank++) {
      const int dev = tile_devs[rank];

      utl::require(
          owned_cell_count + left_cell_count + right_cell_count ==
              this->workspace_->sr_source_cell_capacity(dev),
          function_name,
          "Source-cell segment counts do not match workspace capacity");

      this->workspace_->max_atoms_cell()[dev] = max_atoms_cell;

      this->workspace_->ensure_atom_capacity(dev, range.source_atom_count);

      this->workspace_->set_atom_counts(dev, range.source_atom_count,
                                        range.owned_atom_count);
    }
  }

  // No assignment communication may overwrite a rank's packet or source-cell
  // buffers until preceeding compute work on that rank has finished.
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    cudaCheck(
        cudaEventRecord(this->comp_events_[dev], this->comp_streams_[dev]));

    cudaCheck(cudaStreamWaitEvent(this->comm_streams_[dev],
                                  this->comp_events_[dev], 0));
  }

  const atom_packet *global_packet =
      this->workspace_->global_packet_out().data();

  const unsigned int *global_cell_atom_count =
      this->workspace_->global_cell_atom_count().data();

  // Cell partition 0 is local to GPU 0. Use ordinary asynchronous D2D copies,
  // preserving the exact [owned][left][right] destination order.
  {
    const unsigned int partition = 0u;
    const int root_dev = root_devs[partition];

    const atom_partition_range &range = partition_range[partition];

    const std::array<std::size_t, 3> source_offset = {
        range.owned_source_offset, range.left_source_offset,
        range.right_source_offset};

    const std::array<std::size_t, 3> destination_offset = {
        0u, range.owned_atom_count,
        range.owned_atom_count + range.left_atom_count};

    const std::array<std::size_t, 3> packet_count = {
        range.owned_atom_count, range.left_atom_count, range.right_atom_count};

    atom_packet *destination =
        this->workspace_->source_packets()[root_dev].data();

    cudaCheck(cudaSetDevice(root_dev));

    for (std::size_t segment = 0; segment < packet_count.size(); segment++) {
      if (packet_count[segment] == 0u)
        continue;

      utl::require(packet_count[segment] <=
                       std::numeric_limits<std::size_t>::max() /
                           sizeof(atom_packet),
                   function_name, "Packet-copy byte count overflows size_t");

      const std::size_t byte_count =
          packet_count[segment] * sizeof(atom_packet);

      cudaCheck(cudaMemcpyAsync(
          static_cast<void *>(destination + destination_offset[segment]),
          static_cast<const void *>(global_packet + source_offset[segment]),
          byte_count, cudaMemcpyDeviceToDevice, this->comm_streams_[root_dev]));
    }

    unsigned int *destination_cell_count =
        this->workspace_->sr_source_cell_atom_count()[root_dev].data();

    for (std::size_t segment = 0;
         segment < cell_segment_count[partition].size(); segment++) {
      const std::size_t count = cell_segment_count[partition][segment];

      if (count == 0u)
        continue;

      utl::require(count <= std::numeric_limits<std::size_t>::max() /
                                sizeof(unsigned int),
                   function_name, "Cell-count copy byte overflows size_t");

      const std::size_t byte_count = count * sizeof(unsigned int);

      cudaCheck(cudaMemcpyAsync(
          static_cast<void *>(destination_cell_count +
                              cell_destination_offset[partition][segment]),
          static_cast<const void *>(global_cell_atom_count +
                                    cell_source_offset[partition][segment]),
          byte_count, cudaMemcpyDeviceToDevice, this->comm_streams_[root_dev]));
    }
  }

  bool has_remote_transfer = false;

  for (unsigned int partition = 1; partition < this->cell_partition_count_;
       partition++) {
    if ((partition_range[partition].source_atom_count > 0u) ||
        (cell_segment_count[partition][0] > 0u) ||
        (cell_segment_count[partition][1] > 0u) ||
        (cell_segment_count[partition][2] > 0u)) {
      has_remote_transfer = true;
      break;
    }
  }

  // The current CUDA initiaization deliberately does not own peer-access
  // enablement. Use the existing NCCL cell communicator for all remote roots;
  // NCCL selects its available GPU transport and no application host packet
  // staging is introduced.
  if (has_remote_transfer) {
    ncclCheck(ncclGroupStart());

    for (unsigned int partition = 1; partition < this->cell_partition_count_;
         partition++) {
      const int root_dev = root_devs[partition];

      const atom_partition_range &range = partition_range[partition];

      const std::array<std::size_t, 3> source_offset = {
          range.owned_source_offset, range.left_source_offset,
          range.right_source_offset};

      const std::array<std::size_t, 3> destination_offset = {
          0u, range.owned_atom_count,
          range.owned_atom_count + range.left_atom_count};

      const std::array<std::size_t, 3> packet_count = {range.owned_atom_count,
                                                       range.left_atom_count,
                                                       range.right_atom_count};

      atom_packet *destination =
          this->workspace_->source_packets()[root_dev].data();

      for (std::size_t segment = 0; segment < packet_count.size(); segment++) {
        if (packet_count[segment] == 0u)
          continue;

        utl::require(packet_count[segment] <=
                         std::numeric_limits<std::size_t>::max() /
                             sizeof(atom_packet),
                     function_name, "Packet-copy byte count overflows size_t");

        const std::size_t byte_count =
            packet_count[segment] * sizeof(atom_packet);

        // Cell-communicator rank 0 is GPU 0/cell partition 0.
        cudaCheck(cudaSetDevice(0));

        ncclCheck(ncclSend(
            static_cast<const void *>(global_packet + source_offset[segment]),
            byte_count, ncclChar, static_cast<int>(partition), root_comms[0],
            this->comm_streams_[0]));

        cudaCheck(cudaSetDevice(root_dev));

        ncclCheck(ncclRecv(
            static_cast<void *>(destination + destination_offset[segment]),
            byte_count, ncclChar, 0, root_comms[partition],
            this->comm_streams_[root_dev]));
      }

      unsigned int *destination_cell_count =
          this->workspace_->sr_source_cell_atom_count()[root_dev].data();

      for (std::size_t segment = 0;
           segment < cell_segment_count[partition].size(); segment++) {
        const std::size_t count = cell_segment_count[partition][segment];

        if (count == 0u)
          continue;

        utl::require(count <= std::numeric_limits<std::size_t>::max() /
                                  sizeof(unsigned int),
                     function_name,
                     "Cell-count communication byte count overflows size_t");

        const std::size_t byte_count = count * sizeof(unsigned int);

        cudaCheck(cudaSetDevice(0));

        ncclCheck(ncclSend(
            static_cast<const void *>(global_cell_atom_count +
                                      cell_source_offset[partition][segment]),
            byte_count, ncclChar, static_cast<int>(partition), root_comms[0],
            this->comm_streams_[0]));

        cudaCheck(cudaSetDevice(root_dev));

        ncclCheck(ncclRecv(
            static_cast<void *>(destination_cell_count +
                                cell_destination_offset[partition][segment]),
            byte_count, ncclChar, 0, root_comms[partition],
            this->comm_streams_[root_dev]));
      }
    }

    ncclCheck(ncclGroupEnd());
  }

  // Replicate each cell partition's packed source representation from
  // tile-communicator rank 0 to all tile ranks in that cell partition.
  if (this->tile_partition_count_ > 1u) {
    ncclCheck(ncclGroupStart());

    for (unsigned int partition = 0; partition < this->cell_partition_count_;
         partition++) {
      const atom_partition_range &range = partition_range[partition];

      const std::size_t source_cell_count =
          this->plan_->partition_sr_source_cell_idx(partition).size();

      utl::require(
          range.source_atom_count <=
              std::numeric_limits<std::size_t>::max() / sizeof(atom_packet),
          function_name, "Packet broadcast byte count overflows size_t");

      const std::size_t packet_byte_count =
          range.source_atom_count * sizeof(atom_packet);

      const std::vector<int> &tile_devs = this->tile_comm_devs_[partition];
      const std::vector<ncclComm_t> &tile_comms = this->tile_comms_[partition];

      utl::require(tile_devs.size() == tile_comms.size(), function_name,
                   "Tile communicator device/handle counts differ");

      for (std::size_t rank = 0; rank < tile_devs.size(); rank++) {
        const int dev = tile_devs[rank];

        cudaCheck(cudaSetDevice(dev));

        atom_packet *packet = this->workspace_->source_packets()[dev].data();
        unsigned int *source_cell_atom_count =
            this->workspace_->sr_source_cell_atom_count()[dev].data();

        // Communicator rank 0 is tile partition 0.
        if (packet_byte_count > 0) {
          ncclCheck(ncclBroadcast(static_cast<const void *>(packet),
                                  static_cast<void *>(packet),
                                  packet_byte_count, ncclChar, 0,
                                  tile_comms[rank], this->comm_streams_[dev]));
        }

        if (source_cell_count > 0) {
          ncclCheck(ncclBroadcast(
              static_cast<const void *>(source_cell_atom_count),
              static_cast<void *>(source_cell_atom_count), source_cell_count,
              ncclUint32, 0, tile_comms[rank], this->comm_streams_[dev]));
        }
      }
    }

    ncclCheck(ncclGroupEnd());
  }

  // Publish assignment communication completion to every rank's compute stream.
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    cudaCheck(
        cudaEventRecord(this->comm_events_[dev], this->comm_streams_[dev]));

    cudaCheck(cudaStreamWaitEvent(this->comp_streams_[dev],
                                  this->comm_events_[dev], 0));
  }

  return;
}

void glst_force::validate_atom_scatter(void) const {
  constexpr std::string_view function_name =
      "glst_force::validate_atom_scatter";

  struct atom_scatter_snapshot {
    unsigned int cell_partition = 0u;
    std::size_t owned_atom_count = 0u;
    std::size_t source_atom_count = 0u;
    unsigned int max_atoms_cell = 0u;

    std::vector<atom_packet> packet;
    std::vector<unsigned int> source_cell_atom_count;
    std::vector<unsigned int> source_cell_atom_point;
  };

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  const unsigned int natom = this->plan_->natom();
  const std::size_t device_count = static_cast<std::size_t>(this->cuda_count_);

  utl::require(this->dev_cell_partition_.size() == device_count, function_name,
               "Device cell-partition metadata size is incorrect");

  utl::require(this->dev_tile_partition_.size() == device_count, function_name,
               "Device tile-partition metadata size is incorrect");

  // Production assignment is stream/event ordered. Debug validation introduces
  // one synchronization only after the complete assignment pipeline.
  this->synchronize_compute_streams();

  std::vector<atom_scatter_snapshot> snapshots(device_count);
  std::vector<unsigned int> replica_count(natom, 0u);

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    const unsigned int cell_partition = this->dev_cell_partition_[dev];

    utl::require(cell_partition < this->cell_partition_count_, function_name,
                 "Device cell partition is out of range");

    const std::vector<unsigned int> &owned_cells =
        this->plan_->partition_cell_idx(cell_partition);

    const std::vector<unsigned int> &left_cells =
        this->plan_->partition_left_halo_cell_idx(cell_partition);

    const std::vector<unsigned int> &right_cells =
        this->plan_->partition_right_halo_cell_idx(cell_partition);

    const std::vector<unsigned int> &source_cells =
        this->plan_->partition_sr_source_cell_idx(cell_partition);

    const std::size_t expected_source_cell_count =
        owned_cells.size() + left_cells.size() + right_cells.size();

    utl::require(source_cells.size() == expected_source_cell_count,
                 function_name,
                 "Source-cell list size does not equal owned plus halo cells");

    std::size_t source_cell_offset = 0u;

    for (std::size_t i = 0u; i < owned_cells.size(); i++) {
      utl::require(source_cells[source_cell_offset + i] == owned_cells[i],
                   function_name,
                   "Source-cell list does not begin with owned cells");
    }

    source_cell_offset += owned_cells.size();

    for (std::size_t i = 0u; i < left_cells.size(); i++) {
      utl::require(
          source_cells[source_cell_offset + i] == left_cells[i], function_name,
          "Source-cell list does not place left halo after owned cells");
    }

    source_cell_offset += left_cells.size();

    for (std::size_t i = 0u; i < right_cells.size(); i++) {
      utl::require(source_cells[source_cell_offset + i] == right_cells[i],
                   function_name,
                   "Source-cell list does not place right halo last");
    }

    const std::size_t owned_atom_count =
        this->workspace_->owned_atom_count(dev);

    const std::size_t source_atom_count =
        this->workspace_->source_atom_count(dev);

    const std::size_t atom_capacity = this->workspace_->atom_capacity(dev);

    utl::require(owned_atom_count <= source_atom_count, function_name,
                 "Owned atom count exceeds source atom count");

    utl::require(source_atom_count <= atom_capacity, function_name,
                 "Source atom count exceeds atom capacity");

    utl::require(source_cells.size() ==
                     this->workspace_->sr_source_cell_atom_count()[dev].size(),
                 function_name, "Source-cell count storage size is incorrect");

    utl::require(source_cells.size() ==
                     this->workspace_->sr_source_cell_atom_point()[dev].size(),
                 function_name, "Source-cell point storage size is incorrect");

    atom_scatter_snapshot &snapshot = snapshots[static_cast<std::size_t>(dev)];

    snapshot.cell_partition = cell_partition;
    snapshot.owned_atom_count = owned_atom_count;
    snapshot.source_atom_count = source_atom_count;
    snapshot.max_atoms_cell = this->workspace_->max_atoms_cell()[dev];

    snapshot.packet.resize(source_atom_count);
    snapshot.source_cell_atom_count.resize(source_cells.size());
    snapshot.source_cell_atom_point.resize(source_cells.size());

    cudaCheck(cudaSetDevice(dev));

    // These copies exist only in a __GLST_DEBUG__ caller. Release assignment
    // performs no atom-sized D2H transfer.
    if (source_atom_count > 0u) {
      cudaCheck(cudaMemcpy(static_cast<void *>(snapshot.packet.data()),
                           static_cast<const void *>(
                               this->workspace_->source_packets()[dev].data()),
                           source_atom_count * sizeof(atom_packet),
                           cudaMemcpyDeviceToHost));
    }

    if (!source_cells.empty()) {
      const std::size_t metadata_bytes =
          source_cells.size() * sizeof(unsigned int);

      cudaCheck(cudaMemcpy(
          static_cast<void *>(snapshot.source_cell_atom_count.data()),
          static_cast<const void *>(
              this->workspace_->sr_source_cell_atom_count()[dev].data()),
          metadata_bytes, cudaMemcpyDeviceToHost));

      cudaCheck(cudaMemcpy(
          static_cast<void *>(snapshot.source_cell_atom_point.data()),
          static_cast<const void *>(
              this->workspace_->sr_source_cell_atom_point()[dev].data()),
          metadata_bytes, cudaMemcpyDeviceToHost));
    }

    std::size_t expected_atom_point = 0u;
    std::size_t observed_owned_atom_count = 0u;

    for (std::size_t source_local_cell = 0u;
         source_local_cell < source_cells.size(); source_local_cell++) {
      const unsigned int expected_cell = source_cells[source_local_cell];

      const std::size_t point = static_cast<std::size_t>(
          snapshot.source_cell_atom_point[source_local_cell]);

      const std::size_t count = static_cast<std::size_t>(
          snapshot.source_cell_atom_count[source_local_cell]);

      utl::require(point == expected_atom_point, function_name,
                   "Source-cell atom points are not a contiguous prefix sum");

      utl::require(point <= source_atom_count, function_name,
                   "Source-cell atom point exceeds source atom count");

      utl::require(count <= source_atom_count - point, function_name,
                   "Source-cell atom range exceeds source atom count");

      const bool is_owned_cell = source_local_cell < owned_cells.size();

      if (is_owned_cell)
        observed_owned_atom_count += count;

      unsigned int previous_atom = 0u;

      for (std::size_t i = 0u; i < count; i++) {
        const atom_packet &packet = snapshot.packet[point + i];

        utl::require(packet.i < natom, function_name,
                     "Original atom index is out of range");

        utl::require(packet.cell == expected_cell, function_name,
                     "Source-cell range contains an atom from another cell");

        const unsigned int packet_cell_partition =
            this->plan_->cell_partition_idx(packet.cell);

        if (is_owned_cell) {
          utl::require(packet_cell_partition == cell_partition, function_name,
                       "Owned atom is stored on the wrong cell partition");

          replica_count[packet.i]++;
        } else {
          utl::require(packet_cell_partition != cell_partition, function_name,
                       "Halo atom is owned by the target cell partition");
        }

        if (i > 0u) {
          utl::require(packet.i > previous_atom, function_name,
                       "Atoms within a cell are not ordered by original index");
        }

        previous_atom = packet.i;
      }

      expected_atom_point += count;
    }

    utl::require(expected_atom_point == source_atom_count, function_name,
                 "Source-cell counts do not sum to source atom count");

    utl::require(observed_owned_atom_count == owned_atom_count, function_name,
                 "Owned-cell counts do not sum to owned atom count");
  }

  for (unsigned int atom = 0u; atom < natom; atom++) {
    utl::require(replica_count[atom] == this->tile_partition_count_,
                 function_name,
                 "Owned atom replica count does not equal G_tile");
  }

  std::vector<int> first_dev_by_partition(this->cell_partition_count_, -1);

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    const unsigned int cell_partition = this->dev_cell_partition_[dev];

    int &first_dev = first_dev_by_partition[cell_partition];

    if (first_dev < 0) {
      first_dev = dev;
      continue;
    }

    const atom_scatter_snapshot &lhs =
        snapshots[static_cast<std::size_t>(first_dev)];

    const atom_scatter_snapshot &rhs = snapshots[static_cast<std::size_t>(dev)];

    utl::require(lhs.owned_atom_count == rhs.owned_atom_count, function_name,
                 "Tile ranks have different owned atom counts");

    utl::require(lhs.source_atom_count == rhs.source_atom_count, function_name,
                 "Tile ranks have different source atom counts");

    utl::require(lhs.max_atoms_cell == rhs.max_atoms_cell, function_name,
                 "Tile ranks have different maximum cell populations");

    utl::require(lhs.packet.size() == rhs.packet.size(), function_name,
                 "Tile ranks have different packet-vector sizes");

    utl::require(lhs.source_cell_atom_count.size() ==
                     rhs.source_cell_atom_count.size(),
                 function_name,
                 "Tile ranks have different source-cell count-vector sizes");

    utl::require(lhs.source_cell_atom_point.size() ==
                     rhs.source_cell_atom_point.size(),
                 function_name,
                 "Tile ranks have different source-cell point-vector sizes");

    utl::require(lhs.packet.empty() ||
                     std::memcmp(static_cast<const void *>(lhs.packet.data()),
                                 static_cast<const void *>(rhs.packet.data()),
                                 lhs.packet.size() * sizeof(atom_packet)) == 0,
                 function_name,
                 "Tile ranks do not contain byte-identical packet ordering");

    utl::require(
        lhs.source_cell_atom_count.empty() ||
            std::memcmp(
                static_cast<const void *>(lhs.source_cell_atom_count.data()),
                static_cast<const void *>(rhs.source_cell_atom_count.data()),
                lhs.source_cell_atom_count.size() * sizeof(unsigned int)) == 0,
        function_name, "Tile ranks have different source-cell atom counts");

    utl::require(
        lhs.source_cell_atom_point.empty() ||
            std::memcmp(
                static_cast<const void *>(lhs.source_cell_atom_point.data()),
                static_cast<const void *>(rhs.source_cell_atom_point.data()),
                lhs.source_cell_atom_point.size() * sizeof(unsigned int)) == 0,
        function_name, "Tile ranks have different source-cell atom points");
  }

  for (unsigned int cell_partition = 0u;
       cell_partition < this->cell_partition_count_; cell_partition++) {
    utl::require(first_dev_by_partition[cell_partition] >= 0, function_name,
                 "Cell partition has no device");
  }

  return;
}

void glst_force::assign_atoms_multi_gpu(const double *d_rx, const double *d_ry,
                                        const double *d_rz,
                                        const double *d_qc) {
  constexpr std::string_view function_name =
      "glst_force::assign_atoms_multi_gpu";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  std::chrono::steady_clock::time_point assignment_start;
  if (this->profiling_enabled_)
    assignment_start = std::chrono::steady_clock::now();

  // GPU 0 classifies and sorts once. The resulting owned/left/right ranges are
  // scattered to tile-partition-0 roots and broadcast inside tile
  // communicators.
  this->scatter_sorted_atom_packets(d_rx, d_ry, d_rz, d_qc);

  if (this->profiling_enabled_) {
    const std::chrono::steady_clock::time_point assignment_end =
        std::chrono::steady_clock::now();

    this->profile_.atom_assignment_scatter_ms =
        profile_elapsed_ms(assignment_start, assignment_end);
  }

  std::chrono::steady_clock::time_point source_start;
  if (this->profiling_enabled_)
    source_start = std::chrono::steady_clock::now();

  const std::size_t cub_item_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    const unsigned int cell_partition = this->dev_cell_partition_[dev];
    const std::size_t local_cell_count =
        static_cast<std::size_t>(this->plan_->local_cell_count(cell_partition));
    const std::size_t source_cell_count =
        this->plan_->partition_sr_source_cell_idx(cell_partition).size();
    const std::size_t source_atom_count =
        this->workspace_->source_atom_count(dev);
    const std::size_t owned_atom_count =
        this->workspace_->owned_atom_count(dev);

    utl::require(
        source_cell_count == this->workspace_->sr_source_cell_capacity(dev),
        function_name, "Source-cell count does not match workspace capacity");

    utl::require(source_cell_count <= cub_item_limit, function_name,
                 "Source-cell count exceeds CUB int range");

    utl::require(local_cell_count == this->workspace_->cell_capacity(dev),
                 function_name,
                 "Owned-cell count does not match workspace capacity");

    utl::require(
        local_cell_count <=
            static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        function_name, "Owned-cell count exceeds kernel argument range");

    utl::require(
        source_atom_count <=
            static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        function_name, "Source atom count exceeds unpack-kernel range");

    utl::require(owned_atom_count <= source_atom_count, function_name,
                 "Owned atom count exceeds source atom count");

    utl::require(
        static_cast<std::size_t>(this->workspace_->max_atoms_cell()[dev]) <=
            owned_atom_count,
        function_name,
        "Maximum owned-cell population exceeds owned atom count");

    const cudaStream_t stream = this->comp_streams_[dev];

    // Every rank independently rebuilds source-cell points from the
    // byte-identical source-cell counts it received.
    if (source_cell_count > 0) {
      void *const cub_work_buffer = this->workspace_->cub_work_buffer()[dev];
      const std::size_t cub_work_buffer_capacity =
          this->workspace_->cub_work_buffer_size()[dev];

      utl::require(cub_work_buffer != nullptr, function_name,
                   "CUB work buffer is null");

      utl::require(cub_work_buffer_capacity > 0u, function_name,
                   "CUB work-buffer capacity is zero");

      std::size_t cub_work_buffer_size = cub_work_buffer_capacity;

      cudaCheck(cub::DeviceScan::ExclusiveSum(
          cub_work_buffer, cub_work_buffer_size,
          this->workspace_->sr_source_cell_atom_count()[dev].data(),
          this->workspace_->sr_source_cell_atom_point()[dev].data(),
          static_cast<int>(source_cell_count), stream));
    }

    // Do not communicate the individual Structure of Arrays arrays. Every rank
    // unpacks the one packed source representation locally.
    if (source_atom_count > 0) {
      const unsigned int active_source_atom_count =
          static_cast<unsigned int>(source_atom_count);

      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (active_source_atom_count + num_threads - 1) / num_threads;

      unpack_kernel<<<num_blocks, num_threads, 0, stream>>>(
          this->workspace_->rx()[dev].d_array().data(),
          this->workspace_->ry()[dev].d_array().data(),
          this->workspace_->rz()[dev].d_array().data(),
          this->workspace_->qc()[dev].d_array().data(),
          this->workspace_->sorted_idx()[dev].d_array().data(),
          this->workspace_->source_packets()[dev].data(),
          active_source_atom_count);
      cudaCheck(cudaGetLastError());
    }

    if (this->profiling_enabled_) {
      this->profile_.owned_atom_replicas += owned_atom_count;

      this->profile_.halo_atom_replicas += source_atom_count - owned_atom_count;
    }
  }

  if (this->profiling_enabled_) {
    // Profiling mode already introduces phase boundaries. Normal execution
    // remains entirely stream/event ordered.
    this->synchronize_compute_streams();

    const std::chrono::steady_clock::time_point source_end =
        std::chrono::steady_clock::now();

    this->profile_.owned_halo_source_scatter_ms =
        profile_elapsed_ms(source_start, source_end);
  }

#ifdef __GLST_DEBUG__
  this->validate_atom_scatter();
#endif

  return;
}

void glst_force::calc_sf_tile(const unsigned int tile) {
  constexpr std::string_view function_name = "glst_force::calc_sf_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  const unsigned int tile_node_point = this->plan_->tile_node_point(tile);
  const unsigned int tile_node_count = this->plan_->tile_node_count(tile);

  utl::require(tile_node_count > 0, function_name, "Tile node count is 0");

  utl::require(tile_node_count <= this->plan_->max_tile_nodes(), function_name,
               "Tile exceeds buffer size");

  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);

  const unsigned int nc = tile_node_count;
  const unsigned int off = tile_node_point;

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    if (this->dev_tile_partition_[dev] != tile_partition)
      continue;

    cudaCheck(cudaSetDevice(dev));

    const unsigned int cell_partition = this->dev_cell_partition_[dev];
    const unsigned int local_cell_count =
        this->plan_->local_cell_count(cell_partition);
    const unsigned int first_global_cell =
        this->plan_->first_global_cell(cell_partition);

    const bool use_full_sf_buffer =
        (this->sf_exchange_mode_ ==
         GLST_SF_EXCHANGE_MODE::FULL_GLOBAL_ALLREDUCE);

    const unsigned int sf_cell_count =
        (use_full_sf_buffer) ? this->plan_->ncell() : local_cell_count;

    const unsigned int sf_cell_point =
        (use_full_sf_buffer) ? first_global_cell : 0u;

    const std::size_t sf_entry_count =
        static_cast<std::size_t>(sf_cell_count) * static_cast<std::size_t>(nc);

    const std::size_t sf_bytes = sf_entry_count * sizeof(double);

    utl::require(sf_entry_count <=
                     this->workspace_->sf_tile_buffer_capacity(dev),
                 function_name, "Active SF tile exceeds workspace capacity");

    if (sf_entry_count > 0) {
      cudaCheck(cudaMemsetAsync(
          static_cast<void *>(this->workspace_->sf_re()[dev].d_array().data()),
          0, sf_bytes, this->comp_streams_[dev]));
      cudaCheck(cudaMemsetAsync(
          static_cast<void *>(this->workspace_->sf_im()[dev].d_array().data()),
          0, sf_bytes, this->comp_streams_[dev]));
    }

    if (local_cell_count == 0)
      continue;

    const unsigned int *cell_atom_point = nullptr;
    const unsigned int *cell_atom_count = nullptr;

    if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
      cell_atom_point =
          this->workspace_->cell_atom_point()[dev].d_array().data();
      cell_atom_count =
          this->workspace_->cell_atom_count()[dev].d_array().data();
    } else {
      utl::require(static_cast<std::size_t>(local_cell_count) <=
                       this->workspace_->sr_source_cell_capacity(dev),
                   function_name,
                   "Owned-cell count exceeds source-cell metadata capacity");
      cell_atom_point =
          this->workspace_->sr_source_cell_atom_point()[dev].data();
      cell_atom_count =
          this->workspace_->sr_source_cell_atom_count()[dev].data();
    }

    constexpr dim3 num_threads(128, 1, 1);
    const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                          std::min(65535u, local_cell_count), 1);
    calc_sf_kernel<96>
        <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
            this->workspace_->sf_re()[dev].d_array().data(),
            this->workspace_->sf_im()[dev].d_array().data(),
            this->plan_->x()[dev].d_array().data() + off,
            this->plan_->y()[dev].d_array().data() + off,
            this->plan_->z()[dev].d_array().data() + off, nc,
            this->workspace_->rx()[dev].d_array().data(),
            this->workspace_->ry()[dev].d_array().data(),
            this->workspace_->rz()[dev].d_array().data(),
            this->workspace_->qc()[dev].d_array().data(), cell_atom_point,
            cell_atom_count, local_cell_count, sf_cell_point);

    cudaCheck(cudaGetLastError());
  }

  return;
}

void glst_force::zero_rmt_sum_tile(const unsigned int tile) {
  constexpr std::string_view function_name = "glst_force::zero_rmt_sum_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  const unsigned int nc = this->plan_->tile_node_count(tile);
  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    if (this->dev_tile_partition_[dev] != tile_partition)
      continue;

    const unsigned int cell_partition = this->dev_cell_partition_[dev];
    const unsigned int local_cell_count =
        this->plan_->local_cell_count(cell_partition);

    const std::size_t entry_count = static_cast<std::size_t>(local_cell_count) *
                                    static_cast<std::size_t>(nc);

    utl::require(entry_count <= this->workspace_->rmt_tile_buffer_capacity(dev),
                 function_name, "Active rmt tile exceeds workspace capacity");

    if (entry_count == 0)
      continue;

    const std::size_t byte_count = entry_count * sizeof(double);

    cudaCheck(cudaSetDevice(dev));

    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(
            this->workspace_->rmt_sum_re()[dev].d_array().data()),
        0, byte_count, this->comp_streams_[dev]));

    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(
            this->workspace_->rmt_sum_im()[dev].d_array().data()),
        0, byte_count, this->comp_streams_[dev]));
  }

  return;
}

void glst_force::exchange_sf_tile(const unsigned int tile) {
  constexpr std::string_view function_name = "glst_force::exchange_sf_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  const unsigned int tile_node_count = this->plan_->tile_node_count(tile);

  utl::require(tile_node_count > 0, function_name, "Tile node count is 0");

  utl::require(tile_node_count <= this->plan_->max_tile_nodes(), function_name,
               "Tile exceeds buffer size");

  // No cell-domain communication is required when there is only one cell
  // partition. This covers both single-GPU mode and pure tile decomposition.
  if (this->cell_partition_count_ == 1)
    return;

  utl::require(this->sf_exchange_mode_ ==
                   GLST_SF_EXCHANGE_MODE::FULL_GLOBAL_ALLREDUCE,
               function_name,
               "Full global S_tile exchange called while FULL_GLOBAL_ALLREDUCE "
               "is not selected");

  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);

  utl::require(
      (static_cast<std::size_t>(tile_partition) <
       this->cell_comm_devs_.size()) &&
          (static_cast<std::size_t>(tile_partition) < this->cell_comms_.size()),
      function_name, "Tile partition does not have a cell communicator");

  const std::vector<int> &devs = this->cell_comm_devs_[tile_partition];
  const std::vector<ncclComm_t> &comms = this->cell_comms_[tile_partition];

  utl::require(
      devs.size() == static_cast<std::size_t>(this->cell_partition_count_),
      function_name,
      "Cell communicator device count does not match cell partition count");

  utl::require(comms.size() == devs.size(), function_name,
               "Cell communicator handle count does not match device count");

  const std::size_t sf_entry_count =
      static_cast<std::size_t>(this->plan_->ncell()) *
      static_cast<std::size_t>(tile_node_count);

  // Make each communication stream wait for calc_sf_tile on its corresponding
  // compute stream.
  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    utl::require((dev >= 0) && (dev < this->cuda_count_), function_name,
                 "Cell communicator device is out of range");

    utl::require(this->dev_tile_partition_[dev] == tile_partition,
                 function_name, "Device belongs to the wrong tile partition");

    utl::require(
        this->dev_cell_partition_[dev] == static_cast<unsigned int>(rank),
        function_name, "Cell communicator rank does not match cell partition");

    utl::require(sf_entry_count <=
                     this->workspace_->sf_tile_buffer_capacity(dev),
                 function_name, "Active SF tile exceeds workspace capacity");

    cudaCheck(cudaSetDevice(dev));
    cudaCheck(
        cudaEventRecord(this->comp_events_[dev], this->comp_streams_[dev]));
    cudaCheck(cudaStreamWaitEvent(this->comm_streams_[dev],
                                  this->comp_events_[dev], 0));
  }

  // Each cell partition has nonzero entries only for its owned cells. Summing
  // those disjoint entries reconstructs the complete global S_tile on every
  // member of this tile partition's cell communicator.
  ncclCheck(ncclGroupStart());

  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    cudaCheck(cudaSetDevice(dev));

    double *sf_re = this->workspace_->sf_re()[dev].d_array().data();
    double *sf_im = this->workspace_->sf_im()[dev].d_array().data();

    ncclCheck(ncclAllReduce(static_cast<const void *>(sf_re),
                            static_cast<void *>(sf_re), sf_entry_count,
                            ncclDouble, ncclSum, comms[rank],
                            this->comm_streams_[dev]));

    ncclCheck(ncclAllReduce(static_cast<const void *>(sf_im),
                            static_cast<void *>(sf_im), sf_entry_count,
                            ncclDouble, ncclSum, comms[rank],
                            this->comm_streams_[dev]));
  }

  ncclCheck(ncclGroupEnd());

  // Make subsequent prefix-sum and remote-sum kernels wait for both
  // all-reduces.
  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    cudaCheck(cudaSetDevice(dev));
    cudaCheck(
        cudaEventRecord(this->comm_events_[dev], this->comm_streams_[dev]));
    cudaCheck(cudaStreamWaitEvent(this->comp_streams_[dev],
                                  this->comm_events_[dev], 0));
  }

  return;
}

void glst_force::exchange_sf_chunk_tile(const unsigned int tile,
                                        const unsigned int source_partition,
                                        const unsigned int source_x_offset,
                                        const unsigned int chunk_x_count) {
  constexpr std::string_view function_name =
      "glst_force::exchange_sf_chunk_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  utl::require(this->sf_exchange_mode_ ==
                   GLST_SF_EXCHANGE_MODE::LOCAL_CHUNK_BROADCAST,
               function_name, "Local chunk exchange is not selected");

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  utl::require(source_partition < this->cell_partition_count_, function_name,
               "Source partition is out of bounds");

  utl::require(chunk_x_count > 0, function_name, "Chunk x count is 0");

  const unsigned int source_partition_x_count =
      this->plan_->cell_partition_x_count()[source_partition];

  utl::require(
      (source_x_offset <= source_partition_x_count) &&
          (chunk_x_count <= source_partition_x_count - source_x_offset),
      function_name, "Source x chunk is out of bounds");

  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);
  const unsigned int nc = this->plan_->tile_node_count(tile);

  utl::require(
      (static_cast<std::size_t>(tile_partition) <
       this->cell_comm_devs_.size()) &&
          (static_cast<std::size_t>(tile_partition) < this->cell_comms_.size()),
      function_name, "Tile partition has not cell communicator");

  const std::vector<int> &devs = this->cell_comm_devs_[tile_partition];
  const std::vector<ncclComm_t> &comms = this->cell_comms_[tile_partition];

  utl::require(
      devs.size() == static_cast<std::size_t>(this->cell_partition_count_),
      function_name,
      "Cell communicator device count does not match cell partition count");

  utl::require(
      comms.size() == devs.size(), function_name,
      "Cell communicator handle count does not match cell partition count");

  const std::size_t yz_cell_count =
      static_cast<std::size_t>(this->plan_->ncell_y()) *
      static_cast<std::size_t>(this->plan_->ncell_z());

  const std::size_t chunk_cell_count =
      static_cast<std::size_t>(chunk_x_count) * yz_cell_count;

  const std::size_t entry_count =
      chunk_cell_count * static_cast<std::size_t>(nc);

  const std::size_t source_cell_offset =
      static_cast<std::size_t>(source_x_offset) * yz_cell_count;

  const std::size_t source_entry_offset =
      source_cell_offset * static_cast<std::size_t>(nc);

  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    utl::require((dev >= 0) && (dev < this->cuda_count_), function_name,
                 "Communicator device is out of range");

    utl::require(this->dev_tile_partition_[dev] == tile_partition,
                 function_name, "Device belongs to the wrong tile partition");

    utl::require(
        this->dev_cell_partition_[dev] == static_cast<unsigned int>(rank),
        function_name, "Communicator rank does not match cell partition");

    if (static_cast<unsigned int>(rank) == source_partition) {
      const std::size_t buffer_capacity =
          this->workspace_->sf_tile_buffer_capacity(dev);

      utl::require((source_entry_offset <= buffer_capacity) &&
                       (entry_count <= buffer_capacity - source_entry_offset),
                   function_name, "Source chunk exceeds SF exchange capacity");
    } else {
      utl::require(entry_count <=
                       this->workspace_->sf_exchange_tile_buffer_capacity(dev),
                   function_name,
                   "Received chunk exceeds SF exchange capacity");
    }

    cudaCheck(cudaSetDevice(dev));

    // Also protects the receive buffer from being overwritten before the
    // previous chunk's prefix/shell kernels have finished reading it.
    cudaCheck(
        cudaEventRecord(this->comp_events_[dev], this->comp_streams_[dev]));
    cudaCheck(cudaStreamWaitEvent(this->comm_streams_[dev],
                                  this->comp_events_[dev], 0));
  }

  ncclCheck(ncclGroupStart());

  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    cudaCheck(cudaSetDevice(dev));

    double *active_re = nullptr;
    double *active_im = nullptr;

    if (static_cast<unsigned int>(rank) == source_partition) {
      active_re =
          this->workspace_->sf_re()[dev].d_array().data() + source_entry_offset;
      active_im =
          this->workspace_->sf_im()[dev].d_array().data() + source_entry_offset;
    } else {
      active_re = this->workspace_->sf_exchange_re()[dev].d_array().data();
      active_im = this->workspace_->sf_exchange_im()[dev].d_array().data();
    }

    ncclCheck(ncclBroadcast(static_cast<const void *>(active_re),
                            static_cast<void *>(active_re), entry_count,
                            ncclDouble, static_cast<int>(source_partition),
                            comms[rank], this->comm_streams_[dev]));

    ncclCheck(ncclBroadcast(static_cast<const void *>(active_im),
                            static_cast<void *>(active_im), entry_count,
                            ncclDouble, static_cast<int>(source_partition),
                            comms[rank], this->comm_streams_[dev]));
  }

  ncclCheck(ncclGroupEnd());

  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    cudaCheck(cudaSetDevice(dev));
    cudaCheck(
        cudaEventRecord(this->comm_events_[dev], this->comm_streams_[dev]));
    cudaCheck(cudaStreamWaitEvent(this->comp_streams_[dev],
                                  this->comm_events_[dev], 0));
  }

  return;
}

__global__ static void
calc_prefix_sum_z_kernel(double *__restrict__ sf_re, double *__restrict__ sf_im,
                         const unsigned int nc, const unsigned int nx,
                         const unsigned int ny, const unsigned int nz) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int x = blockIdx.y / ny;
  const unsigned int y = blockIdx.y % ny;

  if (idx >= nc)
    return;

  double sum_re = 0.0, sum_im = 0.0;
  for (unsigned int z = 0; z < nz; z++) {
    const unsigned int cell = (x * ny + y) * nz + z;
    const std::size_t point =
        static_cast<std::size_t>(cell) * static_cast<std::size_t>(nc) +
        static_cast<std::size_t>(idx);
    sum_re += sf_re[point];
    sum_im += sf_im[point];
    sf_re[point] = sum_re;
    sf_im[point] = sum_im;
  }

  return;
}

__global__ static void
calc_prefix_sum_y_kernel(double *__restrict__ sf_re, double *__restrict__ sf_im,
                         const unsigned int nc, const unsigned int nx,
                         const unsigned int ny, const unsigned int nz) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int x = blockIdx.y / nz;
  const unsigned int z = blockIdx.y % nz;

  if (idx >= nc)
    return;

  double sum_re = 0.0, sum_im = 0.0;
  for (unsigned int y = 0; y < ny; y++) {
    const unsigned int cell = (x * ny + y) * nz + z;
    const std::size_t point =
        static_cast<std::size_t>(cell) * static_cast<std::size_t>(nc) +
        static_cast<std::size_t>(idx);
    sum_re += sf_re[point];
    sum_im += sf_im[point];
    sf_re[point] = sum_re;
    sf_im[point] = sum_im;
  }

  return;
}

__global__ static void
calc_prefix_sum_x_kernel(double *__restrict__ sf_re, double *__restrict__ sf_im,
                         const unsigned int nc, const unsigned int nx,
                         const unsigned int ny, const unsigned int nz) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y / nz;
  const unsigned int z = blockIdx.y % nz;

  if (idx >= nc)
    return;

  double sum_re = 0.0, sum_im = 0.0;
  for (unsigned int x = 0; x < nx; x++) {
    const unsigned int cell = (x * ny + y) * nz + z;
    const std::size_t point =
        static_cast<std::size_t>(cell) * static_cast<std::size_t>(nc) +
        static_cast<std::size_t>(idx);
    sum_re += sf_re[point];
    sum_im += sf_im[point];
    sf_re[point] = sum_re;
    sf_im[point] = sum_im;
  }

  return;
}

__global__ static void calc_prefix_base_kernel(
    double *__restrict__ prefix_base_re, double *__restrict__ prefix_base_im,
    const double *__restrict__ partition_total_re,
    const double *__restrict__ partition_total_im,
    const std::size_t plane_entry_count, const unsigned int cell_partition) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) *
                              static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);

  for (std::size_t i = idx; i < plane_entry_count; i += stride) {
    double base_re = 0.0;
    double base_im = 0.0;

    for (unsigned int source_partition = 0; source_partition < cell_partition;
         source_partition++) {
      const std::size_t source =
          static_cast<std::size_t>(source_partition) * plane_entry_count + i;

      base_re += partition_total_re[source];
      base_im += partition_total_im[source];
    }

    prefix_base_re[i] = base_re;
    prefix_base_im[i] = base_im;
  }

  return;
}

__global__ static void
add_prefix_base_kernel(double *__restrict__ sf_re, double *__restrict__ sf_im,
                       const double *__restrict__ prefix_base_re,
                       const double *__restrict__ prefix_base_im,
                       const unsigned int nc, const unsigned int yz_cell_count,
                       const unsigned int local_cell_count) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nc)
    return;

  for (unsigned int local_cell = blockIdx.y; local_cell < local_cell_count;
       local_cell += gridDim.y) {
    const unsigned int yz_cell = local_cell % yz_cell_count;
    const std::size_t local_point =
        static_cast<std::size_t>(local_cell) * static_cast<std::size_t>(nc) +
        static_cast<std::size_t>(idx);
    const std::size_t base_point =
        static_cast<std::size_t>(yz_cell) * static_cast<std::size_t>(nc) +
        static_cast<std::size_t>(idx);

    sf_re[local_point] += prefix_base_re[base_point];
    sf_im[local_point] += prefix_base_im[base_point];
  }

  return;
}

__device__ void box_sum(double &box_re, double &box_im,
                        const double *__restrict__ P_re,
                        const double *__restrict__ P_im, const unsigned int x0,
                        const unsigned int y0, const unsigned int z0,
                        const unsigned int x1, const unsigned int y1,
                        const unsigned int z1, const unsigned int nx,
                        const unsigned int ny, const unsigned int nz,
                        const unsigned int idx, const unsigned int nc) {
  // Need to check if x0-1, y0-1, z0-1 < 0
  const bool xb = (x0 == 0);
  const bool yb = (y0 == 0);
  const bool zb = (z0 == 0);
  const unsigned int xm = (xb) ? 0 : x0 - 1;
  const unsigned int ym = (yb) ? 0 : y0 - 1;
  const unsigned int zm = (zb) ? 0 : z0 - 1;
  const unsigned int cell0 = (x1 * ny + y1) * nz + z1; // ( x1,   y1,   z1   )
  const unsigned int cell1 = (xm * ny + y1) * nz + z1; // ( x0-1, y1,   z1   )
  const unsigned int cell2 = (x1 * ny + ym) * nz + z1; // ( x1,   y0-1, z1   )
  const unsigned int cell3 = (x1 * ny + y1) * nz + zm; // ( x1,   y1,   z0-1 )
  const unsigned int cell4 = (xm * ny + ym) * nz + z1; // ( x0-1, y0-1, z1   )
  const unsigned int cell5 = (xm * ny + y1) * nz + zm; // ( x0-1, y1,   z0-1 )
  const unsigned int cell6 = (x1 * ny + ym) * nz + zm; // ( x1,   y0-1, z0-1 )
  const unsigned int cell7 = (xm * ny + ym) * nz + zm; // ( x0-1, y0-1, z0-1 )

  const double g0_re = P_re[cell0 * nc + idx];                    // Include
  const double g1_re = (xb) ? 0.0 : P_re[cell1 * nc + idx];       // Exclude
  const double g2_re = (yb) ? 0.0 : P_re[cell2 * nc + idx];       // Exclude
  const double g3_re = (zb) ? 0.0 : P_re[cell3 * nc + idx];       // Exclude
  const double g4_re = (xb || yb) ? 0.0 : P_re[cell4 * nc + idx]; // Include
  const double g5_re = (xb || zb) ? 0.0 : P_re[cell5 * nc + idx]; // Include
  const double g6_re = (yb || zb) ? 0.0 : P_re[cell6 * nc + idx]; // Include
  const double g7_re =
      (xb || yb || zb) ? 0.0 : P_re[cell7 * nc + idx]; // Exclude

  const double g0_im = P_im[cell0 * nc + idx];                    // Include
  const double g1_im = (xb) ? 0.0 : P_im[cell1 * nc + idx];       // Exclude
  const double g2_im = (yb) ? 0.0 : P_im[cell2 * nc + idx];       // Exclude
  const double g3_im = (zb) ? 0.0 : P_im[cell3 * nc + idx];       // Exclude
  const double g4_im = (xb || yb) ? 0.0 : P_im[cell4 * nc + idx]; // Include
  const double g5_im = (xb || zb) ? 0.0 : P_im[cell5 * nc + idx]; // Include
  const double g6_im = (yb || zb) ? 0.0 : P_im[cell6 * nc + idx]; // Include
  const double g7_im =
      (xb || yb || zb) ? 0.0 : P_im[cell7 * nc + idx]; // Exclude

  box_re = g0_re - g1_re - g2_re - g3_re + g4_re + g5_re + g6_re - g7_re;
  box_im = g0_im - g1_im - g2_im - g3_im + g4_im + g5_im + g6_im - g7_im;

  return;
}

__device__ void
cube_sum(double &cube_re, double &cube_im, const double *__restrict__ P_re,
         const double *__restrict__ P_im, const unsigned int x,
         const unsigned int y, const unsigned int z, const unsigned int r,
         const unsigned int source_x_point, const unsigned int source_x_count,
         const unsigned int global_nx, const unsigned int ny,
         const unsigned int nz, const unsigned int idx, const unsigned int nc) {
  cube_re = 0.0;
  cube_im = 0.0;

  if ((source_x_count == 0) || (global_nx == 0))
    return;

  // Need to check if x-r, y-r, z-r is < 0
  const unsigned int global_x0 = (x < r) ? 0u : x - r;
  const unsigned int y0 = (y < r) ? 0u : y - r;
  const unsigned int z0 = (z < r) ? 0u : z - r;

  const unsigned int max_x_radius = global_nx - 1u - x;
  const unsigned int max_y_radius = ny - 1u - y;
  const unsigned int max_z_radius = nz - 1u - z;

  const unsigned int global_x1 = (r > max_x_radius) ? global_nx - 1u : x + r;
  const unsigned int y1 = (r > max_y_radius) ? ny - 1u : y + r;
  const unsigned int z1 = (r > max_z_radius) ? nz - 1u : z + r;

  const unsigned int source_x_end = source_x_point + source_x_count;

  if ((global_x1 < source_x_point) || (global_x0 >= source_x_end))
    return;

  const unsigned int intersect_x0 =
      (global_x0 > source_x_point) ? global_x0 : source_x_point;

  const unsigned int source_x_last = source_x_end - 1u;

  const unsigned int intersect_x1 =
      (global_x1 < source_x_last) ? global_x1 : source_x_last;

  const unsigned int local_x0 = intersect_x0 - source_x_point;
  const unsigned int local_x1 = intersect_x1 - source_x_point;

  box_sum(cube_re, cube_im, P_re, P_im, local_x0, y0, z0, local_x1, y1, z1,
          source_x_count, ny, nz, idx, nc);

  return;
}

__device__ void
shell_sum(double &shell_re, double &shell_im, const double *__restrict__ P_re,
          const double *__restrict__ P_im, const unsigned int x,
          const unsigned int y, const unsigned int z, const unsigned int inner,
          const unsigned int outer, const unsigned int source_x_point,
          const unsigned int source_x_count, const unsigned int global_nx,
          const unsigned int ny, const unsigned int nz, const unsigned int idx,
          const unsigned int nc) {
  double osum_re = 0.0;
  double osum_im = 0.0;

  cube_sum(osum_re, osum_im, P_re, P_im, x, y, z, outer, source_x_point,
           source_x_count, global_nx, ny, nz, idx, nc);

  double isum_re = 0.0;
  double isum_im = 0.0;

  cube_sum(isum_re, isum_im, P_re, P_im, x, y, z, inner, source_x_point,
           source_x_count, global_nx, ny, nz, idx, nc);

  shell_re = osum_re - isum_re;
  shell_im = osum_im - isum_im;

  return;
}

__device__ __forceinline__ static void
distributed_prefix_value(double &value_re, double &value_im,
                         const double *__restrict__ local_prefix_re,
                         const double *__restrict__ local_prefix_im,
                         const double *__restrict__ remote_prefix_re,
                         const double *__restrict__ remote_prefix_im,
                         const unsigned int *__restrict__ remote_slot_by_x,
                         const unsigned int global_x, const unsigned int y,
                         const unsigned int z, const unsigned int local_x_point,
                         const unsigned int local_x_count,
                         const unsigned int ny, const unsigned int nz,
                         const unsigned int idx, const unsigned int nc) {
  constexpr unsigned int invalid_slot =
      ~0u; // Device-safe invalid-slot sentinel

  value_re = 0.0;
  value_im = 0.0;

  const unsigned int local_x_end = local_x_point + local_x_count;

  if ((global_x >= local_x_point) && (global_x < local_x_end)) {
    const unsigned int local_x = global_x - local_x_point;
    const std::size_t local_cell =
        (static_cast<std::size_t>(local_x) * static_cast<std::size_t>(ny) +
         static_cast<std::size_t>(y)) *
            static_cast<std::size_t>(nz) +
        static_cast<std::size_t>(z);
    const std::size_t point =
        static_cast<std::size_t>(local_cell) * static_cast<std::size_t>(nc) +
        static_cast<std::size_t>(idx);

    value_re = local_prefix_re[point];
    value_im = local_prefix_im[point];

    return;
  }

  const unsigned int slot = remote_slot_by_x[global_x];

  if (slot == invalid_slot)
    return;

  const std::size_t remote_cell =
      (static_cast<std::size_t>(slot) * static_cast<std::size_t>(ny) +
       static_cast<std::size_t>(y)) *
          static_cast<std::size_t>(nz) +
      static_cast<std::size_t>(z);
  const std::size_t point =
      static_cast<std::size_t>(remote_cell) * static_cast<std::size_t>(nc) +
      static_cast<std::size_t>(idx);

  value_re = remote_prefix_re[point];
  value_im = remote_prefix_im[point];

  return;
}

__device__ __forceinline__ static void distributed_box_sum(
    double &box_re, double &box_im, const double *__restrict__ local_prefix_re,
    const double *__restrict__ local_prefix_im,
    const double *__restrict__ remote_prefix_re,
    const double *__restrict__ remote_prefix_im,
    const unsigned int *__restrict__ remote_slot_by_x, const unsigned int x0,
    const unsigned int y0, const unsigned int z0, const unsigned int x1,
    const unsigned int y1, const unsigned int z1,
    const unsigned int local_x_point, const unsigned int local_x_count,
    const unsigned int ny, const unsigned int nz, const unsigned int idx,
    const unsigned int nc) {
  const bool xb = (x0 == 0);
  const bool yb = (y0 == 0);
  const bool zb = (z0 == 0);

  const unsigned int xm = (xb) ? 0u : x0 - 1u;
  const unsigned int ym = (yb) ? 0u : y0 - 1u;
  const unsigned int zm = (zb) ? 0u : z0 - 1u;

  double g0_re = 0.0, g0_im = 0.0;
  double g1_re = 0.0, g1_im = 0.0;
  double g2_re = 0.0, g2_im = 0.0;
  double g3_re = 0.0, g3_im = 0.0;
  double g4_re = 0.0, g4_im = 0.0;
  double g5_re = 0.0, g5_im = 0.0;
  double g6_re = 0.0, g6_im = 0.0;
  double g7_re = 0.0, g7_im = 0.0;

  // (x1, y1, z1) | Include
  distributed_prefix_value(g0_re, g0_im, local_prefix_re, local_prefix_im,
                           remote_prefix_re, remote_prefix_im, remote_slot_by_x,
                           x1, y1, z1, local_x_point, local_x_count, ny, nz,
                           idx, nc);

  if (!xb) { // (x0 - 1, y1, z1) | Exclude
    distributed_prefix_value(g1_re, g1_im, local_prefix_re, local_prefix_im,
                             remote_prefix_re, remote_prefix_im,
                             remote_slot_by_x, xm, y1, z1, local_x_point,
                             local_x_count, ny, nz, idx, nc);
  }

  if (!yb) { // (x1, y0 - 1, z1) | Exclude
    distributed_prefix_value(g2_re, g2_im, local_prefix_re, local_prefix_im,
                             remote_prefix_re, remote_prefix_im,
                             remote_slot_by_x, x1, ym, z1, local_x_point,
                             local_x_count, ny, nz, idx, nc);
  }

  if (!zb) { // (x1, y1, z0 - 1) | Exclude
    distributed_prefix_value(g3_re, g3_im, local_prefix_re, local_prefix_im,
                             remote_prefix_re, remote_prefix_im,
                             remote_slot_by_x, x1, y1, zm, local_x_point,
                             local_x_count, ny, nz, idx, nc);
  }

  if (!xb && !yb) { // (x0 - 1, y0 - 1, z1) | Include
    distributed_prefix_value(g4_re, g4_im, local_prefix_re, local_prefix_im,
                             remote_prefix_re, remote_prefix_im,
                             remote_slot_by_x, xm, ym, z1, local_x_point,
                             local_x_count, ny, nz, idx, nc);
  }

  if (!xb && !zb) { // (x0 - 1, y1, z0 - 1) | Include
    distributed_prefix_value(g5_re, g5_im, local_prefix_re, local_prefix_im,
                             remote_prefix_re, remote_prefix_im,
                             remote_slot_by_x, xm, y1, zm, local_x_point,
                             local_x_count, ny, nz, idx, nc);
  }

  if (!yb && !zb) { //(x1, y0 - 1, z0 - 1) | Include
    distributed_prefix_value(g6_re, g6_im, local_prefix_re, local_prefix_im,
                             remote_prefix_re, remote_prefix_im,
                             remote_slot_by_x, x1, ym, zm, local_x_point,
                             local_x_count, ny, nz, idx, nc);
  }

  if (!xb && !yb && !zb) { //(x0 - 1, y0 - 1, z0 - 1) | Exclude
    distributed_prefix_value(g7_re, g7_im, local_prefix_re, local_prefix_im,
                             remote_prefix_re, remote_prefix_im,
                             remote_slot_by_x, xm, ym, zm, local_x_point,
                             local_x_count, ny, nz, idx, nc);
  }

  box_re = g0_re - g1_re - g2_re - g3_re + g4_re + g5_re + g6_re - g7_re;
  box_im = g0_im - g1_im - g2_im - g3_im + g4_im + g5_im + g6_im - g7_im;

  return;
}

__device__ __forceinline__ static void distributed_cube_sum(
    double &cube_re, double &cube_im,
    const double *__restrict__ local_prefix_re,
    const double *__restrict__ local_prefix_im,
    const double *__restrict__ remote_prefix_re,
    const double *__restrict__ remote_prefix_im,
    const unsigned int *__restrict__ remote_slot_by_x, const unsigned int x,
    const unsigned int y, const unsigned int z, const unsigned int radius,
    const unsigned int local_x_point, const unsigned int local_x_count,
    const unsigned int global_nx, const unsigned int ny, const unsigned int nz,
    const unsigned int idx, const unsigned int nc) {
  const unsigned int x0 = (x < radius) ? 0u : x - radius;
  const unsigned int y0 = (y < radius) ? 0u : y - radius;
  const unsigned int z0 = (z < radius) ? 0u : z - radius;

  const unsigned int max_x_radius = global_nx - 1u - x;
  const unsigned int max_y_radius = ny - 1u - y;
  const unsigned int max_z_radius = nz - 1u - z;

  const unsigned int x1 = (radius > max_x_radius) ? global_nx - 1u : x + radius;
  const unsigned int y1 = (radius > max_y_radius) ? ny - 1u : y + radius;
  const unsigned int z1 = (radius > max_z_radius) ? nz - 1u : z + radius;

  distributed_box_sum(cube_re, cube_im, local_prefix_re, local_prefix_im,
                      remote_prefix_re, remote_prefix_im, remote_slot_by_x, x0,
                      y0, z0, x1, y1, z1, local_x_point, local_x_count, ny, nz,
                      idx, nc);

  return;
}

__device__ __forceinline__ static void distributed_shell_sum(
    double &shell_re, double &shell_im,
    const double *__restrict__ local_prefix_re,
    const double *__restrict__ local_prefix_im,
    const double *__restrict__ remote_prefix_re,
    const double *__restrict__ remote_prefix_im,
    const unsigned int *__restrict__ remote_slot_by_x, const unsigned int x,
    const unsigned int y, const unsigned int z, const unsigned int inner,
    const unsigned int outer, const unsigned int local_x_point,
    const unsigned int local_x_count, const unsigned int global_nx,
    const unsigned int ny, const unsigned int nz, const unsigned int idx,
    const unsigned int nc) {
  double outer_re = 0.0;
  double outer_im = 0.0;

  distributed_cube_sum(outer_re, outer_im, local_prefix_re, local_prefix_im,
                       remote_prefix_re, remote_prefix_im, remote_slot_by_x, x,
                       y, z, outer, local_x_point, local_x_count, global_nx, ny,
                       nz, idx, nc);

  double inner_re = 0.0;
  double inner_im = 0.0;

  distributed_cube_sum(inner_re, inner_im, local_prefix_re, local_prefix_im,
                       remote_prefix_re, remote_prefix_im, remote_slot_by_x, x,
                       y, z, inner, local_x_point, local_x_count, global_nx, ny,
                       nz, idx, nc);

  shell_re = outer_re - inner_re;
  shell_im = outer_im - inner_im;

  return;
}

template <bool ACCUMULATE>
__global__ static void calc_rmt_sum_kernel(
    double *__restrict__ rmt_sum_re, double *__restrict__ rmt_sum_im,
    const double *__restrict__ sf_re, const double *__restrict__ sf_im,
    const double *__restrict__ cw, const unsigned int *__restrict__ groups,
    const unsigned int nc, const unsigned int *__restrict__ grp_r_in,
    const unsigned int *__restrict__ grp_r_out,
    const unsigned int source_x_point, const unsigned int source_x_count,
    const unsigned int global_nx, const unsigned int ny, const unsigned int nz,
    const unsigned int local_cell_count, const unsigned int first_global_cell) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nc)
    return;

  const double wc = cw[idx];
  const unsigned int group = groups[idx];

  for (unsigned int local_cell = blockIdx.y; local_cell < local_cell_count;
       local_cell += gridDim.y) {
    const unsigned int global_cell = first_global_cell + local_cell;

    const unsigned int x = global_cell / (ny * nz);
    const unsigned int y = (global_cell / nz) % ny;
    const unsigned int z = global_cell % nz;

    const unsigned int inner = grp_r_in[group];
    const unsigned int outer = grp_r_out[group];

    double shell_re = 0.0;
    double shell_im = 0.0;

    shell_sum(shell_re, shell_im, sf_re, sf_im, x, y, z, inner, outer,
              source_x_point, source_x_count, global_nx, ny, nz, idx, nc);

    shell_re *= wc;
    shell_im *= wc;

    const std::size_t output =
        static_cast<std::size_t>(local_cell) * static_cast<std::size_t>(nc) +
        static_cast<std::size_t>(idx);

    if constexpr (ACCUMULATE) {
      rmt_sum_re[output] += shell_re;
      rmt_sum_im[output] += shell_im;
    } else {
      rmt_sum_re[output] = shell_re;
      rmt_sum_im[output] = shell_im;
    }
  }

  return;
}

__global__ static void calc_rmt_sum_distributed_kernel(
    double *__restrict__ rmt_sum_re, double *__restrict__ rmt_sum_im,
    const double *__restrict__ local_prefix_re,
    const double *__restrict__ local_prefix_im,
    const double *__restrict__ remote_prefix_re,
    const double *__restrict__ remote_prefix_im,
    const unsigned int *__restrict__ remote_slot_by_x,
    const double *__restrict__ cw, const unsigned int nc,
    const unsigned int inner, const unsigned int outer,
    const unsigned int local_x_point, const unsigned int local_x_count,
    const unsigned int global_nx, const unsigned int ny, const unsigned int nz,
    const unsigned int local_cell_count, const unsigned int first_global_cell) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nc)
    return;

  const double wc = cw[idx];

  for (unsigned int local_cell = blockIdx.y; local_cell < local_cell_count;
       local_cell += gridDim.y) {
    const unsigned int global_cell = first_global_cell + local_cell;

    const unsigned int x = global_cell / (ny * nz);
    const unsigned int y = (global_cell / nz) % ny;
    const unsigned int z = global_cell % nz;

    double shell_re = 0.0;
    double shell_im = 0.0;

    distributed_shell_sum(shell_re, shell_im, local_prefix_re, local_prefix_im,
                          remote_prefix_re, remote_prefix_im, remote_slot_by_x,
                          x, y, z, inner, outer, local_x_point, local_x_count,
                          global_nx, ny, nz, idx, nc);

    const std::size_t output =
        static_cast<std::size_t>(local_cell) * static_cast<std::size_t>(nc) +
        static_cast<std::size_t>(idx);

    rmt_sum_re[output] = wc * shell_re;
    rmt_sum_im[output] = wc * shell_im;
  }

  return;
}

void glst_force::sum_rmt_sf_tile(const unsigned int tile) {
  constexpr std::string_view function_name = "glst_force::sum_rmt_sf_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  const unsigned int tile_node_point = this->plan_->tile_node_point(tile);
  const unsigned int tile_node_count = this->plan_->tile_node_count(tile);

  utl::require(tile_node_count > 0, function_name, "Tile node count is 0");

  utl::require(tile_node_count <= this->plan_->max_tile_nodes(), function_name,
               "Tile exceeds buffer size");

  if (this->cell_partition_count_ > 1) {
    utl::require(this->sf_exchange_mode_ ==
                     GLST_SF_EXCHANGE_MODE::FULL_GLOBAL_ALLREDUCE,
                 function_name,
                 "Full global S_tile sum called while FULL_GLOBAL_ALLREDUCE is "
                 "not selected");
  }

  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);

  const unsigned int nc = tile_node_count;
  const unsigned int off = tile_node_point;

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    if (this->dev_tile_partition_[dev] != tile_partition)
      continue;

    cudaCheck(cudaSetDevice(dev));

    const unsigned int cell_partition = this->dev_cell_partition_[dev];

    const unsigned int local_cell_count =
        this->plan_->local_cell_count(cell_partition);

    const unsigned int first_global_cell =
        this->plan_->first_global_cell(cell_partition);

    utl::require(static_cast<std::size_t>(local_cell_count) ==
                     this->workspace_->cell_capacity(dev),
                 function_name,
                 "Local cell count does not match workspace capacity");

    const std::size_t rmt_entry_count =
        static_cast<std::size_t>(local_cell_count) *
        static_cast<std::size_t>(nc);

    utl::require(rmt_entry_count <=
                     this->workspace_->rmt_tile_buffer_capacity(dev),
                 function_name, "Active rmt tile exceeds workspace capacity");

    if (local_cell_count == 0)
      continue;

    {
      constexpr dim3 num_threads(512, 1, 1);
      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            this->plan_->ncell_x() * this->plan_->ncell_y(), 1);

      calc_prefix_sum_z_kernel<<<num_blocks, num_threads, 0,
                                 this->comp_streams_[dev]>>>(
          this->workspace_->sf_re()[dev].d_array().data(),
          this->workspace_->sf_im()[dev].d_array().data(), nc,
          this->plan_->ncell_x(), this->plan_->ncell_y(),
          this->plan_->ncell_z());

      cudaCheck(cudaGetLastError());
    }

    {
      constexpr dim3 num_threads(512, 1, 1);
      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            this->plan_->ncell_x() * this->plan_->ncell_z(), 1);

      calc_prefix_sum_y_kernel<<<num_blocks, num_threads, 0,
                                 this->comp_streams_[dev]>>>(
          this->workspace_->sf_re()[dev].d_array().data(),
          this->workspace_->sf_im()[dev].d_array().data(), nc,
          this->plan_->ncell_x(), this->plan_->ncell_y(),
          this->plan_->ncell_z());

      cudaCheck(cudaGetLastError());
    }

    {
      constexpr dim3 num_threads(512, 1, 1);
      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            this->plan_->ncell_y() * this->plan_->ncell_z(), 1);

      calc_prefix_sum_x_kernel<<<num_blocks, num_threads, 0,
                                 this->comp_streams_[dev]>>>(
          this->workspace_->sf_re()[dev].d_array().data(),
          this->workspace_->sf_im()[dev].d_array().data(), nc,
          this->plan_->ncell_x(), this->plan_->ncell_y(),
          this->plan_->ncell_z());

      cudaCheck(cudaGetLastError());
    }

    {
#ifdef __GLST_DEBUG__
      constexpr dim3 num_threads(256, 1, 1);
#else
      constexpr dim3 num_threads(512, 1, 1);
#endif
      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            std::min(65535u, local_cell_count), 1);

      calc_rmt_sum_kernel<false>
          <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
              this->workspace_->rmt_sum_re()[dev].d_array().data(),
              this->workspace_->rmt_sum_im()[dev].d_array().data(),
              this->workspace_->sf_re()[dev].d_array().data(),
              this->workspace_->sf_im()[dev].d_array().data(),
              this->plan_->w()[dev].d_array().data() + off,
              this->plan_->group()[dev].d_array().data() + off, nc,
              this->plan_->grp_r_in()[dev].d_array().data(),
              this->plan_->grp_r_out()[dev].d_array().data(), 0u,
              this->plan_->ncell_x(), this->plan_->ncell_x(),
              this->plan_->ncell_y(), this->plan_->ncell_z(), local_cell_count,
              first_global_cell);

      cudaCheck(cudaGetLastError());
    }
  }

  return;
}

void glst_force::sum_rmt_sf_chunk_tile(const unsigned int tile,
                                       const unsigned int source_partition,
                                       const unsigned int source_x_offset,
                                       const unsigned int chunk_x_count) {
  constexpr std::string_view function_name =
      "glst_force::sum_rmt_sf_chunk_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  utl::require(this->sf_exchange_mode_ ==
                   GLST_SF_EXCHANGE_MODE::LOCAL_CHUNK_BROADCAST,
               function_name, "Local chunk exchange is not selected");

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  utl::require(source_partition < this->cell_partition_count_, function_name,
               "Source partition is out of bounds");

  utl::require(chunk_x_count > 0, function_name, "Chunk x count is 0");

  const unsigned int source_partition_x_count =
      this->plan_->cell_partition_x_count()[source_partition];

  utl::require(
      (source_x_offset <= source_partition_x_count) &&
          (chunk_x_count <= source_partition_x_count - source_x_offset),
      function_name, "Source x chunk is out of bounds");

  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);
  const unsigned int nc = this->plan_->tile_node_count(tile);
  const unsigned int off = this->plan_->tile_node_point(tile);

  const std::vector<int> &devs = this->cell_comm_devs_[tile_partition];

  utl::require(
      devs.size() == static_cast<std::size_t>(this->cell_partition_count_),
      function_name,
      "Cell communicator device count does not match cell partition count");

  const std::size_t yz_cell_count =
      static_cast<std::size_t>(this->plan_->ncell_y()) *
      static_cast<std::size_t>(this->plan_->ncell_z());

  const std::size_t chunk_cell_count =
      static_cast<std::size_t>(chunk_x_count) * yz_cell_count;

  const std::size_t chunk_entry_count =
      chunk_cell_count * static_cast<std::size_t>(nc);

  const std::size_t source_cell_offset =
      static_cast<std::size_t>(source_x_offset) * yz_cell_count;

  const std::size_t source_entry_offset =
      source_cell_offset * static_cast<std::size_t>(nc);

  const unsigned int source_global_x_point =
      this->plan_->cell_partition_x_point()[source_partition] + source_x_offset;

  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];
    const unsigned int cell_partition = this->dev_cell_partition_[dev];

    const unsigned int local_cell_count =
        this->plan_->local_cell_count(cell_partition);

    const unsigned int first_global_cell =
        this->plan_->first_global_cell(cell_partition);

    const std::size_t rmt_entry_count =
        static_cast<std::size_t>(local_cell_count) *
        static_cast<std::size_t>(nc);

    utl::require(rmt_entry_count <=
                     this->workspace_->rmt_tile_buffer_capacity(dev),
                 function_name, "Active rmt tile exceeds workspace capacity");

    if (cell_partition == source_partition) {
      const std::size_t buffer_capacity =
          this->workspace_->sf_tile_buffer_capacity(dev);

      utl::require(
          (source_entry_offset <= buffer_capacity) &&
              (chunk_entry_count <= buffer_capacity - source_entry_offset),
          function_name, "Source chunk exceeds local SF capacity");
    } else {
      utl::require(chunk_entry_count <=
                       this->workspace_->sf_exchange_tile_buffer_capacity(dev),
                   function_name,
                   "Received chunk exceeds SF exchange capacity");
    }

    if (local_cell_count == 0)
      continue;

    cudaCheck(cudaSetDevice(dev));

    double *active_re = nullptr;
    double *active_im = nullptr;

    if (cell_partition == source_partition) {
      active_re =
          this->workspace_->sf_re()[dev].d_array().data() + source_entry_offset;
      active_im =
          this->workspace_->sf_im()[dev].d_array().data() + source_entry_offset;
    } else {
      active_re = this->workspace_->sf_exchange_re()[dev].d_array().data();
      active_im = this->workspace_->sf_exchange_im()[dev].d_array().data();
    }

    {
      constexpr dim3 num_threads(512, 1, 1);

      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            chunk_x_count * this->plan_->ncell_y(), 1);

      calc_prefix_sum_z_kernel<<<num_blocks, num_threads, 0,
                                 this->comp_streams_[dev]>>>(
          active_re, active_im, nc, chunk_x_count, this->plan_->ncell_y(),
          this->plan_->ncell_z());

      cudaCheck(cudaGetLastError());
    }

    {
      constexpr dim3 num_threads(512, 1, 1);

      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            chunk_x_count * this->plan_->ncell_z(), 1);

      calc_prefix_sum_y_kernel<<<num_blocks, num_threads, 0,
                                 this->comp_streams_[dev]>>>(
          active_re, active_im, nc, chunk_x_count, this->plan_->ncell_y(),
          this->plan_->ncell_z());

      cudaCheck(cudaGetLastError());
    }

    {
      constexpr dim3 num_threads(512, 1, 1);

      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            this->plan_->ncell_y() * this->plan_->ncell_z(), 1);

      calc_prefix_sum_x_kernel<<<num_blocks, num_threads, 0,
                                 this->comp_streams_[dev]>>>(
          active_re, active_im, nc, chunk_x_count, this->plan_->ncell_y(),
          this->plan_->ncell_z());

      cudaCheck(cudaGetLastError());
    }

    {
#ifdef __GLST_DEBUG__
      constexpr dim3 num_threads(256, 1, 1);
#else
      constexpr dim3 num_threads(512, 1, 1);
#endif

      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            std::min(65535u, local_cell_count), 1);

      calc_rmt_sum_kernel<true>
          <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
              this->workspace_->rmt_sum_re()[dev].d_array().data(),
              this->workspace_->rmt_sum_im()[dev].d_array().data(), active_re,
              active_im, this->plan_->w()[dev].d_array().data() + off,
              this->plan_->group()[dev].d_array().data() + off, nc,
              this->plan_->grp_r_in()[dev].d_array().data(),
              this->plan_->grp_r_out()[dev].d_array().data(),
              source_global_x_point, chunk_x_count, this->plan_->ncell_x(),
              this->plan_->ncell_y(), this->plan_->ncell_z(), local_cell_count,
              first_global_cell);

      cudaCheck(cudaGetLastError());
    }
  }

  return;
}

void glst_force::build_distributed_prefix_tile(const unsigned int tile) {
  constexpr std::string_view function_name =
      "glst_force::build_distributed_prefix_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  utl::require(this->sf_exchange_mode_ ==
                   GLST_SF_EXCHANGE_MODE::DISTRIBUTED_PREFIX,
               function_name, "Distributed prefix exchange is not selected");

  utl::require(this->cell_partition_count_ > 1, function_name,
               "Distributed prefix exchange requires multiple cell partitions");

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);
  const unsigned int group = this->plan_->tile_group(tile);
  const unsigned int nc = this->plan_->tile_node_count(tile);
  const unsigned int ny = this->plan_->ncell_y();
  const unsigned int nz = this->plan_->ncell_z();
  const unsigned int yz_cell_count = ny * nz;
  const std::size_t plane_entry_count =
      static_cast<std::size_t>(yz_cell_count) * static_cast<std::size_t>(nc);
  const std::size_t plane_byte_count = plane_entry_count * sizeof(double);

  utl::require(
      static_cast<std::size_t>(tile_partition) < this->cell_comm_devs_.size(),
      function_name, "Tile partition has no cell-communicator devices");

  utl::require(static_cast<std::size_t>(tile_partition) <
                   this->cell_comms_.size(),
               function_name, "Tile partition has no cell communicators");

  const std::vector<int> &devs = this->cell_comm_devs_[tile_partition];
  const std::vector<ncclComm_t> &comms = this->cell_comms_[tile_partition];

  utl::require(
      devs.size() == static_cast<std::size_t>(this->cell_partition_count_),
      function_name,
      "Cell communicator device count does not match cell partition count");

  utl::require(comms.size() == devs.size(), function_name,
               "Cell communicator handle count does not match device count");

  // Build local z, y, and x prefixes. The final local x-plane is the total
  // contribution of this cell partition.
  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    utl::require((dev >= 0) && (dev < this->cuda_count_), function_name,
                 "Cell communicator device is out of range");

    const unsigned int cell_partition = static_cast<unsigned int>(rank);

    utl::require(this->dev_cell_partition_[dev] == cell_partition,
                 function_name,
                 "Cell communicator rank does not match cell partition");

    utl::require(this->dev_tile_partition_[dev] == tile_partition,
                 function_name, "Device belongs to the wrong tile partition");

    const unsigned int local_x_count =
        this->plan_->cell_partition_x_count()[cell_partition];
    const unsigned int local_cell_count =
        this->plan_->local_cell_count(cell_partition);
    const std::size_t local_entry_count =
        static_cast<std::size_t>(local_cell_count) *
        static_cast<std::size_t>(nc);

    utl::require(local_entry_count <=
                     this->workspace_->sf_tile_buffer_capacity(dev),
                 function_name, "Local prefix exceeds SF workspace capacity");

    const std::size_t partition_total_entry_count =
        static_cast<std::size_t>(this->cell_partition_count_) *
        plane_entry_count;

    utl::require(
        partition_total_entry_count <=
            this->workspace_->prefix_partition_total_buffer_capacity(dev),
        function_name,
        "Partition-total prefix data exceeds workspace capacity");

    utl::require(plane_entry_count <=
                     this->workspace_->prefix_base_buffer_capacity(dev),
                 function_name, "Prefix base exceeds workspace capacity");

    const std::vector<unsigned int> &prefix_x =
        this->plan_->partition_group_prefix_x(cell_partition, group);
    const std::size_t imported_entry_count =
        prefix_x.size() * plane_entry_count;

    utl::require(imported_entry_count <=
                     this->workspace_->sf_exchange_tile_buffer_capacity(dev),
                 function_name,
                 "Imported prefix planes exceed exchange capacity");

    cudaCheck(cudaSetDevice(dev));

    if (local_x_count > 0) {
      {
        constexpr dim3 num_threads(512, 1, 1);
        const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                              local_x_count * ny, 1);

        calc_prefix_sum_z_kernel<<<num_blocks, num_threads, 0,
                                   this->comp_streams_[dev]>>>(
            this->workspace_->sf_re()[dev].d_array().data(),
            this->workspace_->sf_im()[dev].d_array().data(), nc, local_x_count,
            ny, nz);
        cudaCheck(cudaGetLastError());
      }

      {
        constexpr dim3 num_threads(512, 1, 1);
        const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                              local_x_count * nz, 1);

        calc_prefix_sum_y_kernel<<<num_blocks, num_threads, 0,
                                   this->comp_streams_[dev]>>>(
            this->workspace_->sf_re()[dev].d_array().data(),
            this->workspace_->sf_im()[dev].d_array().data(), nc, local_x_count,
            ny, nz);
        cudaCheck(cudaGetLastError());
      }

      {
        constexpr dim3 num_threads(512, 1, 1);
        const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                              yz_cell_count, 1);

        calc_prefix_sum_x_kernel<<<num_blocks, num_threads, 0,
                                   this->comp_streams_[dev]>>>(
            this->workspace_->sf_re()[dev].d_array().data(),
            this->workspace_->sf_im()[dev].d_array().data(), nc, local_x_count,
            ny, nz);
        cudaCheck(cudaGetLastError());
      }
    } else {
      // A zero-width partition contributes a zero slab total. prefix_base is
      // reused temporarily as its all-gather send buffer.
      cudaCheck(cudaMemsetAsync(
          static_cast<void *>(
              this->workspace_->prefix_base_re()[dev].d_array().data()),
          0, plane_byte_count, this->comp_streams_[dev]));

      cudaCheck(cudaMemsetAsync(
          static_cast<void *>(
              this->workspace_->prefix_base_im()[dev].d_array().data()),
          0, plane_byte_count, this->comp_streams_[dev]));
    }

    cudaCheck(
        cudaEventRecord(this->comp_events_[dev], this->comp_streams_[dev]));

    cudaCheck(cudaStreamWaitEvent(this->comm_streams_[dev],
                                  this->comp_events_[dev], 0));
  }

  // All-gather each slab's final local x-prefix plane.
  ncclCheck(ncclGroupStart());

  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    const unsigned int cell_partition = static_cast<unsigned int>(rank);
    const unsigned int local_x_count =
        this->plan_->cell_partition_x_count()[cell_partition];

    cudaCheck(cudaSetDevice(dev));

    const double *send_re = nullptr;
    const double *send_im = nullptr;

    if (local_x_count > 0) {
      const std::size_t last_plane_offset =
          static_cast<std::size_t>(local_x_count - 1u) * plane_entry_count;

      send_re =
          this->workspace_->sf_re()[dev].d_array().data() + last_plane_offset;
      send_im =
          this->workspace_->sf_im()[dev].d_array().data() + last_plane_offset;
    } else {
      send_re = this->workspace_->prefix_base_re()[dev].d_array().data();
      send_im = this->workspace_->prefix_base_im()[dev].d_array().data();
    }

    ncclCheck(ncclAllGather(
        static_cast<const void *>(send_re),
        static_cast<void *>(this->workspace_->prefix_partition_total_re()[dev]
                                .d_array()
                                .data()),
        plane_entry_count, ncclDouble, comms[rank], this->comm_streams_[dev]));

    ncclCheck(ncclAllGather(
        static_cast<const void *>(send_im),
        static_cast<void *>(this->workspace_->prefix_partition_total_im()[dev]
                                .d_array()
                                .data()),
        plane_entry_count, ncclDouble, comms[rank], this->comm_streams_[dev]));
  }

  ncclCheck(ncclGroupEnd());

  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    cudaCheck(cudaSetDevice(dev));

    cudaCheck(
        cudaEventRecord(this->comm_events_[dev], this->comm_streams_[dev]));

    cudaCheck(cudaStreamWaitEvent(this->comp_streams_[dev],
                                  this->comm_events_[dev], 0));
  }

  // Construct the exclusive prefix base for each partition and add it to
  // every locally stored x-plane
  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    const unsigned int cell_partition = static_cast<unsigned int>(rank);
    const unsigned int local_cell_count =
        this->plan_->local_cell_count(cell_partition);

    cudaCheck(cudaSetDevice(dev));

    {
      constexpr unsigned int num_threads = 256;
      const std::size_t required_blocks =
          (plane_entry_count + static_cast<std::size_t>(num_threads) - 1) /
          static_cast<std::size_t>(num_threads);
      const unsigned int num_blocks = static_cast<unsigned int>(
          std::min<std::size_t>(required_blocks, 65535u));

      calc_prefix_base_kernel<<<num_blocks, num_threads, 0,
                                this->comp_streams_[dev]>>>(
          this->workspace_->prefix_base_re()[dev].d_array().data(),
          this->workspace_->prefix_base_im()[dev].d_array().data(),
          this->workspace_->prefix_partition_total_re()[dev].d_array().data(),
          this->workspace_->prefix_partition_total_im()[dev].d_array().data(),
          plane_entry_count, cell_partition);
      cudaCheck(cudaGetLastError());
    }

    if (local_cell_count > 0) {
      constexpr dim3 num_threads(512, 1, 1);
      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            std::min(65535u, local_cell_count), 1);

      add_prefix_base_kernel<<<num_blocks, num_threads, 0,
                               this->comp_streams_[dev]>>>(
          this->workspace_->sf_re()[dev].d_array().data(),
          this->workspace_->sf_im()[dev].d_array().data(),
          this->workspace_->prefix_base_re()[dev].d_array().data(),
          this->workspace_->prefix_base_im()[dev].d_array().data(), nc,
          yz_cell_count, local_cell_count);
      cudaCheck(cudaGetLastError());
    }

    cudaCheck(
        cudaEventRecord(this->comp_events_[dev], this->comp_streams_[dev]));

    cudaCheck(cudaStreamWaitEvent(this->comm_streams_[dev],
                                  this->comp_events_[dev], 0));
  }

  std::size_t imported_plane_count = 0;

  for (unsigned int target_partition = 0;
       target_partition < this->cell_partition_count_; target_partition++) {
    const std::vector<unsigned int> &prefix_x =
        this->plan_->partition_group_prefix_x(target_partition, group);

    utl::require(prefix_x.size() <= std::numeric_limits<std::size_t>::max() -
                                        imported_plane_count,
                 function_name, "Imported prefix-plane count overflow");

    imported_plane_count += prefix_x.size();
  }

  if (imported_plane_count == 0)
    return;

  // Import the required global-prefix x-planes. Ordering is receiving
  // partition, then increasing global x.
  ncclCheck(ncclGroupStart());

  for (unsigned int recv_partition = 0;
       recv_partition < this->cell_partition_count_; recv_partition++) {
    const int recv_dev = devs[recv_partition];

    const std::vector<unsigned int> &prefix_x =
        this->plan_->partition_group_prefix_x(recv_partition, group);

    for (std::size_t slot = 0; slot < prefix_x.size(); slot++) {
      const unsigned int global_x = prefix_x[slot];

      const unsigned int representative_cell = global_x * yz_cell_count;
      const unsigned int send_partition =
          this->plan_->cell_partition_idx(representative_cell);

      utl::require(send_partition != recv_partition, function_name,
                   "Imported prefix plane is locally owned");

      const int send_dev = devs[send_partition];

      const unsigned int send_x_point =
          this->plan_->cell_partition_x_point()[send_partition];
      const unsigned int send_x_count =
          this->plan_->cell_partition_x_count()[send_partition];

      utl::require(global_x >= send_x_point, function_name,
                   "Prefix-plane owner begins after the plane");

      const unsigned int send_local_x = global_x - send_x_point;

      utl::require(send_local_x < send_x_count, function_name,
                   "Prefix plane is outside its owner partition");

      const std::size_t send_offset =
          static_cast<std::size_t>(send_local_x) * plane_entry_count;
      const std::size_t recv_offset = slot * plane_entry_count;

      cudaCheck(cudaSetDevice(send_dev));

      ncclCheck(ncclSend(
          static_cast<const void *>(
              this->workspace_->sf_re()[send_dev].d_array().data() +
              send_offset),
          plane_entry_count, ncclDouble, static_cast<int>(recv_partition),
          comms[send_partition], this->comm_streams_[send_dev]));

      ncclCheck(ncclSend(
          static_cast<const void *>(
              this->workspace_->sf_im()[send_dev].d_array().data() +
              send_offset),
          plane_entry_count, ncclDouble, static_cast<int>(recv_partition),
          comms[send_partition], this->comm_streams_[send_dev]));

      cudaCheck(cudaSetDevice(recv_dev));

      ncclCheck(ncclRecv(
          static_cast<void *>(
              this->workspace_->sf_exchange_re()[recv_dev].d_array().data() +
              recv_offset),
          plane_entry_count, ncclDouble, static_cast<int>(send_partition),
          comms[recv_partition], this->comm_streams_[recv_dev]));

      ncclCheck(ncclRecv(
          static_cast<void *>(
              this->workspace_->sf_exchange_im()[recv_dev].d_array().data() +
              recv_offset),
          plane_entry_count, ncclDouble, static_cast<int>(send_partition),
          comms[recv_partition], this->comm_streams_[recv_dev]));
    }
  }

  ncclCheck(ncclGroupEnd());

  for (std::size_t rank = 0; rank < devs.size(); rank++) {
    const int dev = devs[rank];

    cudaCheck(cudaSetDevice(dev));

    cudaCheck(
        cudaEventRecord(this->comm_events_[dev], this->comm_streams_[dev]));

    cudaCheck(cudaStreamWaitEvent(this->comp_streams_[dev],
                                  this->comm_events_[dev], 0));
  }

  return;
}

void glst_force::sum_rmt_sf_prefix_tile(const unsigned int tile) {
  constexpr std::string_view function_name =
      "glst_force::sum_rmt_sf_prefix_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  utl::require(this->sf_exchange_mode_ ==
                   GLST_SF_EXCHANGE_MODE::DISTRIBUTED_PREFIX,
               function_name, "Distributed prefix exchange is not selected");

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);
  const unsigned int group = this->plan_->tile_group(tile);
  const unsigned int nc = this->plan_->tile_node_count(tile);
  const unsigned int off = this->plan_->tile_node_point(tile);
  const unsigned int inner = this->plan_->grp_r_in()[0][group];
  const unsigned int outer = this->plan_->grp_r_out()[0][group];
  const unsigned int nx = this->plan_->ncell_x();
  const unsigned int ny = this->plan_->ncell_y();
  const unsigned int nz = this->plan_->ncell_z();
  const std::size_t slot_group_offset =
      static_cast<std::size_t>(group) * static_cast<std::size_t>(nx);
  const std::size_t required_slot_count =
      slot_group_offset + static_cast<std::size_t>(nx);

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    if (this->dev_tile_partition_[dev] != tile_partition)
      continue;

    const unsigned int cell_partition = this->dev_cell_partition_[dev];
    const unsigned int local_x_point =
        this->plan_->cell_partition_x_point()[cell_partition];
    const unsigned int local_x_count =
        this->plan_->cell_partition_x_count()[cell_partition];
    const unsigned int local_cell_count =
        this->plan_->local_cell_count(cell_partition);

    if (local_cell_count == 0)
      continue;

    const unsigned int first_global_cell =
        this->plan_->first_global_cell(cell_partition);
    const std::size_t rmt_entry_count =
        static_cast<std::size_t>(local_cell_count) *
        static_cast<std::size_t>(nc);

    utl::require(
        rmt_entry_count <= this->workspace_->rmt_tile_buffer_capacity(dev),
        function_name, "Distributed rmt_sum exceeds workspace capacity");

    utl::require(required_slot_count <=
                     this->workspace_->prefix_plane_slot_capacity(dev),
                 function_name, "Prefix-plane slot-map capacity is too small");

    cudaCheck(cudaSetDevice(dev));

    const unsigned int *remote_slot_by_x =
        this->workspace_->prefix_plane_slot()[dev].d_array().data() +
        slot_group_offset;

#ifdef __GLST_DEBUG__
    constexpr dim3 num_threads(256, 1, 1);
#else
    constexpr dim3 num_threads(512, 1, 1);
#endif

    const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                          std::min(65535u, local_cell_count), 1);

    calc_rmt_sum_distributed_kernel<<<num_blocks, num_threads, 0,
                                      this->comp_streams_[dev]>>>(
        this->workspace_->rmt_sum_re()[dev].d_array().data(),
        this->workspace_->rmt_sum_im()[dev].d_array().data(),
        this->workspace_->sf_re()[dev].d_array().data(),
        this->workspace_->sf_im()[dev].d_array().data(),
        this->workspace_->sf_exchange_re()[dev].d_array().data(),
        this->workspace_->sf_exchange_im()[dev].d_array().data(),
        remote_slot_by_x, this->plan_->w()[dev].d_array().data() + off, nc,
        inner, outer, local_x_point, local_x_count, nx, ny, nz,
        local_cell_count, first_global_cell);
    cudaCheck(cudaGetLastError());
  }

  return;
}

void glst_force::build_rmt_sum_tile(const unsigned int tile) {
  constexpr std::string_view function_name = "glst_force::build_rmt_sum_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);

  if ((this->sf_exchange_mode_ ==
       GLST_SF_EXCHANGE_MODE::FULL_GLOBAL_ALLREDUCE) ||
      (this->cell_partition_count_ == 1)) {
    if (!this->profiling_enabled_) {
      this->exchange_sf_tile(tile);
      this->sum_rmt_sf_tile(tile);
      return;
    }

    std::chrono::steady_clock::time_point phase_start =
        std::chrono::steady_clock::now();

    this->exchange_sf_tile(tile);
    this->synchronize_tile_partition_compute_streams(tile_partition);

    std::chrono::steady_clock::time_point phase_end =
        std::chrono::steady_clock::now();

    this->profile_.exchange_sf_ms += profile_elapsed_ms(phase_start, phase_end);

    if (this->cell_partition_count_ > 1) {
      const std::size_t entry_count =
          static_cast<std::size_t>(this->plan_->ncell()) *
          static_cast<std::size_t>(this->plan_->tile_node_count(tile));

      this->profile_.sf_collective_input_bytes +=
          2 * entry_count * sizeof(double) *
          static_cast<std::size_t>(this->cell_partition_count_);
    }

    phase_start = std::chrono::steady_clock::now();

    this->sum_rmt_sf_tile(tile);
    this->synchronize_tile_partition_compute_streams(tile_partition);

    phase_end = std::chrono::steady_clock::now();

    this->profile_.sum_rmt_sf_ms += profile_elapsed_ms(phase_start, phase_end);

    return;
  }

  if (this->sf_exchange_mode_ == GLST_SF_EXCHANGE_MODE::DISTRIBUTED_PREFIX) {
    if (!this->profiling_enabled_) {
      this->build_distributed_prefix_tile(tile);
      this->sum_rmt_sf_prefix_tile(tile);
      return;
    }

    std::chrono::steady_clock::time_point phase_start =
        std::chrono::steady_clock::now();
    this->build_distributed_prefix_tile(tile);
    this->synchronize_tile_partition_compute_streams(tile_partition);
    std::chrono::steady_clock::time_point phase_end =
        std::chrono::steady_clock::now();
    this->profile_.exchange_sf_ms += profile_elapsed_ms(phase_start, phase_end);

    const unsigned int group = this->plan_->tile_group(tile);
    const std::size_t plane_entry_count =
        static_cast<std::size_t>(this->plan_->ncell_y()) *
        static_cast<std::size_t>(this->plan_->ncell_z()) *
        static_cast<std::size_t>(this->plan_->tile_node_count(tile));
    const std::size_t prefix_base_input_bytes =
        2 * static_cast<std::size_t>(this->cell_partition_count_) *
        plane_entry_count * sizeof(double);

    std::size_t prefix_plane_import_count = 0;

    for (unsigned int cell_partition = 0;
         cell_partition < this->cell_partition_count_; cell_partition++) {
      const std::vector<unsigned int> &prefix_x =
          this->plan_->partition_group_prefix_x(cell_partition, group);

      utl::require(prefix_x.size() <= std::numeric_limits<std::size_t>::max() -
                                          prefix_plane_import_count,
                   function_name, "Profiled prefix-plane count overflow");

      prefix_plane_import_count += prefix_x.size();
    }

    const std::size_t prefix_plane_input_bytes =
        2 * prefix_plane_import_count * plane_entry_count * sizeof(double);

    this->profile_.prefix_base_input_bytes += prefix_base_input_bytes;
    this->profile_.prefix_plane_input_bytes += prefix_plane_input_bytes;
    this->profile_.prefix_plane_import_count += prefix_plane_import_count;
    this->profile_.sf_collective_input_bytes +=
        prefix_base_input_bytes + prefix_plane_input_bytes;

    phase_start = std::chrono::steady_clock::now();
    this->sum_rmt_sf_prefix_tile(tile);
    this->synchronize_tile_partition_compute_streams(tile_partition);
    phase_end = std::chrono::steady_clock::now();
    this->profile_.sum_rmt_sf_ms += profile_elapsed_ms(phase_start, phase_end);

    return;
  }

  utl::require(this->sf_exchange_chunk_x_count_ > 0, function_name,
               "Local exchange chunk size is 0");

  if (!this->profiling_enabled_)
    this->zero_rmt_sum_tile(tile);
  else {
    const std::chrono::steady_clock::time_point phase_start =
        std::chrono::steady_clock::now();
    this->zero_rmt_sum_tile(tile);
    this->synchronize_tile_partition_compute_streams(tile_partition);
    const std::chrono::steady_clock::time_point phase_end =
        std::chrono::steady_clock::now();
    this->profile_.sum_rmt_sf_ms += profile_elapsed_ms(phase_start, phase_end);
  }

  for (unsigned int source_partition = 0;
       source_partition < this->cell_partition_count_; source_partition++) {
    const unsigned int source_x_count =
        this->plan_->cell_partition_x_count()[source_partition];

    for (unsigned int source_x_offset = 0; source_x_offset < source_x_count;
         source_x_offset += this->sf_exchange_chunk_x_count_) {
      const unsigned int remaining = source_x_count - source_x_offset;

      const unsigned int chunk_x_count =
          std::min(this->sf_exchange_chunk_x_count_, remaining);

      if (!this->profiling_enabled_) {
        this->exchange_sf_chunk_tile(tile, source_partition, source_x_offset,
                                     chunk_x_count);

        this->sum_rmt_sf_chunk_tile(tile, source_partition, source_x_offset,
                                    chunk_x_count);

        continue;
      }

      std::chrono::steady_clock::time_point phase_start =
          std::chrono::steady_clock::now();
      this->exchange_sf_chunk_tile(tile, source_partition, source_x_offset,
                                   chunk_x_count);
      this->synchronize_tile_partition_compute_streams(tile_partition);
      std::chrono::steady_clock::time_point phase_end =
          std::chrono::steady_clock::now();
      this->profile_.exchange_sf_ms +=
          profile_elapsed_ms(phase_start, phase_end);

      const std::size_t chunk_cell_count =
          static_cast<std::size_t>(chunk_x_count) *
          static_cast<std::size_t>(this->plan_->ncell_y()) *
          static_cast<std::size_t>(this->plan_->ncell_z());

      const std::size_t chunk_entry_count =
          chunk_cell_count *
          static_cast<std::size_t>(this->plan_->tile_node_count(tile));

      // One real and one imaginary source payload. This deliberately does not
      // multiply by G_cell; The receivers can be reported separately.
      this->profile_.sf_collective_input_bytes +=
          2 * chunk_entry_count * sizeof(double);

      phase_start = std::chrono::steady_clock::now();
      this->sum_rmt_sf_chunk_tile(tile, source_partition, source_x_offset,
                                  chunk_x_count);
      this->synchronize_tile_partition_compute_streams(tile_partition);
      phase_end = std::chrono::steady_clock::now();
      this->profile_.sum_rmt_sf_ms +=
          profile_elapsed_ms(phase_start, phase_end);
    }
  }

  return;
}

template <unsigned int BLOCK>
__global__ static void
calc_lr_ef_kernel(double *__restrict__ fx, double *__restrict__ fy,
                  double *__restrict__ fz, double *__restrict__ en,
                  const double *__restrict__ rx, const double *__restrict__ ry,
                  const double *__restrict__ rz, const double *__restrict__ qc,
                  const unsigned int *__restrict__ local_cell_atom_points,
                  const unsigned int *__restrict__ local_cell_atom_counts,
                  const double *__restrict__ cx, const double *__restrict__ cy,
                  const double *__restrict__ cz,
                  const double *__restrict__ rmt_sum_re,
                  const double *__restrict__ rmt_sum_im, const unsigned int nc,
                  const unsigned int local_cell_count) {
  __shared__ double s_cache[BLOCK * 5];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int local_cell = blockIdx.y; local_cell < local_cell_count;
       local_cell += gridDim.y) {
    const unsigned int apnt = local_cell_atom_points[local_cell];
    const unsigned int acnt = local_cell_atom_counts[local_cell];
    const bool active = (idx < acnt);

    const std::size_t rmt_point =
        static_cast<std::size_t>(local_cell) * static_cast<std::size_t>(nc);

    double xa = 0.0, ya = 0.0, za = 0.0, qa = 0.0;
    if (active) {
      xa = rx[apnt + idx];
      ya = ry[apnt + idx];
      za = rz[apnt + idx];
      qa = qc[apnt + idx];
    }

    double fx0 = 0.0, fy0 = 0.0, fz0 = 0.0, en0 = 0.0;
    for (unsigned int i = 0; i < nc; i += BLOCK) {
      __syncthreads();
      if (i + threadIdx.x < nc) {
        const unsigned int local_node = i + threadIdx.x;

        s_cache[threadIdx.x * 5 + 0] = cx[local_node];
        s_cache[threadIdx.x * 5 + 1] = cy[local_node];
        s_cache[threadIdx.x * 5 + 2] = cz[local_node];
        s_cache[threadIdx.x * 5 + 3] = rmt_sum_re[rmt_point + local_node];
        s_cache[threadIdx.x * 5 + 4] = rmt_sum_im[rmt_point + local_node];
      }
      __syncthreads();

      if (active) { // Only "active" threads do expensive sincos work
        const unsigned int n = min(BLOCK, nc - i);

        for (unsigned int j = 0; j < n; j++) {
          const double xc = s_cache[j * 5 + 0];
          const double yc = s_cache[j * 5 + 1];
          const double zc = s_cache[j * 5 + 2];
          const double rmt_re = s_cache[j * 5 + 3];
          const double rmt_im = s_cache[j * 5 + 4];

          const double theta = xc * xa + yc * ya + zc * za;
          double re = 0.0, im = 0.0;
          sincos(theta, &im, &re);

          const double dre = qa * (re * rmt_re - im * rmt_im);
          const double dim = qa * (re * rmt_im + im * rmt_re);
          fx0 += dim * xc;
          fy0 += dim * yc;
          fz0 += dim * zc;
          en0 += dre;
        }
      }
    }

    if (active) {
      fx[apnt + idx] += fx0;
      fy[apnt + idx] += fy0;
      fz[apnt + idx] += fz0;
      en[apnt + idx] += en0;
    }
  }

  return;
}

void glst_force::calc_lr_ef_tile(const unsigned int tile) {
  constexpr std::string_view function_name = "glst_force::calc_lr_ef_tile";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  utl::require(tile < this->plan_->tile_count(), function_name,
               "Tile is out of bounds");

  const unsigned int tile_node_point = this->plan_->tile_node_point(tile);
  const unsigned int tile_node_count = this->plan_->tile_node_count(tile);

  utl::require(tile_node_count > 0, function_name, "Tile node count is 0");

  utl::require(tile_node_count <= this->plan_->max_tile_nodes(), function_name,
               "Tile exceeds buffer size");

  const unsigned int tile_partition = this->plan_->tile_partition_idx(tile);

  const unsigned int nc = tile_node_count;
  const unsigned int off = tile_node_point;

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    if (this->dev_tile_partition_[dev] != tile_partition)
      continue;

    cudaCheck(cudaSetDevice(dev));

    const unsigned int cell_partition = this->dev_cell_partition_[dev];

    const unsigned int local_cell_count =
        this->plan_->local_cell_count(cell_partition);

    utl::require(static_cast<std::size_t>(local_cell_count) ==
                     this->workspace_->cell_capacity(dev),
                 function_name,
                 "Local ell count does not match workspace capacity");

    const std::size_t rmt_entry_count =
        static_cast<std::size_t>(local_cell_count) *
        static_cast<std::size_t>(nc);

    utl::require(rmt_entry_count <=
                     this->workspace_->rmt_tile_buffer_capacity(dev),
                 function_name, "Active rmt tile exceeds workspace capacity");

    if (local_cell_count == 0)
      continue;

    const unsigned int max_atoms_cell = this->workspace_->max_atoms_cell()[dev];

    if (max_atoms_cell == 0)
      continue;

    const unsigned int *cell_atom_point = nullptr;
    const unsigned int *cell_atom_count = nullptr;

    if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
      cell_atom_point =
          this->workspace_->cell_atom_point()[dev].d_array().data();
      cell_atom_count =
          this->workspace_->cell_atom_count()[dev].d_array().data();
    } else {
      utl::require(static_cast<std::size_t>(local_cell_count) <=
                       this->workspace_->sr_source_cell_capacity(dev),
                   function_name,
                   "Owned-cell count exceeds source-cell metadata capacity");
      cell_atom_point =
          this->workspace_->sr_source_cell_atom_point()[dev].data();
      cell_atom_count =
          this->workspace_->sr_source_cell_atom_count()[dev].data();
    }

    constexpr dim3 num_threads(64, 1, 1);
    const dim3 num_blocks((max_atoms_cell + num_threads.x - 1) / num_threads.x,
                          std::min(65535u, local_cell_count), 1);

    calc_lr_ef_kernel<num_threads.x>
        <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
            this->workspace_->fx()[dev].d_array().data(),
            this->workspace_->fy()[dev].d_array().data(),
            this->workspace_->fz()[dev].d_array().data(),
            this->workspace_->en()[dev].d_array().data(),
            this->workspace_->rx()[dev].d_array().data(),
            this->workspace_->ry()[dev].d_array().data(),
            this->workspace_->rz()[dev].d_array().data(),
            this->workspace_->qc()[dev].d_array().data(), cell_atom_point,
            cell_atom_count, this->plan_->x()[dev].d_array().data() + off,
            this->plan_->y()[dev].d_array().data() + off,
            this->plan_->z()[dev].d_array().data() + off,
            this->workspace_->rmt_sum_re()[dev].d_array().data(),
            this->workspace_->rmt_sum_im()[dev].d_array().data(), nc,
            local_cell_count);

    cudaCheck(cudaGetLastError());
  }

  return;
}

void glst_force::reduce_tile_partition_ef(void) {
  constexpr std::string_view function_name =
      "glst_force::reduce_tile_partition_ef";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  utl::require(this->workspace_ != nullptr, function_name,
               "Workspace is not initialized");

  // This covers single-GPU mode and pure cell decomposition.
  if (this->tile_partition_count_ == 1)
    return;

  utl::require(this->tile_comm_devs_.size() ==
                   static_cast<std::size_t>(this->cell_partition_count_),
               function_name,
               "Tile communicator device topology does not match cell "
               "partition count");

  utl::require(
      this->tile_comms_.size() ==
          static_cast<std::size_t>(this->cell_partition_count_),
      function_name,
      "Tile communicator topology does not match cell partition count");

  for (unsigned int cell_partition = 0;
       cell_partition < this->cell_partition_count_; cell_partition++) {
    const std::vector<int> &devs = this->tile_comm_devs_[cell_partition];
    const std::vector<ncclComm_t> &comms = this->tile_comms_[cell_partition];

    utl::require(
        devs.size() == static_cast<std::size_t>(this->tile_partition_count_),
        function_name,
        "Tile communicator device count does not match tile partition count");

    utl::require(comms.size() == devs.size(), function_name,
                 "Tile communicator handle count does not match device count");

    // Validate every rank before beginning the collective. This is important:
    // Every rank in a NCCL collective must use the same count and datatype.
    std::size_t owned_atom_count = 0;

    for (std::size_t rank = 0; rank < devs.size(); rank++) {
      const int dev = devs[rank];

      utl::require((dev >= 0) && (dev < this->cuda_count_), function_name,
                   "Tile communicator device is out of range");

      utl::require(this->dev_cell_partition_[dev] == cell_partition,
                   function_name, "Device belongs to the wrong cell partition");

      utl::require(this->dev_tile_partition_[dev] ==
                       static_cast<unsigned int>(rank),
                   function_name,
                   "Tile communicator rank does not match tile partition");

      const std::size_t dev_owned_atom_count =
          this->workspace_->owned_atom_count(dev);

      if (rank == 0)
        owned_atom_count = dev_owned_atom_count;
      else {
        utl::require(dev_owned_atom_count == owned_atom_count, function_name,
                     "Tile ranks in one cell partition have different owned "
                     "atom counts");
      }

      utl::require(dev_owned_atom_count <= this->workspace_->atom_capacity(dev),
                   function_name,
                   "Owned atom count exceeds workspace atom capacity");
    }

    if (owned_atom_count == 0)
      continue;

    // Ensure that the communication streams see all preceding long-range work
    // and the root-only short-range work.
    for (std::size_t rank = 0; rank < devs.size(); rank++) {
      const int dev = devs[rank];

      cudaCheck(cudaSetDevice(dev));
      cudaCheck(
          cudaEventRecord(this->comp_events_[dev], this->comp_streams_[dev]));
      cudaCheck(cudaStreamWaitEvent(this->comm_streams_[dev],
                                    this->comp_events_[dev], 0));
    }

    // Root rank is communicator rank 0, which is tile partition 0. Use four
    // root reductions rather than four all-reduces because only tile
    // partition 0 needs complete local results.
    ncclCheck(ncclGroupStart());

    for (std::size_t rank = 0; rank < devs.size(); rank++) {
      const int dev = devs[rank];

      cudaCheck(cudaSetDevice(dev));

      double *fx = this->workspace_->fx()[dev].d_array().data();
      double *fy = this->workspace_->fy()[dev].d_array().data();
      double *fz = this->workspace_->fz()[dev].d_array().data();
      double *en = this->workspace_->en()[dev].d_array().data();

      ncclCheck(ncclReduce(static_cast<const void *>(fx),
                           static_cast<void *>(fx), owned_atom_count,
                           ncclDouble, ncclSum, 0, comms[rank],
                           this->comm_streams_[dev]));

      ncclCheck(ncclReduce(static_cast<const void *>(fy),
                           static_cast<void *>(fy), owned_atom_count,
                           ncclDouble, ncclSum, 0, comms[rank],
                           this->comm_streams_[dev]));

      ncclCheck(ncclReduce(static_cast<const void *>(fz),
                           static_cast<void *>(fz), owned_atom_count,
                           ncclDouble, ncclSum, 0, comms[rank],
                           this->comm_streams_[dev]));

      ncclCheck(ncclReduce(static_cast<const void *>(en),
                           static_cast<void *>(en), owned_atom_count,
                           ncclDouble, ncclSum, 0, comms[rank],
                           this->comm_streams_[dev]));
    }

    ncclCheck(ncclGroupEnd());

    // Prevent subsequent compute work, zeroing, or result gathering from
    // touching these arrays before the reduction has completed.
    for (std::size_t rank = 0; rank < devs.size(); rank++) {
      const int dev = devs[rank];

      cudaCheck(cudaSetDevice(dev));
      cudaCheck(
          cudaEventRecord(this->comm_events_[dev], this->comm_streams_[dev]));
      cudaCheck(cudaStreamWaitEvent(this->comp_streams_[dev],
                                    this->comm_events_[dev], 0));
    }
  }

  return;
}

void glst_force::zero_ef(void) {
  constexpr std::string_view function_name = "glst_force::zero_ef";

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    const std::size_t atom_capacity = this->workspace_->atom_capacity(dev);
    const std::size_t source_atom_count =
        this->workspace_->source_atom_count(dev);
    const std::size_t owned_atom_count =
        this->workspace_->owned_atom_count(dev);

    utl::require(owned_atom_count <= source_atom_count, function_name,
                 "Owned atom count exceeds source atom count");

    utl::require(source_atom_count <= atom_capacity, function_name,
                 "Source atom count exceeds atom capacity");

    const std::size_t nbytes = owned_atom_count * sizeof(double);
    if (nbytes == 0)
      continue;

    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(this->workspace_->fx()[dev].d_array().data()), 0,
        nbytes, this->comp_streams_[dev]));
    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(this->workspace_->fy()[dev].d_array().data()), 0,
        nbytes, this->comp_streams_[dev]));
    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(this->workspace_->fz()[dev].d_array().data()), 0,
        nbytes, this->comp_streams_[dev]));
    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(this->workspace_->en()[dev].d_array().data()), 0,
        nbytes, this->comp_streams_[dev]));
  }

  return;
}

void glst_force::reset_profile(void) {
  this->profile_ = glst_profile();
  return;
}

void glst_force::synchronize_compute_streams(void) const {
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamSynchronize(this->comp_streams_[dev]));
  }
  return;
}

void glst_force::synchronize_tile_partition_compute_streams(
    const unsigned int tile_partition) const {
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    if (this->dev_tile_partition_[dev] != tile_partition)
      continue;
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamSynchronize(this->comp_streams_[dev]));
  }
  return;
}

void glst_force::deallocate(void) {
  if (!this->cuda_initialized_)
    return;

  if (this->execution_mode_ != GLST_EXECUTION_MODE::SINGLE_GPU_TILED)
    this->destroy_nccl_topology();

  // Destroy CUDA streams and events
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamDestroy(this->comp_streams_[dev]));

    if (this->execution_mode_ != GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
      cudaCheck(cudaStreamDestroy(this->comm_streams_[dev]));
      cudaCheck(cudaEventDestroy(this->comp_events_[dev]));
      cudaCheck(cudaEventDestroy(this->comm_events_[dev]));
    }
  }

  // if (this->execution_mode_ != GLST_EXECUTION_MODE::SINGLE_GPU_TILED)
  //   disable_p2p(this->cuda_count_);

  this->comp_streams_.clear();
  this->comm_streams_.clear();
  this->comp_events_.clear();
  this->comm_events_.clear();
  this->nccl_devs_.clear();
  this->nccl_comms_.clear();

  this->cuda_initialized_ = false;

  return;
}

void glst_force::init_cuda_resources(void) {
  if (this->cuda_initialized_)
    return;

  int device_count = 0;
  cudaCheck(cudaGetDeviceCount(&device_count));

  this->init_gpu_layout(device_count);

  if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
    this->comp_streams_.resize(1);
    this->comm_streams_.clear();
    this->comp_events_.clear();
    this->comm_events_.clear();
    this->nccl_devs_.clear();
    this->nccl_comms_.clear();
    this->cell_comm_devs_.clear();
    this->tile_comm_devs_.clear();
    this->cell_comms_.clear();
    this->tile_comms_.clear();

    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaStreamCreate(&this->comp_streams_[0]));

    this->cuda_initialized_ = true;

    return;
  }

  // NCCL enables peer access as needed when communicator transports connect.
  // Pre-enabling every device pair here causes NCCL to encounter
  // cudaErrorPeerAccessAlreadyEnabled during lazy connection setup, which
  // compute-sanitizer reports as an API error even though the collective
  // completes successfully.
  // enable_p2p(this->cuda_count_);

  // Initialize streams and events
  this->comp_streams_.resize(this->cuda_count_);
  this->comm_streams_.resize(this->cuda_count_);
  this->comp_events_.resize(this->cuda_count_);
  this->comm_events_.resize(this->cuda_count_);
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamCreate(&this->comp_streams_[dev]));
    cudaCheck(cudaStreamCreate(&this->comm_streams_[dev]));
    cudaCheck(cudaEventCreateWithFlags(&this->comp_events_[dev],
                                       cudaEventDisableTiming));
    cudaCheck(cudaEventCreateWithFlags(&this->comm_events_[dev],
                                       cudaEventDisableTiming));
  }

  this->init_nccl_topology();

  this->cuda_initialized_ = true;

  return;
}

void glst_force::init_nccl_topology(void) {
  constexpr std::string_view function_name = "glst_force::init_nccl_topology";

  this->nccl_devs_.clear();
  this->nccl_comms_.clear();
  this->cell_comm_devs_.clear();
  this->tile_comm_devs_.clear();
  this->cell_comms_.clear();
  this->tile_comms_.clear();

  if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED)
    return;

  this->nccl_devs_.resize(this->cuda_count_);
  for (int dev = 0; dev < this->cuda_count_; dev++)
    this->nccl_devs_[dev] = dev;

  this->cell_comm_devs_.resize(this->tile_partition_count_);
  this->cell_comms_.resize(this->tile_partition_count_);

  for (unsigned int tile_part = 0; tile_part < this->tile_partition_count_;
       tile_part++) {
    std::vector<int> &devs = this->cell_comm_devs_[tile_part];
    std::vector<ncclComm_t> &comms = this->cell_comms_[tile_part];

    devs.resize(this->cell_partition_count_);
    comms.resize(this->cell_partition_count_);

    for (unsigned int cell_part = 0; cell_part < this->cell_partition_count_;
         cell_part++) {
      const unsigned int udev =
          cell_part * this->tile_partition_count_ + tile_part;

      utl::require(udev < static_cast<unsigned int>(this->cuda_count_),
                   function_name, "Cell communicator device is out of range");

      utl::require((this->dev_cell_partition_[udev] == cell_part) &&
                       (this->dev_tile_partition_[udev] == tile_part),
                   function_name, "Cell communicator layout mismatch");

      devs[cell_part] = static_cast<int>(udev);
    }

    ncclCheck(ncclCommInitAll(comms.data(),
                              static_cast<int>(this->cell_partition_count_),
                              devs.data()));
  }

  this->tile_comm_devs_.resize(this->cell_partition_count_);
  this->tile_comms_.resize(this->cell_partition_count_);

  for (unsigned int cell_part = 0; cell_part < this->cell_partition_count_;
       cell_part++) {
    std::vector<int> &devs = this->tile_comm_devs_[cell_part];
    std::vector<ncclComm_t> &comms = this->tile_comms_[cell_part];

    devs.resize(this->tile_partition_count_);
    comms.resize(this->tile_partition_count_);

    for (unsigned int tile_part = 0; tile_part < this->tile_partition_count_;
         tile_part++) {
      const unsigned int udev =
          cell_part * this->tile_partition_count_ + tile_part;

      utl::require(udev < static_cast<unsigned int>(this->cuda_count_),
                   function_name, "Tile communicator device is out of range");

      utl::require((this->dev_cell_partition_[udev] == cell_part) &&
                       (this->dev_tile_partition_[udev] == tile_part),
                   function_name, "Tile communicator layout mismatch");

      devs[tile_part] = static_cast<int>(udev);
    }

    ncclCheck(ncclCommInitAll(comms.data(),
                              static_cast<int>(this->tile_partition_count_),
                              devs.data()));
  }

  // Keep the old global communicator only for the existing MULTI_GPU_CELL
  // fallback. Do not initialize it for MULTI_GPU_TILE or MULTI_GPU_CELL_TILE.
  if (this->execution_mode_ == GLST_EXECUTION_MODE::MULTI_GPU_CELL) {
    this->nccl_comms_.resize(this->cuda_count_);
    ncclCheck(ncclCommInitAll(this->nccl_comms_.data(), this->cuda_count_,
                              this->nccl_devs_.data()));
  }

  return;
}

void glst_force::destroy_nccl_topology(void) {
  for (std::size_t tile_part = 0; tile_part < this->cell_comms_.size();
       tile_part++) {
    for (std::size_t rank = 0; rank < this->cell_comms_[tile_part].size();
         rank++) {
      const int dev = this->cell_comm_devs_[tile_part][rank];
      cudaCheck(cudaSetDevice(dev));
      ncclCheck(ncclCommDestroy(this->cell_comms_[tile_part][rank]));
    }
  }

  for (std::size_t cell_part = 0; cell_part < this->tile_comms_.size();
       cell_part++) {
    for (std::size_t rank = 0; rank < this->tile_comms_[cell_part].size();
         rank++) {
      const int dev = this->tile_comm_devs_[cell_part][rank];
      cudaCheck(cudaSetDevice(dev));
      ncclCheck(ncclCommDestroy(this->tile_comms_[cell_part][rank]));
    }
  }

  for (std::size_t i = 0; i < this->nccl_comms_.size(); i++) {
    const int dev = this->nccl_devs_[i];
    cudaCheck(cudaSetDevice(dev));
    ncclCheck(ncclCommDestroy(this->nccl_comms_[i]));
  }

  this->cell_comm_devs_.clear();
  this->tile_comm_devs_.clear();
  this->cell_comms_.clear();
  this->tile_comms_.clear();
  this->nccl_devs_.clear();
  this->nccl_comms_.clear();

  return;
}

void glst_force::print_nccl_topology(std::ostream &os) const {
  if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
    os << "  NCCL topology: disabled for single-GPU tiled mode" << std::endl;
    return;
  }

  os << "  NCCL cell communicators:" << std::endl;
  for (std::size_t tile_part = 0; tile_part < this->cell_comm_devs_.size();
       tile_part++) {
    os << "         tile partition " << tile_part << ": devices ";
    for (std::size_t rank = 0; rank < this->cell_comm_devs_[tile_part].size();
         rank++) {
      if (rank > 0)
        os << ", ";
      os << this->cell_comm_devs_[tile_part][rank];
    }
    os << std::endl;
  }

  os << "  NCCL tile communicators:" << std::endl;
  for (std::size_t cell_part = 0; cell_part < this->tile_comm_devs_.size();
       cell_part++) {
    os << "         cell partition " << cell_part << ": devices ";
    for (std::size_t rank = 0; rank < this->tile_comm_devs_[cell_part].size();
         rank++) {
      if (rank > 0)
        os << ", ";
      os << this->tile_comm_devs_[cell_part][rank];
    }
    os << std::endl;
  }

  os << "  NCCL global fallback communicator: "
     << (this->nccl_comms_.empty() ? "disabled" : "enabled") << std::endl;

  return;
}

void glst_force::select_automatic_gpu_layout(
    unsigned int &cell_partition_count, unsigned int &tile_partition_count,
    std::size_t &estimated_peak_bytes, std::string &reason,
    const glst_plan &plan, const int device_count,
    const GLST_SF_EXCHANGE_MODE sf_exchange_mode,
    const std::vector<std::size_t> &free_memory_bytes) {
  constexpr std::string_view function_name =
      "glst_force::select_automatic_gpu_layout";

  // Atom positions are normally close to uniformly distributed for the target
  // molecular systems, but halos and local density fluctuations require
  // additional capacity beyond the uniform-density estimate.
  constexpr double source_atom_headroom = 1.25;

  // Above this atom count per cell partition, prefer another spatial partition
  // even when the allocation would technically fit.
  constexpr std::size_t target_atoms_per_cell_partition = 4 * 1024 * 1024;

  // Require four max-sized tiles' worth of actual cubature-node work on every
  // tile partition before preferring tile decomposition. Tile count alone is
  // insufficient because tiles cannot cross alpha-group boundaries and may be
  // only partially filled.
  const unsigned int minimum_tile_node_factor = 4;

  utl::require(device_count >= 1, function_name,
               "device_count must be positive");

  utl::require(free_memory_bytes.size() ==
                   static_cast<std::size_t>(device_count),
               function_name, "free-memory count does not match device_count");

  utl::require(plan.natom() > 0, function_name, "natom is 0");

  utl::require((plan.ncell_x() > 0) && (plan.ncell_y() > 0) &&
                   (plan.ncell_z() > 0),
               function_name, "cell dimensions are invalid");

  utl::require(plan.max_tile_nodes() > 0, function_name, "max_tile_nodes is 0");

  utl::require(plan.tile_count() > 0, function_name, "tile_count is 0");

  struct layout_candidate {
    unsigned int cell_partitions = 1;
    unsigned int tile_partitions = 1;
    std::size_t minimum_tile_partition_node_count = 0;
    std::size_t estimated_peak_bytes = 0;
    bool fits_memory_budget = false;
  };

  const std::size_t natom = static_cast<std::size_t>(plan.natom());
  const std::size_t ncell_x = static_cast<std::size_t>(plan.ncell_x());
  const std::size_t ncell_y = static_cast<std::size_t>(plan.ncell_y());
  const std::size_t ncell_z = static_cast<std::size_t>(plan.ncell_z());

  const std::size_t yz_cell_count = ncell_y * ncell_z;
  const std::size_t ncell = ncell_x * yz_cell_count;

  const std::size_t max_tile_nodes =
      static_cast<std::size_t>(plan.max_tile_nodes());

  const std::size_t minimum_tile_nodes_per_tile_partition =
      minimum_tile_node_factor * max_tile_nodes;

  std::vector<layout_candidate> candidates;

  for (unsigned int candidate_cell_partitions = 1;
       candidate_cell_partitions <= static_cast<unsigned int>(device_count);
       candidate_cell_partitions++) {
    if (static_cast<unsigned int>(device_count) % candidate_cell_partitions !=
        0)
      continue;

    // Empty x-slab partitions do not improve either memory or performance.
    if (static_cast<std::size_t>(candidate_cell_partitions) > ncell_x)
      continue;

    const unsigned int candidate_tile_partitions =
        static_cast<unsigned int>(device_count) / candidate_cell_partitions;

    std::vector<std::size_t> tile_partition_node_count(
        candidate_tile_partitions, 0);

    for (unsigned int tile = 0; tile < plan.tile_count(); tile++) {
      const unsigned int tile_partition = tile % candidate_tile_partitions;

      tile_partition_node_count[tile_partition] +=
          static_cast<std::size_t>(plan.tile_node_count(tile));
    }

    const std::size_t minimum_tile_partition_node_count = *std::min_element(
        tile_partition_node_count.begin(), tile_partition_node_count.end());

    const std::size_t min_local_x_count =
        ncell_x / static_cast<std::size_t>(candidate_cell_partitions);

    const std::size_t max_local_x_count =
        (ncell_x + static_cast<std::size_t>(candidate_cell_partitions) - 1) /
        static_cast<std::size_t>(candidate_cell_partitions);

    unsigned int halo_x_count = 0;

    if (candidate_cell_partitions == 2)
      halo_x_count = 1;
    else if (candidate_cell_partitions > 2)
      halo_x_count = 2;

    const std::size_t max_source_x_count = std::min(
        ncell_x, max_local_x_count + static_cast<std::size_t>(halo_x_count));

    double estimated_source_atom_count = std::ceil(
        static_cast<double>(natom) * static_cast<double>(max_source_x_count) /
        static_cast<double>(ncell_x) * source_atom_headroom);

    if (estimated_source_atom_count > static_cast<double>(natom))
      estimated_source_atom_count = static_cast<double>(natom);

    const double max_local_cell_count = static_cast<double>(max_local_x_count) *
                                        static_cast<double>(yz_cell_count);

    const double max_source_cell_count =
        static_cast<double>(max_source_x_count) *
        static_cast<double>(yz_cell_count);

    const double local_tile_entry_count =
        max_local_cell_count * static_cast<double>(max_tile_nodes);

    const double prefix_plane_entry_count = static_cast<double>(yz_cell_count) *
                                            static_cast<double>(max_tile_nodes);

    double local_atom_bytes = 0.0;
    double local_metadata_bytes = 0.0;
    double local_cub_reserve_bytes = 0.0;

    if (device_count == 1) {
      // idx, sorted_idx, atom_cell_idx, atom_cell_sorted_idx, packets,
      // sorted_packets, rx/ry/rz/qc, and fx/fy/fz/en.
      local_atom_bytes =
          estimated_source_atom_count *
          static_cast<double>(4 * sizeof(unsigned int) +
                              2 * sizeof(atom_packet) + 8 * sizeof(double));

      local_metadata_bytes = static_cast<double>(ncell) *
                             static_cast<double>(2 * sizeof(unsigned int));

      // Conservative reserve for the larger of the single-GPU packet and value
      // radix-sort workspaces.
      local_cub_reserve_bytes =
          static_cast<double>(natom) *
          static_cast<double>(sizeof(atom_sort_key) + sizeof(atom_packet));
    } else {
      // source_packets, sorted_idx, rx/ry/rz/qc, and fx/fy/fz/en.
      local_atom_bytes =
          estimated_source_atom_count *
          static_cast<double>(sizeof(atom_packet) + sizeof(unsigned int) +
                              8 * sizeof(double));

      local_metadata_bytes =
          max_source_cell_count * static_cast<double>(2 * sizeof(unsigned int));

      // Local multi-GPU CUB use is primarily the source-cell prefix scan.
      local_cub_reserve_bytes =
          max_source_cell_count * static_cast<double>(sizeof(unsigned int));
    }

    double tile_buffer_bytes = 0.0;
    double prefix_slot_bytes = 0.0;

    if (sf_exchange_mode == GLST_SF_EXCHANGE_MODE::FULL_GLOBAL_ALLREDUCE) {
      const double global_tile_entry_count =
          static_cast<double>(ncell) * static_cast<double>(max_tile_nodes);

      // Complex global S_tile plus complex local rmt_sum.
      tile_buffer_bytes = 2.0 *
                          (global_tile_entry_count + local_tile_entry_count) *
                          static_cast<double>(sizeof(double));
    } else if (candidate_cell_partitions == 1) {
      // Complex local S_tile plus complex local rmt_sum. There is no
      // inter-cell-partition exchange buffer.
      tile_buffer_bytes =
          4.0 * local_tile_entry_count * static_cast<double>(sizeof(double));
    } else if (sf_exchange_mode ==
               GLST_SF_EXCHANGE_MODE::LOCAL_CHUNK_BROADCAST) {
      const std::size_t exchange_chunk_x_count = (max_local_x_count + 1) / 2;

      const double exchange_tile_entry_count =
          static_cast<double>(exchange_chunk_x_count) *
          prefix_plane_entry_count;

      tile_buffer_bytes = 2.0 *
                          (local_tile_entry_count + exchange_tile_entry_count +
                           local_tile_entry_count) *
                          static_cast<double>(sizeof(double));
    } else {
      // Each local x plane contributes at most four prefix-plane candidates for
      // one alpha group. Uneven x-slab partitions can make the smaller
      // partition require more imported planes than the larger partition.
      const std::size_t max_partition_imported_prefix_x_count =
          std::min(ncell_x - max_local_x_count, 4 * max_local_x_count);

      const std::size_t min_partition_imported_prefix_x_count =
          std::min(ncell_x - min_local_x_count, 4 * min_local_x_count);

      const std::size_t imported_prefix_x_count =
          std::max(max_partition_imported_prefix_x_count,
                   min_partition_imported_prefix_x_count);

      const double imported_prefix_entry_count =
          static_cast<double>(imported_prefix_x_count) *
          prefix_plane_entry_count;

      const double partition_total_entry_count =
          static_cast<double>(candidate_cell_partitions) *
          prefix_plane_entry_count;

      // Complex local S_tile, imported planes, partition totals, one prefix
      // base plane, and local rmt_sum.
      tile_buffer_bytes =
          2.0 *
          (local_tile_entry_count + imported_prefix_entry_count +
           partition_total_entry_count + prefix_plane_entry_count +
           local_tile_entry_count) *
          static_cast<double>(sizeof(double));

      prefix_slot_bytes = static_cast<double>(plan.ngroup()) *
                          static_cast<double>(ncell_x) *
                          static_cast<double>(sizeof(unsigned int));
    }

    double gpu_zero_only_bytes = 0.0;

    if (device_count > 1) {
      // GPU 0 owns two global sort-key arrays and two global packet arrays. Add
      // one key+packet per atom as a conservative radix-sort workspace reserve.
      gpu_zero_only_bytes = static_cast<double>(natom) *
                            static_cast<double>(3 * sizeof(atom_sort_key) +
                                                3 * sizeof(atom_packet));

      // global_cell_atom_count, global_cell_atom_point,
      // global_x_plane_atom_point, and global_max_atoms_cell.
      gpu_zero_only_bytes += static_cast<double>(2 * ncell + 1 + ncell_x + 1 +
                                                 candidate_cell_partitions) *
                             static_cast<double>(sizeof(unsigned int));
    }

    const double local_required_bytes =
        local_atom_bytes + local_metadata_bytes + local_cub_reserve_bytes +
        tile_buffer_bytes + prefix_slot_bytes;

    const double gpu_zero_required_bytes =
        local_required_bytes + gpu_zero_only_bytes;

    const double peak_required_bytes =
        std::max(local_required_bytes, gpu_zero_required_bytes);

    layout_candidate candidate;
    candidate.cell_partitions = candidate_cell_partitions;
    candidate.tile_partitions = candidate_tile_partitions;
    candidate.minimum_tile_partition_node_count =
        minimum_tile_partition_node_count;

    if (peak_required_bytes >=
        static_cast<double>(std::numeric_limits<std::size_t>::max())) {
      candidate.estimated_peak_bytes = std::numeric_limits<std::size_t>::max();
    } else {
      candidate.estimated_peak_bytes =
          static_cast<std::size_t>(std::ceil(peak_required_bytes));
    }

    candidate.fits_memory_budget = true;

    for (int dev = 0; dev < device_count; dev++) {
      const double required_bytes =
          (dev == 0) ? gpu_zero_required_bytes : local_required_bytes;

      const double budget_bytes = static_cast<double>(free_memory_bytes[dev]) *
                                  gpu_layout_memory_budget_fraction;

      if (required_bytes > budget_bytes) {
        candidate.fits_memory_budget = false;
        break;
      }
    }

    candidates.push_back(candidate);
  }

  utl::require(!candidates.empty(), function_name,
               "No valid G_cell divisor is <= ncell_x");

  unsigned int minimum_memory_cell_partitions = 0;

  for (std::size_t i = 0; i < candidates.size(); i++) {
    if (candidates[i].fits_memory_budget) {
      minimum_memory_cell_partitions = candidates[i].cell_partitions;
      break;
    }
  }

  // The estimates are deliberately conservative. When every factorization
  // exceeds the soft 80% budget, select the smallest estimated footprint and
  // allow the real workspace allocation to remain authoritative.
  if (minimum_memory_cell_partitions == 0) {
    const layout_candidate *fallback = &(candidates[0]);

    for (std::size_t i = 1; i < candidates.size(); i++) {
      if (candidates[i].estimated_peak_bytes < fallback->estimated_peak_bytes)
        fallback = &(candidates[i]);
    }

    cell_partition_count = fallback->cell_partitions;
    tile_partition_count = fallback->tile_partitions;
    estimated_peak_bytes = fallback->estimated_peak_bytes;
    reason = "no factorization satisfied the 80% free-memory budget; selected "
             "the minimum-estimated-memory fallback " +
             std::to_string(cell_partition_count) + " x " +
             std::to_string(tile_partition_count);

    return;
  }

  const std::size_t atom_partition_requirement =
      (natom + target_atoms_per_cell_partition - 1) /
      target_atoms_per_cell_partition;

  unsigned int minimum_cell_partitions = minimum_memory_cell_partitions;

  if (atom_partition_requirement >
      static_cast<std::size_t>(minimum_cell_partitions)) {
    minimum_cell_partitions = static_cast<unsigned int>(std::min(
        atom_partition_requirement, static_cast<std::size_t>(device_count)));
  }

  const layout_candidate *selected = nullptr;

  // First preference: Satisfy memory/atom requirements while leaving enough
  // actual cubature-node work on every tile partition. Candidates are ordered
  // by increasing G_cell, so this chooses the largest useful G_tile.
  for (std::size_t i = 0; i < candidates.size(); i++) {
    const layout_candidate &candidate = candidates[i];

    if ((!candidate.fits_memory_budget) ||
        (candidate.cell_partitions < minimum_cell_partitions)) {
      continue;
    }

    if (candidate.minimum_tile_partition_node_count >=
        minimum_tile_nodes_per_tile_partition) {
      selected = &candidate;
      break;
    }
  }

  // Second preference: If no candidate has enough cubature-node work on every
  // tile partition, maximize spatial decomposition. This avoids tile-reduction
  // overhead when the long-range cubature is too small to benefit from G_tile.
  if (selected == nullptr) {
    for (std::size_t i = candidates.size(); i > 0; i--) {
      const layout_candidate &candidate = candidates[i - 1];

      if ((!candidate.fits_memory_budget) ||
          (candidate.cell_partitions < minimum_cell_partitions)) {
        continue;
      }

      selected = &candidate;
      break;
    }
  }

  // ncell_x can prevent reaching the preferred atom-partition count. In that
  // case, use the largest memory-fitting spatial factor.
  if (selected == nullptr) {
    for (std::size_t i = 0; i < candidates.size(); i++) {
      if (candidates[i].fits_memory_budget)
        selected = &(candidates[i]);
    }
  }

  utl::require(selected != nullptr, function_name,
               "Could not select a memory-fitting GPU layout");

  cell_partition_count = selected->cell_partitions;
  tile_partition_count = selected->tile_partitions;
  estimated_peak_bytes = selected->estimated_peak_bytes;
  reason = "memory requires G_cell >= " +
           std::to_string(minimum_memory_cell_partitions) + ", the " +
           std::to_string(target_atoms_per_cell_partition) +
           "-atom target requires G_cell >= " +
           std::to_string(atom_partition_requirement) + "; selected " +
           std::to_string(cell_partition_count) + " x " +
           std::to_string(tile_partition_count) + " for " +
           std::to_string(plan.tile_count()) + " tiles and " +
           std::to_string(plan.tot_num_nodes()) +
           " cubature nodes; minimum tile-partition work is " +
           std::to_string(selected->minimum_tile_partition_node_count) +
           " nodes with a target of " +
           std::to_string(minimum_tile_nodes_per_tile_partition);

  return;
}

void glst_force::init_gpu_layout(const int device_count) {
  constexpr std::string_view function_name = "glst_force::init_gpu_layout";

  utl::require(device_count >= 1, function_name,
               "Could not find any CUDA capable devices");

  this->gpu_layout_min_free_bytes_ = 0;
  this->gpu_layout_memory_budget_bytes_ = 0;
  this->gpu_layout_estimated_peak_bytes_ = 0;
  this->gpu_layout_reason_.clear();

  unsigned int cell_partition_count = this->cell_partition_count_;
  unsigned int tile_partition_count = this->tile_partition_count_;

  if (!this->gpu_layout_user_set_) {
    utl::require(this->plan_ != nullptr, function_name,
                 "Automatic GPU layout requires initialized plan");
    std::vector<std::size_t> free_memory_bytes(device_count, 0);

    this->gpu_layout_min_free_bytes_ = std::numeric_limits<std::size_t>::max();

    for (int dev = 0; dev < device_count; dev++) {
      cudaCheck(cudaSetDevice(dev));

      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;

      cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));

      utl::require(total_bytes >= free_bytes, function_name,
                   "CUDA reported free memory larger than total memory");

      free_memory_bytes[dev] = free_bytes;

      if (free_bytes < this->gpu_layout_min_free_bytes_)
        this->gpu_layout_min_free_bytes_ = free_bytes;
    }

    this->gpu_layout_memory_budget_bytes_ = static_cast<std::size_t>(
        static_cast<double>(this->gpu_layout_min_free_bytes_) *
        gpu_layout_memory_budget_fraction);

    glst_force::select_automatic_gpu_layout(
        cell_partition_count, tile_partition_count,
        this->gpu_layout_estimated_peak_bytes_, this->gpu_layout_reason_,
        *(this->plan_), device_count, this->sf_exchange_mode_,
        free_memory_bytes);
  } else {
    this->gpu_layout_reason_ = "explicit set_gpu_layout_override";
  }

  const unsigned long long int product =
      static_cast<unsigned long long int>(cell_partition_count) *
      static_cast<unsigned long long int>(tile_partition_count);

  if (product != static_cast<unsigned long long int>(device_count)) {
    utl::throw_error(
        function_name,
        "G_cell * G_tile must equal the visible CUDA device count; observed " +
            std::to_string(cell_partition_count) + " * " +
            std::to_string(tile_partition_count) +
            " != " + std::to_string(device_count));
  }

  this->cuda_count_ = device_count;
  this->cell_partition_count_ = cell_partition_count;
  this->tile_partition_count_ = tile_partition_count;

  if (this->cuda_count_ == 1)
    this->execution_mode_ = GLST_EXECUTION_MODE::SINGLE_GPU_TILED;
  else if (this->tile_partition_count_ == 1)
    this->execution_mode_ = GLST_EXECUTION_MODE::MULTI_GPU_CELL;
  else if (this->cell_partition_count_ == 1)
    this->execution_mode_ = GLST_EXECUTION_MODE::MULTI_GPU_TILE;
  else
    this->execution_mode_ = GLST_EXECUTION_MODE::MULTI_GPU_CELL_TILE;

  this->dev_cell_partition_.resize(this->cuda_count_);
  this->dev_tile_partition_.resize(this->cuda_count_);

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    const unsigned int udev = static_cast<unsigned int>(dev);
    this->dev_cell_partition_[dev] = udev / this->tile_partition_count_;
    this->dev_tile_partition_[dev] = udev % this->tile_partition_count_;
  }

  return;
}

void glst_force::cells2dev(void) {
  constexpr std::string_view function_name = "glst_force::cells2dev";

  utl::require(this->plan_ != nullptr, function_name,
               "Plan is not initialized");

  this->dev_cell_idx_.resize(this->cuda_count_);

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    const unsigned int partition = this->dev_cell_partition_[dev];

    utl::require(partition < this->plan_->cell_partition_count(), function_name,
                 "Device cell partition is out of range");

    cudaCheck(cudaSetDevice(dev));
    this->dev_cell_idx_[dev] = this->plan_->partition_cell_idx(partition);
  }

  return;
}

void glst_force::require_single_gpu_runtime(
    const std::string_view method) const {
  if (this->execution_mode_ != GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
    utl::throw_error(
        "glst_force::" + std::string(method),
        "Multi-GPU local-workspace runtime is not implemented yet");
  }
  return;
}
