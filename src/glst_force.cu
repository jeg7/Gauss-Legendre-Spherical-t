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
#include "device_comm.hcu"
#include "reduce.hcu"

#include <algorithm>
#include <cstddef>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

glst_force::glst_force(void)
    : plan_(nullptr), workspace_(nullptr),
      execution_mode_(GLST_EXECUTION_MODE::SINGLE_GPU_TILED),
      cell_partition_count_(1), tile_partition_count_(1), dev_cell_partition_(),
      dev_tile_partition_(), cuda_count_(-1), dev_cell_idx_(), comp_streams_(),
      comm_streams_(), comp_events_(), comm_events_(), nccl_devs_(),
      nccl_comms_(), cell_comm_devs_(), tile_comm_devs_(), cell_comms_(),
      tile_comms_(), gpu_layout_user_set_(false), cuda_initialized_(false) {}

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
  this->require_single_gpu_runtime("get_ef");

  cudaCheck(cudaSetDevice(0));
  fx.resize(this->plan_->natom());
  fy.resize(this->plan_->natom());
  fz.resize(this->plan_->natom());
  en.resize(this->plan_->natom());

  cub::DeviceRadixSort::SortPairs(
      this->workspace_->cub_work_buffer()[0],
      this->workspace_->cub_work_buffer_size()[0],
      this->workspace_->sorted_idx()[0].d_array().data(),
      this->workspace_->idx()[0].d_array().data(),
      this->workspace_->fx()[0].d_array().data(), fx.d_array().data(),
      this->plan_->natom(), 0, static_cast<int>(8 * sizeof(unsigned int)),
      this->comp_streams_[0]);
  cub::DeviceRadixSort::SortPairs(
      this->workspace_->cub_work_buffer()[0],
      this->workspace_->cub_work_buffer_size()[0],
      this->workspace_->sorted_idx()[0].d_array().data(),
      this->workspace_->idx()[0].d_array().data(),
      this->workspace_->fy()[0].d_array().data(), fy.d_array().data(),
      this->plan_->natom(), 0, static_cast<int>(8 * sizeof(unsigned int)),
      this->comp_streams_[0]);
  cub::DeviceRadixSort::SortPairs(
      this->workspace_->cub_work_buffer()[0],
      this->workspace_->cub_work_buffer_size()[0],
      this->workspace_->sorted_idx()[0].d_array().data(),
      this->workspace_->idx()[0].d_array().data(),
      this->workspace_->fz()[0].d_array().data(), fz.d_array().data(),
      this->plan_->natom(), 0, static_cast<int>(8 * sizeof(unsigned int)),
      this->comp_streams_[0]);
  cub::DeviceRadixSort::SortPairs(
      this->workspace_->cub_work_buffer()[0],
      this->workspace_->cub_work_buffer_size()[0],
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

void glst_force::init(const unsigned int natom, const double tol,
                      const double box_dim_x, const double box_dim_y,
                      const double box_dim_z, const double rcut) {
  this->init_cuda_resources();

  this->plan_ = std::make_unique<glst_plan>();
  this->plan_->init_cells(natom, box_dim_x, box_dim_y, box_dim_z, rcut);
  this->plan_->init_cell_partitions(this->cell_partition_count_);
  this->plan_->init_alpha_groups(tol);
  this->plan_->init_cubature(tol);
  this->plan_->init_tile_schedule(2048);
  this->plan_->init_tile_partitions(this->tile_partition_count_);

  this->workspace_ = std::make_unique<glst_workspace>(
      *(this->plan_), this->dev_cell_partition_, this->cuda_count_);

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    const std::size_t atom_capacity = this->workspace_->atom_capacity(dev);
    for (std::size_t i = 0; i < atom_capacity; i++)
      this->workspace_->idx()[dev][i] = static_cast<unsigned int>(i);

    if (atom_capacity > 0)
      this->workspace_->idx()[dev].transfer_to_device();
  }

  // Layout validation
  if (this->cell_partition_count_ == 0) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::init: cell_partition_count == 0");
  }

  if (this->tile_partition_count_ == 0) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::init: tile_partition_count == 0");
  }

  if (this->dev_cell_partition_.size() !=
      static_cast<std::size_t>(this->cuda_count_)) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::init: dev_cell_partition size does not match "
        "cuda_count");
  }

  if (this->dev_tile_partition_.size() !=
      static_cast<std::size_t>(this->cuda_count_)) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::init: dev_tile_partition size does not match "
        "cuda_count");
  }

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    if (this->dev_cell_partition_[dev] >= this->cell_partition_count_) {
      throw std::runtime_error("FATAL ERROR: glst_force::init: Device cell "
                               "partition is out of range");
    }

    if (this->dev_tile_partition_[dev] >= this->tile_partition_count_) {
      throw std::runtime_error("FATAL ERROR: glst_force::init: Device tile "
                               "partition is out of range");
    }
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
    const std::size_t rmt_capacity =
        this->workspace_->rmt_tile_buffer_capacity(dev);

    const double sf_mib =
        static_cast<double>(2 * sf_capacity * sizeof(double)) /
        (1024.0 * 1024.0);
    const double rmt_mib =
        static_cast<double>(2 * rmt_capacity * sizeof(double)) /
        (1024.0 * 1024.0);

    std::cout << "    GPU " << dev << ": atom capacity " << atom_capacity
              << ", local cells " << cell_capacity << ", sf tile entries "
              << sf_capacity << " (" << sf_mib << " MiB), rmt tile entries "
              << rmt_capacity << " (" << rmt_mib << " MiB)" << std::endl;
  }

  return;
}

void glst_force::set_gpu_layout(const unsigned int cell_partition_count,
                                const unsigned int tile_partition_count) {
  if (this->cuda_initialized_) {
    throw std::runtime_error("FATAL ERROR: glst_force::set_gpu_layout: GPU "
                             "layout must be set before init");
  }

  if ((cell_partition_count == 0) || (tile_partition_count == 0)) {
    throw std::runtime_error("FATAL ERROR: glst_force::set_gpu_layout: "
                             "Partition counts must be positive");
  }

  this->cell_partition_count_ = cell_partition_count;
  this->tile_partition_count_ = tile_partition_count;
  this->gpu_layout_user_set_ = true;

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
  if (this->plan_ == nullptr) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::calc_lr_ef: Plan is not initialized");
  }

  this->require_single_gpu_runtime("calc_lr_ef");

  this->zero_ef();

  for (unsigned int tile = 0; tile < this->plan_->tile_count(); tile++) {
    this->calc_sf_tile(tile);
    this->sum_rmt_sf_tile(tile);
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

    const unsigned int *source_cell_atom_point = nullptr;
    const unsigned int *source_cell_atom_count = nullptr;

    if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
      source_cell_atom_point =
          this->workspace_->cell_atom_point()[dev].d_array().data();
      source_cell_atom_count =
          this->workspace_->cell_atom_count()[dev].d_array().data();
    } else {
      source_cell_atom_point =
          this->workspace_->sr_source_cell_atom_point()[dev].d_array().data();
      source_cell_atom_count =
          this->workspace_->sr_source_cell_atom_count()[dev].d_array().data();
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
            this->workspace_->qc()[dev].d_array().data(),
            this->workspace_->cell_atom_point()[dev].d_array().data(),
            this->workspace_->cell_atom_count()[dev].d_array().data(),
            this->dev_cell_idx_[dev].d_array().data(),
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
            this->workspace_->qc()[dev].d_array().data(),
            this->workspace_->cell_atom_point()[dev].d_array().data(),
            this->workspace_->cell_atom_count()[dev].d_array().data(),
            source_cell_atom_point, source_cell_atom_count,
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
  this->require_single_gpu_runtime("comm_ef");

  if (this->execution_mode_ == GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaStreamSynchronize(this->comp_streams_[0]));
    return;
  }

  // Synchronize to ensure that all devices are done computing
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamSynchronize(this->comp_streams_[dev]));
  }

  nccl_root_reduce_sum_ef_ip(this->workspace_->fx(), this->workspace_->fy(),
                             this->workspace_->fz(), this->workspace_->en(),
                             this->plan_->natom(), this->nccl_comms_,
                             this->comm_streams_, this->cuda_count_);

  // Synchronize to ensure that all devices are done communicating
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamSynchronize(this->comm_streams_[dev]));
  }

  return;
}

void glst_force::calc_ener_force(const double *d_rx, const double *d_ry,
                                 const double *d_rz, const double *d_qc) {
  if (this->plan_ == nullptr) {
    throw std::runtime_error("FATAL ERROR: glst_force::calc_ener_force: "
                             "Plan is not initialized");
  }

  this->require_single_gpu_runtime("calc_ener_force");

  this->assign_atoms(d_rx, d_ry, d_rz, d_qc);

  this->zero_ef();

  for (unsigned int tile = 0; tile < this->plan_->tile_count(); tile++) {
    this->calc_sf_tile(tile);
    this->sum_rmt_sf_tile(tile);
    this->calc_lr_ef_tile(tile);
  }

  this->calc_sr_ef();
  this->comm_ef();

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
               const unsigned int ncell) {
  __shared__ double s_cache[ATOM_TILE * 4];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = (idx < nc);

  double xc = 0.0, yc = 0.0, zc = 0.0;
  if (active) {
    xc = cx[idx];
    yc = cy[idx];
    zc = cz[idx];
  }

  for (unsigned int cell = blockIdx.y; cell < ncell; cell += gridDim.y) {
    const unsigned int apnt = cell_atom_points[cell];
    const unsigned int acnt = cell_atom_counts[cell];

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
      sf_re[cell * nc + idx] = sf_re0;
      sf_im[cell * nc + idx] = sf_im0;
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
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

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
      cub::DeviceRadixSort::SortPairs(
          this->workspace_->cub_work_buffer()[dev],
          this->workspace_->cub_work_buffer_size()[dev],
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

  return;
}

void glst_force::validate_atom_scatter(void) const {
  const unsigned int natom = this->plan_->natom();

  std::vector<unsigned int> replica_count(natom, 0);

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    const unsigned int expected_cell_partition = this->dev_cell_partition_[dev];

    const std::vector<atom_packet> &packets =
        this->workspace_->sorted_packets()[dev].h_array();

    const std::size_t owned_atom_count =
        this->workspace_->owned_atom_count(dev);

    if (owned_atom_count > packets.size()) {
      throw std::runtime_error(
          "FATAL ERROR: glst_force::validate_atom_scatter: Owned atom count "
          "exceeds source atom count");
    }

    for (std::size_t i = 0; i < packets.size(); i++) {
      const atom_packet &packet = packets[i];

      if (packet.i >= natom) {
        throw std::runtime_error(
            "FATAL ERROR: glst_force::validate_atom_scatter: Original atom "
            "index is out of range");
      }

      if (packet.cell >= this->plan_->ncell()) {
        throw std::runtime_error(
            "FATAL ERROR: glst_force::validate_atom_scatter: Global cell index "
            "is out of range");
      }

      const unsigned int observed_cell_partition =
          this->plan_->cell_partition_idx(packet.cell);

      if (i < owned_atom_count) {
        if (observed_cell_partition != expected_cell_partition) {
          throw std::runtime_error(
              "FATAL ERROR: glst_force::validate_atom_scatter: Owned atom is "
              "stored on the wrong cell partition");
        }

        replica_count[packet.i]++;
      } else {
        if (observed_cell_partition == expected_cell_partition) {
          throw std::runtime_error(
              "FATAL ERROR: glst_force::validate_atom_scatter: Halo atom is "
              "owned by the target partition");
        }
      }
    }

    const std::vector<unsigned int> &source_cells =
        this->plan_->partition_sr_source_cell_idx(expected_cell_partition);

    const std::size_t owned_cell_count = static_cast<std::size_t>(
        this->plan_->local_cell_count(expected_cell_partition));
    std::size_t observed_source_atom_count = 0;
    std::size_t observed_owned_atom_count = 0;

    if ((source_cells.size() !=
         this->workspace_->sr_source_cell_atom_count()[dev].h_array().size()) ||
        (source_cells.size() !=
         this->workspace_->sr_source_cell_atom_point()[dev].h_array().size())) {
      throw std::runtime_error(
          "FATAL ERROR: glst_force::validate_atom_scatter: Source cell "
          "metadata size mismatch");
    }

    for (std::size_t source_local_cell = 0;
         source_local_cell < source_cells.size(); source_local_cell++) {
      const unsigned int expected_cell = source_cells[source_local_cell];
      const unsigned int point =
          this->workspace_->sr_source_cell_atom_point()[dev][source_local_cell];
      const unsigned int count =
          this->workspace_->sr_source_cell_atom_count()[dev][source_local_cell];

      observed_source_atom_count += static_cast<std::size_t>(count);
      if (source_local_cell < owned_cell_count)
        observed_owned_atom_count += static_cast<std::size_t>(count);

      if (static_cast<std::size_t>(point) + static_cast<std::size_t>(count) >
          packets.size()) {
        throw std::runtime_error(
            "FATAL ERROR: glst_force::validate_atom_scatter: Source cell atom "
            "range is out of bounds");
      }

      unsigned int last_atom = 0;
      for (unsigned int j = 0; j < count; j++) {
        const atom_packet &packet = packets[point + j];

        if (packet.cell != expected_cell) {
          throw std::runtime_error(
              "FATAL ERROR: glst_force::validate_atom_scatter: Source cell "
              "range contains an atom from the wrong global cell");
        }

        if ((j > 0) && (packet.i <= last_atom)) {
          throw std::runtime_error(
              "FATAL ERROR: glst_force::validate_atom_scatter: Source cell "
              "atom order is not deterministic");
        }

        last_atom = packet.i;
      }
    }

    if (observed_source_atom_count != packets.size()) {
      throw std::runtime_error(
          "FATAL ERROR: glst_force::validate_atom_scatter: Source cell counts "
          "do not sum to source atoms");
    }

    if (observed_owned_atom_count != owned_atom_count) {
      throw std::runtime_error(
          "FATAL ERROR: glst_force::validate_atom_scatter: Owned source cell "
          "counts do not sum to owned atoms");
    }
  }

  for (unsigned int atom = 0; atom < natom; atom++) {
    if (replica_count[atom] != this->tile_partition_count_) {
      throw std::runtime_error(
          "FATAL ERROR: glst_force::validate_atom_scatter: Atom replica count "
          "does not equal G_tile");
    }
  }

  for (unsigned int cell_partition = 0;
       cell_partition < this->cell_partition_count_; cell_partition++) {
    int first_dev = -1;

    for (int dev = 0; dev < this->cuda_count_; dev++) {
      if (this->dev_cell_partition_[dev] != cell_partition)
        continue;

      if (first_dev < 0) {
        first_dev = dev;
        continue;
      }

      const std::vector<atom_packet> &lhs =
          this->workspace_->sorted_packets()[first_dev].h_array();
      const std::vector<atom_packet> &rhs =
          this->workspace_->sorted_packets()[dev].h_array();

      if (lhs.size() != rhs.size()) {
        throw std::runtime_error(
            "FATAL ERROR: glst_force::validate_atom_scatter: Tile ranks in a "
            "cell partition have different atom counts");
      }

      for (std::size_t i = 0; i < lhs.size(); i++) {
        if ((lhs[i].i != rhs[i].i) || (lhs[i].cell != rhs[i].cell) ||
            (lhs[i].x != rhs[i].x) || (lhs[i].y != rhs[i].y) ||
            (lhs[i].z != rhs[i].z) || (lhs[i].q != rhs[i].q)) {
          throw std::runtime_error(
              "FATAL ERROR: glst_force::validate_atom_scatter: Tile ranks in a "
              "cell partition have different atom ordering");
        }
      }
    }
  }

  return;
}

void glst_force::assign_atoms_multi_gpu(const double *d_rx, const double *d_ry,
                                        const double *d_rz,
                                        const double *d_qc) {
  if (this->plan_ == nullptr) {
    throw std::runtime_error("FATAL ERROR: glst_force::assign_atoms_multi_gpu: "
                             "Plan is not initialized");
  }

  const unsigned int natom = this->plan_->natom();
  const unsigned int ncell_x = this->plan_->ncell_x();
  const unsigned int ncell_y = this->plan_->ncell_y();
  const unsigned int ncell_z = this->plan_->ncell_z();

  std::vector<double> h_rx(natom);
  std::vector<double> h_ry(natom);
  std::vector<double> h_rz(natom);
  std::vector<double> h_qc(natom);

  cudaCheck(cudaSetDevice(0));
  cudaCheck(cudaMemcpy(h_rx.data(), d_rx, natom * sizeof(double),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_ry.data(), d_ry, natom * sizeof(double),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_rz.data(), d_rz, natom * sizeof(double),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_qc.data(), d_qc, natom * sizeof(double),
                       cudaMemcpyDeviceToHost));

  std::vector<std::vector<atom_packet>> cell_packets(this->plan_->ncell());
  std::vector<std::vector<atom_packet>> partition_owned_packets(
      this->cell_partition_count_);
  std::vector<std::vector<atom_packet>> partition_source_packets(
      this->cell_partition_count_);

  const double inv_cell_dim_x = 1.0 / this->plan_->cell_dim_x();
  const double inv_cell_dim_y = 1.0 / this->plan_->cell_dim_y();
  const double inv_cell_dim_z = 1.0 / this->plan_->cell_dim_z();

  for (unsigned int atom = 0; atom < natom; atom++) {
    int cx = static_cast<int>(h_rx[atom] * inv_cell_dim_x);
    int cy = static_cast<int>(h_ry[atom] * inv_cell_dim_y);
    int cz = static_cast<int>(h_rz[atom] * inv_cell_dim_z);

    cx = (cx >= static_cast<int>(ncell_x)) ? static_cast<int>(ncell_x - 1) : cx;
    cy = (cy >= static_cast<int>(ncell_y)) ? static_cast<int>(ncell_y - 1) : cy;
    cz = (cz >= static_cast<int>(ncell_z)) ? static_cast<int>(ncell_z - 1) : cz;

    cx = (cx < 0) ? 0 : cx;
    cy = (cy < 0) ? 0 : cy;
    cz = (cz < 0) ? 0 : cz;

    const unsigned int cell = (static_cast<unsigned int>(cx) * ncell_y +
                               static_cast<unsigned int>(cy)) *
                                  ncell_z +
                              static_cast<unsigned int>(cz);

    const atom_packet packet(atom, cell, h_rx[atom], h_ry[atom], h_rz[atom],
                             h_qc[atom]);

    cell_packets[cell].push_back(packet);
  }

  for (unsigned int cell = 0; cell < this->plan_->ncell(); cell++) {
    std::vector<atom_packet> &packets = cell_packets[cell];

    std::sort(packets.begin(), packets.end(),
              [](const atom_packet &lhs, const atom_packet &rhs) {
                return lhs.i < rhs.i;
              });
  }

  for (unsigned int partition = 0; partition < this->cell_partition_count_;
       partition++) {
    std::vector<atom_packet> &owned_packets =
        partition_owned_packets[partition];
    std::vector<atom_packet> &source_packets =
        partition_source_packets[partition];

    const std::vector<unsigned int> &owned_cells =
        this->plan_->partition_cell_idx(partition);
    for (std::size_t i = 0; i < owned_cells.size(); i++) {
      const unsigned int cell = owned_cells[i];
      const std::vector<atom_packet> &cell_atoms = cell_packets[cell];
      owned_packets.insert(owned_packets.end(), cell_atoms.begin(),
                           cell_atoms.end());
    }

    const std::vector<unsigned int> &source_cells =
        this->plan_->partition_sr_source_cell_idx(partition);
    for (std::size_t i = 0; i < source_cells.size(); i++) {
      const unsigned int cell = source_cells[i];
      const std::vector<atom_packet> &cell_atoms = cell_packets[cell];
      source_packets.insert(source_packets.end(), cell_atoms.begin(),
                            cell_atoms.end());
    }
  }

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    const unsigned int cell_partition = this->dev_cell_partition_[dev];

    const std::vector<atom_packet> &owned_packets =
        partition_owned_packets[cell_partition];
    const std::vector<atom_packet> &source_packets =
        partition_source_packets[cell_partition];

    const std::size_t owned_atom_count = owned_packets.size();
    const std::size_t source_atom_count = source_packets.size();

    const std::size_t local_cell_count =
        static_cast<std::size_t>(this->plan_->local_cell_count(cell_partition));

    const std::vector<unsigned int> &source_cells =
        this->plan_->partition_sr_source_cell_idx(cell_partition);
    const std::size_t source_cell_count = source_cells.size();

    if (source_cell_count != this->workspace_->sr_source_cell_capacity(dev)) {
      throw std::runtime_error(
          "FATAL ERROR: glst_force::assign_atoms_multi_gpu: Source cell count "
          "does not match workspace capacity");
    }

    this->workspace_->resize_atom_storage(dev, source_atom_count);
    this->workspace_->set_owned_atom_count(dev, owned_atom_count);

    std::fill(this->workspace_->cell_atom_count()[dev].h_array().begin(),
              this->workspace_->cell_atom_count()[dev].h_array().end(), 0u);
    std::fill(this->workspace_->cell_atom_point()[dev].h_array().begin(),
              this->workspace_->cell_atom_point()[dev].h_array().end(), 0u);
    std::fill(
        this->workspace_->sr_source_cell_atom_count()[dev].h_array().begin(),
        this->workspace_->sr_source_cell_atom_count()[dev].h_array().end(), 0u);
    std::fill(
        this->workspace_->sr_source_cell_atom_point()[dev].h_array().begin(),
        this->workspace_->sr_source_cell_atom_point()[dev].h_array().end(), 0u);

    std::size_t atom_point = 0;
    this->workspace_->max_atoms_cell()[dev] = 0;

    for (std::size_t source_local_cell = 0;
         source_local_cell < source_cell_count; source_local_cell++) {
      const unsigned int global_cell = source_cells[source_local_cell];
      const std::vector<atom_packet> &cell_atoms = cell_packets[global_cell];

      if (cell_atoms.size() >
          static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::runtime_error(
            "FATAL ERROR: glst_force::assign_atoms_multi_gpu: Cell atom count "
            "exceeds unsigned int range");
      }

      const unsigned int point = static_cast<unsigned int>(atom_point);
      const unsigned int count = static_cast<unsigned int>(cell_atoms.size());

      this->workspace_->sr_source_cell_atom_point()[dev][source_local_cell] =
          point;

      this->workspace_->sr_source_cell_atom_count()[dev][source_local_cell] =
          count;

      if (source_local_cell < local_cell_count) {
        this->workspace_->cell_atom_point()[dev][source_local_cell] = point;
        this->workspace_->cell_atom_count()[dev][source_local_cell] = count;

        if (count > this->workspace_->max_atoms_cell()[dev])
          this->workspace_->max_atoms_cell()[dev] = count;
      }

      for (std::size_t i = 0; i < cell_atoms.size(); i++) {
        const atom_packet &packet = cell_atoms[i];

        this->workspace_->idx()[dev][atom_point] = packet.i;
        this->workspace_->sorted_idx()[dev][atom_point] = packet.i;
        this->workspace_->rx()[dev][atom_point] = packet.x;
        this->workspace_->ry()[dev][atom_point] = packet.y;
        this->workspace_->rz()[dev][atom_point] = packet.z;
        this->workspace_->qc()[dev][atom_point] = packet.q;
        this->workspace_->packets()[dev][atom_point] = packet;
        this->workspace_->sorted_packets()[dev][atom_point] = packet;

        this->workspace_->atom_cell_idx()[dev][atom_point] =
            static_cast<unsigned int>(source_local_cell);
        this->workspace_->atom_cell_sorted_idx()[dev][atom_point] =
            static_cast<unsigned int>(source_local_cell);

        atom_point++;
      }
    }

    if (atom_point != source_atom_count) {
      throw std::runtime_error(
          "FATAL ERROR: glst_force::assign_atoms_multi_gpu: Source cell counts "
          "do not sum to source atoms");
    }

    for (std::size_t i = 0; i < owned_atom_count; i++) {
      if ((source_packets[i].i != owned_packets[i].i) ||
          (source_packets[i].cell != owned_packets[i].cell) ||
          (source_packets[i].x != owned_packets[i].x) ||
          (source_packets[i].y != owned_packets[i].y) ||
          (source_packets[i].z != owned_packets[i].z) ||
          (source_packets[i].q != owned_packets[i].q)) {
        throw std::runtime_error(
            "FATAL ERROR: glst_force::assign_atoms_multi_gpu: Source packet "
            "list does not start with owned packets");
      }
    }

    if (source_atom_count > 0) {
      this->workspace_->idx()[dev].transfer_to_device();
      this->workspace_->sorted_idx()[dev].transfer_to_device();
      this->workspace_->rx()[dev].transfer_to_device();
      this->workspace_->ry()[dev].transfer_to_device();
      this->workspace_->rz()[dev].transfer_to_device();
      this->workspace_->qc()[dev].transfer_to_device();
      this->workspace_->packets()[dev].transfer_to_device();
      this->workspace_->sorted_packets()[dev].transfer_to_device();
      this->workspace_->atom_cell_idx()[dev].transfer_to_device();
      this->workspace_->atom_cell_sorted_idx()[dev].transfer_to_device();
    }

    this->workspace_->cell_atom_count()[dev].transfer_to_device();
    this->workspace_->cell_atom_point()[dev].transfer_to_device();

    this->workspace_->sr_source_cell_atom_count()[dev].transfer_to_device();
    this->workspace_->sr_source_cell_atom_point()[dev].transfer_to_device();
  }

#ifdef __GLST_DEBUG__
  this->validate_atom_scatter();
#endif

  return;
}

void glst_force::calc_sf_tile(const unsigned int tile) {
  if (this->plan_ == nullptr) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::calc_sf_tile: Plan is not initialized");
  }

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  if (tile >= this->plan_->tile_count()) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::calc_sf_tile: Tile is out of bounds");
  }

  const unsigned int tile_node_point = this->plan_->tile_node_point(tile);
  const unsigned int tile_node_count = this->plan_->tile_node_count(tile);

  if (tile_node_count == 0) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::calc_sf_tile: Tile node count is 0");
  }

  if (tile_node_count > this->plan_->max_tile_nodes()) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::calc_sf_tile: Tile exceeds buffer size");
  }

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    const unsigned int nc = tile_node_count;
    const unsigned int off = tile_node_point;

    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(this->workspace_->sf_re()[dev].d_array().data()), 0,
        this->plan_->ncell() * nc * sizeof(double), this->comp_streams_[dev]));
    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(this->workspace_->sf_im()[dev].d_array().data()), 0,
        this->plan_->ncell() * nc * sizeof(double), this->comp_streams_[dev]));

    constexpr dim3 num_threads(128, 1, 1);
    const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                          std::min(65535u, this->plan_->ncell()), 1);
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
            this->workspace_->qc()[dev].d_array().data(),
            this->workspace_->cell_atom_point()[dev].d_array().data(),
            this->workspace_->cell_atom_count()[dev].d_array().data(),
            this->plan_->ncell());

    cudaCheck(cudaGetLastError());
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
    sum_re += sf_re[cell * nc + idx];
    sum_im += sf_im[cell * nc + idx];
    sf_re[cell * nc + idx] = sum_re;
    sf_im[cell * nc + idx] = sum_im;
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
    sum_re += sf_re[cell * nc + idx];
    sum_im += sf_im[cell * nc + idx];
    sf_re[cell * nc + idx] = sum_re;
    sf_im[cell * nc + idx] = sum_im;
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
    sum_re += sf_re[cell * nc + idx];
    sum_im += sf_im[cell * nc + idx];
    sf_re[cell * nc + idx] = sum_re;
    sf_im[cell * nc + idx] = sum_im;
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

__device__ void cube_sum(double &cube_re, double &cube_im,
                         const double *__restrict__ P_re,
                         const double *__restrict__ P_im, const unsigned int x,
                         const unsigned int y, const unsigned int z,
                         const unsigned int r, const unsigned int nx,
                         const unsigned int ny, const unsigned int nz,
                         const unsigned int idx, const unsigned int nc) {
  // Need to check if x-r, y-r, z-r is < 0
  const unsigned int x0 = (x < r) ? 0 : x - r;
  const unsigned int y0 = (y < r) ? 0 : y - r;
  const unsigned int z0 = (z < r) ? 0 : z - r;

  // Need to check if x+r, y+r, z+r is > nx-1, ny-1, nz-1
  unsigned int x1 = x + r;
  unsigned int y1 = y + r;
  unsigned int z1 = z + r;
  x1 = (x1 >= nx) ? nx - 1 : x1;
  y1 = (y1 >= ny) ? ny - 1 : y1;
  z1 = (z1 >= nz) ? nz - 1 : z1;

  box_sum(cube_re, cube_im, P_re, P_im, x0, y0, z0, x1, y1, z1, nx, ny, nz, idx,
          nc);

  return;
}

__device__ void shell_sum(double &shell_re, double &shell_im,
                          const double *__restrict__ P_re,
                          const double *__restrict__ P_im, const unsigned int x,
                          const unsigned int y, const unsigned int z,
                          const unsigned int inner, const unsigned int outer,
                          const unsigned int nx, const unsigned int ny,
                          const unsigned int nz, const unsigned int idx,
                          const unsigned int nc) {
  double osum_re = static_cast<double>(0.0), osum_im = static_cast<double>(0.0);
  cube_sum(osum_re, osum_im, P_re, P_im, x, y, z, outer, nx, ny, nz, idx, nc);

  double isum_re = static_cast<double>(0.0), isum_im = static_cast<double>(0.0);
  cube_sum(isum_re, isum_im, P_re, P_im, x, y, z, inner, nx, ny, nz, idx, nc);

  shell_re = osum_re - isum_re;
  shell_im = osum_im - isum_im;

  return;
}

__global__ static void calc_rmt_sum_kernel(
    double *__restrict__ rmt_sum_re, double *__restrict__ rmt_sum_im,
    const double *__restrict__ sf_re, const double *__restrict__ sf_im,
    const double *__restrict__ cw, const unsigned int *__restrict__ groups,
    const unsigned int nc, const unsigned int *__restrict__ grp_r_in,
    const unsigned int *__restrict__ grp_r_out, const unsigned int nx,
    const unsigned int ny, const unsigned int nz, const unsigned int ncell) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nc)
    return;

  const double wc = cw[idx];
  const unsigned int grp = groups[idx];

  for (unsigned int cell = blockIdx.y; cell < ncell; cell += gridDim.y) {
    const unsigned int x = cell / (ny * nz);
    const unsigned int y = (cell / nz) % ny;
    const unsigned int z = cell % nz;
    const unsigned int inner = grp_r_in[grp];
    const unsigned int outer = grp_r_out[grp];

    double shell_re = 0.0, shell_im = 0.0;
    shell_sum(shell_re, shell_im, sf_re, sf_im, x, y, z, inner, outer, nx, ny,
              nz, idx, nc);

    shell_re *= wc;
    shell_im *= wc;

    rmt_sum_re[cell * nc + idx] = shell_re;
    rmt_sum_im[cell * nc + idx] = shell_im;
  }

  return;
}

void glst_force::sum_rmt_sf_tile(const unsigned int tile) {
  if (this->plan_ == nullptr) {
    throw std::runtime_error("FATAL ERROR: glst_force::sum_rmt_sf_tile: "
                             "Plan is not initialized");
  }

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  if (tile >= this->plan_->tile_count()) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::sum_rmt_sf_tile: Tile is out of bounds");
  }

  const unsigned int tile_node_point = this->plan_->tile_node_point(tile);
  const unsigned int tile_node_count = this->plan_->tile_node_count(tile);

  if (tile_node_count == 0) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::sum_rmt_sf_tile: Tile node count is 0");
  }

  if (tile_node_count > this->plan_->max_tile_nodes()) {
    throw std::runtime_error("FATAL ERROR: glst_force::sum_rmt_sf_tile: "
                             "Tile exceeds buffer size");
  }

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    const unsigned int nc = tile_node_count;
    const unsigned int off = tile_node_point;

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
                            std::min(65535u, this->plan_->ncell()), 1);

      calc_rmt_sum_kernel<<<num_blocks, num_threads, 0,
                            this->comp_streams_[dev]>>>(
          this->workspace_->rmt_sum_re()[dev].d_array().data(),
          this->workspace_->rmt_sum_im()[dev].d_array().data(),
          this->workspace_->sf_re()[dev].d_array().data(),
          this->workspace_->sf_im()[dev].d_array().data(),
          this->plan_->w()[dev].d_array().data() + off,
          this->plan_->group()[dev].d_array().data() + off, nc,
          this->plan_->grp_r_in()[dev].d_array().data(),
          this->plan_->grp_r_out()[dev].d_array().data(),
          this->plan_->ncell_x(), this->plan_->ncell_y(),
          this->plan_->ncell_z(), this->plan_->ncell());

      cudaCheck(cudaGetLastError());
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
                  const unsigned int *__restrict__ cell_atom_points,
                  const unsigned int *__restrict__ cell_atom_counts,
                  const double *__restrict__ cx, const double *__restrict__ cy,
                  const double *__restrict__ cz,
                  const double *__restrict__ rmt_sum_re,
                  const double *__restrict__ rmt_sum_im, const unsigned int nc,
                  const unsigned int ncell) {
  __shared__ double s_cache[BLOCK * 5];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int cell = blockIdx.y; cell < ncell; cell += gridDim.y) {
    const unsigned int apnt = cell_atom_points[cell];
    const unsigned int acnt = cell_atom_counts[cell];
    const bool active = (idx < acnt);

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
        s_cache[threadIdx.x * 5 + 0] = cx[i + threadIdx.x];
        s_cache[threadIdx.x * 5 + 1] = cy[i + threadIdx.x];
        s_cache[threadIdx.x * 5 + 2] = cz[i + threadIdx.x];
        s_cache[threadIdx.x * 5 + 3] = rmt_sum_re[cell * nc + i + threadIdx.x];
        s_cache[threadIdx.x * 5 + 4] = rmt_sum_im[cell * nc + i + threadIdx.x];
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
  if (this->plan_ == nullptr) {
    throw std::runtime_error("FATAL ERROR: glst_force::calc_lr_ef_tile: "
                             "Plan is not initialized");
  }

  const bool has_long_range_cells =
      ((this->plan_->ncell_x() > 2) && (this->plan_->ncell_y() > 2) &&
       (this->plan_->ncell_z() > 2));
  if (!has_long_range_cells)
    return;

  if (tile >= this->plan_->tile_count()) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::calc_lr_ef_tile: Tile is out of bounds");
  }

  const unsigned int tile_node_point = this->plan_->tile_node_point(tile);
  const unsigned int tile_node_count = this->plan_->tile_node_count(tile);

  if (tile_node_count == 0) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::calc_lr_ef_tile: Tile node count is 0");
  }

  if (tile_node_count > this->plan_->max_tile_nodes()) {
    throw std::runtime_error("FATAL ERROR: glst_force::calc_lr_ef_tile: "
                             "Tile exceeds buffer size");
  }

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    const unsigned int nc = tile_node_count;
    const unsigned int off = tile_node_point;

    constexpr dim3 num_threads(64, 1, 1);
    const dim3 num_blocks(
        (this->workspace_->max_atoms_cell()[dev] + num_threads.x - 1) /
            num_threads.x,
        std::min(65535u, this->plan_->ncell()), 1);

    calc_lr_ef_kernel<num_threads.x>
        <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
            this->workspace_->fx()[dev].d_array().data(),
            this->workspace_->fy()[dev].d_array().data(),
            this->workspace_->fz()[dev].d_array().data(),
            this->workspace_->en()[dev].d_array().data(),
            this->workspace_->rx()[dev].d_array().data(),
            this->workspace_->ry()[dev].d_array().data(),
            this->workspace_->rz()[dev].d_array().data(),
            this->workspace_->qc()[dev].d_array().data(),
            this->workspace_->cell_atom_point()[dev].d_array().data(),
            this->workspace_->cell_atom_count()[dev].d_array().data(),
            this->plan_->x()[dev].d_array().data() + off,
            this->plan_->y()[dev].d_array().data() + off,
            this->plan_->z()[dev].d_array().data() + off,
            this->workspace_->rmt_sum_re()[dev].d_array().data(),
            this->workspace_->rmt_sum_im()[dev].d_array().data(), nc,
            this->plan_->ncell());

    cudaCheck(cudaGetLastError());
  }

  return;
}

void glst_force::zero_ef(void) {
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));

    const std::size_t nbytes =
        this->workspace_->atom_capacity(dev) * sizeof(double);

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

  if (this->execution_mode_ != GLST_EXECUTION_MODE::SINGLE_GPU_TILED)
    disable_p2p(this->cuda_count_);

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

  enable_p2p(this->cuda_count_);

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

      if (udev >= static_cast<unsigned int>(this->cuda_count_)) {
        throw std::runtime_error("FATAL ERROR: glst_force::init_nccl_topology: "
                                 "Cell communicator device is out of range");
      }

      if ((this->dev_cell_partition_[udev] != cell_part) ||
          (this->dev_tile_partition_[udev] != tile_part)) {
        throw std::runtime_error("FATAL ERROR: glst_force::init_nccl_topology: "
                                 "Cell communicator layout mismatch");
      }

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

      if (udev >= static_cast<unsigned int>(this->cuda_count_)) {
        throw std::runtime_error("FATAL ERROR: glst_force::init_nccl_topology: "
                                 "Tile communicator device is out of range");
      }

      if ((this->dev_cell_partition_[udev] != cell_part) ||
          (this->dev_tile_partition_[udev] != tile_part)) {
        throw std::runtime_error("FATAL ERROR: glst_force::init_nccl_topology: "
                                 "Tile communicator layout mismatch");
      }

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

void glst_force::init_gpu_layout(const int device_count) {
  if (device_count < 1) {
    throw std::runtime_error("FATAL ERROR: glst_force::init_gpu_layout: Could "
                             "not find any CUDA capable devices");
  }

  unsigned int cell_partition_count = this->cell_partition_count_;
  unsigned int tile_partition_count = this->tile_partition_count_;

  if (!this->gpu_layout_user_set_) {
    if (device_count == 1) {
      cell_partition_count = 1;
      tile_partition_count = 1;
    } else {
      cell_partition_count = static_cast<unsigned int>(device_count);
      tile_partition_count = 1;
    }
  }

  const unsigned long long int product =
      static_cast<unsigned long long int>(cell_partition_count) *
      static_cast<unsigned long long int>(tile_partition_count);

  if (product != static_cast<unsigned long long int>(device_count)) {
    throw std::runtime_error("FATAL ERROR: glst_force::init_gpu_layout: "
                             "GLST_CELL_PARTITION * GLST_TILE_PARTITION must "
                             "equal the visible CUDA device count; observed " +
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
  if (this->plan_ == nullptr) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::cells2dev: Plan is not initialized");
  }

  this->dev_cell_idx_.resize(this->cuda_count_);

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    const unsigned int partition = this->dev_cell_partition_[dev];

    if (partition >= this->plan_->cell_partition_count()) {
      throw std::runtime_error("FATAL ERROR: glst_force::cells2dev: Device "
                               "cell partition is out of range");
    }

    cudaCheck(cudaSetDevice(dev));
    this->dev_cell_idx_[dev] = this->plan_->partition_cell_idx(partition);
  }

  return;
}

void glst_force::require_single_gpu_runtime(
    const std::string_view method) const {
  if (this->execution_mode_ != GLST_EXECUTION_MODE::SINGLE_GPU_TILED) {
    throw std::runtime_error(
        "FATAL ERROR: glst_force::" + std::string(method) +
        ": Multi-GPU local-workspace runtime is not implemented yet");
  }
  return;
}
