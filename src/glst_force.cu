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

#include "cell_decomp.hpp"
#include "cuda_utils.hcu"
#include "device_comm.hcu"
#include "reduce.hcu"
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <type_traits>

template <typename CT>
glst_force<CT>::glst_force(void)
    : natom_(0), idx_(), sorted_idx_(), rx_(), ry_(), rz_(), qc_(), packets_(),
      sorted_packets_(), fx_(), fy_(), fz_(), en_(), atom_cell_idx_(),
      atom_cell_sorted_idx_(), ncell_x_(0), ncell_y_(0), ncell_z_(0), ncell_(0),
      cell_dim_x_(static_cast<CT>(0.0)), cell_dim_y_(static_cast<CT>(0.0)),
      cell_dim_z_(static_cast<CT>(0.0)), ngroup_(0), grp_r_in_(), grp_r_out_(),
      cubature_(nullptr), dev_cub_counts_(), dev_cub_points_(),
      cell_atom_point_(), cell_atom_count_(), max_atoms_cell_(), sf_re_(),
      sf_im_(), rmt_sum_re_(), rmt_sum_im_(), cub_work_buffer_(),
      cub_work_buffer_size_(), cuda_count_(-1), cell_dev_idx_(),
      dev_cell_idx_(), comp_streams_(), comm_streams_(), comp_events_(),
      comm_events_(), nccl_devs_(), nccl_comms_() {
  // Ensure that CT is of a correct type
  static_assert(std::is_floating_point_v<CT>,
                "CT must be a floating-point type (float or double)");

  cudaCheck(cudaGetDeviceCount(&this->cuda_count_));
  if (this->cuda_count_ < 1) {
    throw std::runtime_error(
        "glst_force<CT>::init: Could not find any CUDA capable devices");
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

  // Initialize NCCL
  this->nccl_devs_.resize(this->cuda_count_);
  for (int dev = 0; dev < this->cuda_count_; dev++)
    this->nccl_devs_[dev] = dev;
  this->nccl_comms_.resize(this->cuda_count_);
  ncclCheck(ncclCommInitAll(this->nccl_comms_.data(), this->cuda_count_,
                            this->nccl_devs_.data()));
}

template <typename CT>
glst_force<CT>::glst_force(const unsigned int natom, const double tol,
                           const double box_dim_x, const double box_dim_y,
                           const double box_dim_z, const double rcut)
    : glst_force() {
  this->init(natom, tol, box_dim_x, box_dim_y, box_dim_z, rcut);
}

template <typename CT>
glst_force<CT>::glst_force(const unsigned int natom, const double tol,
                           const double box_dim_x, const double box_dim_y,
                           const double box_dim_z, const unsigned int ncell_x,
                           const unsigned int ncell_y,
                           const unsigned int ncell_z)
    : glst_force() {
  double rcxd = box_dim_x / static_cast<double>(ncell_x);
  double rcyd = box_dim_y / static_cast<double>(ncell_y);
  double rczd = box_dim_z / static_cast<double>(ncell_z);
  double rcut = rcxd;
  rcut = (rcyd < rcut) ? rcyd : rcut;
  rcut = (rczd < rcut) ? rczd : rcut;

  this->init(natom, tol, box_dim_x, box_dim_y, box_dim_z, rcut);
}

template <typename CT> glst_force<CT>::~glst_force(void) {
  this->deallocate();
  disable_p2p(this->cuda_count_);
}

template <typename CT>
const std::vector<cuda_container<double>> &glst_force<CT>::fx(void) const {
  return this->fx_;
}

template <typename CT>
const std::vector<cuda_container<double>> &glst_force<CT>::fy(void) const {
  return this->fy_;
}

template <typename CT>
const std::vector<cuda_container<double>> &glst_force<CT>::fz(void) const {
  return this->fz_;
}

template <typename CT>
const std::vector<cuda_container<double>> &glst_force<CT>::en(void) const {
  return this->en_;
}

template <typename CT>
std::vector<cuda_container<double>> &glst_force<CT>::fx(void) {
  return this->fx_;
}

template <typename CT>
std::vector<cuda_container<double>> &glst_force<CT>::fy(void) {
  return this->fy_;
}

template <typename CT>
std::vector<cuda_container<double>> &glst_force<CT>::fz(void) {
  return this->fz_;
}

template <typename CT>
std::vector<cuda_container<double>> &glst_force<CT>::en(void) {
  return this->en_;
}

template <typename CT>
void glst_force<CT>::get_ef(cuda_container<double> &fx,
                            cuda_container<double> &fy,
                            cuda_container<double> &fz,
                            cuda_container<double> &en) {
  cudaCheck(cudaSetDevice(0));
  fx.resize(this->natom_);
  fy.resize(this->natom_);
  fz.resize(this->natom_);
  en.resize(this->natom_);

  cub::DeviceRadixSort::SortPairs(
      this->cub_work_buffer_[0], this->cub_work_buffer_size_[0],
      this->sorted_idx_[0].d_array().data(), this->idx_[0].d_array().data(),
      this->fx_[0].d_array().data(), fx.d_array().data(), this->natom_);
  cub::DeviceRadixSort::SortPairs(
      this->cub_work_buffer_[0], this->cub_work_buffer_size_[0],
      this->sorted_idx_[0].d_array().data(), this->idx_[0].d_array().data(),
      this->fy_[0].d_array().data(), fy.d_array().data(), this->natom_);
  cub::DeviceRadixSort::SortPairs(
      this->cub_work_buffer_[0], this->cub_work_buffer_size_[0],
      this->sorted_idx_[0].d_array().data(), this->idx_[0].d_array().data(),
      this->fz_[0].d_array().data(), fz.d_array().data(), this->natom_);
  cub::DeviceRadixSort::SortPairs(
      this->cub_work_buffer_[0], this->cub_work_buffer_size_[0],
      this->sorted_idx_[0].d_array().data(), this->idx_[0].d_array().data(),
      this->en_[0].d_array().data(), en.d_array().data(), this->natom_);

  fx.transfer_to_host();
  fy.transfer_to_host();
  fz.transfer_to_host();
  en.transfer_to_host();

  return;
}

template <typename CT>
void glst_force<CT>::init(const unsigned int natom, const double tol,
                          const double box_dim_x, const double box_dim_y,
                          const double box_dim_z, const double rcut) {
  this->idx_.resize(this->cuda_count_);
  this->sorted_idx_.resize(this->cuda_count_);
  this->rx_.resize(this->cuda_count_);
  this->ry_.resize(this->cuda_count_);
  this->rz_.resize(this->cuda_count_);
  this->qc_.resize(this->cuda_count_);
  this->packets_.resize(this->cuda_count_);
  this->sorted_packets_.resize(this->cuda_count_);
  this->fx_.resize(this->cuda_count_);
  this->fy_.resize(this->cuda_count_);
  this->fz_.resize(this->cuda_count_);
  this->en_.resize(this->cuda_count_);
  this->atom_cell_idx_.resize(this->cuda_count_);
  this->atom_cell_sorted_idx_.resize(this->cuda_count_);

  this->natom_ = natom;
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    this->idx_[dev].resize(natom);
    this->sorted_idx_[dev].resize(natom);
    this->rx_[dev].resize(natom);
    this->ry_[dev].resize(natom);
    this->rz_[dev].resize(natom);
    this->qc_[dev].resize(natom);
    this->packets_[dev].resize(natom);
    this->sorted_packets_[dev].resize(natom);
    this->fx_[dev].resize(natom);
    this->fy_[dev].resize(natom);
    this->fz_[dev].resize(natom);
    this->en_[dev].resize(natom);
    this->atom_cell_idx_[dev].resize(natom);
    this->atom_cell_sorted_idx_[dev].resize(natom);
  }

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    for (unsigned int i = 0; i < natom; i++)
      this->idx_[dev][i] = i;
    this->idx_[dev].transfer_to_device();
  }

  this->cell_dim_x_ = static_cast<CT>(rcut);
  this->cell_dim_y_ = static_cast<CT>(rcut);
  this->cell_dim_z_ = static_cast<CT>(rcut);
  this->ncell_x_ =
      static_cast<unsigned int>(static_cast<CT>(box_dim_x) / this->cell_dim_x_);
  this->ncell_y_ =
      static_cast<unsigned int>(static_cast<CT>(box_dim_y) / this->cell_dim_y_);
  this->ncell_z_ =
      static_cast<unsigned int>(static_cast<CT>(box_dim_z) / this->cell_dim_z_);
  if (static_cast<CT>(this->ncell_x_) * this->cell_dim_x_ <
      static_cast<CT>(box_dim_x))
    this->ncell_x_++;
  if (static_cast<CT>(this->ncell_y_) * this->cell_dim_y_ <
      static_cast<CT>(box_dim_y))
    this->ncell_y_++;
  if (static_cast<CT>(this->ncell_z_) * this->cell_dim_z_ <
      static_cast<CT>(box_dim_z))
    this->ncell_z_++;
  this->ncell_ = this->ncell_x_ * this->ncell_y_ * this->ncell_z_;

  this->cells2dev();

  std::cout << std::endl;
  std::cout << "          Number of atoms: " << this->natom_ << std::endl;
  std::cout << "    System dimensions [A]: " << box_dim_x << " x " << box_dim_y
            << " x " << box_dim_z << std::endl;
  std::cout << "          Number of cells: " << this->ncell_x_ << ", "
            << this->ncell_y_ << ", " << this->ncell_z_ << std::endl;
  std::cout << "      Cell dimensions [A]: " << this->cell_dim_x_ << " x "
            << this->cell_dim_y_ << " x " << this->cell_dim_z_ << std::endl;
  std::cout << "  Total space covered [A]: "
            << static_cast<CT>(this->ncell_x_) * this->cell_dim_x_ << " x "
            << static_cast<CT>(this->ncell_y_) * this->cell_dim_y_ << " x "
            << static_cast<CT>(this->ncell_z_) * this->cell_dim_z_ << std::endl;
  std::cout << "           Number of GPUs: " << this->cuda_count_ << std::endl;

  std::vector<unsigned int> ncell_alpha_group;
  std::vector<double> rmax, alpha, zcut;
  this->init_alpha_groups(ncell_alpha_group, rmax, alpha, zcut, tol);

  std::cout << std::endl;
  std::cout << "  Number of alpha groups: " << this->ngroup_ << std::endl;

  this->cubature_ =
      std::make_unique<cubature<CT>>(tol, this->ngroup_, rmax, alpha, zcut);

  std::cout << "       Total number of cubature nodes: "
            << this->cubature_->tot_num_nodes() << std::endl;
  for (unsigned int grp = 0; grp < this->ngroup_; grp++) {
    std::cout << "  Number of cubature nodes in group " << grp << ": "
              << this->cubature_->num_nodes()[0][grp] << std::endl;
  }

  { // Distribute cubature nodes across devices
    const unsigned int size = this->cubature_->tot_num_nodes() /
                              static_cast<unsigned int>(this->cuda_count_);
    const int rmdr =
        static_cast<int>(this->cubature_->tot_num_nodes()) % this->cuda_count_;
    this->dev_cub_counts_.resize(this->cuda_count_);
    this->dev_cub_points_.resize(this->cuda_count_);
    this->dev_cub_points_[0] = 0;
    for (int dev = 0; dev < this->cuda_count_; dev++) {
      this->dev_cub_counts_[dev] = (dev < rmdr) ? size + 1 : size;
      if (dev > 0) {
        this->dev_cub_points_[dev] =
            this->dev_cub_points_[dev - 1] + this->dev_cub_counts_[dev - 1];
      }
    }
    std::cout << "  Number of cubature nodes per device: ~"
              << this->dev_cub_counts_[0] << std::endl;
  }

  // Allocate cell memory
  this->cell_atom_point_.resize(this->cuda_count_);
  this->cell_atom_count_.resize(this->cuda_count_);
  this->max_atoms_cell_.resize(this->cuda_count_);
  this->sf_re_.resize(this->cuda_count_);
  this->sf_im_.resize(this->cuda_count_);
  this->rmt_sum_re_.resize(this->cuda_count_);
  this->rmt_sum_im_.resize(this->cuda_count_);
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    this->cell_atom_point_[dev].resize(this->ncell_);
    this->cell_atom_count_[dev].resize(this->ncell_);
    this->max_atoms_cell_[dev].resize(1);
    this->sf_re_[dev].resize(this->ncell_ * this->dev_cub_counts_[dev]);
    this->sf_im_[dev].resize(this->ncell_ * this->dev_cub_counts_[dev]);
    this->rmt_sum_re_[dev].resize(this->ncell_ * this->dev_cub_counts_[dev]);
    this->rmt_sum_im_[dev].resize(this->ncell_ * this->dev_cub_counts_[dev]);
  }

  this->allocate();

  // Call NCCL collective once to avoid initialization penalty
  nccl_all_reduce_sum_ip(this->fx_, this->natom_, this->nccl_comms_,
                         this->comm_streams_, this->cuda_count_);

  // Synchronize on CUDA stream to wait for completion of NCCL operation
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamSynchronize(this->comm_streams_[dev]));
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

template <typename CT>
__global__ void copy_coords_kernel(CT *__restrict__ rx, CT *__restrict__ ry,
                                   CT *__restrict__ rz, CT *__restrict__ qc,
                                   const double *__restrict__ d_rx,
                                   const double *__restrict__ d_ry,
                                   const double *__restrict__ d_rz,
                                   const double *__restrict__ d_qc,
                                   const unsigned int natom) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < natom) {
    rx[idx] = static_cast<CT>(d_rx[idx]);
    ry[idx] = static_cast<CT>(d_ry[idx]);
    rz[idx] = static_cast<CT>(d_rz[idx]);
    qc[idx] = static_cast<CT>(d_qc[idx]);
  }
  return;
}

template <typename CT>
__global__ static void calc_cell_atom_count_kernel(
    unsigned int *__restrict__ atom_cell_idx,
    unsigned int *__restrict__ cell_atom_count, const CT *__restrict__ rx,
    const CT *__restrict__ ry, const CT *__restrict__ rz,
    const unsigned int natom, const CT cell_dim_x, const CT cell_dim_y,
    const CT cell_dim_z, const unsigned int ncell_x, const unsigned int ncell_y,
    const unsigned int ncell_z) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CT inv_cell_dim_x = static_cast<CT>(1.0) / cell_dim_x;
  const CT inv_cell_dim_y = static_cast<CT>(1.0) / cell_dim_y;
  const CT inv_cell_dim_z = static_cast<CT>(1.0) / cell_dim_z;
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

template <typename CT>
__global__ static void
pack_kernel(atom_packet<CT> *__restrict__ packet, const CT *__restrict__ rx,
            const CT *__restrict__ ry, const CT *__restrict__ rz,
            const CT *__restrict__ qc, const unsigned int *__restrict__ id,
            const unsigned int natom) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < natom)
    packet[idx] = atom_packet<CT>(id[idx], rx[idx], ry[idx], rz[idx], qc[idx]);
  return;
}

template <typename CT>
__global__ static void unpack_kernel(CT *__restrict__ rx, CT *__restrict__ ry,
                                     CT *__restrict__ rz, CT *__restrict__ qc,
                                     unsigned int *__restrict__ id,
                                     const atom_packet<CT> *__restrict__ packet,
                                     const unsigned int natom) {
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

template <typename CT>
void glst_force<CT>::assign_atoms(const double *d_rx, const double *d_ry,
                                  const double *d_rz, const double *d_qc) {
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    { // Fast reset of cell atom count array
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->ncell_ + num_threads - 1) / num_threads;
      init_cell_atom_count_kernel<<<num_blocks, num_threads>>>(
          this->cell_atom_count_[dev].d_array().data(), this->ncell_);
    }

    { // Store input double precision coordinates as private CT coordinates
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->natom_ + num_threads - 1) / num_threads;
      copy_coords_kernel<CT><<<num_blocks, num_threads>>>(
          this->rx_[dev].d_array().data(), this->ry_[dev].d_array().data(),
          this->rz_[dev].d_array().data(), this->qc_[dev].d_array().data(),
          d_rx, d_ry, d_rz, d_qc, this->natom_);
    }

    { // Determine which cell each atom is in and count how many atoms are in
      // each cell
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->natom_ + num_threads - 1) / num_threads;
      calc_cell_atom_count_kernel<CT>
          <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
              this->atom_cell_idx_[dev].d_array().data(),
              this->cell_atom_count_[dev].d_array().data(),
              this->rx_[dev].d_array().data(), this->ry_[dev].d_array().data(),
              this->rz_[dev].d_array().data(), this->natom_, this->cell_dim_x_,
              this->cell_dim_y_, this->cell_dim_z_, this->ncell_x_,
              this->ncell_y_, this->ncell_z_);
    }

    // JEG260127: Find optimial place to do this
    this->cell_atom_count_[dev].transfer_to_host();
    this->max_atoms_cell_[dev][0] = 0;
    for (unsigned int cell = 0; cell < this->ncell_; cell++) {
      this->max_atoms_cell_[dev][0] =
          (this->cell_atom_count_[dev][cell] > this->max_atoms_cell_[dev][0])
              ? this->cell_atom_count_[dev][cell]
              : this->max_atoms_cell_[dev][0];
    }

    { // Determine where each cell's atom data is stored
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->ncell_ + num_threads - 1) / num_threads;
      calc_cell_atom_point_kernel<<<num_blocks, num_threads, 0,
                                    this->comp_streams_[dev]>>>(
          this->cell_atom_point_[dev].d_array().data(),
          this->cell_atom_count_[dev].d_array().data(), this->ncell_);
    }

    { // Pack atom data
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->natom_ + num_threads - 1) / num_threads;
      pack_kernel<CT><<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
          this->packets_[dev].d_array().data(), this->rx_[dev].d_array().data(),
          this->ry_[dev].d_array().data(), this->rz_[dev].d_array().data(),
          this->qc_[dev].d_array().data(), this->idx_[dev].d_array().data(),
          this->natom_);
    }

    // JEG260211: Fix this later
    cudaCheck(cudaStreamSynchronize(this->comp_streams_[dev]));

    { // Sort atoms based on cell indices
      cub::DeviceRadixSort::SortPairs(
          this->cub_work_buffer_[dev], this->cub_work_buffer_size_[dev],
          this->atom_cell_idx_[dev].d_array().data(),
          this->atom_cell_sorted_idx_[dev].d_array().data(),
          this->packets_[dev].d_array().data(),
          this->sorted_packets_[dev].d_array().data(), this->natom_);
    }

    { // Unpack atom data
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks =
          (this->natom_ + num_threads - 1) / num_threads;
      unpack_kernel<CT>
          <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
              this->rx_[dev].d_array().data(), this->ry_[dev].d_array().data(),
              this->rz_[dev].d_array().data(), this->qc_[dev].d_array().data(),
              this->sorted_idx_[dev].d_array().data(),
              this->sorted_packets_[dev].d_array().data(), this->natom_);
    }
  }

  return;
}

// SINGLE-PRECISION
template <unsigned int ATOM_TILE>
__global__ static void
calc_sf_kernel(float *__restrict__ sf_re, float *__restrict__ sf_im,
               const float *__restrict__ cx, const float *__restrict__ cy,
               const float *__restrict__ cz, const unsigned int nc,
               const float *__restrict__ rx, const float *__restrict__ ry,
               const float *__restrict__ rz, const float *__restrict__ qc,
               const unsigned int *__restrict__ cell_atom_points,
               const unsigned int *__restrict__ cell_atom_counts,
               const unsigned int ncell) {
  __shared__ float s_cache[ATOM_TILE * 4];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = (idx < nc);

  float xc = 0.0f, yc = 0.0f, zc = 0.0f;
  if (active) {
    xc = cx[idx];
    yc = cy[idx];
    zc = cz[idx];
  }

  for (unsigned int cell = blockIdx.y; cell < ncell; cell += gridDim.y) {
    const unsigned int apnt = cell_atom_points[cell];
    const unsigned int acnt = cell_atom_counts[cell];

    float sf_re0 = 0.0f, sf_im0 = 0.0f;
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
          const float xa = s_cache[j * 4 + 0];
          const float ya = s_cache[j * 4 + 1];
          const float za = s_cache[j * 4 + 2];
          const float qa = s_cache[j * 4 + 3];

          const float theta = xc * xa + yc * ya + zc * za;
          float re = 0.0, im = 0.0;
          sincosf(theta, &im, &re);

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

// DOUBLE-PRECISION
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

template <typename CT> void glst_force<CT>::calc_sf(void) {
  if ((this->ncell_x_ < 3) && (this->ncell_y_ < 3) && (this->ncell_z_ < 3))
    return;

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    const unsigned int nc = this->dev_cub_counts_[dev];
    const unsigned int off = this->dev_cub_points_[dev];
    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(this->sf_re_[dev].d_array().data()), 0,
        this->ncell_ * nc * sizeof(CT), this->comp_streams_[dev]));
    cudaCheck(cudaMemsetAsync(
        static_cast<void *>(this->sf_im_[dev].d_array().data()), 0,
        this->ncell_ * nc * sizeof(CT), this->comp_streams_[dev]));
    constexpr dim3 num_threads(128, 1, 1);
    const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                          std::min(65535u, this->ncell_), 1);
    calc_sf_kernel<96>
        <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
            this->sf_re_[dev].d_array().data(),
            this->sf_im_[dev].d_array().data(),
            this->cubature_->x()[dev].d_array().data() + off,
            this->cubature_->y()[dev].d_array().data() + off,
            this->cubature_->z()[dev].d_array().data() + off, nc,
            this->rx_[dev].d_array().data(), this->ry_[dev].d_array().data(),
            this->rz_[dev].d_array().data(), this->qc_[dev].d_array().data(),
            this->cell_atom_point_[dev].d_array().data(),
            this->cell_atom_count_[dev].d_array().data(), this->ncell_);
  }

  return;
}

template <typename CT>
__global__ static void
calc_prefix_sum_z_kernel(CT *__restrict__ sf_re, CT *__restrict__ sf_im,
                         const unsigned int nc, const unsigned int nx,
                         const unsigned int ny, const unsigned int nz) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int x = blockIdx.y / ny;
  const unsigned int y = blockIdx.y % ny;

  if (idx >= nc)
    return;

  CT sum_re = static_cast<CT>(0.0), sum_im = static_cast<CT>(0.0);
  for (unsigned int z = 0; z < nz; z++) {
    const unsigned int cell = (x * ny + y) * nz + z;
    sum_re += sf_re[cell * nc + idx];
    sum_im += sf_im[cell * nc + idx];
    sf_re[cell * nc + idx] = sum_re;
    sf_im[cell * nc + idx] = sum_im;
  }

  return;
}

template <typename CT>
__global__ static void
calc_prefix_sum_y_kernel(CT *__restrict__ sf_re, CT *__restrict__ sf_im,
                         const unsigned int nc, const unsigned int nx,
                         const unsigned int ny, const unsigned int nz) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int x = blockIdx.y / nz;
  const unsigned int z = blockIdx.y % nz;

  if (idx >= nc)
    return;

  CT sum_re = static_cast<CT>(0.0), sum_im = static_cast<CT>(0.0);
  for (unsigned int y = 0; y < ny; y++) {
    const unsigned int cell = (x * ny + y) * nz + z;
    sum_re += sf_re[cell * nc + idx];
    sum_im += sf_im[cell * nc + idx];
    sf_re[cell * nc + idx] = sum_re;
    sf_im[cell * nc + idx] = sum_im;
  }

  return;
}

template <typename CT>
__global__ static void
calc_prefix_sum_x_kernel(CT *__restrict__ sf_re, CT *__restrict__ sf_im,
                         const unsigned int nc, const unsigned int nx,
                         const unsigned int ny, const unsigned int nz) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y / nz;
  const unsigned int z = blockIdx.y % nz;

  if (idx >= nc)
    return;

  CT sum_re = static_cast<CT>(0.0), sum_im = static_cast<CT>(0.0);
  for (unsigned int x = 0; x < nx; x++) {
    const unsigned int cell = (x * ny + y) * nz + z;
    sum_re += sf_re[cell * nc + idx];
    sum_im += sf_im[cell * nc + idx];
    sf_re[cell * nc + idx] = sum_re;
    sf_im[cell * nc + idx] = sum_im;
  }

  return;
}

template <typename CT>
__device__ void box_sum(CT &box_re, CT &box_im, const CT *__restrict__ P_re,
                        const CT *__restrict__ P_im, const unsigned int x0,
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

  const CT g0_re = P_re[cell0 * nc + idx]; // Include
  const CT g1_re =
      (xb) ? static_cast<CT>(0.0) : P_re[cell1 * nc + idx]; // Exclude
  const CT g2_re =
      (yb) ? static_cast<CT>(0.0) : P_re[cell2 * nc + idx]; // Exclude
  const CT g3_re =
      (zb) ? static_cast<CT>(0.0) : P_re[cell3 * nc + idx]; // Exclude
  const CT g4_re =
      (xb || yb) ? static_cast<CT>(0.0) : P_re[cell4 * nc + idx]; // Include
  const CT g5_re =
      (xb || zb) ? static_cast<CT>(0.0) : P_re[cell5 * nc + idx]; // Include
  const CT g6_re =
      (yb || zb) ? static_cast<CT>(0.0) : P_re[cell6 * nc + idx]; // Include
  const CT g7_re = (xb || yb || zb) ? static_cast<CT>(0.0)
                                    : P_re[cell7 * nc + idx]; // Exclude

  const CT g0_im = P_im[cell0 * nc + idx]; // Include
  const CT g1_im =
      (xb) ? static_cast<CT>(0.0) : P_im[cell1 * nc + idx]; // Exclude
  const CT g2_im =
      (yb) ? static_cast<CT>(0.0) : P_im[cell2 * nc + idx]; // Exclude
  const CT g3_im =
      (zb) ? static_cast<CT>(0.0) : P_im[cell3 * nc + idx]; // Exclude
  const CT g4_im =
      (xb || yb) ? static_cast<CT>(0.0) : P_im[cell4 * nc + idx]; // Include
  const CT g5_im =
      (xb || zb) ? static_cast<CT>(0.0) : P_im[cell5 * nc + idx]; // Include
  const CT g6_im =
      (yb || zb) ? static_cast<CT>(0.0) : P_im[cell6 * nc + idx]; // Include
  const CT g7_im = (xb || yb || zb) ? static_cast<CT>(0.0)
                                    : P_im[cell7 * nc + idx]; // Exclude

  box_re = g0_re - g1_re - g2_re - g3_re + g4_re + g5_re + g6_re - g7_re;
  box_im = g0_im - g1_im - g2_im - g3_im + g4_im + g5_im + g6_im - g7_im;

  return;
}

template <typename CT>
__device__ void cube_sum(CT &cube_re, CT &cube_im, const CT *__restrict__ P_re,
                         const CT *__restrict__ P_im, const unsigned int x,
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

  box_sum<CT>(cube_re, cube_im, P_re, P_im, x0, y0, z0, x1, y1, z1, nx, ny, nz,
              idx, nc);
  return;
}

template <typename CT>
__device__ void
shell_sum(CT &shell_re, CT &shell_im, const CT *__restrict__ P_re,
          const CT *__restrict__ P_im, const unsigned int x,
          const unsigned int y, const unsigned int z, const unsigned int inner,
          const unsigned int outer, const unsigned int nx,
          const unsigned int ny, const unsigned int nz, const unsigned int idx,
          const unsigned int nc) {
  CT osum_re = static_cast<CT>(0.0), osum_im = static_cast<CT>(0.0);
  cube_sum<CT>(osum_re, osum_im, P_re, P_im, x, y, z, outer, nx, ny, nz, idx,
               nc);
  CT isum_re = static_cast<CT>(0.0), isum_im = static_cast<CT>(0.0);
  cube_sum<CT>(isum_re, isum_im, P_re, P_im, x, y, z, inner, nx, ny, nz, idx,
               nc);
  shell_re = osum_re - isum_re;
  shell_im = osum_im - isum_im;
  return;
}

template <typename CT>
__global__ static void calc_rmt_sum_kernel(
    CT *__restrict__ rmt_sum_re, CT *__restrict__ rmt_sum_im,
    const CT *__restrict__ sf_re, CT *__restrict__ sf_im,
    const CT *__restrict__ cw, const unsigned int *__restrict__ groups,
    const unsigned int nc, const unsigned int *__restrict__ grp_r_in,
    const unsigned int *__restrict__ grp_r_out, const unsigned int nx,
    const unsigned int ny, const unsigned int nz, const unsigned int ncell) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nc)
    return;

  const CT wc = cw[idx];
  const unsigned int grp = groups[idx];

  for (unsigned int cell = blockIdx.y; cell < ncell; cell += gridDim.y) {
    const unsigned int x = cell / (ny * nz);
    const unsigned int y = (cell / nz) % ny;
    const unsigned int z = cell % nz;
    const unsigned int inner = grp_r_in[grp];
    const unsigned int outer = grp_r_out[grp];

    CT shell_re = static_cast<CT>(0.0), shell_im = static_cast<CT>(0.0);
    shell_sum<CT>(shell_re, shell_im, sf_re, sf_im, x, y, z, inner, outer, nx,
                  ny, nz, idx, nc);
    shell_re *= wc;
    shell_im *= wc;
    rmt_sum_re[cell * nc + idx] = shell_re;
    rmt_sum_im[cell * nc + idx] = shell_im;
  }

  return;
}

template <typename CT> void glst_force<CT>::sum_rmt_sf(void) {
  if ((this->ncell_x_ < 3) && (this->ncell_y_ < 3) && (this->ncell_z_ < 3))
    return;

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    const unsigned int nc = this->dev_cub_counts_[dev];
    const unsigned int off = this->dev_cub_points_[dev];
    {
      constexpr dim3 num_threads(512, 1, 1);
      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            this->ncell_x_ * this->ncell_y_, 1);
      calc_prefix_sum_z_kernel<CT>
          <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
              this->sf_re_[dev].d_array().data(),
              this->sf_im_[dev].d_array().data(), nc, this->ncell_x_,
              this->ncell_y_, this->ncell_z_);
    }

    {
      constexpr dim3 num_threads(512, 1, 1);
      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            this->ncell_x_ * this->ncell_z_, 1);
      calc_prefix_sum_y_kernel<CT>
          <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
              this->sf_re_[dev].d_array().data(),
              this->sf_im_[dev].d_array().data(), nc, this->ncell_x_,
              this->ncell_y_, this->ncell_z_);
    }

    {
      constexpr dim3 num_threads(512, 1, 1);
      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            this->ncell_y_ * this->ncell_z_, 1);
      calc_prefix_sum_x_kernel<CT>
          <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
              this->sf_re_[dev].d_array().data(),
              this->sf_im_[dev].d_array().data(), nc, this->ncell_x_,
              this->ncell_y_, this->ncell_z_);
    }

    {
      constexpr dim3 num_threads(512, 1, 1);
      const dim3 num_blocks((nc + num_threads.x - 1) / num_threads.x,
                            std::min(65535u, this->ncell_), 1);
      calc_rmt_sum_kernel<CT>
          <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
              this->rmt_sum_re_[dev].d_array().data(),
              this->rmt_sum_im_[dev].d_array().data(),
              this->sf_re_[dev].d_array().data(),
              this->sf_im_[dev].d_array().data(),
              this->cubature_->w()[dev].d_array().data() + off,
              this->cubature_->group()[dev].d_array().data() + off, nc,
              this->grp_r_in_[dev].d_array().data(),
              this->grp_r_out_[dev].d_array().data(), this->ncell_x_,
              this->ncell_y_, this->ncell_z_, this->ncell_);
    }
  }

  return;
}

// SINGLE-PRECISION
template <unsigned int BLOCK>
__global__ static void
calc_lr_ef_kernel(double *__restrict__ fx, double *__restrict__ fy,
                  double *__restrict__ fz, double *__restrict__ en,
                  const float *__restrict__ rx, const float *__restrict__ ry,
                  const float *__restrict__ rz, const float *__restrict__ qc,
                  const unsigned int *__restrict__ cell_atom_points,
                  const unsigned int *__restrict__ cell_atom_counts,
                  const float *__restrict__ cx, const float *__restrict__ cy,
                  const float *__restrict__ cz,
                  const float *__restrict__ rmt_sum_re,
                  const float *__restrict__ rmt_sum_im, const unsigned int nc,
                  const unsigned int ncell) {
  __shared__ float s_cache[BLOCK * 5];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int cell = blockIdx.y; cell < ncell; cell += gridDim.y) {
    const unsigned int apnt = cell_atom_points[cell];
    const unsigned int acnt = cell_atom_counts[cell];
    const bool active = (idx < acnt);

    float xa = 0.0f, ya = 0.0f, za = 0.0f, qa = 0.0f;
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
          const float xc = s_cache[j * 5 + 0];
          const float yc = s_cache[j * 5 + 1];
          const float zc = s_cache[j * 5 + 2];
          const float rmt_re = s_cache[j * 5 + 3];
          const float rmt_im = s_cache[j * 5 + 4];

          const float theta = xc * xa + yc * ya + zc * za;
          float re = 0.0f, im = 0.0f;
          sincosf(theta, &im, &re);

          const float dre = qa * (re * rmt_re - im * rmt_im);
          const float dim = qa * (re * rmt_im + im * rmt_re);
          fx0 += static_cast<double>(dim * xc);
          fy0 += static_cast<double>(dim * yc);
          fz0 += static_cast<double>(dim * zc);
          en0 += static_cast<double>(dre);
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

// DOUBLE-PRECISION
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

template <typename CT> void glst_force<CT>::calc_lr_ef(void) {
  if ((this->ncell_x_ < 3) && (this->ncell_y_ < 3) && (this->ncell_z_ < 3))
    return;

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    const unsigned int nc = this->dev_cub_counts_[dev];
    const unsigned int off = this->dev_cub_points_[dev];
    constexpr dim3 num_threads(64, 1, 1);
    const dim3 num_blocks((this->max_atoms_cell_[dev][0] + num_threads.x - 1) /
                              num_threads.x,
                          std::min(65535u, this->ncell_), 1);
    calc_lr_ef_kernel<num_threads.x>
        <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
            this->fx_[dev].d_array().data(), this->fy_[dev].d_array().data(),
            this->fz_[dev].d_array().data(), this->en_[dev].d_array().data(),
            this->rx_[dev].d_array().data(), this->ry_[dev].d_array().data(),
            this->rz_[dev].d_array().data(), this->qc_[dev].d_array().data(),
            this->cell_atom_point_[dev].d_array().data(),
            this->cell_atom_count_[dev].d_array().data(),
            this->cubature_->x()[dev].d_array().data() + off,
            this->cubature_->y()[dev].d_array().data() + off,
            this->cubature_->z()[dev].d_array().data() + off,
            this->rmt_sum_re_[dev].d_array().data(),
            this->rmt_sum_im_[dev].d_array().data(), nc, this->ncell_);
  }

  return;
}

// SINGLE-PRECISION
template <unsigned int BLOCK>
__global__ static void calc_sr_ef_intra_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const float *__restrict__ rx,
    const float *__restrict__ ry, const float *__restrict__ rz,
    const float *__restrict__ qc,
    const unsigned int *__restrict__ cell_atom_points,
    const unsigned int *__restrict__ cell_atom_counts,
    const unsigned int *__restrict__ cells, const unsigned int cell_count) {
  __shared__ float s_cache[BLOCK * 4];
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int c = blockIdx.y; c < cell_count; c += gridDim.y) {
    const unsigned int cell = cells[c];
    const unsigned int apnt = cell_atom_points[cell];
    const unsigned int acnt = cell_atom_counts[cell];
    const bool active = (idx < acnt);

    float xi = 0.0f, yi = 0.0f, zi = 0.0f, qi = 0.0f;
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
          const float xj = s_cache[k * 4 + 0];
          const float yj = s_cache[k * 4 + 1];
          const float zj = s_cache[k * 4 + 2];
          const float qj = s_cache[k * 4 + 3];
          const float qij = qi * qj;
          const float xij = xi - xj;
          const float yij = yi - yj;
          const float zij = zi - zj;
          const float rij2 = xij * xij + yij * yij + zij * zij;
          const float rij = sqrt(rij2);
          const float irij = 1.0 / rij;
          const float dudr = qij / rij2; // u = qij / rij
          const float drdx = xij * irij;
          const float drdy = yij * irij;
          const float drdz = zij * irij;
          fx0 += static_cast<double>(dudr * drdx);
          fy0 += static_cast<double>(dudr * drdy);
          fz0 += static_cast<double>(dudr * drdz);
          en0 += static_cast<double>(qij * irij);
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

// DOUBLE-PRECISION
template <unsigned int BLOCK>
__global__ static void calc_sr_ef_intra_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const double *__restrict__ rx,
    const double *__restrict__ ry, const double *__restrict__ rz,
    const double *__restrict__ qc,
    const unsigned int *__restrict__ cell_atom_points,
    const unsigned int *__restrict__ cell_atom_counts,
    const unsigned int *__restrict__ cells, const unsigned int cell_count) {
  __shared__ double s_cache[BLOCK * 4];
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int c = blockIdx.y; c < cell_count; c += gridDim.y) {
    const unsigned int cell = cells[c];
    const unsigned int apnt = cell_atom_points[cell];
    const unsigned int acnt = cell_atom_counts[cell];
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

// SINGLE-PRECISION
template <unsigned int BLOCK>
__global__ static void calc_sr_ef_inter_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const float *__restrict__ rx,
    const float *__restrict__ ry, const float *__restrict__ rz,
    const float *__restrict__ qc,
    const unsigned int *__restrict__ cell_atom_points,
    const unsigned int *__restrict__ cell_atom_counts,
    const unsigned int *__restrict__ cells, const unsigned int cell_count,
    const unsigned int ncell_x, const unsigned int ncell_y,
    const unsigned int ncell_z) {
  __shared__ float s_cache[BLOCK * 4];
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int c = blockIdx.y; c < cell_count; c += gridDim.y) {
    const unsigned int cell = cells[c];
    const unsigned int xcell = cell / (ncell_y * ncell_z);
    const unsigned int ycell = (cell / ncell_z) % ncell_y;
    const unsigned int zcell = cell % ncell_z;
    const unsigned int apnt = cell_atom_points[cell];
    const unsigned int acnt = cell_atom_counts[cell];
    const bool active = (idx < acnt);

    float xi = 0.0f, yi = 0.0f, zi = 0.0f, qi = 0.0f;
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
          const unsigned int bpnt = cell_atom_points[nbr];
          const unsigned int bcnt = cell_atom_counts[nbr];
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
                const float xj = s_cache[k * 4 + 0];
                const float yj = s_cache[k * 4 + 1];
                const float zj = s_cache[k * 4 + 2];
                const float qj = s_cache[k * 4 + 3];
                const float qij = qi * qj;
                const float xij = xi - xj;
                const float yij = yi - yj;
                const float zij = zi - zj;
                const float rij2 = xij * xij + yij * yij + zij * zij;
                const float rij = sqrt(rij2);
                const float irij = 1.0 / rij;
                const float dudr = qij / rij2; // u = qij / rij
                const float drdx = xij * irij;
                const float drdy = yij * irij;
                const float drdz = zij * irij;
                fx0 += static_cast<float>(dudr * drdx);
                fy0 += static_cast<float>(dudr * drdy);
                fz0 += static_cast<float>(dudr * drdz);
                en0 += static_cast<float>(qij * irij);
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

// DOUBLE-PRECISION
template <unsigned int BLOCK>
__global__ static void calc_sr_ef_inter_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const double *__restrict__ rx,
    const double *__restrict__ ry, const double *__restrict__ rz,
    const double *__restrict__ qc,
    const unsigned int *__restrict__ cell_atom_points,
    const unsigned int *__restrict__ cell_atom_counts,
    const unsigned int *__restrict__ cells, const unsigned int cell_count,
    const unsigned int ncell_x, const unsigned int ncell_y,
    const unsigned int ncell_z) {
  __shared__ double s_cache[BLOCK * 4];
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int c = blockIdx.y; c < cell_count; c += gridDim.y) {
    const unsigned int cell = cells[c];
    const unsigned int xcell = cell / (ncell_y * ncell_z);
    const unsigned int ycell = (cell / ncell_z) % ncell_y;
    const unsigned int zcell = cell % ncell_z;
    const unsigned int apnt = cell_atom_points[cell];
    const unsigned int acnt = cell_atom_counts[cell];
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
          const unsigned int bpnt = cell_atom_points[nbr];
          const unsigned int bcnt = cell_atom_counts[nbr];
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

template <typename CT> void glst_force<CT>::calc_sr_ef(void) {
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    constexpr dim3 num_threads(64, 1, 1);
    const dim3 num_blocks(
        (this->max_atoms_cell_[dev][0] + num_threads.x - 1) / num_threads.x,
        std::min(65535u,
                 static_cast<unsigned int>(this->dev_cell_idx_[dev].size())),
        1);
    calc_sr_ef_intra_kernel<num_threads.x>
        <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
            this->fx_[dev].d_array().data(), this->fy_[dev].d_array().data(),
            this->fz_[dev].d_array().data(), this->en_[dev].d_array().data(),
            this->rx_[dev].d_array().data(), this->ry_[dev].d_array().data(),
            this->rz_[dev].d_array().data(), this->qc_[dev].d_array().data(),
            this->cell_atom_point_[dev].d_array().data(),
            this->cell_atom_count_[dev].d_array().data(),
            this->dev_cell_idx_[dev].d_array().data(),
            static_cast<unsigned int>(this->dev_cell_idx_[dev].size()));
    calc_sr_ef_inter_kernel<num_threads.x>
        <<<num_blocks, num_threads, 0, this->comp_streams_[dev]>>>(
            this->fx_[dev].d_array().data(), this->fy_[dev].d_array().data(),
            this->fz_[dev].d_array().data(), this->en_[dev].d_array().data(),
            this->rx_[dev].d_array().data(), this->ry_[dev].d_array().data(),
            this->rz_[dev].d_array().data(), this->qc_[dev].d_array().data(),
            this->cell_atom_point_[dev].d_array().data(),
            this->cell_atom_count_[dev].d_array().data(),
            this->dev_cell_idx_[dev].d_array().data(),
            static_cast<unsigned int>(this->dev_cell_idx_[dev].size()),
            this->ncell_x_, this->ncell_y_, this->ncell_z_);
  }

  return;
}

template <typename CT> void glst_force<CT>::comm_ef(void) {
  // Synchronize to ensure that all devices are done computing
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamSynchronize(this->comp_streams_[dev]));
  }

  nccl_root_reduce_sum_ef_ip(this->fx_, this->fy_, this->fz_, this->en_,
                             this->natom_, this->nccl_comms_,
                             this->comm_streams_, this->cuda_count_);

  // Synchronize to ensure that all devices are done communicating
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamSynchronize(this->comm_streams_[dev]));
  }

  return;
}

template <typename CT>
void glst_force<CT>::calc_ener_force(const double *d_rx, const double *d_ry,
                                     const double *d_rz, const double *d_qc) {
  this->assign_atoms(d_rx, d_ry, d_rz, d_qc);
  this->calc_sf();
  this->sum_rmt_sf();
  this->calc_lr_ef();
  this->calc_sr_ef();
  this->comm_ef();
  return;
}

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
    x -= err / deriv; // Newton-Raphson forumla: x = x - f(x) / f'(x)
  }

  if (std::abs(err) >= tol) {
    throw std::runtime_error("FATAL ERROR: erfc_inv(const double, const "
                             "double): Inverse of erfc not found after " +
                             std::to_string(MAX_IT) + " iterations");
  }

  return x;
}

template <typename CT>
void glst_force<CT>::init_alpha_groups(
    std::vector<unsigned int> &ncell_alpha_group, std::vector<double> &rmax,
    std::vector<double> &alpha, std::vector<double> &zcut, const double tol) {
  ncell_alpha_group.clear();
  rmax.clear();
  alpha.clear();
  zcut.clear();

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
    ncell_alpha_group.push_back(static_cast<unsigned int>(width));

    double lmin = this->cell_dim_x_;
    lmin = (this->cell_dim_y_ < lmin) ? this->cell_dim_y_ : lmin;
    lmin = (this->cell_dim_z_ < lmin) ? this->cell_dim_z_ : lmin;

    double lmax = this->cell_dim_x_;
    lmax = (this->cell_dim_y_ > lmax) ? this->cell_dim_y_ : lmax;
    lmax = (this->cell_dim_z_ > lmax) ? this->cell_dim_z_ : lmax;

    const double rmin0 = lmin * static_cast<double>(tot_width + 1);
    const double rmax0 =
        std::sqrt(3.0) * (rmin0 + lmax * static_cast<double>(width + 1));
    rmax.push_back(rmax0);

    const double alpha0 =
        erfc_inv(tol * rmin0, std::numeric_limits<double>::epsilon()) / rmin0;
    alpha.push_back(alpha0);

    const double zcut0 = erfc_inv(0.5 * std::sqrt(M_PI) / alpha0 * tol,
                                  std::numeric_limits<double>::epsilon());
    zcut.push_back(zcut0);

    ncell_remain -= width;
    tot_width += width;
    if (ncell_remain < 1)
      break;

    // Determine the next alpha group width
    int width1 = 2 * width; // Width of next complete alpha group
    int width2 =
        6 * width; // Sum of the next two alpha group width (2 + 4) * width
    if (ncell_remain <= width1)
      width = ncell_remain; // The last group
    else if ((width1 < ncell_remain) && (ncell_remain < width2))
      width = ncell_remain / 2; // Two groups left
    else
      width = width1; // Keep going
  }

  this->ngroup_ = static_cast<unsigned int>(ncell_alpha_group.size());

  std::vector<unsigned int> r_in((this->ngroup_ > 0) ? this->ngroup_ : 1);
  std::vector<unsigned int> r_out((this->ngroup_ > 0) ? this->ngroup_ : 1);
  r_in[0] = 1;
  for (unsigned int group = 0; group < this->ngroup_; group++) {
    if (group > 0)
      r_in[group] = r_out[group - 1];
    r_out[group] = r_in[group] + ncell_alpha_group[group];
  }

  this->grp_r_in_.resize(this->cuda_count_);
  this->grp_r_out_.resize(this->cuda_count_);
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    this->grp_r_in_[dev] = r_in;
    this->grp_r_out_[dev] = r_out;
  }

  return;
}

template <typename CT> void glst_force<CT>::allocate(void) {
  this->cub_work_buffer_.resize(this->cuda_count_);
  this->cub_work_buffer_size_.resize(this->cuda_count_);
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    this->cub_work_buffer_[dev] = nullptr;
    this->cub_work_buffer_size_[dev] = 0;

    // Determine storage requirements for CUB functions
    std::size_t size0 = 0, size1 = 0, size2 = 0;
    cub::DeviceRadixSort::SortPairs(
        this->cub_work_buffer_[dev], size0,
        this->atom_cell_idx_[dev].d_array().data(),
        this->atom_cell_sorted_idx_[dev].d_array().data(),
        this->idx_[dev].d_array().data(),
        this->sorted_idx_[dev].d_array().data(), this->natom_);
    cub::DeviceRadixSort::SortPairs(
        this->cub_work_buffer_[dev], size1,
        this->atom_cell_idx_[dev].d_array().data(),
        this->atom_cell_sorted_idx_[dev].d_array().data(),
        this->fx_[dev].d_array().data(), this->fx_[dev].d_array().data(),
        this->natom_);
    cub::DeviceRadixSort::SortPairs(
        this->cub_work_buffer_[dev], size2,
        this->atom_cell_idx_[dev].d_array().data(),
        this->atom_cell_sorted_idx_[dev].d_array().data(),
        this->packets_[dev].d_array().data(),
        this->sorted_packets_[dev].d_array().data(), this->natom_);

    this->cub_work_buffer_size_[dev] = size0;
    if (size1 > this->cub_work_buffer_size_[dev])
      this->cub_work_buffer_size_[dev] = size1;
    if (size2 > this->cub_work_buffer_size_[dev])
      this->cub_work_buffer_size_[dev] = size2;

    cudaCheck(cudaMalloc(&(this->cub_work_buffer_[dev]),
                         this->cub_work_buffer_size_[dev]));
  }

  return;
}

template <typename CT> void glst_force<CT>::deallocate(void) {
  if (static_cast<int>(this->cub_work_buffer_.size()) == this->cuda_count_) {
    for (int dev = 0; dev < this->cuda_count_; dev++) {
      cudaCheck(cudaSetDevice(dev));
      if (this->cub_work_buffer_[dev] != nullptr) {
        cudaCheck(cudaFree(this->cub_work_buffer_[dev]));
        this->cub_work_buffer_[dev] = nullptr;
        this->cub_work_buffer_size_[dev] = 0;
      }
    }
  }

  // Finalize NCCL
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    ncclCheck(ncclCommDestroy(this->nccl_comms_[dev]));
  }

  // Destroy CUDA streams and events
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaStreamDestroy(this->comp_streams_[dev]));
    cudaCheck(cudaStreamDestroy(this->comm_streams_[dev]));
    cudaCheck(cudaEventDestroy(this->comp_events_[dev]));
    cudaCheck(cudaEventDestroy(this->comm_events_[dev]));
  }

  return;
}

template <typename CT> void glst_force<CT>::cells2dev(void) {
  /* */
  { // Assign each cell to a CUDA device
    this->cell_dev_idx_.resize(this->ncell_);
    int dev = 0;
    for (unsigned int cell = 0; cell < this->ncell_; cell++) {
      if (dev >= this->cuda_count_)
        dev = 0;
      this->cell_dev_idx_[cell] = dev++;
    }
  }

  { // Store a list of the cells each device is responsible for
    this->dev_cell_idx_.resize(this->cuda_count_);
    for (int dev = 0; dev < this->cuda_count_; dev++) {
      cudaCheck(cudaSetDevice(dev));
      std::vector<unsigned int> tmp;
      for (unsigned int cell = 0; cell < this->ncell_; cell++) {
        if (this->cell_dev_idx_[cell] == dev)
          tmp.push_back(cell);
      }
      this->dev_cell_idx_[dev] = tmp;
    }
  }
  /* */

  /* *
  { // Assign each cell to a CUDA device
    // Each device owns a continguous range of x-cell indices
    this->cell_dev_idx_.resize(this->ncell_);
    const unsigned int nx = this->ncell_x_ / this->cuda_count_;
    const unsigned int rem = this->ncell_x_ % this->cuda_count_;
    std::vector<unsigned int> xpoint(this->cuda_count_, 0);
    std::vector<unsigned int> xcount(this->cuda_count_, 0);
    for (int dev = 0; dev < this->cuda_count_; dev++) {
      if (dev > 0)
        xpoint[dev] = xpoint[dev - 1] + xcount[dev - 1];
      xcount[dev] = (dev < static_cast<int>(rem)) ? nx + 1 : nx;
    }
    for (int dev = 0; dev < this->cuda_count_; dev++) {
      for (unsigned int x0 = 0; x0 < xcount[dev]; x0++) {
        const unsigned int x = xpoint[dev] + x0;
        for (unsigned int y = 0; y < this->ncell_y_; y++) {
          for (unsigned int z = 0; z < this->ncell_z_; z++) {
            const unsigned int cell =
                (x * this->ncell_y_ + y) * this->ncell_z_ + z;
            this->cell_dev_idx_[cell] = dev;
          }
        }
      }
    }
  }

  { // Store a list of the cells each device is responsible for
    this->dev_cell_idx_.resize(this->cuda_count_);
    for (int dev = 0; dev < this->cuda_count_; dev++) {
      cudaCheck(cudaSetDevice(dev));
      std::vector<unsigned int> tmp;
      for (unsigned int cell = 0; cell < this->ncell_; cell++) {
        if (this->cell_dev_idx_[cell] == dev)
          tmp.push_back(cell);
      }
      this->dev_cell_idx_[dev] = tmp;
    }
  }
  * */

  return;
}
