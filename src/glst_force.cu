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
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

glst_force::glst_force(void)
    : natom_(0), idx_(), sorted_idx_(), rx_(), ry_(), rz_(), qc_(), packets_(),
      sorted_packets_(), fx_(), fy_(), fz_(), en_(), atom_cell_idx_(),
      atom_cell_sorted_idx_(), ncell_x_(0), ncell_y_(0), ncell_z_(0), ncell_(0),
      cell_dim_x_(0.0), cell_dim_y_(0.0), cell_dim_z_(0.0), ngroup_(0),
      dir_nghbr_point_(), dir_nghbr_count_(), dir_nghbr_list_(),
      rmt_nghbr_point_(), rmt_nghbr_count_(), rmt_nghbr_group_(),
      rmt_nghbr_list_(), cubature_(nullptr), cell_atom_point_(),
      cell_atom_count_(), sf_re_(), sf_im_(), rmt_sum_re_(), rmt_sum_im_(),
      cell_stream_(), cub_work_buffer_(), cub_work_buffer_size_(),
      cuda_count_(-1), cell_dev_idx_() {
  cudaCheck(cudaGetDeviceCount(&this->cuda_count_));
  if (this->cuda_count_ < 1) {
    throw std::runtime_error(
        "glst_force::init: Could not find any CUDA capable devices");
  }
  enable_p2p(this->cuda_count_);
}

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

glst_force::~glst_force(void) {
  this->deallocate();
  disable_p2p(this->cuda_count_);
}

const std::vector<cuda_container<double>> &glst_force::fx(void) const {
  return this->fx_;
}

const std::vector<cuda_container<double>> &glst_force::fy(void) const {
  return this->fy_;
}

const std::vector<cuda_container<double>> &glst_force::fz(void) const {
  return this->fz_;
}

const std::vector<cuda_container<double>> &glst_force::en(void) const {
  return this->en_;
}

std::vector<cuda_container<double>> &glst_force::fx(void) { return this->fx_; }

std::vector<cuda_container<double>> &glst_force::fy(void) { return this->fy_; }

std::vector<cuda_container<double>> &glst_force::fz(void) { return this->fz_; }

std::vector<cuda_container<double>> &glst_force::en(void) { return this->en_; }

void glst_force::get_ef(cuda_container<double> &fx, cuda_container<double> &fy,
                        cuda_container<double> &fz,
                        cuda_container<double> &en) {
  fx.resize(this->natom_);
  fy.resize(this->natom_);
  fz.resize(this->natom_);
  en.resize(this->natom_);

  fx.set(0);
  fy.set(0);
  fz.set(0);
  en.set(0);

  std::vector<cuda_container<double>> tmp(this->cuda_count_);
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    tmp[dev].resize(this->natom_);
  }

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    tmp[dev].set(0);
    cub::DeviceRadixSort::SortPairs(
        this->cub_work_buffer_[dev], this->cub_work_buffer_size_[dev],
        this->sorted_idx_[dev].d_array().data(),
        this->idx_[dev].d_array().data(), this->fx_[dev].d_array().data(),
        tmp[dev].d_array().data(), this->natom_);
    tmp[dev].transfer_to_host();
    for (unsigned int i = 0; i < this->natom_; i++)
      fx[i] += tmp[dev][i];

    tmp[dev].set(0);
    cub::DeviceRadixSort::SortPairs(
        this->cub_work_buffer_[dev], this->cub_work_buffer_size_[dev],
        this->sorted_idx_[dev].d_array().data(),
        this->idx_[dev].d_array().data(), this->fy_[dev].d_array().data(),
        tmp[dev].d_array().data(), this->natom_);
    tmp[dev].transfer_to_host();
    for (unsigned int i = 0; i < this->natom_; i++)
      fy[i] += tmp[dev][i];

    tmp[dev].set(0);
    cub::DeviceRadixSort::SortPairs(
        this->cub_work_buffer_[dev], this->cub_work_buffer_size_[dev],
        this->sorted_idx_[dev].d_array().data(),
        this->idx_[dev].d_array().data(), this->fz_[dev].d_array().data(),
        tmp[dev].d_array().data(), this->natom_);
    tmp[dev].transfer_to_host();
    for (unsigned int i = 0; i < this->natom_; i++)
      fz[i] += tmp[dev][i];

    tmp[dev].set(0);
    cub::DeviceRadixSort::SortPairs(
        this->cub_work_buffer_[dev], this->cub_work_buffer_size_[dev],
        this->sorted_idx_[dev].d_array().data(),
        this->idx_[dev].d_array().data(), this->en_[dev].d_array().data(),
        tmp[dev].d_array().data(), this->natom_);
    tmp[dev].transfer_to_host();
    for (unsigned int i = 0; i < this->natom_; i++)
      en[i] += tmp[dev][i];
  }

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

  { // Assign each cell to a CUDA device
    this->cell_dev_idx_.resize(this->ncell_);
    int dev = 0;
    for (unsigned int cell = 0; cell < this->ncell_; cell++) {
      if (dev >= this->cuda_count_)
        dev = 0;
      this->cell_dev_idx_[cell] = dev++;
    }
  }

  std::cout << std::endl;
  std::cout << "          Number of atoms: " << this->natom_ << std::endl;
  std::cout << "    System dimensions [A]: " << box_dim_x << " x " << box_dim_y
            << " x " << box_dim_z << std::endl;
  std::cout << "          Number of cells: " << this->ncell_x_ << ", "
            << this->ncell_y_ << ", " << this->ncell_z_ << std::endl;
  std::cout << "      Cell dimensions [A]: " << this->cell_dim_x_ << " x "
            << this->cell_dim_y_ << " x " << this->cell_dim_z_ << std::endl;
  std::cout << "  Total space covered [A]: "
            << static_cast<double>(this->ncell_x_) * this->cell_dim_x_ << " x "
            << static_cast<double>(this->ncell_y_) * this->cell_dim_y_ << " x "
            << static_cast<double>(this->ncell_z_) * this->cell_dim_z_
            << std::endl;

  std::vector<unsigned int> ncell_alpha_group;
  std::vector<double> rmax, alpha, zcut;
  this->init_alpha_groups(ncell_alpha_group, rmax, alpha, zcut, tol);

  std::cout << std::endl;
  std::cout << "  Number of alpha groups: " << this->ngroup_ << std::endl;

  this->init_cell_nghbr_lists(ncell_alpha_group);

  this->cubature_ =
      std::make_unique<cubature>(tol, this->ngroup_, rmax, alpha, zcut);

  std::cout << "       Total number of cubature nodes: "
            << this->cubature_->tot_num_nodes() << std::endl;
  for (unsigned int grp = 0; grp < this->ngroup_; grp++) {
    std::cout << "  Number of cubature nodes in group " << grp << ": "
              << this->cubature_->num_nodes()[0][grp] << std::endl;
  }

  // Allocate cell memory
  this->cell_atom_point_.resize(this->cuda_count_);
  this->cell_atom_count_.resize(this->cuda_count_);
  this->sf_re_.resize(this->cuda_count_);
  this->sf_im_.resize(this->cuda_count_);
  this->rmt_sum_re_.resize(this->cuda_count_);
  this->rmt_sum_im_.resize(this->cuda_count_);
  this->cell_stream_.resize(this->cuda_count_);
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    this->cell_atom_point_[dev].resize(this->ncell_);
    this->cell_atom_count_[dev].resize(this->ncell_);
    this->sf_re_[dev].resize(this->ncell_ * this->cubature_->tot_num_nodes());
    this->sf_im_[dev].resize(this->ncell_ * this->cubature_->tot_num_nodes());
    this->rmt_sum_re_[dev].resize(this->ncell_ *
                                  this->cubature_->tot_num_nodes());
    this->rmt_sum_im_[dev].resize(this->ncell_ *
                                  this->cubature_->tot_num_nodes());
    this->cell_stream_[dev].resize(this->ncell_);
    for (unsigned int cell = 0; cell < this->ncell_; cell++) {
      cudaCheck(cudaStreamCreateWithFlags(&(this->cell_stream_[dev][cell]),
                                          cudaStreamNonBlocking));
    }
  }

  this->allocate();

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
            const unsigned int natom) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < natom)
    packet[idx] = atom_packet(id[idx], rx[idx], ry[idx], rz[idx], qc[idx]);
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

void glst_force::assign_atoms(const double *d_rx, const double *d_ry,
                              const double *d_rz, const double *d_qc) {
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    { // Fast reset of cell atom count array
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks = this->ncell_ / num_threads + 1;
      init_cell_atom_count_kernel<<<num_blocks, num_threads>>>(
          this->cell_atom_count_[dev].d_array().data(), this->ncell_);
    }

    { // Determine which cell each atom is in and count how many atoms are in
      // each cell
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks = this->natom_ / num_threads + 1;
      calc_cell_atom_count_kernel<<<num_blocks, num_threads>>>(
          this->atom_cell_idx_[dev].d_array().data(),
          this->cell_atom_count_[dev].d_array().data(), d_rx, d_ry, d_rz,
          this->natom_, this->cell_dim_x_, this->cell_dim_y_, this->cell_dim_z_,
          this->ncell_x_, this->ncell_y_, this->ncell_z_);
      cudaCheck(cudaMemcpyAsync(
          static_cast<void *>(this->cell_atom_count_[dev].h_array().data()),
          static_cast<const void *>(
              this->cell_atom_count_[dev].d_array().data()),
          this->ncell_ * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    { // Determine where each cell's atom data is stored
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks = this->ncell_ / num_threads + 1;
      calc_cell_atom_point_kernel<<<num_blocks, num_threads>>>(
          this->cell_atom_point_[dev].d_array().data(),
          this->cell_atom_count_[dev].d_array().data(), this->ncell_);
      cudaCheck(cudaMemcpyAsync(
          static_cast<void *>(this->cell_atom_point_[dev].h_array().data()),
          static_cast<const void *>(
              this->cell_atom_point_[dev].d_array().data()),
          this->ncell_ * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    { // Pack atom data
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks = this->natom_ / num_threads + 1;
      pack_kernel<<<num_blocks, num_threads>>>(
          this->packets_[dev].d_array().data(), d_rx, d_ry, d_rz, d_qc,
          this->idx_[dev].d_array().data(), this->natom_);
    }

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
      const unsigned int num_blocks = this->natom_ / num_threads + 1;
      unpack_kernel<<<num_blocks, num_threads>>>(
          this->rx_[dev].d_array().data(), this->ry_[dev].d_array().data(),
          this->rz_[dev].d_array().data(), this->qc_[dev].d_array().data(),
          this->sorted_idx_[dev].d_array().data(),
          this->sorted_packets_[dev].d_array().data(), this->natom_);
    }
  }

  return;
}

template <unsigned int num_threads>
__global__ static void
calc_sf_kernel(double *__restrict__ sf_re, double *__restrict__ sf_im,
               const double *__restrict__ cx, const double *__restrict__ cy,
               const double *__restrict__ cz, const unsigned int nc,
               const double *__restrict__ rx, const double *__restrict__ ry,
               const double *__restrict__ rz, const double *__restrict__ qc,
               const unsigned int natom) {
  __shared__ double s_cache[num_threads * 4];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double xc = 0.0, yc = 0.0, zc = 0.0;
  if (idx < nc) {
    xc = cx[idx];
    yc = cy[idx];
    zc = cz[idx];
  }

  double sf_re0 = 0.0, sf_im0 = 0.0;
  for (unsigned int i = 0; i < natom; i += num_threads) {
    // Each thread reads atom data from global to local memory
    __syncthreads();
    s_cache[threadIdx.x * 4 + 0] = 0.0;
    s_cache[threadIdx.x * 4 + 1] = 0.0;
    s_cache[threadIdx.x * 4 + 2] = 0.0;
    s_cache[threadIdx.x * 4 + 3] = 0.0;
    if (i + threadIdx.x < natom) {
      s_cache[threadIdx.x * 4 + 0] = rx[i + threadIdx.x];
      s_cache[threadIdx.x * 4 + 1] = ry[i + threadIdx.x];
      s_cache[threadIdx.x * 4 + 2] = rz[i + threadIdx.x];
      s_cache[threadIdx.x * 4 + 3] = qc[i + threadIdx.x];
    }
    __syncthreads();

    for (unsigned int j = 0; j < num_threads; j++) {
      const double xa = s_cache[j * 4 + 0];
      const double ya = s_cache[j * 4 + 1];
      const double za = s_cache[j * 4 + 2];
      const double qa = s_cache[j * 4 + 3];

      // const double theta = xc * xa + yc * ya + zc * za;
      // double re = 0.0, im = 0.0;
      // sincos(theta, &im, &re);

      const float theta = static_cast<float>(xc * xa + yc * ya + zc * za);
      float ref = 0.0f, imf = 0.0f;
      sincosf(theta, &imf, &ref);
      const double re = static_cast<double>(ref);
      const double im = static_cast<double>(imf);

      sf_re0 += qa * re;
      sf_im0 -= qa * im;
    }
  }

  if (idx < nc) {
    sf_re[idx] = sf_re0;
    sf_im[idx] = sf_im0;
  }

  return;
}

void glst_force::calc_sf(void) {
  if ((this->ncell_x_ < 3) && (this->ncell_y_ < 3) && (this->ncell_z_ < 3))
    return;

  for (unsigned int cell = 0; cell < this->ncell_; cell++) {
    const int dev = this->cell_dev_idx_[cell];
    cudaCheck(cudaSetDevice(dev));
    const unsigned int point = this->cell_atom_point_[dev][cell];
    const unsigned int count = this->cell_atom_count_[dev][cell];
    const unsigned int nc = this->cubature_->tot_num_nodes();
    constexpr unsigned int num_threads = 32;
    const int num_blocks = nc / num_threads + 1;
    calc_sf_kernel<num_threads>
        <<<num_blocks, num_threads, 0, this->cell_stream_[dev][cell]>>>(
            this->sf_re_[dev].d_array().data() + cell * nc,
            this->sf_im_[dev].d_array().data() + cell * nc,
            this->cubature_->nodes_x()[dev].d_array().data(),
            this->cubature_->nodes_y()[dev].d_array().data(),
            this->cubature_->nodes_z()[dev].d_array().data(), nc,
            this->rx_[dev].d_array().data() + point,
            this->ry_[dev].d_array().data() + point,
            this->rz_[dev].d_array().data() + point,
            this->qc_[dev].d_array().data() + point, count);
  }

  return;
}

__global__ static void copy_kernel(double *__restrict__ r1,
                                   double *__restrict__ i1,
                                   const double *__restrict__ r2,
                                   const double *__restrict__ i2,
                                   const unsigned int n) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    r1[idx] = r2[idx];
    i1[idx] = i2[idx];
  }
  return;
}

__global__ static void sum_rmt_sf_kernel(
    double *__restrict__ rmt_sum_re, double *__restrict__ rmt_sum_im,
    const double *__restrict__ sf_re, const double *__restrict__ sf_im,
    const unsigned int nc, const unsigned int *__restrict__ cub_points,
    const unsigned int *__restrict__ cub_counts,
    const unsigned int rmt_nbr_count,
    const unsigned int *__restrict__ rmt_nbr_grp,
    const unsigned int *__restrict__ rmt_nbr_list) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double sum_re = 0.0, sum_im = 0.0;

  for (unsigned int nbr = 0; nbr < rmt_nbr_count; nbr++) {
    const unsigned int grp = rmt_nbr_grp[nbr];
    const unsigned int cell = rmt_nbr_list[nbr];
    const unsigned int cub_point = cub_points[grp];
    const unsigned int cub_count = cub_counts[grp];

    if ((idx >= cub_point) && (idx < cub_point + cub_count)) {
      sum_re += sf_re[cell * nc + idx];
      sum_im += sf_im[cell * nc + idx];
    }
  }

  if (idx < nc) {
    rmt_sum_re[idx] = sum_re;
    rmt_sum_im[idx] = sum_im;
  }

  return;
}

void glst_force::sum_rmt_sf(void) {
  if ((this->ncell_x_ < 3) && (this->ncell_y_ < 3) && (this->ncell_z_ < 3))
    return;

  const unsigned int nc = this->cubature_->tot_num_nodes();

  for (unsigned int cell = 0; cell < this->ncell_; cell++) {
    const int idev = this->cell_dev_idx_[cell];
    cudaCheck(cudaSetDevice(idev));

    // Avoid race condition
    cudaCheck(cudaStreamSynchronize(this->cell_stream_[idev][cell]));

    for (int jdev = 0; jdev < this->cuda_count_; jdev++) {
      if (idev == jdev)
        continue;
      cudaCheck(cudaSetDevice(jdev));
      constexpr unsigned int num_threads = 512;
      const unsigned int num_blocks = nc / num_threads + 1;
      copy_kernel<<<num_blocks, num_threads>>>(
          this->sf_re_[jdev].d_array().data() + cell * nc,
          this->sf_im_[jdev].d_array().data() + cell * nc,
          this->sf_re_[idev].d_array().data() + cell * nc,
          this->sf_im_[idev].d_array().data() + cell * nc, nc);
    }
  }

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaDeviceSynchronize());
  }

  for (unsigned int cell = 0; cell < this->ncell_; cell++) {
    const int dev = this->cell_dev_idx_[cell];
    cudaCheck(cudaSetDevice(dev));
    constexpr unsigned int num_threads = 512;
    const unsigned int num_blocks = nc / num_threads + 1;
    sum_rmt_sf_kernel<<<num_blocks, num_threads, 0,
                        this->cell_stream_[dev][cell]>>>(
        this->rmt_sum_re_[dev].d_array().data() + cell * nc,
        this->rmt_sum_im_[dev].d_array().data() + cell * nc,
        this->sf_re_[dev].d_array().data(), this->sf_im_[dev].d_array().data(),
        nc, this->cubature_->points()[dev].d_array().data(),
        this->cubature_->num_nodes()[dev].d_array().data(),
        this->rmt_nghbr_count_[dev][cell],
        this->rmt_nghbr_group_[dev].d_array().data() +
            this->rmt_nghbr_point_[dev][cell],
        this->rmt_nghbr_list_[dev].d_array().data() +
            this->rmt_nghbr_point_[dev][cell]);
  }

  return;
}

template <unsigned int num_threads>
__global__ static void
calc_lr_ef_kernel(double *__restrict__ fx, double *__restrict__ fy,
                  double *__restrict__ fz, double *__restrict__ en,
                  const double *__restrict__ rx, const double *__restrict__ ry,
                  const double *__restrict__ rz, const double *__restrict__ qc,
                  const unsigned int natom, const double *__restrict__ cx,
                  const double *__restrict__ cy, const double *__restrict__ cz,
                  const double *__restrict__ cw, const unsigned int nc,
                  const double *__restrict__ rmt_sum_re,
                  const double *__restrict__ rmt_sum_im) {
  __shared__ double s_cache[num_threads * 6];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double xa = 0.0, ya = 0.0, za = 0.0, qa = 0.0;
  if (idx < natom) {
    xa = rx[idx];
    ya = ry[idx];
    za = rz[idx];
    qa = qc[idx];
  }

  double fx0 = 0.0, fy0 = 0.0, fz0 = 0.0, en0 = 0.0;
  for (unsigned int i = 0; i < nc; i += num_threads) {
    // Each thread reads cubature data from global to shared memory
    __syncthreads();
    s_cache[threadIdx.x * 6 + 0] = 0.0;
    s_cache[threadIdx.x * 6 + 1] = 0.0;
    s_cache[threadIdx.x * 6 + 2] = 0.0;
    s_cache[threadIdx.x * 6 + 3] = 0.0;
    s_cache[threadIdx.x * 6 + 4] = 0.0;
    s_cache[threadIdx.x * 6 + 5] = 0.0;
    if (i + threadIdx.x < nc) {
      s_cache[threadIdx.x * 6 + 0] = cx[i + threadIdx.x];
      s_cache[threadIdx.x * 6 + 1] = cy[i + threadIdx.x];
      s_cache[threadIdx.x * 6 + 2] = cz[i + threadIdx.x];
      s_cache[threadIdx.x * 6 + 3] = cw[i + threadIdx.x];
      s_cache[threadIdx.x * 6 + 4] = rmt_sum_re[i + threadIdx.x];
      s_cache[threadIdx.x * 6 + 5] = rmt_sum_im[i + threadIdx.x];
    }
    __syncthreads();

    for (unsigned int j = 0; j < num_threads; j++) {
      const double xc = s_cache[j * 6 + 0];
      const double yc = s_cache[j * 6 + 1];
      const double zc = s_cache[j * 6 + 2];
      const double wc = s_cache[j * 6 + 3];
      const double rmt_re = s_cache[j * 6 + 4];
      const double rmt_im = s_cache[j * 6 + 5];

      // const double theta = xc * xa + yc * ya + zc * za;
      // double re = 0.0, im = 0.0;
      // sincos(theta, &im, &re);

      const float theta = static_cast<float>(xc * xa + yc * ya + zc * za);
      float ref = 0.0f, imf = 0.0f;
      sincosf(theta, &imf, &ref);
      double re = static_cast<double>(ref);
      double im = static_cast<double>(imf);

      const double dre = qa * wc * (re * rmt_re - im * rmt_im);
      const double dim = qa * wc * (re * rmt_im + im * rmt_re);
      fx0 += dim * xc;
      fy0 += dim * yc;
      fz0 += dim * zc;
      en0 += dre;
    }
  }

  if (idx < natom) {
    fx[idx] += fx0;
    fy[idx] += fy0;
    fz[idx] += fz0;
    en[idx] += en0;
  }

  return;
}

void glst_force::calc_lr_ef(void) {
  if ((this->ncell_x_ < 3) && (this->ncell_y_ < 3) && (this->ncell_z_ < 3))
    return;

  for (unsigned int cell = 0; cell < this->ncell_; cell++) {
    const int dev = this->cell_dev_idx_[cell];
    cudaCheck(cudaSetDevice(dev));
    const unsigned int point = this->cell_atom_point_[dev][cell];
    const unsigned int count = this->cell_atom_count_[dev][cell];
    const unsigned int nc = this->cubature_->tot_num_nodes();
    if (count == 0)
      continue;

    constexpr unsigned int num_threads = 256;
    const unsigned int num_blocks = count / num_threads + 1;
    calc_lr_ef_kernel<num_threads>
        <<<num_blocks, num_threads, 0, this->cell_stream_[dev][cell]>>>(
            this->fx_[dev].d_array().data() + point,
            this->fy_[dev].d_array().data() + point,
            this->fz_[dev].d_array().data() + point,
            this->en_[dev].d_array().data() + point,
            this->rx_[dev].d_array().data() + point,
            this->ry_[dev].d_array().data() + point,
            this->rz_[dev].d_array().data() + point,
            this->qc_[dev].d_array().data() + point, count,
            this->cubature_->nodes_x()[dev].d_array().data(),
            this->cubature_->nodes_y()[dev].d_array().data(),
            this->cubature_->nodes_z()[dev].d_array().data(),
            this->cubature_->weights()[dev].d_array().data(), nc,
            this->rmt_sum_re_[dev].d_array().data() + cell * nc,
            this->rmt_sum_im_[dev].d_array().data() + cell * nc);
  }

  return;
}

template <unsigned int num_threads>
__global__ static void calc_sr_ef_intra_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const double *__restrict__ rx,
    const double *__restrict__ ry, const double *__restrict__ rz,
    const double *__restrict__ qc, const unsigned int natom) {
  __shared__ double s_cache[num_threads * 4];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double xi = 0.0, yi = 0.0, zi = 0.0, qi = 0.0;
  if (idx < natom) {
    xi = rx[idx];
    yi = ry[idx];
    zi = rz[idx];
    qi = qc[idx];
  }

  double fx0 = 0.0, fy0 = 0.0, fz0 = 0.0, en0 = 0.0;
  for (unsigned int j = 0; j < natom; j += num_threads) {
    // Read block of atom data into shared memory
    __syncthreads();
    s_cache[threadIdx.x * 4 + 0] = 0.0;
    s_cache[threadIdx.x * 4 + 1] = 0.0;
    s_cache[threadIdx.x * 4 + 2] = 0.0;
    s_cache[threadIdx.x * 4 + 3] = 0.0;
    if (j + threadIdx.x < natom) {
      s_cache[threadIdx.x * 4 + 0] = rx[j + threadIdx.x];
      s_cache[threadIdx.x * 4 + 1] = ry[j + threadIdx.x];
      s_cache[threadIdx.x * 4 + 2] = rz[j + threadIdx.x];
      s_cache[threadIdx.x * 4 + 3] = qc[j + threadIdx.x];
    }
    __syncthreads();

    // Process block of atom data
    for (unsigned int k = 0; k < num_threads; k++) {
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

  if (idx < natom) {
    fx[idx] += fx0;
    fy[idx] += fy0;
    fz[idx] += fz0;
    en[idx] += en0;
  }

  return;
}

template <unsigned int num_threads>
__global__ static void calc_sr_ef_inter_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const double *__restrict__ rx1,
    const double *__restrict__ ry1, const double *__restrict__ rz1,
    const double *__restrict__ qc1, const unsigned int natom1,
    const double *__restrict__ rx2, const double *__restrict__ ry2,
    const double *__restrict__ rz2, const double *__restrict__ qc2,
    const unsigned int natom2) {
  __shared__ double s_cache[num_threads * 4];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double xi = 0.0, yi = 0.0, zi = 0.0, qi = 0.0;
  if (idx < natom1) {
    xi = rx1[idx];
    yi = ry1[idx];
    zi = rz1[idx];
    qi = qc1[idx];
  }

  double fx0 = 0.0, fy0 = 0.0, fz0 = 0.0, en0 = 0.0;
  for (unsigned int j = 0; j < natom2; j += num_threads) {
    // Read block of atom data into shared memory
    __syncthreads();
    s_cache[threadIdx.x * 4 + 0] = 0.0;
    s_cache[threadIdx.x * 4 + 1] = 0.0;
    s_cache[threadIdx.x * 4 + 2] = 0.0;
    s_cache[threadIdx.x * 4 + 3] = 0.0;
    if (j + threadIdx.x < natom2) {
      s_cache[threadIdx.x * 4 + 0] = rx2[j + threadIdx.x];
      s_cache[threadIdx.x * 4 + 1] = ry2[j + threadIdx.x];
      s_cache[threadIdx.x * 4 + 2] = rz2[j + threadIdx.x];
      s_cache[threadIdx.x * 4 + 3] = qc2[j + threadIdx.x];
    }
    __syncthreads();

    // Process block of atom data
    for (unsigned int k = 0; k < num_threads; k++) {
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

  if (idx < natom1) {
    fx[idx] += fx0;
    fy[idx] += fy0;
    fz[idx] += fz0;
    en[idx] += en0;
  }

  return;
}

void glst_force::calc_sr_ef(void) {
  for (unsigned int icell = 0; icell < this->ncell_; icell++) {
    const int idev = this->cell_dev_idx_[icell];
    cudaCheck(cudaSetDevice(idev));
    const unsigned int ipoint = this->cell_atom_point_[idev][icell];
    const unsigned int icount = this->cell_atom_count_[idev][icell];
    if (icount == 0)
      continue;

    constexpr unsigned int num_threads = 32;
    const unsigned int num_blocks = icount / num_threads + 1;
    calc_sr_ef_intra_kernel<num_threads>
        <<<num_blocks, num_threads, 0, this->cell_stream_[idev][icell]>>>(
            this->fx_[idev].d_array().data() + ipoint,
            this->fy_[idev].d_array().data() + ipoint,
            this->fz_[idev].d_array().data() + ipoint,
            this->en_[idev].d_array().data() + ipoint,
            this->rx_[idev].d_array().data() + ipoint,
            this->ry_[idev].d_array().data() + ipoint,
            this->rz_[idev].d_array().data() + ipoint,
            this->qc_[idev].d_array().data() + ipoint, icount);

    const unsigned int dir_nghbr_point = this->dir_nghbr_point_[idev][icell];
    const unsigned int dir_nghbr_count = this->dir_nghbr_count_[idev][icell];
    const unsigned int *dir_nghbr_list =
        this->dir_nghbr_list_[idev].h_array().data() + dir_nghbr_point;
    for (unsigned int nghbr = 0; nghbr < dir_nghbr_count; nghbr++) {
      const unsigned int jcell = dir_nghbr_list[nghbr];
      const unsigned int jpoint = this->cell_atom_point_[idev][jcell];
      const unsigned int jcount = this->cell_atom_count_[idev][jcell];
      if (jcount == 0)
        continue;

      calc_sr_ef_inter_kernel<num_threads>
          <<<num_blocks, num_threads, 0, this->cell_stream_[idev][icell]>>>(
              this->fx_[idev].d_array().data() + ipoint,
              this->fy_[idev].d_array().data() + ipoint,
              this->fz_[idev].d_array().data() + ipoint,
              this->en_[idev].d_array().data() + ipoint,
              this->rx_[idev].d_array().data() + ipoint,
              this->ry_[idev].d_array().data() + ipoint,
              this->rz_[idev].d_array().data() + ipoint,
              this->qc_[idev].d_array().data() + ipoint, icount,
              this->rx_[idev].d_array().data() + jpoint,
              this->ry_[idev].d_array().data() + jpoint,
              this->rz_[idev].d_array().data() + jpoint,
              this->qc_[idev].d_array().data() + jpoint, jcount);
    }
  }

  return;
}

void glst_force::calc_ener_force(const double *d_rx, const double *d_ry,
                                 const double *d_rz, const double *d_qc) {
  this->assign_atoms(d_rx, d_ry, d_rz, d_qc);
  this->calc_sf();
  this->sum_rmt_sf();
  this->calc_lr_ef();
  this->calc_sr_ef();
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

void glst_force::init_alpha_groups(std::vector<unsigned int> &ncell_alpha_group,
                                   std::vector<double> &rmax,
                                   std::vector<double> &alpha,
                                   std::vector<double> &zcut,
                                   const double tol) {
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

  return;
}

void glst_force::init_cell_nghbr_lists(
    const std::vector<unsigned int> &ncell_alpha_group) {
  std::vector<unsigned int> dir_nghbr_point(this->ncell_, 0);
  std::vector<unsigned int> dir_nghbr_count(this->ncell_, 0);
  std::vector<unsigned int> dir_nghbr_list;
  std::vector<unsigned int> rmt_nghbr_point(this->ncell_, 0);
  std::vector<unsigned int> rmt_nghbr_count(this->ncell_, 0);
  std::vector<unsigned int> rmt_nghbr_group;
  std::vector<unsigned int> rmt_nghbr_list;

  for (int cx = 0; cx < static_cast<int>(this->ncell_x_); cx++) {
    for (int cy = 0; cy < static_cast<int>(this->ncell_y_); cy++) {
      for (int cz = 0; cz < static_cast<int>(this->ncell_z_); cz++) {
        const unsigned int cell =
            (static_cast<unsigned int>(cx) * this->ncell_y_ +
             static_cast<unsigned int>(cy)) *
                this->ncell_z_ +
            static_cast<unsigned int>(cz);

        // Generate direct neighbor lists
        int width0 = 1, width1 = 0;
        for (int sx = -width0; sx <= width0; sx++) {
          const int nx = cx + sx;
          if ((nx < 0) || (nx >= static_cast<int>(this->ncell_x_)))
            continue;
          for (int sy = -width0; sy <= width0; sy++) {
            const int ny = cy + sy;
            if ((ny < 0) || (ny >= static_cast<int>(this->ncell_y_)))
              continue;
            for (int sz = -width0; sz <= width0; sz++) {
              const int nz = cz + sz;
              if ((nz < 0) || (nz >= static_cast<int>(this->ncell_z_)))
                continue;
              if ((sx == 0) && (sy == 0) && (sz == 0)) // Skip self
                continue;
              const unsigned int nbr =
                  (static_cast<unsigned int>(nx) * this->ncell_y_ +
                   static_cast<unsigned int>(ny)) *
                      this->ncell_z_ +
                  static_cast<unsigned int>(nz);
              dir_nghbr_list.push_back(nbr);
              dir_nghbr_count[cell]++;
            }
          }
        }

        // Generate remote neighbor lists
        for (unsigned int grp = 0; grp < this->ngroup_; grp++) {
          width1 = width0;
          width0 += ncell_alpha_group[grp];
          for (int sx = -width0; sx <= width0; sx++) {
            const int nx = cx + sx;
            if ((nx < 0) || (nx >= static_cast<int>(this->ncell_x_)))
              continue;
            for (int sy = -width0; sy <= width0; sy++) {
              const int ny = cy + sy;
              if ((ny < 0) || (ny >= static_cast<int>(this->ncell_y_)))
                continue;
              for (int sz = -width0; sz <= width0; sz++) {
                const int nz = cz + sz;
                if ((nz < 0) || (nz >= static_cast<int>(this->ncell_z_)))
                  continue;
                if ((sx == 0) && (sy == 0) && (sz == 0)) // Skip self
                  continue;
                // Check if neighbor has already been assigned in previous
                // alpha group
                if ((std::abs(sx) <= width1) && (std::abs(sy) <= width1) &&
                    (std::abs(sz) <= width1))
                  continue;
                const unsigned int nbr =
                    (static_cast<unsigned int>(nx) * this->ncell_y_ +
                     static_cast<unsigned int>(ny)) *
                        this->ncell_z_ +
                    static_cast<unsigned int>(nz);
                rmt_nghbr_group.push_back(grp);
                rmt_nghbr_list.push_back(nbr);
                rmt_nghbr_count[cell]++;
              }
            }
          }
        }

        // Set starting indices for direct and remote neighbor lists
        if (cell < this->ncell_ - 1) {
          dir_nghbr_point[cell + 1] =
              dir_nghbr_point[cell] + dir_nghbr_count[cell];
          rmt_nghbr_point[cell + 1] =
              rmt_nghbr_point[cell] + rmt_nghbr_count[cell];
        }
      }
    }
  }

  this->dir_nghbr_point_.resize(this->cuda_count_);
  this->dir_nghbr_count_.resize(this->cuda_count_);
  this->dir_nghbr_list_.resize(this->cuda_count_);
  this->rmt_nghbr_point_.resize(this->cuda_count_);
  this->rmt_nghbr_count_.resize(this->cuda_count_);
  this->rmt_nghbr_group_.resize(this->cuda_count_);
  this->rmt_nghbr_list_.resize(this->cuda_count_);
  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    this->dir_nghbr_point_[dev] = dir_nghbr_point;
    this->dir_nghbr_count_[dev] = dir_nghbr_count;
    this->dir_nghbr_list_[dev] = dir_nghbr_list;
    this->rmt_nghbr_point_[dev] = rmt_nghbr_point;
    this->rmt_nghbr_count_[dev] = rmt_nghbr_count;
    this->rmt_nghbr_group_[dev] = rmt_nghbr_group;
    this->rmt_nghbr_list_[dev] = rmt_nghbr_list;
  }

  return;
}

void glst_force::allocate(void) {
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

void glst_force::deallocate(void) {
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

  for (int dev = 0; dev < this->cuda_count_; dev++) {
    cudaCheck(cudaSetDevice(dev));
    for (unsigned int cell = 0; cell < this->ncell_; cell++)
      cudaCheck(cudaStreamDestroy(this->cell_stream_[dev][cell]));
  }

  return;
}
