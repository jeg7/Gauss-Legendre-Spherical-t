// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include "coulomb.hcu"
#include <vector>

template <unsigned int BLOCK>
__global__ static void compute_coulomb_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const double *__restrict__ rx,
    const double *__restrict__ ry, const double *__restrict__ rz,
    const double *__restrict__ qc, const std::size_t point,
    const std::size_t count, const std::size_t natom) {
  __shared__ double s_cache[BLOCK * 4];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double xi = 0.0, yi = 0.0, zi = 0.0, qi = 0.0;
  if (idx < count) {
    xi = rx[point + idx];
    yi = ry[point + idx];
    zi = rz[point + idx];
    qi = qc[point + idx];
  }

  double fx0 = 0.0, fy0 = 0.0, fz0 = 0.0, en0 = 0.0;
  for (std::size_t j = 0; j < natom; j += BLOCK) {
    // Read block of atom data into shared memory
    __syncthreads();
    std::size_t jdx = j + static_cast<std::size_t>(threadIdx.x);
    if (jdx < natom) {
      s_cache[threadIdx.x * 4 + 0] = rx[jdx];
      s_cache[threadIdx.x * 4 + 1] = ry[jdx];
      s_cache[threadIdx.x * 4 + 2] = rz[jdx];
      s_cache[threadIdx.x * 4 + 3] = qc[jdx];
    }
    __syncthreads();

    // Process block of atom data
    const unsigned int n = min(BLOCK, static_cast<unsigned int>(natom - j));
    for (unsigned int k = 0; k < n; k++) {
      jdx = j + static_cast<std::size_t>(k);
      if (point + idx == jdx)
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

  if (idx < count) {
    fx[idx] = fx0;
    fy[idx] = fy0;
    fz[idx] = fz0;
    en[idx] = en0;
  }

  return;
}

void compute_coulomb_cuda(double *fx, double *fy, double *fz, double *en,
                          const double *rx, const double *ry, const double *rz,
                          const double *qc, const std::size_t natom) {
  int cuda_count = 0;
  cudaGetDeviceCount(&cuda_count);

  const std::size_t base = natom / static_cast<std::size_t>(cuda_count);
  const std::size_t rem = natom % static_cast<std::size_t>(cuda_count);
  std::vector<std::size_t> points(cuda_count, 0), counts(cuda_count, 0);
  for (int dev = 0; dev < cuda_count; dev++) {
    if (dev > 0)
      points[dev] = points[dev - 1] + counts[dev - 1];
    counts[dev] = base + ((static_cast<std::size_t>(dev) < rem) ? 1 : 0);
  }

  std::vector<double *> d_rx(cuda_count, nullptr);
  std::vector<double *> d_ry(cuda_count, nullptr);
  std::vector<double *> d_rz(cuda_count, nullptr);
  std::vector<double *> d_qc(cuda_count, nullptr);
  std::vector<double *> d_fx(cuda_count, nullptr);
  std::vector<double *> d_fy(cuda_count, nullptr);
  std::vector<double *> d_fz(cuda_count, nullptr);
  std::vector<double *> d_en(cuda_count, nullptr);
  for (int dev = 0; dev < cuda_count; dev++) {
    cudaSetDevice(dev);
    cudaMalloc(reinterpret_cast<void **>(&d_rx[dev]), natom * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d_ry[dev]), natom * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d_rz[dev]), natom * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d_qc[dev]), natom * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d_fx[dev]),
               counts[dev] * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d_fy[dev]),
               counts[dev] * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d_fz[dev]),
               counts[dev] * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d_en[dev]),
               counts[dev] * sizeof(double));
    cudaMemcpy(static_cast<void *>(d_rx[dev]), static_cast<const void *>(rx),
               natom * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(static_cast<void *>(d_ry[dev]), static_cast<const void *>(ry),
               natom * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(static_cast<void *>(d_rz[dev]), static_cast<const void *>(rz),
               natom * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(static_cast<void *>(d_qc[dev]), static_cast<const void *>(qc),
               natom * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaSetDevice(dev);
    constexpr unsigned int num_threads = 128;
    const unsigned int num_blocks =
        (counts[dev] + num_threads - 1) / num_threads;
    compute_coulomb_kernel<num_threads><<<num_blocks, num_threads>>>(
        d_fx[dev], d_fy[dev], d_fz[dev], d_en[dev], d_rx[dev], d_ry[dev],
        d_rz[dev], d_qc[dev], points[dev], counts[dev], natom);
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaSetDevice(dev);
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    cudaMemcpy(static_cast<void *>(fx + points[dev]),
               static_cast<const void *>(d_fx[dev]),
               counts[dev] * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(static_cast<void *>(fy + points[dev]),
               static_cast<const void *>(d_fy[dev]),
               counts[dev] * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(static_cast<void *>(fz + points[dev]),
               static_cast<const void *>(d_fz[dev]),
               counts[dev] * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(static_cast<void *>(en + points[dev]),
               static_cast<const void *>(d_en[dev]),
               counts[dev] * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaSetDevice(dev);
    cudaFree(static_cast<void *>(d_rx[dev]));
    cudaFree(static_cast<void *>(d_ry[dev]));
    cudaFree(static_cast<void *>(d_rz[dev]));
    cudaFree(static_cast<void *>(d_qc[dev]));
    cudaFree(static_cast<void *>(d_fx[dev]));
    cudaFree(static_cast<void *>(d_fy[dev]));
    cudaFree(static_cast<void *>(d_fz[dev]));
    cudaFree(static_cast<void *>(d_en[dev]));
  }

  return;
}

template <unsigned int BLOCK>
__global__ static void compute_coulomb_kernel(double4 *__restrict__ fxyzen,
                                              const double4 *__restrict__ xyzq,
                                              const std::size_t count,
                                              const std::size_t point,
                                              const std::size_t natom) {
  __shared__ double4 s_cache[BLOCK];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double4 xyzqi = make_double4(0.0, 0.0, 0.0, 0.0);
  if (idx < count)
    xyzqi = xyzq[point + idx];

  double4 fxyzen0 = make_double4(0.0, 0.0, 0.0, 0.0);
  for (std::size_t j = 0; j < natom; j += BLOCK) {
    // Read block of atom data into shared memory
    std::size_t jdx = j + static_cast<std::size_t>(threadIdx.x);
    __syncthreads();
    s_cache[threadIdx.x] = make_double4(0.0, 0.0, 0.0, 0.0);
    if (jdx < natom)
      s_cache[threadIdx.x] = xyzq[jdx];
    __syncthreads();

    // Process block of atom data
    const unsigned int n = min(BLOCK, static_cast<unsigned int>(natom - j));
    for (unsigned int k = 0; k < n; k++) {
      jdx = j + static_cast<std::size_t>(k);
      if (point + idx == jdx)
        continue;
      const double4 xyzqj = s_cache[k];
      const double qij = xyzqi.w * xyzqj.w;
      const double xij = xyzqi.x - xyzqj.x;
      const double yij = xyzqi.y - xyzqj.y;
      const double zij = xyzqi.z - xyzqj.z;
      const double rij2 = xij * xij + yij * yij + zij * zij;
      const double rij = sqrt(rij2);
      const double irij = 1.0 / rij;
      const double dudr = qij / rij2; // u = qij / rij
      const double drdx = xij * irij;
      const double drdy = yij * irij;
      const double drdz = zij * irij;
      fxyzen0.x += dudr * drdx;
      fxyzen0.y += dudr * drdy;
      fxyzen0.z += dudr * drdz;
      fxyzen0.w += qij * irij;
    }
  }

  if (idx < count)
    fxyzen[idx] = fxyzen0;

  return;
}

void compute_coulomb_cuda(double4 *fxyzen, const double4 *xyzq,
                          const std::size_t natom) {
  int cuda_count = 0;
  cudaGetDeviceCount(&cuda_count);

  const std::size_t base = natom / static_cast<std::size_t>(cuda_count);
  const std::size_t rem = natom % static_cast<std::size_t>(cuda_count);
  std::vector<std::size_t> points(cuda_count, 0), counts(cuda_count, 0);
  for (int dev = 0; dev < cuda_count; dev++) {
    if (dev > 0)
      points[dev] = points[dev - 1] + counts[dev - 1];
    counts[dev] = base + ((static_cast<std::size_t>(dev) < rem) ? 1 : 0);
  }

  std::vector<double4 *> d_xyzq(cuda_count, nullptr);
  std::vector<double4 *> d_fxyzen(cuda_count, nullptr);
  for (int dev = 0; dev < cuda_count; dev++) {
    cudaSetDevice(dev);
    cudaMalloc(reinterpret_cast<void **>(&d_xyzq[dev]),
               natom * sizeof(double4));
    cudaMalloc(reinterpret_cast<void **>(&d_fxyzen[dev]),
               counts[dev] * sizeof(double4));
    cudaMemcpy(static_cast<void *>(d_xyzq[dev]),
               static_cast<const void *>(xyzq), natom * sizeof(double4),
               cudaMemcpyDeviceToDevice);
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaSetDevice(dev);
    constexpr unsigned int num_threads = 128;
    const unsigned int num_blocks =
        (counts[dev] + num_threads - 1) / num_threads;
    compute_coulomb_kernel<num_threads><<<num_blocks, num_threads>>>(
        d_fxyzen[dev], d_xyzq[dev], points[dev], counts[dev], natom);
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaSetDevice(dev);
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    cudaMemcpy(static_cast<void *>(fxyzen + points[dev]),
               static_cast<const void *>(d_fxyzen[dev]),
               counts[dev] * sizeof(double4), cudaMemcpyDeviceToDevice);
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaSetDevice(dev);
    cudaFree(static_cast<void *>(d_xyzq[dev]));
    cudaFree(static_cast<void *>(d_fxyzen[dev]));
  }

  return;
}
