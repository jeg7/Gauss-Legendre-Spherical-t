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

template <unsigned int nthread>
__global__ static void compute_coulomb_kernel(
    double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
    double *__restrict__ en, const double *__restrict__ rx,
    const double *__restrict__ ry, const double *__restrict__ rz,
    const double *__restrict__ q, const std::size_t natom) {
  __shared__ double s_cache[nthread * 4];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double xi = 0.0, yi = 0.0, zi = 0.0, qi = 0.0;
  if (idx < natom) {
    xi = rx[idx];
    yi = ry[idx];
    zi = rz[idx];
    qi = q[idx];
  }

  double fx0 = 0.0, fy0 = 0.0, fz0 = 0.0, en0 = 0.0;
  for (std::size_t j = 0; j < natom; j += nthread) {
    { // Read block of atom data into shared memory
      __syncthreads();
      const std::size_t jdx = j + static_cast<std::size_t>(threadIdx.x);
      s_cache[threadIdx.x * 4 + 0] = 0.0;
      s_cache[threadIdx.x * 4 + 1] = 0.0;
      s_cache[threadIdx.x * 4 + 2] = 0.0;
      s_cache[threadIdx.x * 4 + 3] = 0.0;
      if (jdx < natom) {
        s_cache[threadIdx.x * 4 + 0] = rx[jdx];
        s_cache[threadIdx.x * 4 + 1] = ry[jdx];
        s_cache[threadIdx.x * 4 + 2] = rz[jdx];
        s_cache[threadIdx.x * 4 + 3] = q[jdx];
      }
      __syncthreads();
    }

    { // Process block of atom data
      for (unsigned int k = 0; k < nthread; k++) {
        const std::size_t jdx = j + static_cast<std::size_t>(k);
        if (idx == jdx)
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

  if (idx < natom) {
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
  constexpr unsigned int nthread = 128;
  const unsigned int nblock = natom / nthread + 1;
  compute_coulomb_kernel<nthread>
      <<<nblock, nthread>>>(fx, fy, fz, en, rx, ry, rz, qc, natom);
  cudaDeviceSynchronize();
  return;
}

template <unsigned int nthread>
__global__ static void compute_coulomb_kernel(double4 *__restrict__ fxyzen,
                                              const double4 *__restrict__ xyzq,
                                              const std::size_t natom) {
  __shared__ double4 s_cache[nthread];

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double4 xyzqi = make_double4(0.0, 0.0, 0.0, 0.0);
  if (idx < natom)
    xyzqi = xyzq[idx];

  double4 fxyzen0 = make_double4(0.0, 0.0, 0.0, 0.0);
  for (std::size_t j = 0; j < natom; j += nthread) {
    // Read block of atom data into shared memory
    std::size_t jdx = j + static_cast<std::size_t>(threadIdx.x);
    __syncthreads();
    s_cache[threadIdx.x] = make_double4(0.0, 0.0, 0.0, 0.0);
    if (jdx < natom)
      s_cache[threadIdx.x] = xyzq[jdx];
    __syncthreads();

    // Process block of atom data
    for (unsigned int k = 0; k < nthread; k++) {
      jdx = j + static_cast<std::size_t>(k);
      if (idx == jdx)
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

  if (idx < natom)
    fxyzen[idx] = fxyzen0;

  return;
}

void compute_coulomb_cuda(double4 *fxyzen, const double4 *xyzq,
                          const std::size_t natom) {
  constexpr unsigned int nthread = 128;
  const unsigned int nblock = natom / nthread + 1;
  compute_coulomb_kernel<nthread><<<nblock, nthread>>>(fxyzen, xyzq, natom);
  cudaDeviceSynchronize();
  return;
}
