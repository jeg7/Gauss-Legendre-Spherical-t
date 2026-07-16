// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include <cuda_container.hcu>
#include <cuda_utils.hcu>
#include <glst_force.hcu>

#include <cmath>
#include <cstdlib>

int main(void) {
  constexpr unsigned int natom = 2959;
  constexpr double tol = 1.0e-6;
  constexpr double box = 32.0;
  constexpr double rcut = 12.0;

  int cuda_count = 0;
  cudaCheck(cudaGetDeviceCount(&cuda_count));
  if (cuda_count < 1)
    return EXIT_FAILURE;

  cudaCheck(cudaSetDevice(0));

  cuda_container<double> rx(natom), ry(natom), rz(natom), qc(natom);
  for (unsigned int i = 0; i < natom; i++) {
    rx[i] = 0.001 + std::fmod(0.137 * static_cast<double>(i), box - 0.002);
    ry[i] = 0.001 + std::fmod(0.173 * static_cast<double>(i), box - 0.002);
    rz[i] = 0.001 + std::fmod(0.197 * static_cast<double>(i), box - 0.002);
    qc[i] = (i % 2 == 0) ? 1.0 : -1.0;
  }
  rx.transfer_to_device();
  ry.transfer_to_device();
  rz.transfer_to_device();
  qc.transfer_to_device();

  glst_force glst;

  if (cuda_count == 1)
    glst.set_gpu_layout(1, 1);
  else if (cuda_count == 2)
    glst.set_gpu_layout(2, 1);
  else
    glst.set_gpu_layout(2, static_cast<unsigned int>(cuda_count / 2));

  glst.init(natom, tol, box, box, box, rcut);
  glst.assign_atoms(rx.d_array().data(), ry.d_array().data(),
                    rz.d_array().data(), qc.d_array().data());

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaDeviceSynchronize());
  }

  return EXIT_SUCCESS;
}
