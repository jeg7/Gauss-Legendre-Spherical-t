// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include <chrono>
#include <coulomb.hcu>
#include <cstdlib>
#include <cuda_container.hcu>
#include <cuda_utils.hcu>
#include <glst_force.hcu>
#include <io.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utils.hpp>

int main(int argc, char **argv) {
  // Input check and error catch
  if ((argc != 5) && (argc != 7)) {
    std::cout << "Usage: " << argv[0] << " [sys] [tol] [box_dim] [rcut]"
              << std::endl;
    std::cout << "    OR" << std::endl;
    std::cout << "       " << argv[0]
              << " [sys] [tol] [box_dim] [ncell.x] [ncell.y] [ncell.z]"
              << std::endl;
  }

  int cuda_count = 0;
  cudaGetDeviceCount(&cuda_count);

  // Parse input
  std::string file_name = argv[1];
  double tol = std::stod(argv[2]);
  double box_dim_x = std::stod(argv[3]);
  double box_dim_y = std::stod(argv[3]);
  double box_dim_z = std::stod(argv[3]);
  double rcut = 0.0;
  int ncell_x = 0.0;
  int ncell_y = 0.0;
  int ncell_z = 0.0;
  if (argc == 5)
    rcut = std::stod(argv[4]);
  else if (argc == 7) {
    ncell_x = std::stoi(argv[4]);
    ncell_y = std::stoi(argv[5]);
    ncell_z = std::stoi(argv[6]);
  }

  // Allocate host memory and perform IO
  std::size_t natom = get_natom_psf(file_name + ".psf");
  cuda_container<double> rx(natom), ry(natom), rz(natom), qc(natom);
  read_charmm_cor(rx.h_array(), ry.h_array(), rz.h_array(), natom,
                  file_name + ".cor");
  read_charmm_psf(qc.h_array(), natom, file_name + ".psf");
  recenter(rx.h_array(), ry.h_array(), rz.h_array(), natom);

  // Copy coordinates and charges to device
  rx.transfer_to_device();
  ry.transfer_to_device();
  rz.transfer_to_device();
  qc.transfer_to_device();

  // Compute GLST energy and forces
  cuda_container<double> fx_glst(natom), fy_glst(natom), fz_glst(natom),
      en_glst(natom);

  std::unique_ptr<glst_force> glst = nullptr;
  if (argc == 5)
    glst = std::make_unique<glst_force>(natom, tol, box_dim_x, box_dim_y,
                                        box_dim_z, rcut);
  else if (argc == 7)
    glst = std::make_unique<glst_force>(natom, tol, box_dim_x, box_dim_y,
                                        box_dim_z, ncell_x, ncell_y, ncell_z);

  std::cout << std::endl;

  constexpr std::size_t MAX_ITER = 50;
  std::vector<std::vector<double>> times(5, std::vector<double>(MAX_ITER));
  for (std::size_t ITER = 0; ITER < MAX_ITER; ITER++) {
    std::cout << "\rIteration " << ITER << std::flush;
    auto start_glst = std::chrono::high_resolution_clock::now();
    auto start_assign = std::chrono::high_resolution_clock::now();
    glst->assign_atoms(rx.d_array().data(), ry.d_array().data(),
                       rz.d_array().data(), qc.d_array().data());
    for (int dev = 0; dev < cuda_count; dev++) {
      cudaSetDevice(dev);
      cudaDeviceSynchronize();
    }
    auto end_assign = std::chrono::high_resolution_clock::now();
    auto start_sf = std::chrono::high_resolution_clock::now();
    glst->calc_sf();
    for (int dev = 0; dev < cuda_count; dev++) {
      cudaSetDevice(dev);
      cudaDeviceSynchronize();
    }
    auto end_sf = std::chrono::high_resolution_clock::now();
    auto start_sum = std::chrono::high_resolution_clock::now();
    glst->sum_rmt_sf();
    for (int dev = 0; dev < cuda_count; dev++) {
      cudaSetDevice(dev);
      cudaDeviceSynchronize();
    }
    auto end_sum = std::chrono::high_resolution_clock::now();
    auto start_lr = std::chrono::high_resolution_clock::now();
    glst->calc_lr_ef();
    for (int dev = 0; dev < cuda_count; dev++) {
      cudaSetDevice(dev);
      cudaDeviceSynchronize();
    }
    auto end_lr = std::chrono::high_resolution_clock::now();
    auto end_glst = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_assign = end_assign - start_assign;
    std::chrono::duration<double> time_sf = end_sf - start_sf;
    std::chrono::duration<double> time_sum = end_sum - start_sum;
    std::chrono::duration<double> time_lr = end_lr - start_lr;
    std::chrono::duration<double> time_glst = end_glst - start_glst;
    times[0][ITER] = 1000.0 * time_assign.count();
    times[1][ITER] = 1000.0 * time_sf.count();
    times[2][ITER] = 1000.0 * time_sum.count();
    times[3][ITER] = 1000.0 * time_lr.count();
    times[4][ITER] = 1000.0 * time_glst.count();
    if (ITER < MAX_ITER - 1) {
      for (int dev = 0; dev < cuda_count; dev++) {
        cudaSetDevice(dev);
        glst->fx()[dev].set(0.0);
        glst->fy()[dev].set(0.0);
        glst->fz()[dev].set(0.0);
        glst->en()[dev].set(0.0);
      }
    }
  }
  glst->calc_sr_ef(); // Don't include in timing

  // Transfer GLST results to host
  glst->get_ef(fx_glst, fy_glst, fz_glst, en_glst);

  std::cout << "\rFinished " << MAX_ITER << " calculations" << std::endl;
  std::cout << std::endl;
  std::cout << "                  Assign atoms to cells: " << avg(times[0])
            << " ms (" << stdev(times[0]) << ") " << std::endl;
  std::cout << "            Calculate structure factors: " << avg(times[1])
            << " ms (" << stdev(times[1]) << ")" << std::endl;
  std::cout << "           Sum remote structure factors: " << avg(times[2])
            << " ms (" << stdev(times[2]) << ")" << std::endl;
  std::cout << " Calculate long-range energy and forces: " << avg(times[3])
            << " ms (" << stdev(times[3]) << ")" << std::endl;
  std::cout << "---------------------------------------------------------------"
               "-----------"
            << std::endl;
  std::cout << "                           GLST Runtime: " << avg(times[4])
            << " ms (" << stdev(times[4]) << ")" << std::endl;

  // Compute Coulomb energy and forces
  cuda_container<double> fx_coul(natom), fy_coul(natom), fz_coul(natom),
      en_coul(natom);

  auto start_coul = std::chrono::high_resolution_clock::now();
  compute_coulomb_cuda(fx_coul.d_array().data(), fy_coul.d_array().data(),
                       fz_coul.d_array().data(), en_coul.d_array().data(),
                       rx.d_array().data(), ry.d_array().data(),
                       rz.d_array().data(), qc.d_array().data(), natom);
  auto end_coul = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_coul = end_coul - start_coul;
  std::cout << std::endl;
  std::cout << "Coulomb Runtime: " << 1000.0 * time_coul.count() << " ms"
            << std::endl;

  // Transfer Coulomb results to host
  fx_coul.transfer_to_host();
  fy_coul.transfer_to_host();
  fz_coul.transfer_to_host();
  en_coul.transfer_to_host();

  // Print and compute errors
  print_error_report(fx_glst.h_array(), fy_glst.h_array(), fz_glst.h_array(),
                     en_glst.h_array(), fx_coul.h_array(), fy_coul.h_array(),
                     fz_coul.h_array(), en_coul.h_array(), natom, tol);

  return 0;
}
