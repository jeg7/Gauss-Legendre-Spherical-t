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
#include <iostream>
#include <stdexcept>
#include <vector>

static void calc_reference(std::vector<double> &fx, std::vector<double> &fy,
                           std::vector<double> &fz, std::vector<double> &en,
                           const std::vector<double> &rx,
                           const std::vector<double> &ry,
                           const std::vector<double> &rz,
                           const std::vector<double> &qc,
                           const unsigned int ncell_axis) {
  const unsigned int natom = ncell_axis * ncell_axis * ncell_axis;

  fx.assign(natom, 0.0);
  fy.assign(natom, 0.0);
  fz.assign(natom, 0.0);
  en.assign(natom, 0.0);

  for (unsigned int x = 0; x < ncell_axis; x++) {
    for (unsigned int y = 0; y < ncell_axis; y++) {
      for (unsigned int z = 0; z < ncell_axis; z++) {
        const unsigned int atom = (x * ncell_axis + y) * ncell_axis + z;

        const double xi = rx[atom];
        const double yi = ry[atom];
        const double zi = rz[atom];
        const double qi = qc[atom];

        for (int dx = -1; dx <= 1; dx++) {
          const int nx = static_cast<int>(x) + dx;
          if ((nx < 0) || (nx >= static_cast<int>(ncell_axis)))
            continue;

          for (int dy = -1; dy <= 1; dy++) {
            const int ny = static_cast<int>(y) + dy;
            if ((ny < 0) || (ny >= static_cast<int>(ncell_axis)))
              continue;

            for (int dz = -1; dz <= 1; dz++) {
              const int nz = static_cast<int>(z) + dz;
              if ((nz < 0) || (nz >= static_cast<int>(ncell_axis)))
                continue;

              if ((dx == 0) && (dy == 0) && (dz == 0))
                continue;

              const unsigned int nbr =
                  (static_cast<unsigned int>(nx) * ncell_axis +
                   static_cast<unsigned int>(ny)) *
                      ncell_axis +
                  static_cast<unsigned int>(nz);

              const double xj = rx[nbr];
              const double yj = ry[nbr];
              const double zj = rz[nbr];
              const double qj = qc[nbr];

              const double qij = qi * qj;
              const double xij = xi - xj;
              const double yij = yi - yj;
              const double zij = zi - zj;
              const double rij2 = xij * xij + yij * yij + zij * zij;
              const double rij = std::sqrt(rij2);
              const double irij = 1.0 / rij;
              const double dudr = qij / rij2;

              fx[atom] += dudr * xij * irij;
              fy[atom] += dudr * yij * irij;
              fz[atom] += dudr * zij * irij;
              en[atom] += qij * irij;
            }
          }
        }
      }
    }
  }

  return;
}

int main(void) {
  try {
    int cuda_count = 0;
    cudaCheck(cudaGetDeviceCount(&cuda_count));

    if (cuda_count < 2) {
      std::cout << "SKIP short_range_halo_exchange: requires at least 2 GPUs"
                << std::endl;
      return EXIT_SUCCESS;
    }

    const unsigned int ncell_axis = static_cast<unsigned int>(cuda_count);
    const unsigned int natom = ncell_axis * ncell_axis * ncell_axis;
    constexpr double tol = 1.0e-6;
    constexpr double rcut = 12.0;
    const double box = rcut * static_cast<double>(ncell_axis);

    std::vector<double> h_rx(natom);
    std::vector<double> h_ry(natom);
    std::vector<double> h_rz(natom);
    std::vector<double> h_qc(natom);

    for (unsigned int x = 0; x < ncell_axis; x++) {
      for (unsigned int y = 0; y < ncell_axis; y++) {
        for (unsigned int z = 0; z < ncell_axis; z++) {
          const unsigned int atom = (x * ncell_axis + y) * ncell_axis + z;

          h_rx[atom] = (static_cast<double>(x) + 0.5) * rcut;
          h_ry[atom] = (static_cast<double>(y) + 0.5) * rcut;
          h_rz[atom] = (static_cast<double>(z) + 0.5) * rcut;
          h_qc[atom] = (atom % 2 == 0) ? 1.0 : -1.0;
        }
      }
    }

    cudaCheck(cudaSetDevice(0));

    cuda_container<double> rx(natom), ry(natom), rz(natom), qc(natom);
    for (unsigned int atom = 0; atom < natom; atom++) {
      rx[atom] = h_rx[atom];
      ry[atom] = h_ry[atom];
      rz[atom] = h_rz[atom];
      qc[atom] = h_qc[atom];
    }

    rx.transfer_to_device();
    ry.transfer_to_device();
    rz.transfer_to_device();
    qc.transfer_to_device();

    glst_force glst;
    glst.set_gpu_layout(static_cast<unsigned int>(cuda_count), 1);
    glst.init(natom, tol, box, box, box, rcut);
    glst.assign_atoms(rx.d_array().data(), ry.d_array().data(),
                      rz.d_array().data(), qc.d_array().data());

    for (int dev = 0; dev < cuda_count; dev++) {
      cudaCheck(cudaSetDevice(dev));
      glst.fx()[dev].set(0.0);
      glst.fy()[dev].set(0.0);
      glst.fz()[dev].set(0.0);
      glst.en()[dev].set(0.0);
    }

    glst.calc_sr_ef();

    for (int dev = 0; dev < cuda_count; dev++) {
      cudaCheck(cudaSetDevice(dev));
      cudaCheck(cudaDeviceSynchronize());

      glst.fx()[dev].transfer_to_host();
      glst.fy()[dev].transfer_to_host();
      glst.fz()[dev].transfer_to_host();
      glst.en()[dev].transfer_to_host();
    }

    std::vector<double> ref_fx, ref_fy, ref_fz, ref_en;
    calc_reference(ref_fx, ref_fy, ref_fz, ref_en, h_rx, h_ry, h_rz, h_qc,
                   ncell_axis);

    double max_err = 0.0;
    double rmse = 0.0;
    unsigned int count = 0;

    for (int dev = 0; dev < cuda_count; dev++) {
      const unsigned int x = static_cast<unsigned int>(dev);

      for (unsigned int y = 0; y < ncell_axis; y++) {
        for (unsigned int z = 0; z < ncell_axis; z++) {
          const unsigned int global_atom =
              (x * ncell_axis + y) * ncell_axis + z;
          const unsigned int local_atom = y * ncell_axis + z;

          const double dfx = glst.fx()[dev][local_atom] - ref_fx[global_atom];
          const double dfy = glst.fy()[dev][local_atom] - ref_fy[global_atom];
          const double dfz = glst.fz()[dev][local_atom] - ref_fz[global_atom];
          const double den = glst.en()[dev][local_atom] - ref_en[global_atom];

          const double afx = std::abs(dfx);
          const double afy = std::abs(dfy);
          const double afz = std::abs(dfz);
          const double aen = std::abs(den);

          max_err = (afx > max_err) ? afx : max_err;
          max_err = (afy > max_err) ? afy : max_err;
          max_err = (afz > max_err) ? afz : max_err;
          max_err = (aen > max_err) ? aen : max_err;

          rmse += dfx * dfx + dfy * dfy + dfz * dfz + den * den;
          count += 4;
        }
      }
    }

    rmse = std::sqrt(rmse / static_cast<double>(count));

    constexpr double threshold = 1.0e-10;
    if (max_err > threshold) {
      std::cerr << "FAIL short_range_halo_exchange: max error " << max_err
                << " exceeds " << threshold << ", RMSE " << rmse << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "PASS short_range_halo_exchange: max error " << max_err
              << ", RMSE " << rmse << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "FAIL short_range_halo_exchange: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
