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
#include <glst_plan.hcu>
#include <glst_workspace.hcu>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class glst_force_test_access {
public:
  static const glst_plan &plan(const glst_force &force) {
    if (force.plan_ == nullptr) {
      throw std::runtime_error(
          "glst_force_test_access::plan: Plan is not initialized");
    }
    return *(force.plan_);
  }

  static glst_workspace &workspace(glst_force &force) {
    if (force.workspace_ == nullptr) {
      throw std::runtime_error(
          "glst_force_test_access::workspace: Workspace is not initialized");
    }
    return *(force.workspace_);
  }

  static int cuda_count(const glst_force &force) { return force.cuda_count_; }

  static unsigned int dev_cell_partition(const glst_force &force,
                                         const int dev) {
    if ((dev < 0) ||
        (static_cast<std::size_t>(dev) >= force.dev_cell_partition_.size())) {
      throw std::runtime_error(
          "glst_force_test_access::dev_cell_partition: Device out of range");
    }
    return force.dev_cell_partition_[dev];
  }

  static unsigned int dev_tile_partition(const glst_force &force,
                                         const int dev) {
    if ((dev < 0) ||
        (static_cast<std::size_t>(dev) >= force.dev_tile_partition_.size())) {
      throw std::runtime_error(
          "glst_force_test_access::dev_tile_partition: Device out of range");
    }
    return force.dev_tile_partition_[dev];
  }

  static void calc_sf_tile(glst_force &force, const unsigned int tile) {
    force.calc_sf_tile(tile);
  }

  static void exchange_sf_tile(glst_force &force, const unsigned int tile) {
    force.exchange_sf_tile(tile);
  }

  static void sum_rmt_sf_tile(glst_force &force, const unsigned int tile) {
    force.sum_rmt_sf_tile(tile);
  }

  static void calc_lr_ef_tile(glst_force &force, const unsigned int tile) {
    force.calc_lr_ef_tile(tile);
  }

  static void zero_ef(glst_force &force) { force.zero_ef(); }
};

static void require(const bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

static void append_unique_tile(std::vector<unsigned int> &tiles,
                               const unsigned int tile) {
  if (std::find(tiles.begin(), tiles.end(), tile) == tiles.end())
    tiles.push_back(tile);
}

static void cube_sum_reference(
    double &cube_re, double &cube_im, const std::vector<double> &prefix_re,
    const std::vector<double> &prefix_im, const unsigned int x,
    const unsigned int y, const unsigned int z, const unsigned int radius,
    const unsigned int nx, const unsigned int ny, const unsigned int nz,
    const unsigned int local_node, const unsigned int node_count) {
  const unsigned int x0 = (x < radius) ? 0u : x - radius;
  const unsigned int y0 = (y < radius) ? 0u : y - radius;
  const unsigned int z0 = (z < radius) ? 0u : z - radius;

  unsigned int x1 = x + radius;
  unsigned int y1 = y + radius;
  unsigned int z1 = z + radius;

  x1 = (x1 >= nx) ? nx - 1 : x1;
  y1 = (y1 >= ny) ? ny - 1 : y1;
  z1 = (z1 >= nz) ? nz - 1 : z1;

  const bool xb = (x0 == 0);
  const bool yb = (y0 == 0);
  const bool zb = (z0 == 0);

  const unsigned int xm = xb ? 0u : x0 - 1;
  const unsigned int ym = yb ? 0u : y0 - 1;
  const unsigned int zm = zb ? 0u : z0 - 1;

  const unsigned int cell0 = (x1 * ny + y1) * nz + z1;
  const unsigned int cell1 = (xm * ny + y1) * nz + z1;
  const unsigned int cell2 = (x1 * ny + ym) * nz + z1;
  const unsigned int cell3 = (x1 * ny + y1) * nz + zm;
  const unsigned int cell4 = (xm * ny + ym) * nz + z1;
  const unsigned int cell5 = (xm * ny + y1) * nz + zm;
  const unsigned int cell6 = (x1 * ny + ym) * nz + zm;
  const unsigned int cell7 = (xm * ny + ym) * nz + zm;

  const std::size_t entry0 =
      static_cast<std::size_t>(cell0) * static_cast<std::size_t>(node_count) +
      static_cast<std::size_t>(local_node);
  const std::size_t entry1 =
      static_cast<std::size_t>(cell1) * static_cast<std::size_t>(node_count) +
      static_cast<std::size_t>(local_node);
  const std::size_t entry2 =
      static_cast<std::size_t>(cell2) * static_cast<std::size_t>(node_count) +
      static_cast<std::size_t>(local_node);
  const std::size_t entry3 =
      static_cast<std::size_t>(cell3) * static_cast<std::size_t>(node_count) +
      static_cast<std::size_t>(local_node);
  const std::size_t entry4 =
      static_cast<std::size_t>(cell4) * static_cast<std::size_t>(node_count) +
      static_cast<std::size_t>(local_node);
  const std::size_t entry5 =
      static_cast<std::size_t>(cell5) * static_cast<std::size_t>(node_count) +
      static_cast<std::size_t>(local_node);
  const std::size_t entry6 =
      static_cast<std::size_t>(cell6) * static_cast<std::size_t>(node_count) +
      static_cast<std::size_t>(local_node);
  const std::size_t entry7 =
      static_cast<std::size_t>(cell7) * static_cast<std::size_t>(node_count) +
      static_cast<std::size_t>(local_node);

  const double g0_re = prefix_re[entry0];
  const double g1_re = xb ? 0.0 : prefix_re[entry1];
  const double g2_re = yb ? 0.0 : prefix_re[entry2];
  const double g3_re = zb ? 0.0 : prefix_re[entry3];
  const double g4_re = (xb || yb) ? 0.0 : prefix_re[entry4];
  const double g5_re = (xb || zb) ? 0.0 : prefix_re[entry5];
  const double g6_re = (yb || zb) ? 0.0 : prefix_re[entry6];
  const double g7_re = (xb || yb || zb) ? 0.0 : prefix_re[entry7];

  const double g0_im = prefix_im[entry0];
  const double g1_im = xb ? 0.0 : prefix_im[entry1];
  const double g2_im = yb ? 0.0 : prefix_im[entry2];
  const double g3_im = zb ? 0.0 : prefix_im[entry3];
  const double g4_im = (xb || yb) ? 0.0 : prefix_im[entry4];
  const double g5_im = (xb || zb) ? 0.0 : prefix_im[entry5];
  const double g6_im = (yb || zb) ? 0.0 : prefix_im[entry6];
  const double g7_im = (xb || yb || zb) ? 0.0 : prefix_im[entry7];

  cube_re = g0_re - g1_re - g2_re - g3_re + g4_re + g5_re + g6_re - g7_re;
  cube_im = g0_im - g1_im - g2_im - g3_im + g4_im + g5_im + g6_im - g7_im;

  return;
}

static double verify_tile(glst_force &force, const std::vector<double> &atom_rx,
                          const std::vector<double> &atom_ry,
                          const std::vector<double> &atom_rz,
                          const std::vector<double> &atom_qc,
                          const unsigned int tile) {
  constexpr double sentinel = 12345.0;
  constexpr double sf_threshold = 1.0e-12;
  constexpr double rmt_threshold = 1.0e-11;
  constexpr double ef_threshold = 1.0e-10;

  const glst_plan &plan = glst_force_test_access::plan(force);
  glst_workspace &workspace = glst_force_test_access::workspace(force);
  const int cuda_count = glst_force_test_access::cuda_count(force);

  require(cuda_count > 0, "glst_force has no CUDA devices");
  require(atom_rx.size() == static_cast<std::size_t>(plan.ncell()),
          "x-coordinate reference size does not match ncell");
  require(atom_ry.size() == static_cast<std::size_t>(plan.ncell()),
          "y-coordinate reference size does not match ncell");
  require(atom_rz.size() == static_cast<std::size_t>(plan.ncell()),
          "z-coordinate reference size does not match ncell");
  require(atom_qc.size() == static_cast<std::size_t>(plan.ncell()),
          "charge reference size does not match ncell");

  require(workspace.sf_re().size() == static_cast<std::size_t>(cuda_count),
          "Real SF device count does not match glst_force");
  require(workspace.sf_im().size() == static_cast<std::size_t>(cuda_count),
          "Imaginary SF device count does not match glst_force");
  require(workspace.rmt_sum_re().size() == static_cast<std::size_t>(cuda_count),
          "Real RMT device count does not match glst_force");
  require(workspace.rmt_sum_im().size() == static_cast<std::size_t>(cuda_count),
          "Imaginary RMT device count does not match glst_force");

  require(plan.x().size() == static_cast<std::size_t>(cuda_count),
          "Cubature x device count does not match glst_force");
  require(plan.y().size() == static_cast<std::size_t>(cuda_count),
          "Cubature y device count does not match glst_force");
  require(plan.z().size() == static_cast<std::size_t>(cuda_count),
          "Cubature z device count does not match glst_force");
  require(plan.w().size() == static_cast<std::size_t>(cuda_count),
          "Cubature weight device count does not match glst_force");
  require(plan.group().size() == static_cast<std::size_t>(cuda_count),
          "Cubature group device count does not match glst_force");
  require(plan.grp_r_in().size() == static_cast<std::size_t>(cuda_count),
          "Inner-radius device count does not match glst_force");
  require(plan.grp_r_out().size() == static_cast<std::size_t>(cuda_count),
          "Outer-radius device count does not match glst_force");

  require(tile < plan.tile_count(), "Tile index is out of range");

  const unsigned int tile_partition = plan.tile_partition_idx(tile);
  const unsigned int node_point = plan.tile_node_point(tile);
  const unsigned int node_count = plan.tile_node_count(tile);

  require(node_count > 0, "Tile node count is zero");
  require(node_count <= plan.max_tile_nodes(),
          "Tile node count exceeds max_tile_nodes");
  require(node_point <= plan.tot_num_nodes(),
          "Tile node point exceeds total cubature nodes");
  require(node_count <= plan.tot_num_nodes() - node_point,
          "Tile node range exceeds total cubature nodes");

  const std::size_t active_sf_entry_count =
      static_cast<std::size_t>(plan.ncell()) *
      static_cast<std::size_t>(node_count);

  for (int dev = 0; dev < cuda_count; dev++) {
    const unsigned int cell_partition =
        glst_force_test_access::dev_cell_partition(force, dev);
    const unsigned int device_tile_partition =
        glst_force_test_access::dev_tile_partition(force, dev);
    const unsigned int local_cell_count = plan.local_cell_count(cell_partition);

    const std::size_t sf_buffer_capacity =
        workspace.sf_tile_buffer_capacity(dev);
    const std::size_t rmt_buffer_capacity =
        workspace.rmt_tile_buffer_capacity(dev);
    const std::size_t active_rmt_entry_count =
        static_cast<std::size_t>(local_cell_count) *
        static_cast<std::size_t>(node_count);

    require(workspace.sf_re()[dev].size() == sf_buffer_capacity,
            "Real SF container size does not match workspace capacity");
    require(workspace.sf_im()[dev].size() == sf_buffer_capacity,
            "Imaginary SF container size does not match workspace capacity");
    require(workspace.rmt_sum_re()[dev].size() == rmt_buffer_capacity,
            "Real RMT container size does not match workspace capacity");
    require(workspace.rmt_sum_im()[dev].size() == rmt_buffer_capacity,
            "Imaginary RMT container size does not match workspace capacity");
    require(workspace.cell_capacity(dev) ==
                static_cast<std::size_t>(local_cell_count),
            "Workspace cell capacity does not match local cell count");

    if (device_tile_partition == tile_partition) {
      require(active_sf_entry_count <= sf_buffer_capacity,
              "Active SF tile exceeds workspace capacity");
      require(active_rmt_entry_count <= rmt_buffer_capacity,
              "Active RMT tile exceeds workspace capacity");
    }

    cudaCheck(cudaSetDevice(dev));
    workspace.sf_re()[dev].set(sentinel);
    workspace.sf_im()[dev].set(sentinel);
    workspace.rmt_sum_re()[dev].set(sentinel);
    workspace.rmt_sum_im()[dev].set(sentinel);
  }

  glst_force_test_access::calc_sf_tile(force, tile);

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaDeviceSynchronize());
    workspace.sf_re()[dev].transfer_to_host();
    workspace.sf_im()[dev].transfer_to_host();
  }

  double max_error = 0.0;

  for (int dev = 0; dev < cuda_count; dev++) {
    const unsigned int cell_partition =
        glst_force_test_access::dev_cell_partition(force, dev);
    const unsigned int device_tile_partition =
        glst_force_test_access::dev_tile_partition(force, dev);
    const bool owns_tile = (device_tile_partition == tile_partition);
    const std::size_t sf_buffer_capacity =
        workspace.sf_tile_buffer_capacity(dev);

    if (!owns_tile) {
      for (std::size_t entry = 0; entry < sf_buffer_capacity; entry++) {
        if ((workspace.sf_re()[dev][entry] != sentinel) ||
            (workspace.sf_im()[dev][entry] != sentinel)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": calc_sf_tile modified a non-participating tile "
                     "partition at SF entry "
                  << entry;
          throw std::runtime_error(message.str());
        }
      }
      continue;
    }

    for (unsigned int global_cell = 0; global_cell < plan.ncell();
         global_cell++) {
      const bool owns_cell =
          (plan.cell_partition_idx(global_cell) == cell_partition);

      for (unsigned int local_node = 0; local_node < node_count; local_node++) {
        const std::size_t entry = static_cast<std::size_t>(global_cell) *
                                      static_cast<std::size_t>(node_count) +
                                  static_cast<std::size_t>(local_node);

        const double observed_re = workspace.sf_re()[dev][entry];
        const double observed_im = workspace.sf_im()[dev][entry];

        if (!owns_cell) {
          if ((observed_re != 0.0) || (observed_im != 0.0)) {
            std::ostringstream message;
            message << "GPU " << dev << ", tile " << tile
                    << ": non-owned global cell " << global_cell
                    << " is not zero at local node " << local_node
                    << " (re=" << std::setprecision(17) << observed_re
                    << ", im=" << observed_im << ")";
            throw std::runtime_error(message.str());
          }
          continue;
        }

        const unsigned int global_node = node_point + local_node;
        const double theta = plan.x()[dev][global_node] * atom_rx[global_cell] +
                             plan.y()[dev][global_node] * atom_ry[global_cell] +
                             plan.z()[dev][global_node] * atom_rz[global_cell];

        const double expected_re = atom_qc[global_cell] * std::cos(theta);
        const double expected_im = -atom_qc[global_cell] * std::sin(theta);

        if (!std::isfinite(observed_re) || !std::isfinite(observed_im)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": non-finite structure factor for global cell "
                  << global_cell << ", local node " << local_node;
          throw std::runtime_error(message.str());
        }

        const double error_re = std::abs(observed_re - expected_re);
        const double error_im = std::abs(observed_im - expected_im);

        max_error = std::max(max_error, error_re);
        max_error = std::max(max_error, error_im);

        if ((error_re > sf_threshold) || (error_im > sf_threshold)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": structure-factor mismatch for global cell "
                  << global_cell << ", local node " << local_node
                  << ", global node " << global_node << std::setprecision(17)
                  << " (observed re=" << observed_re
                  << ", expected re=" << expected_re
                  << ", observed im=" << observed_im
                  << ", expected im=" << expected_im
                  << ", re error=" << error_re << ", im error=" << error_im
                  << ")";
          throw std::runtime_error(message.str());
        }
      }
    }

    for (std::size_t entry = active_sf_entry_count; entry < sf_buffer_capacity;
         entry++) {
      if ((workspace.sf_re()[dev][entry] != sentinel) ||
          (workspace.sf_im()[dev][entry] != sentinel)) {
        std::ostringstream message;
        message << "GPU " << dev << ", tile " << tile
                << ": calc_sf_tile wrote beyond the active compact SF range "
                   "at entry "
                << entry << " (active entries " << active_sf_entry_count
                << ", capacity " << sf_buffer_capacity << ")";
        throw std::runtime_error(message.str());
      }
    }
  }

  glst_force_test_access::exchange_sf_tile(force, tile);

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaDeviceSynchronize());
    workspace.sf_re()[dev].transfer_to_host();
    workspace.sf_im()[dev].transfer_to_host();
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    const unsigned int device_tile_partition =
        glst_force_test_access::dev_tile_partition(force, dev);
    const bool owns_tile = (device_tile_partition == tile_partition);
    const std::size_t sf_buffer_capacity =
        workspace.sf_tile_buffer_capacity(dev);

    if (!owns_tile) {
      for (std::size_t entry = 0; entry < sf_buffer_capacity; entry++) {
        if ((workspace.sf_re()[dev][entry] != sentinel) ||
            (workspace.sf_im()[dev][entry] != sentinel)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": exchange_sf_tile modified a non-participating tile "
                     "partition at SF entry "
                  << entry;
          throw std::runtime_error(message.str());
        }
      }
      continue;
    }

    for (unsigned int global_cell = 0; global_cell < plan.ncell();
         global_cell++) {
      for (unsigned int local_node = 0; local_node < node_count; local_node++) {
        const std::size_t entry = static_cast<std::size_t>(global_cell) *
                                      static_cast<std::size_t>(node_count) +
                                  static_cast<std::size_t>(local_node);
        const unsigned int global_node = node_point + local_node;

        const double theta = plan.x()[dev][global_node] * atom_rx[global_cell] +
                             plan.y()[dev][global_node] * atom_ry[global_cell] +
                             plan.z()[dev][global_node] * atom_rz[global_cell];

        const double expected_re = atom_qc[global_cell] * std::cos(theta);
        const double expected_im = -atom_qc[global_cell] * std::sin(theta);
        const double observed_re = workspace.sf_re()[dev][entry];
        const double observed_im = workspace.sf_im()[dev][entry];

        if (!std::isfinite(observed_re) || !std::isfinite(observed_im)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": non-finite post-exchange structure factor for global "
                     "cell "
                  << global_cell << ", local node " << local_node;
          throw std::runtime_error(message.str());
        }

        const double error_re = std::abs(observed_re - expected_re);
        const double error_im = std::abs(observed_im - expected_im);

        max_error = std::max(max_error, error_re);
        max_error = std::max(max_error, error_im);

        if ((error_re > sf_threshold) || (error_im > sf_threshold)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": post-exchange structure-factor mismatch for global "
                     "cell "
                  << global_cell << ", local node " << local_node
                  << ", global node " << global_node << std::setprecision(17)
                  << " (observed re=" << observed_re
                  << ", expected re=" << expected_re
                  << ", observed im=" << observed_im
                  << ", expected im=" << expected_im
                  << ", re error=" << error_re << ", im error=" << error_im
                  << ")";
          throw std::runtime_error(message.str());
        }
      }
    }

    for (std::size_t entry = active_sf_entry_count; entry < sf_buffer_capacity;
         entry++) {
      if ((workspace.sf_re()[dev][entry] != sentinel) ||
          (workspace.sf_im()[dev][entry] != sentinel)) {
        std::ostringstream message;
        message << "GPU " << dev << ", tile " << tile
                << ": exchange_sf_tile wrote beyond the active compact SF "
                   "range at entry "
                << entry << " (active entries " << active_sf_entry_count
                << ", capacity " << sf_buffer_capacity << ")";
        throw std::runtime_error(message.str());
      }
    }
  }

  int reference_dev = -1;
  std::size_t covered_target_cell_count = 0;
  std::vector<unsigned int> cell_partition_visit_count(
      plan.cell_partition_count(), 0u);

  for (int dev = 0; dev < cuda_count; dev++) {
    const unsigned int device_tile_partition =
        glst_force_test_access::dev_tile_partition(force, dev);

    if (device_tile_partition != tile_partition)
      continue;

    const unsigned int cell_partition =
        glst_force_test_access::dev_cell_partition(force, dev);

    require(cell_partition < plan.cell_partition_count(),
            "Participating device has an invalid cell partition");

    cell_partition_visit_count[cell_partition]++;
    covered_target_cell_count +=
        static_cast<std::size_t>(plan.local_cell_count(cell_partition));

    if (reference_dev < 0)
      reference_dev = dev;
  }

  require(reference_dev >= 0, "No device participates in the tile partition");
  require(covered_target_cell_count == static_cast<std::size_t>(plan.ncell()),
          "Participating devices do not cover every target cell");

  for (unsigned int cell_partition = 0;
       cell_partition < plan.cell_partition_count(); cell_partition++) {
    if (cell_partition_visit_count[cell_partition] != 1) {
      std::ostringstream message;
      message << "Tile partition " << tile_partition << " has "
              << cell_partition_visit_count[cell_partition]
              << " devices for cell partition " << cell_partition
              << "; expected exactly one";
      throw std::runtime_error(message.str());
    }
  }

  require(plan.w()[reference_dev].size() ==
              static_cast<std::size_t>(plan.tot_num_nodes()),
          "Reference cubature weight array has the wrong size");
  require(plan.group()[reference_dev].size() ==
              static_cast<std::size_t>(plan.tot_num_nodes()),
          "Reference cubature group array has the wrong size");
  require(plan.grp_r_in()[reference_dev].size() ==
              static_cast<std::size_t>(plan.ngroup()),
          "Reference inner-radius array has the wrong size");
  require(plan.grp_r_out()[reference_dev].size() ==
              static_cast<std::size_t>(plan.ngroup()),
          "Reference outer-radius array has the wrong size");

  std::vector<double> prefix_re(active_sf_entry_count);
  std::vector<double> prefix_im(active_sf_entry_count);

  for (std::size_t entry = 0; entry < active_sf_entry_count; entry++) {
    prefix_re[entry] = workspace.sf_re()[reference_dev][entry];
    prefix_im[entry] = workspace.sf_im()[reference_dev][entry];
  }

  const unsigned int nx = plan.ncell_x();
  const unsigned int ny = plan.ncell_y();
  const unsigned int nz = plan.ncell_z();

  for (unsigned int local_node = 0; local_node < node_count; local_node++) {
    for (unsigned int x = 0; x < nx; x++) {
      for (unsigned int y = 0; y < ny; y++) {
        double sum_re = 0.0;
        double sum_im = 0.0;

        for (unsigned int z = 0; z < nz; z++) {
          const unsigned int global_cell = (x * ny + y) * nz + z;
          const std::size_t entry = static_cast<std::size_t>(global_cell) *
                                        static_cast<std::size_t>(node_count) +
                                    static_cast<std::size_t>(local_node);

          sum_re += prefix_re[entry];
          sum_im += prefix_im[entry];
          prefix_re[entry] = sum_re;
          prefix_im[entry] = sum_im;
        }
      }
    }
  }

  for (unsigned int local_node = 0; local_node < node_count; local_node++) {
    for (unsigned int x = 0; x < nx; x++) {
      for (unsigned int z = 0; z < nz; z++) {
        double sum_re = 0.0;
        double sum_im = 0.0;

        for (unsigned int y = 0; y < ny; y++) {
          const unsigned int global_cell = (x * ny + y) * nz + z;
          const std::size_t entry = static_cast<std::size_t>(global_cell) *
                                        static_cast<std::size_t>(node_count) +
                                    static_cast<std::size_t>(local_node);

          sum_re += prefix_re[entry];
          sum_im += prefix_im[entry];
          prefix_re[entry] = sum_re;
          prefix_im[entry] = sum_im;
        }
      }
    }
  }

  for (unsigned int local_node = 0; local_node < node_count; local_node++) {
    for (unsigned int y = 0; y < ny; y++) {
      for (unsigned int z = 0; z < nz; z++) {
        double sum_re = 0.0;
        double sum_im = 0.0;

        for (unsigned int x = 0; x < nx; x++) {
          const unsigned int global_cell = (x * ny + y) * nz + z;
          const std::size_t entry = static_cast<std::size_t>(global_cell) *
                                        static_cast<std::size_t>(node_count) +
                                    static_cast<std::size_t>(local_node);

          sum_re += prefix_re[entry];
          sum_im += prefix_im[entry];
          prefix_re[entry] = sum_re;
          prefix_im[entry] = sum_im;
        }
      }
    }
  }

  glst_force_test_access::sum_rmt_sf_tile(force, tile);

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaDeviceSynchronize());
    workspace.sf_re()[dev].transfer_to_host();
    workspace.sf_im()[dev].transfer_to_host();
    workspace.rmt_sum_re()[dev].transfer_to_host();
    workspace.rmt_sum_im()[dev].transfer_to_host();
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    const unsigned int cell_partition =
        glst_force_test_access::dev_cell_partition(force, dev);
    const unsigned int device_tile_partition =
        glst_force_test_access::dev_tile_partition(force, dev);
    const bool owns_tile = (device_tile_partition == tile_partition);
    const unsigned int local_cell_count = plan.local_cell_count(cell_partition);

    const std::size_t sf_buffer_capacity =
        workspace.sf_tile_buffer_capacity(dev);
    const std::size_t rmt_buffer_capacity =
        workspace.rmt_tile_buffer_capacity(dev);
    const std::size_t active_rmt_entry_count =
        static_cast<std::size_t>(local_cell_count) *
        static_cast<std::size_t>(node_count);

    if (!owns_tile) {
      for (std::size_t entry = 0; entry < sf_buffer_capacity; entry++) {
        if ((workspace.sf_re()[dev][entry] != sentinel) ||
            (workspace.sf_im()[dev][entry] != sentinel)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": sum_rmt_sf_tile modified a non-participating SF "
                     "buffer at entry "
                  << entry;
          throw std::runtime_error(message.str());
        }
      }

      for (std::size_t entry = 0; entry < rmt_buffer_capacity; entry++) {
        if ((workspace.rmt_sum_re()[dev][entry] != sentinel) ||
            (workspace.rmt_sum_im()[dev][entry] != sentinel)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": sum_rmt_sf_tile modified a non-participating RMT "
                     "buffer at entry "
                  << entry;
          throw std::runtime_error(message.str());
        }
      }

      continue;
    }

    require(active_sf_entry_count <= sf_buffer_capacity,
            "Active prefix tile exceeds SF workspace capacity");
    require(active_rmt_entry_count <= rmt_buffer_capacity,
            "Active remote-sum tile exceeds RMT workspace capacity");

    for (std::size_t entry = 0; entry < active_sf_entry_count; entry++) {
      const double expected_re = prefix_re[entry];
      const double expected_im = prefix_im[entry];
      const double observed_re = workspace.sf_re()[dev][entry];
      const double observed_im = workspace.sf_im()[dev][entry];

      if (!std::isfinite(observed_re) || !std::isfinite(observed_im)) {
        std::ostringstream message;
        message << "GPU " << dev << ", tile " << tile
                << ": non-finite prefix value at SF entry " << entry;
        throw std::runtime_error(message.str());
      }

      const double error_re = std::abs(observed_re - expected_re);
      const double error_im = std::abs(observed_im - expected_im);
      const double allowed_re =
          rmt_threshold * std::max(1.0, std::abs(expected_re));
      const double allowed_im =
          rmt_threshold * std::max(1.0, std::abs(expected_im));

      max_error = std::max(max_error, error_re);
      max_error = std::max(max_error, error_im);

      if ((error_re > allowed_re) || (error_im > allowed_im)) {
        std::ostringstream message;
        message << "GPU " << dev << ", tile " << tile
                << ": prefix mismatch at SF entry " << entry
                << std::setprecision(17) << " (observed re=" << observed_re
                << ", expected re=" << expected_re
                << ", observed im=" << observed_im
                << ", expected im=" << expected_im << ", re error=" << error_re
                << ", re tolerance=" << allowed_re << ", im error=" << error_im
                << ", im tolerance=" << allowed_im << ")";
        throw std::runtime_error(message.str());
      }
    }

    for (std::size_t entry = active_sf_entry_count; entry < sf_buffer_capacity;
         entry++) {
      if ((workspace.sf_re()[dev][entry] != sentinel) ||
          (workspace.sf_im()[dev][entry] != sentinel)) {
        std::ostringstream message;
        message << "GPU " << dev << ", tile " << tile
                << ": prefix calculation wrote beyond the active SF range at "
                   "entry "
                << entry << " (active entries " << active_sf_entry_count
                << ", capacity " << sf_buffer_capacity << ")";
        throw std::runtime_error(message.str());
      }
    }

    for (unsigned int local_cell = 0; local_cell < local_cell_count;
         local_cell++) {
      const unsigned int global_cell =
          plan.global_cell_from_local_cell(cell_partition, local_cell);
      const unsigned int expected_global_cell =
          plan.first_global_cell(cell_partition) + local_cell;

      if (global_cell != expected_global_cell) {
        std::ostringstream message;
        message << "GPU " << dev << ": local cell " << local_cell
                << " maps to global cell " << global_cell << ", expected "
                << expected_global_cell;
        throw std::runtime_error(message.str());
      }

      if (plan.cell_partition_idx(global_cell) != cell_partition) {
        std::ostringstream message;
        message << "GPU " << dev << ": local cell " << local_cell
                << " maps to global cell " << global_cell
                << " owned by a different cell partition";
        throw std::runtime_error(message.str());
      }

      if (plan.local_cell_from_global_cell(cell_partition, global_cell) !=
          local_cell) {
        std::ostringstream message;
        message << "GPU " << dev
                << ": local/global cell mapping is not reversible for local "
                   "cell "
                << local_cell << " and global cell " << global_cell;
        throw std::runtime_error(message.str());
      }

      unsigned int x = 0;
      unsigned int y = 0;
      unsigned int z = 0;
      plan.global_cell_coords(x, y, z, global_cell);

      for (unsigned int local_node = 0; local_node < node_count; local_node++) {
        const unsigned int global_node = node_point + local_node;
        const unsigned int group = plan.group()[reference_dev][global_node];

        if (group >= plan.ngroup()) {
          std::ostringstream message;
          message << "Tile " << tile << ", global node " << global_node
                  << ": cubature group " << group << " is out of range";
          throw std::runtime_error(message.str());
        }

        const unsigned int inner = plan.grp_r_in()[reference_dev][group];
        const unsigned int outer = plan.grp_r_out()[reference_dev][group];

        if (inner > outer) {
          std::ostringstream message;
          message << "Tile " << tile << ", global node " << global_node
                  << ": inner radius " << inner << " exceeds outer radius "
                  << outer;
          throw std::runtime_error(message.str());
        }

        double outer_re = 0.0;
        double outer_im = 0.0;
        cube_sum_reference(outer_re, outer_im, prefix_re, prefix_im, x, y, z,
                           outer, nx, ny, nz, local_node, node_count);

        double inner_re = 0.0;
        double inner_im = 0.0;
        cube_sum_reference(inner_re, inner_im, prefix_re, prefix_im, x, y, z,
                           inner, nx, ny, nz, local_node, node_count);

        const double weight = plan.w()[reference_dev][global_node];
        const double expected_re = (outer_re - inner_re) * weight;
        const double expected_im = (outer_im - inner_im) * weight;

        const std::size_t entry = static_cast<std::size_t>(local_cell) *
                                      static_cast<std::size_t>(node_count) +
                                  static_cast<std::size_t>(local_node);

        const double observed_re = workspace.rmt_sum_re()[dev][entry];
        const double observed_im = workspace.rmt_sum_im()[dev][entry];

        if (!std::isfinite(observed_re) || !std::isfinite(observed_im)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": non-finite remote sum for local cell " << local_cell
                  << ", global cell " << global_cell << ", local node "
                  << local_node;
          throw std::runtime_error(message.str());
        }

        const double error_re = std::abs(observed_re - expected_re);
        const double error_im = std::abs(observed_im - expected_im);
        const double allowed_re =
            rmt_threshold * std::max(1.0, std::abs(expected_re));
        const double allowed_im =
            rmt_threshold * std::max(1.0, std::abs(expected_im));

        max_error = std::max(max_error, error_re);
        max_error = std::max(max_error, error_im);

        if ((error_re > allowed_re) || (error_im > allowed_im)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": local remote-sum mismatch for local cell " << local_cell
                  << ", global cell " << global_cell << ", local node "
                  << local_node << ", global node " << global_node
                  << std::setprecision(17) << " (observed re=" << observed_re
                  << ", expected re=" << expected_re
                  << ", observed im=" << observed_im
                  << ", expected im=" << expected_im
                  << ", re error=" << error_re
                  << ", re tolerance=" << allowed_re
                  << ", im error=" << error_im
                  << ", im tolerance=" << allowed_im << ")";
          throw std::runtime_error(message.str());
        }
      }
    }

    for (std::size_t entry = active_rmt_entry_count;
         entry < rmt_buffer_capacity; entry++) {
      if ((workspace.rmt_sum_re()[dev][entry] != sentinel) ||
          (workspace.rmt_sum_im()[dev][entry] != sentinel)) {
        std::ostringstream message;
        message << "GPU " << dev << ", tile " << tile
                << ": remote summation wrote beyond the active local RMT "
                   "range at entry "
                << entry << " (active entries " << active_rmt_entry_count
                << ", capacity " << rmt_buffer_capacity << ")";
        throw std::runtime_error(message.str());
      }
    }
  }

  glst_force_test_access::zero_ef(force);
  glst_force_test_access::calc_lr_ef_tile(force, tile);

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaDeviceSynchronize());

    workspace.cell_atom_point()[dev].transfer_to_host();
    workspace.cell_atom_count()[dev].transfer_to_host();

    workspace.fx()[dev].transfer_to_host();
    workspace.fy()[dev].transfer_to_host();
    workspace.fz()[dev].transfer_to_host();
    workspace.en()[dev].transfer_to_host();
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    const unsigned int cell_partition =
        glst_force_test_access::dev_cell_partition(force, dev);

    const unsigned int device_tile_partition =
        glst_force_test_access::dev_tile_partition(force, dev);

    const bool owns_tile = (device_tile_partition == tile_partition);

    const unsigned int local_cell_count = plan.local_cell_count(cell_partition);

    const std::size_t atom_capacity = workspace.atom_capacity(dev);

    const std::size_t owned_atom_count = workspace.owned_atom_count(dev);

    require(owned_atom_count <= atom_capacity,
            "Owned atom count exceeds atom capacity");

    if (!owns_tile) {
      for (std::size_t atom = 0; atom < atom_capacity; atom++) {
        if ((workspace.fx()[dev][atom] != 0.0) ||
            (workspace.fy()[dev][atom] != 0.0) ||
            (workspace.fz()[dev][atom] != 0.0) ||
            (workspace.en()[dev][atom] != 0.0)) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile " << tile
                  << ": calc_lr_ef_tile modified a non-participating tile "
                     "rank at atom entry "
                  << atom;
          throw std::runtime_error(message.str());
        }
      }

      continue;
    }

    for (unsigned int local_cell = 0; local_cell < local_cell_count;
         local_cell++) {
      const unsigned int global_cell =
          plan.global_cell_from_local_cell(cell_partition, local_cell);

      const unsigned int atom_point =
          workspace.cell_atom_point()[dev][local_cell];

      const unsigned int atom_count =
          workspace.cell_atom_count()[dev][local_cell];

      require(atom_count == 1,
              "calc_sf_hybrid LR test requires one atom per owned cell");

      require(static_cast<std::size_t>(atom_point) < owned_atom_count,
              "Owned target atom point is outside the owned atom range");

      const double xa = atom_rx[global_cell];
      const double ya = atom_ry[global_cell];
      const double za = atom_rz[global_cell];
      const double qa = atom_qc[global_cell];

      double expected_fx = 0.0;
      double expected_fy = 0.0;
      double expected_fz = 0.0;
      double expected_en = 0.0;

      for (unsigned int local_node = 0; local_node < node_count; local_node++) {
        const unsigned int global_node = node_point + local_node;

        const double xc = plan.x()[dev][global_node];
        const double yc = plan.y()[dev][global_node];
        const double zc = plan.z()[dev][global_node];

        const std::size_t rmt_entry = static_cast<std::size_t>(local_cell) *
                                          static_cast<std::size_t>(node_count) +
                                      static_cast<std::size_t>(local_node);

        const double rmt_re = workspace.rmt_sum_re()[dev][rmt_entry];

        const double rmt_im = workspace.rmt_sum_im()[dev][rmt_entry];

        const double theta = xc * xa + yc * ya + zc * za;

        const double re = std::cos(theta);
        const double im = std::sin(theta);

        const double dre = qa * (re * rmt_re - im * rmt_im);

        const double dim = qa * (re * rmt_im + im * rmt_re);

        expected_fx += dim * xc;
        expected_fy += dim * yc;
        expected_fz += dim * zc;
        expected_en += dre;
      }

      const double observed_fx = workspace.fx()[dev][atom_point];

      const double observed_fy = workspace.fy()[dev][atom_point];

      const double observed_fz = workspace.fz()[dev][atom_point];

      const double observed_en = workspace.en()[dev][atom_point];

      if (!std::isfinite(observed_fx) || !std::isfinite(observed_fy) ||
          !std::isfinite(observed_fz) || !std::isfinite(observed_en)) {
        std::ostringstream message;
        message << "GPU " << dev << ", tile " << tile
                << ": non-finite long-range result for local cell "
                << local_cell << ", global cell " << global_cell;
        throw std::runtime_error(message.str());
      }

      const double error_fx = std::abs(observed_fx - expected_fx);

      const double error_fy = std::abs(observed_fy - expected_fy);

      const double error_fz = std::abs(observed_fz - expected_fz);

      const double error_en = std::abs(observed_en - expected_en);

      const double allowed_fx =
          ef_threshold * std::max(1.0, std::abs(expected_fx));

      const double allowed_fy =
          ef_threshold * std::max(1.0, std::abs(expected_fy));

      const double allowed_fz =
          ef_threshold * std::max(1.0, std::abs(expected_fz));

      const double allowed_en =
          ef_threshold * std::max(1.0, std::abs(expected_en));

      max_error = std::max(max_error, error_fx);
      max_error = std::max(max_error, error_fy);
      max_error = std::max(max_error, error_fz);
      max_error = std::max(max_error, error_en);

      if ((error_fx > allowed_fx) || (error_fy > allowed_fy) ||
          (error_fz > allowed_fz) || (error_en > allowed_en)) {
        std::ostringstream message;

        message << "GPU " << dev << ", tile " << tile
                << ": local long-range mismatch for local cell " << local_cell
                << ", global cell " << global_cell << std::setprecision(17)
                << " (observed fx=" << observed_fx
                << ", expected fx=" << expected_fx
                << ", observed fy=" << observed_fy
                << ", expected fy=" << expected_fy
                << ", observed fz=" << observed_fz
                << ", expected fz=" << expected_fz
                << ", observed en=" << observed_en
                << ", expected en=" << expected_en << ")";

        throw std::runtime_error(message.str());
      }
    }

    // Atom storage after owned_atom_count contains short-range halo atoms.
    // Long-range target evaluation must not modify them.
    for (std::size_t atom = owned_atom_count; atom < atom_capacity; atom++) {
      if ((workspace.fx()[dev][atom] != 0.0) ||
          (workspace.fy()[dev][atom] != 0.0) ||
          (workspace.fz()[dev][atom] != 0.0) ||
          (workspace.en()[dev][atom] != 0.0)) {
        std::ostringstream message;

        message << "GPU " << dev << ", tile " << tile
                << ": calc_lr_ef_tile modified halo atom entry " << atom
                << " (owned atoms " << owned_atom_count << ", capacity "
                << atom_capacity << ")";

        throw std::runtime_error(message.str());
      }
    }
  }

  return max_error;
}

int main(void) {
  try {
    int cuda_count = 0;
    cudaCheck(cudaGetDeviceCount(&cuda_count));

    if (cuda_count < 1) {
      std::cerr << "FAIL calc_sf_hybrid: no CUDA-capable device found"
                << std::endl;
      return EXIT_FAILURE;
    }

    constexpr unsigned int ncell_axis = 4;
    constexpr unsigned int natom = ncell_axis * ncell_axis * ncell_axis;
    constexpr double tol = 1.0e-6;
    constexpr double rcut = 12.0;
    constexpr double box = rcut * static_cast<double>(ncell_axis);

    std::vector<double> atom_rx(natom);
    std::vector<double> atom_ry(natom);
    std::vector<double> atom_rz(natom);
    std::vector<double> atom_qc(natom);

    cudaCheck(cudaSetDevice(0));

    cuda_container<double> rx(natom), ry(natom), rz(natom), qc(natom);

    for (unsigned int x = 0; x < ncell_axis; x++) {
      for (unsigned int y = 0; y < ncell_axis; y++) {
        for (unsigned int z = 0; z < ncell_axis; z++) {
          const unsigned int global_cell =
              (x * ncell_axis + y) * ncell_axis + z;

          atom_rx[global_cell] = (static_cast<double>(x) + 0.5) * rcut;
          atom_ry[global_cell] = (static_cast<double>(y) + 0.5) * rcut;
          atom_rz[global_cell] = (static_cast<double>(z) + 0.5) * rcut;
          atom_qc[global_cell] = (global_cell % 2 == 0) ? 1.0 : -1.0;

          rx[global_cell] = atom_rx[global_cell];
          ry[global_cell] = atom_ry[global_cell];
          rz[global_cell] = atom_rz[global_cell];
          qc[global_cell] = atom_qc[global_cell];
        }
      }
    }

    rx.transfer_to_device();
    ry.transfer_to_device();
    rz.transfer_to_device();
    qc.transfer_to_device();

    unsigned int cell_partition_count = 1;
    unsigned int tile_partition_count = 1;

    if (cuda_count == 1) {
      cell_partition_count = 1;
      tile_partition_count = 1;
    } else if ((cuda_count % 2) == 0) {
      cell_partition_count = 2;
      tile_partition_count = static_cast<unsigned int>(cuda_count / 2);
    } else {
      cell_partition_count = 1;
      tile_partition_count = static_cast<unsigned int>(cuda_count);
    }

    glst_force force;
    force.set_gpu_layout(cell_partition_count, tile_partition_count);
    force.init(natom, tol, box, box, box, rcut);
    force.assign_atoms(rx.d_array().data(), ry.d_array().data(),
                       rz.d_array().data(), qc.d_array().data());

    const glst_plan &plan = glst_force_test_access::plan(force);

    require(glst_force_test_access::cuda_count(force) == cuda_count,
            "glst_force CUDA device count does not match cudaGetDeviceCount");
    require(plan.ncell_x() == ncell_axis, "Unexpected x-cell count");
    require(plan.ncell_y() == ncell_axis, "Unexpected y-cell count");
    require(plan.ncell_z() == ncell_axis, "Unexpected z-cell count");
    require(plan.ncell() == natom,
            "Test requires exactly one atom per global cell");
    require(plan.cell_partition_count() == cell_partition_count,
            "Unexpected cell partition count");
    require(plan.tile_partition_count() == tile_partition_count,
            "Unexpected tile partition count");
    require(plan.tile_count() > 0, "Plan contains no cubature tiles");

    std::vector<unsigned int> tiles;
    append_unique_tile(tiles, 0);

    bool has_partial_tile = false;

    for (unsigned int partition = 0; partition < plan.tile_partition_count();
         partition++) {
      const std::vector<unsigned int> &partition_tiles =
          plan.partition_tile_idx(partition);

      if (partition_tiles.empty())
        continue;

      append_unique_tile(tiles, partition_tiles.front());
      append_unique_tile(tiles, partition_tiles.back());

      for (std::size_t i = 0; i < partition_tiles.size(); i++) {
        const unsigned int partition_tile = partition_tiles[i];

        if (plan.tile_node_count(partition_tile) < plan.max_tile_nodes()) {
          has_partial_tile = true;
          append_unique_tile(tiles, partition_tile);
          break;
        }
      }
    }

    append_unique_tile(tiles, plan.tile_count() - 1);

    require(has_partial_tile,
            "Test setup did not produce a partial cubature-node tile");

    double max_error = 0.0;
    for (std::size_t i = 0; i < tiles.size(); i++) {
      const double tile_error =
          verify_tile(force, atom_rx, atom_ry, atom_rz, atom_qc, tiles[i]);
      max_error = std::max(max_error, tile_error);
    }

    std::cout << "PASS calc_sf_hybrid: tested " << tiles.size() << " tiles on "
              << cuda_count << " GPU(s), layout " << cell_partition_count
              << " x " << tile_partition_count << ", max error "
              << std::scientific << std::setprecision(6) << max_error
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "FAIL calc_sf_hybrid: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
