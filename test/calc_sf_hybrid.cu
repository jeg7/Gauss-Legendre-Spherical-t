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

static double verify_tile(glst_force &force, const std::vector<double> &atom_rx,
                          const std::vector<double> &atom_ry,
                          const std::vector<double> &atom_rz,
                          const std::vector<double> &atom_qc,
                          const unsigned int tile) {
  constexpr double sentinel = 12345.0;
  constexpr double threshold = 1.0e-12;

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
  require(plan.x().size() == static_cast<std::size_t>(cuda_count),
          "Cubature x device count does not match glst_force");
  require(plan.y().size() == static_cast<std::size_t>(cuda_count),
          "Cubature y device count does not match glst_force");
  require(plan.z().size() == static_cast<std::size_t>(cuda_count),
          "Cubature z device count does not match glst_force");

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

  const std::size_t active_entry_count =
      static_cast<std::size_t>(plan.ncell()) *
      static_cast<std::size_t>(node_count);

  for (int dev = 0; dev < cuda_count; dev++) {
    const std::size_t buffer_capacity = workspace.sf_tile_buffer_capacity(dev);

    require(workspace.sf_re()[dev].size() == buffer_capacity,
            "Real SF container size does not match workspace capacity");
    require(workspace.sf_im()[dev].size() == buffer_capacity,
            "Imaginary SF container size does not match workspace capacity");

    const unsigned int device_tile_partition =
        glst_force_test_access::dev_tile_partition(force, dev);
    if (device_tile_partition == tile_partition) {
      require(active_entry_count <= buffer_capacity,
              "Active SF tile exceeds workspace capacity");
    }

    cudaCheck(cudaSetDevice(dev));
    workspace.sf_re()[dev].set(sentinel);
    workspace.sf_im()[dev].set(sentinel);
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

    const std::size_t buffer_capacity = workspace.sf_tile_buffer_capacity(dev);

    require(workspace.sf_re()[dev].size() == buffer_capacity,
            "Real SF container size does not match workspace capacity");
    require(workspace.sf_im()[dev].size() == buffer_capacity,
            "Imaginary SF container size does not match workspace capacity");

    if (!owns_tile) {
      for (std::size_t entry = 0; entry < buffer_capacity; entry++) {
        if ((workspace.sf_re()[dev][entry] != sentinel) ||
            (workspace.sf_im()[dev][entry] != sentinel)) {
          std::ostringstream message;
          message << "Device " << dev << " modified tile " << tile
                  << " owned by tile partition " << tile_partition
                  << " at SF entry " << entry;
          throw std::runtime_error(message.str());
        }
      }
      continue;
    }

    if (active_entry_count > buffer_capacity) {
      std::ostringstream message;
      message << "Active SF entries for tile " << tile << " ("
              << active_entry_count << ") exceed GPU " << dev << " capacity ("
              << buffer_capacity << ")";
      throw std::runtime_error(message.str());
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

        if ((error_re > threshold) || (error_im > threshold)) {
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

    for (std::size_t entry = active_entry_count; entry < buffer_capacity;
         entry++) {
      if ((workspace.sf_re()[dev][entry] != sentinel) ||
          (workspace.sf_im()[dev][entry] != sentinel)) {
        std::ostringstream message;
        message << "GPU " << dev << ", tile " << tile
                << ": write beyond active compact tile range at SF entry "
                << entry << " (active entries " << active_entry_count
                << ", capacity " << buffer_capacity << ")";
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

    for (unsigned int partition = 0; partition < plan.tile_partition_count();
         partition++) {
      const std::vector<unsigned int> &partition_tiles =
          plan.partition_tile_idx(partition);
      if (!partition_tiles.empty())
        append_unique_tile(tiles, partition_tiles.front());
    }

    const unsigned int final_tile = plan.tile_count() - 1;
    append_unique_tile(tiles, final_tile);

    bool has_partial_tile = false;
    for (unsigned int tile = 0; tile < plan.tile_count(); tile++) {
      if (plan.tile_node_count(tile) < plan.max_tile_nodes()) {
        has_partial_tile = true;
        append_unique_tile(tiles, tile);
        break;
      }
    }

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
