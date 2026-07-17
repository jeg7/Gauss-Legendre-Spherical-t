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
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

  static unsigned int cell_partition_count(const glst_force &force) {
    return force.cell_partition_count_;
  }

  static unsigned int tile_partition_count(const glst_force &force) {
    return force.tile_partition_count_;
  }

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

struct ef_snapshot {
  std::vector<std::vector<double>> fx;
  std::vector<std::vector<double>> fy;
  std::vector<std::vector<double>> fz;
  std::vector<std::vector<double>> en;
};

static void require(const bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

static int find_device(const glst_force &force,
                       const unsigned int cell_partition,
                       const unsigned int tile_partition) {
  const int cuda_count = glst_force_test_access::cuda_count(force);

  for (int dev = 0; dev < cuda_count; dev++) {
    if ((glst_force_test_access::dev_cell_partition(force, dev) ==
         cell_partition) &&
        (glst_force_test_access::dev_tile_partition(force, dev) ==
         tile_partition)) {
      return dev;
    }
  }

  std::ostringstream message;
  message << "No device maps to cell partition " << cell_partition
          << " and tile partition " << tile_partition;
  throw std::runtime_error(message.str());
}

static ef_snapshot snapshot_ef(glst_workspace &workspace,
                               const int cuda_count) {
  ef_snapshot snapshot;

  snapshot.fx.resize(cuda_count);
  snapshot.fy.resize(cuda_count);
  snapshot.fz.resize(cuda_count);
  snapshot.en.resize(cuda_count);

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaDeviceSynchronize());

    workspace.fx()[dev].transfer_to_host();
    workspace.fy()[dev].transfer_to_host();
    workspace.fz()[dev].transfer_to_host();
    workspace.en()[dev].transfer_to_host();

    snapshot.fx[dev] = workspace.fx()[dev].h_array();
    snapshot.fy[dev] = workspace.fy()[dev].h_array();
    snapshot.fz[dev] = workspace.fz()[dev].h_array();
    snapshot.en[dev] = workspace.en()[dev].h_array();
  }

  return snapshot;
}

static double halo_sentinel(const int dev, const unsigned int component,
                            const std::size_t atom) {
  const double value =
      100000000.0 + 1000000.0 * static_cast<double>(component + 1) +
      10000.0 * static_cast<double>(dev + 1) + static_cast<double>(atom + 1);

  return ((component % 2) == 0) ? value : -value;
}

static double verify_close(const double observed, const double expected,
                           const double threshold,
                           const std::string &description) {
  if (!std::isfinite(observed)) {
    throw std::runtime_error(description + ": observed value is not finite");
  }

  if (!std::isfinite(expected)) {
    throw std::runtime_error(description + ": expected value is not finite");
  }

  const double error = std::abs(observed - expected);
  const double allowed = threshold * std::max(1.0, std::abs(expected));

  if (error > allowed) {
    std::ostringstream message;
    message << description << std::setprecision(17) << ": observed=" << observed
            << ", expected=" << expected << ", error=" << error
            << ", tolerance=" << allowed;
    throw std::runtime_error(message.str());
  }

  return error;
}

static void
append_layout(std::vector<std::pair<unsigned int, unsigned int>> &layouts,
              const unsigned int cell_partition_count,
              const unsigned int tile_partition_count) {
  const std::pair<unsigned int, unsigned int> layout(cell_partition_count,
                                                     tile_partition_count);

  if (std::find(layouts.begin(), layouts.end(), layout) == layouts.end())
    layouts.push_back(layout);

  return;
}

static double run_case(const unsigned int cell_partition_count,
                       const unsigned int tile_partition_count) {
  const unsigned long long int layout_product =
      static_cast<unsigned long long int>(cell_partition_count) *
      static_cast<unsigned long long int>(tile_partition_count);

  require(cell_partition_count > 0, "Cell partition count is zero");
  require(tile_partition_count > 0, "Tile partition count is zero");
  require(layout_product <= static_cast<unsigned long long int>(
                                std::numeric_limits<int>::max()),
          "GPU layout product exceeds int range");

  int visible_cuda_count = 0;
  cudaCheck(cudaGetDeviceCount(&visible_cuda_count));

  require(layout_product ==
              static_cast<unsigned long long int>(visible_cuda_count),
          "Requested GPU layout does not match visible CUDA device count");

  const unsigned int ncell_axis = std::max(5u, cell_partition_count + 1u);
  const std::size_t natom_size = static_cast<std::size_t>(ncell_axis) *
                                 static_cast<std::size_t>(ncell_axis) *
                                 static_cast<std::size_t>(ncell_axis);

  require(natom_size <= static_cast<std::size_t>(
                            std::numeric_limits<unsigned int>::max()),
          "Test atom count exceeds unsigned int range");

  const unsigned int natom = static_cast<unsigned int>(natom_size);
  constexpr double tol = 1.0e-6;
  constexpr double rcut = 12.0;
  const double box = rcut * static_cast<double>(ncell_axis);

  std::vector<double> atom_rx(natom);
  std::vector<double> atom_ry(natom);
  std::vector<double> atom_rz(natom);
  std::vector<double> atom_qc(natom);

  cudaCheck(cudaSetDevice(0));

  cuda_container<double> rx(natom), ry(natom), rz(natom), qc(natom);

  for (unsigned int x = 0; x < ncell_axis; x++) {
    for (unsigned int y = 0; y < ncell_axis; y++) {
      for (unsigned int z = 0; z < ncell_axis; z++) {
        const unsigned int atom = (x * ncell_axis + y) * ncell_axis + z;

        atom_rx[atom] = (static_cast<double>(x) + 0.5) * rcut;
        atom_ry[atom] = (static_cast<double>(y) + 0.5) * rcut;
        atom_rz[atom] = (static_cast<double>(z) + 0.5) * rcut;

        switch (atom % 4) {
        case 0:
          atom_qc[atom] = 1.0;
          break;
        case 1:
          atom_qc[atom] = -0.75;
          break;
        case 2:
          atom_qc[atom] = 0.5;
          break;
        default:
          atom_qc[atom] = -0.25;
          break;
        }

        rx[atom] = atom_rx[atom];
        ry[atom] = atom_ry[atom];
        rz[atom] = atom_rz[atom];
        qc[atom] = atom_qc[atom];
      }
    }
  }

  rx.transfer_to_device();
  ry.transfer_to_device();
  rz.transfer_to_device();
  qc.transfer_to_device();

  glst_force force;
  force.set_gpu_layout(cell_partition_count, tile_partition_count);
  force.init(natom, tol, box, box, box, rcut);
  force.assign_atoms(rx.d_array().data(), ry.d_array().data(),
                     rz.d_array().data(), qc.d_array().data());

  const glst_plan &plan = glst_force_test_access::plan(force);
  glst_workspace &workspace = glst_force_test_access::workspace(force);
  const int cuda_count = glst_force_test_access::cuda_count(force);

  require(cuda_count == visible_cuda_count,
          "glst_force CUDA count does not match visible CUDA count");
  require(glst_force_test_access::cell_partition_count(force) ==
              cell_partition_count,
          "glst_force cell partition count does not match requested layout");
  require(glst_force_test_access::tile_partition_count(force) ==
              tile_partition_count,
          "glst_force tile partition count does not match requested layout");

  require(plan.natom() == natom, "Unexpected atom count in plan");
  require(plan.ncell_x() == ncell_axis, "Unexpected x-cell count");
  require(plan.ncell_y() == ncell_axis, "Unexpected y-cell count");
  require(plan.ncell_z() == ncell_axis, "Unexpected z-cell count");
  require(plan.ncell() == natom,
          "Test requires exactly one atom per global cell");
  require(plan.cell_partition_count() == cell_partition_count,
          "Plan cell partition count does not match requested layout");
  require(plan.tile_partition_count() == tile_partition_count,
          "Plan tile partition count does not match requested layout");
  require(plan.tile_count() > 0, "Plan contains no cubature tiles");

  std::vector<unsigned int> layout_visit_count(
      static_cast<std::size_t>(cell_partition_count) *
          static_cast<std::size_t>(tile_partition_count),
      0u);

  for (int dev = 0; dev < cuda_count; dev++) {
    const unsigned int cell_partition =
        glst_force_test_access::dev_cell_partition(force, dev);
    const unsigned int tile_partition =
        glst_force_test_access::dev_tile_partition(force, dev);

    require(cell_partition < cell_partition_count,
            "Device cell partition is out of range");
    require(tile_partition < tile_partition_count,
            "Device tile partition is out of range");

    const std::size_t layout_index =
        static_cast<std::size_t>(cell_partition) *
            static_cast<std::size_t>(tile_partition_count) +
        static_cast<std::size_t>(tile_partition);
    layout_visit_count[layout_index]++;
  }

  for (std::size_t i = 0; i < layout_visit_count.size(); i++) {
    require(
        layout_visit_count[i] == 1,
        "GPU layout does not contain exactly one device per partition pair");
  }

  for (unsigned int cell_partition = 0; cell_partition < cell_partition_count;
       cell_partition++) {
    const int root_dev = find_device(force, cell_partition, 0);
    const std::size_t root_owned_atom_count =
        workspace.owned_atom_count(root_dev);
    const std::size_t root_atom_capacity = workspace.atom_capacity(root_dev);

    require(root_owned_atom_count > 0, "Test cell partition owns no atoms");
    require(root_owned_atom_count <= root_atom_capacity,
            "Root owned atom count exceeds atom capacity");

    for (unsigned int tile_partition = 0; tile_partition < tile_partition_count;
         tile_partition++) {
      const int dev = find_device(force, cell_partition, tile_partition);

      require(workspace.owned_atom_count(dev) == root_owned_atom_count,
              "Tile ranks in one cell partition have different owned atom "
              "counts");
      require(workspace.atom_capacity(dev) == root_atom_capacity,
              "Tile ranks in one cell partition have different atom "
              "capacities");

      for (std::size_t atom = 0; atom < root_owned_atom_count; atom++) {
        if (workspace.idx()[dev][atom] != workspace.idx()[root_dev][atom]) {
          std::ostringstream message;
          message << "Cell partition " << cell_partition << ", tile partition "
                  << tile_partition
                  << ": owned atom ordering differs at local atom " << atom;
          throw std::runtime_error(message.str());
        }
      }
    }
  }

  glst_force_test_access::zero_ef(force);

  for (unsigned int tile = 0; tile < plan.tile_count(); tile++) {
    glst_force_test_access::calc_sf_tile(force, tile);
    glst_force_test_access::exchange_sf_tile(force, tile);
    glst_force_test_access::sum_rmt_sf_tile(force, tile);
    glst_force_test_access::calc_lr_ef_tile(force, tile);
  }

  const ef_snapshot pre_short_range = snapshot_ef(workspace, cuda_count);

  force.calc_sr_ef();

  const ef_snapshot post_short_range = snapshot_ef(workspace, cuda_count);

  std::vector<bool> short_range_changed_root(cell_partition_count, false);

  for (int dev = 0; dev < cuda_count; dev++) {
    const unsigned int cell_partition =
        glst_force_test_access::dev_cell_partition(force, dev);
    const unsigned int tile_partition =
        glst_force_test_access::dev_tile_partition(force, dev);
    const std::size_t owned_atom_count = workspace.owned_atom_count(dev);
    const std::size_t atom_capacity = workspace.atom_capacity(dev);

    require(owned_atom_count <= atom_capacity,
            "Owned atom count exceeds atom capacity after short range");

    if (tile_partition != 0) {
      for (std::size_t atom = 0; atom < atom_capacity; atom++) {
        if ((post_short_range.fx[dev][atom] != pre_short_range.fx[dev][atom]) ||
            (post_short_range.fy[dev][atom] != pre_short_range.fy[dev][atom]) ||
            (post_short_range.fz[dev][atom] != pre_short_range.fz[dev][atom]) ||
            (post_short_range.en[dev][atom] != pre_short_range.en[dev][atom])) {
          std::ostringstream message;
          message << "GPU " << dev << ", tile partition " << tile_partition
                  << ": short-range calculation modified non-root atom entry "
                  << atom;
          throw std::runtime_error(message.str());
        }
      }
    } else {
      for (std::size_t atom = 0; atom < owned_atom_count; atom++) {
        if ((post_short_range.fx[dev][atom] != pre_short_range.fx[dev][atom]) ||
            (post_short_range.fy[dev][atom] != pre_short_range.fy[dev][atom]) ||
            (post_short_range.fz[dev][atom] != pre_short_range.fz[dev][atom]) ||
            (post_short_range.en[dev][atom] != pre_short_range.en[dev][atom])) {
          short_range_changed_root[cell_partition] = true;
        }
      }

      for (std::size_t atom = owned_atom_count; atom < atom_capacity; atom++) {
        if ((post_short_range.fx[dev][atom] != pre_short_range.fx[dev][atom]) ||
            (post_short_range.fy[dev][atom] != pre_short_range.fy[dev][atom]) ||
            (post_short_range.fz[dev][atom] != pre_short_range.fz[dev][atom]) ||
            (post_short_range.en[dev][atom] != pre_short_range.en[dev][atom])) {
          std::ostringstream message;
          message << "GPU " << dev
                  << ": short-range calculation modified halo atom entry "
                  << atom << " (owned atoms " << owned_atom_count
                  << ", capacity " << atom_capacity << ")";
          throw std::runtime_error(message.str());
        }
      }
    }
  }

  for (unsigned int cell_partition = 0; cell_partition < cell_partition_count;
       cell_partition++) {
    if (!short_range_changed_root[cell_partition]) {
      std::ostringstream message;
      message << "Short-range calculation did not modify any owned atom for "
                 "cell partition "
              << cell_partition << " on tile partition 0";
      throw std::runtime_error(message.str());
    }
  }

  for (int dev = 0; dev < cuda_count; dev++) {
    cudaCheck(cudaSetDevice(dev));

    const std::size_t owned_atom_count = workspace.owned_atom_count(dev);
    const std::size_t atom_capacity = workspace.atom_capacity(dev);

    for (std::size_t atom = owned_atom_count; atom < atom_capacity; atom++) {
      workspace.fx()[dev][atom] = halo_sentinel(dev, 0, atom);
      workspace.fy()[dev][atom] = halo_sentinel(dev, 1, atom);
      workspace.fz()[dev][atom] = halo_sentinel(dev, 2, atom);
      workspace.en()[dev][atom] = halo_sentinel(dev, 3, atom);
    }

    workspace.fx()[dev].transfer_to_device();
    workspace.fy()[dev].transfer_to_device();
    workspace.fz()[dev].transfer_to_device();
    workspace.en()[dev].transfer_to_device();
  }

  const ef_snapshot pre_reduce = snapshot_ef(workspace, cuda_count);

  std::vector<int> root_devs(cell_partition_count, -1);
  std::vector<std::size_t> owned_atom_counts(cell_partition_count, 0);
  std::vector<std::vector<double>> expected_fx(cell_partition_count);
  std::vector<std::vector<double>> expected_fy(cell_partition_count);
  std::vector<std::vector<double>> expected_fz(cell_partition_count);
  std::vector<std::vector<double>> expected_en(cell_partition_count);

  for (unsigned int cell_partition = 0; cell_partition < cell_partition_count;
       cell_partition++) {
    const int root_dev = find_device(force, cell_partition, 0);
    const std::size_t owned_atom_count = workspace.owned_atom_count(root_dev);

    root_devs[cell_partition] = root_dev;
    owned_atom_counts[cell_partition] = owned_atom_count;

    expected_fx[cell_partition].assign(owned_atom_count, 0.0);
    expected_fy[cell_partition].assign(owned_atom_count, 0.0);
    expected_fz[cell_partition].assign(owned_atom_count, 0.0);
    expected_en[cell_partition].assign(owned_atom_count, 0.0);

    for (unsigned int tile_partition = 0; tile_partition < tile_partition_count;
         tile_partition++) {
      const int dev = find_device(force, cell_partition, tile_partition);

      require(workspace.owned_atom_count(dev) == owned_atom_count,
              "Tile communicator ranks have different owned atom counts");

      for (std::size_t atom = 0; atom < owned_atom_count; atom++) {
        expected_fx[cell_partition][atom] += pre_reduce.fx[dev][atom];
        expected_fy[cell_partition][atom] += pre_reduce.fy[dev][atom];
        expected_fz[cell_partition][atom] += pre_reduce.fz[dev][atom];
        expected_en[cell_partition][atom] += pre_reduce.en[dev][atom];
      }
    }
  }

  force.comm_ef();

  const ef_snapshot post_reduce = snapshot_ef(workspace, cuda_count);

  constexpr double threshold = 1.0e-10;
  double max_error = 0.0;
  std::size_t reduced_value_count = 0;
  std::size_t checked_halo_count = 0;

  for (unsigned int cell_partition = 0; cell_partition < cell_partition_count;
       cell_partition++) {
    const int root_dev = root_devs[cell_partition];
    const std::size_t owned_atom_count = owned_atom_counts[cell_partition];
    const std::size_t atom_capacity = workspace.atom_capacity(root_dev);

    for (std::size_t atom = 0; atom < owned_atom_count; atom++) {
      std::ostringstream prefix;
      prefix << "cell partition " << cell_partition << ", root GPU " << root_dev
             << ", local atom " << atom;

      double error = verify_close(post_reduce.fx[root_dev][atom],
                                  expected_fx[cell_partition][atom], threshold,
                                  prefix.str() + ", fx");
      max_error = std::max(max_error, error);

      error = verify_close(post_reduce.fy[root_dev][atom],
                           expected_fy[cell_partition][atom], threshold,
                           prefix.str() + ", fy");
      max_error = std::max(max_error, error);

      error = verify_close(post_reduce.fz[root_dev][atom],
                           expected_fz[cell_partition][atom], threshold,
                           prefix.str() + ", fz");
      max_error = std::max(max_error, error);

      error = verify_close(post_reduce.en[root_dev][atom],
                           expected_en[cell_partition][atom], threshold,
                           prefix.str() + ", en");
      max_error = std::max(max_error, error);

      reduced_value_count += 4;
    }

    for (std::size_t atom = owned_atom_count; atom < atom_capacity; atom++) {
      if ((post_reduce.fx[root_dev][atom] != pre_reduce.fx[root_dev][atom]) ||
          (post_reduce.fy[root_dev][atom] != pre_reduce.fy[root_dev][atom]) ||
          (post_reduce.fz[root_dev][atom] != pre_reduce.fz[root_dev][atom]) ||
          (post_reduce.en[root_dev][atom] != pre_reduce.en[root_dev][atom])) {
        std::ostringstream message;
        message << "Cell partition " << cell_partition << ", root GPU "
                << root_dev << ": reduction modified halo atom entry " << atom
                << " (owned atoms " << owned_atom_count << ", capacity "
                << atom_capacity << ")";
        throw std::runtime_error(message.str());
      }

      checked_halo_count += 4;
    }
  }

  require(reduced_value_count > 0,
          "Reduction test did not check any owned force/energy values");

  // Run the same calculation through the public orchestration method. The
  // workspace is deliberately nonzero here, so this also verifies that
  // calc_ener_force zeroes old force/energy values before accumulating.
  force.calc_ener_force(rx.d_array().data(), ry.d_array().data(),
                        rz.d_array().data(), qc.d_array().data());

  const ef_snapshot orchestrated = snapshot_ef(workspace, cuda_count);

  double orchestration_max_error = 0.0;
  std::size_t orchestration_value_count = 0;
  std::size_t zero_halo_value_count = 0;

  for (unsigned int cell_partition = 0; cell_partition < cell_partition_count;
       cell_partition++) {
    const int root_dev = root_devs[cell_partition];
    const std::size_t owned_atom_count = owned_atom_counts[cell_partition];
    const std::size_t atom_capacity = workspace.atom_capacity(root_dev);

    require(workspace.owned_atom_count(root_dev) == owned_atom_count,
            "calc_ener_force changed the root owned atom count");

    for (std::size_t atom = 0; atom < owned_atom_count; atom++) {
      std::ostringstream prefix;
      prefix << "calc_ener_force, cell partition " << cell_partition
             << ", root GPU " << root_dev << ", local atom " << atom;

      double error = verify_close(orchestrated.fx[root_dev][atom],
                                  expected_fx[cell_partition][atom], threshold,
                                  prefix.str() + ", fx");
      orchestration_max_error = std::max(orchestration_max_error, error);

      error = verify_close(orchestrated.fy[root_dev][atom],
                           expected_fy[cell_partition][atom], threshold,
                           prefix.str() + ", fy");
      orchestration_max_error = std::max(orchestration_max_error, error);

      error = verify_close(orchestrated.fz[root_dev][atom],
                           expected_fz[cell_partition][atom], threshold,
                           prefix.str() + ", fz");
      orchestration_max_error = std::max(orchestration_max_error, error);

      error = verify_close(orchestrated.en[root_dev][atom],
                           expected_en[cell_partition][atom], threshold,
                           prefix.str() + ", en");
      orchestration_max_error = std::max(orchestration_max_error, error);

      orchestration_value_count += 4;
    }

    // calc_ener_force must zero the complete local allocation, while all
    // calculation and reduction kernels are restricted to owned atoms.
    for (std::size_t atom = owned_atom_count; atom < atom_capacity; atom++) {
      if ((orchestrated.fx[root_dev][atom] != 0.0) ||
          (orchestrated.fy[root_dev][atom] != 0.0) ||
          (orchestrated.fz[root_dev][atom] != 0.0) ||
          (orchestrated.en[root_dev][atom] != 0.0)) {
        std::ostringstream message;
        message << "calc_ener_force, cell partition " << cell_partition
                << ", root GPU " << root_dev
                << ": halo force/energy entry was modified at local atom "
                << atom << " (owned atoms " << owned_atom_count << ", capacity "
                << atom_capacity << ")";
        throw std::runtime_error(message.str());
      }

      zero_halo_value_count += 4;
    }
  }

  require(orchestration_value_count == reduced_value_count,
          "calc_ener_force checked value count does not match the manual "
          "reduction reference");

  max_error = std::max(max_error, orchestration_max_error);

  std::cout << "PASS reduce_tile_partition_ef: layout " << cell_partition_count
            << " x " << tile_partition_count << ", " << ncell_axis
            << "^3 cells, " << reduced_value_count << " reduced values, "
            << orchestration_value_count << " calc_ener_force values, "
            << checked_halo_count << " preserved sentinel halo values, "
            << zero_halo_value_count << " zeroed halo values, max error "
            << std::scientific << std::setprecision(6) << max_error
            << std::endl;

  return max_error;
}

int main(void) {
  try {
    int cuda_count = 0;
    cudaCheck(cudaGetDeviceCount(&cuda_count));

    if (cuda_count < 1) {
      std::cerr << "FAIL reduce_tile_partition_ef: no CUDA-capable device found"
                << std::endl;
      return EXIT_FAILURE;
    }

    std::vector<std::pair<unsigned int, unsigned int>> layouts;

    if (cuda_count == 1) {
      append_layout(layouts, 1, 1);
    } else {
      append_layout(layouts, static_cast<unsigned int>(cuda_count), 1);
      append_layout(layouts, 1, static_cast<unsigned int>(cuda_count));

      for (unsigned int cell_partition_count = 2;
           cell_partition_count < static_cast<unsigned int>(cuda_count);
           cell_partition_count++) {
        const unsigned int visible_count =
            static_cast<unsigned int>(cuda_count);

        if ((visible_count % cell_partition_count) != 0)
          continue;

        const unsigned int tile_partition_count =
            visible_count / cell_partition_count;

        if (tile_partition_count <= 1)
          continue;

        append_layout(layouts, cell_partition_count, tile_partition_count);
        break;
      }
    }

    double max_error = 0.0;

    for (std::size_t i = 0; i < layouts.size(); i++) {
      const double case_error = run_case(layouts[i].first, layouts[i].second);
      max_error = std::max(max_error, case_error);
    }

    std::cout << "PASS reduce_tile_partition_ef: completed " << layouts.size()
              << " layout(s) on " << cuda_count
              << " visible GPU(s), overall max error " << std::scientific
              << std::setprecision(6) << max_error << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "FAIL reduce_tile_partition_ef: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
