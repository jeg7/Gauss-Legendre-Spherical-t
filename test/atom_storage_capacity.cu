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
#include <glst_workspace.hcu>

#include <cub/cub.cuh>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

class glst_force_test_access {
public:
  static const glst_workspace &workspace(const glst_force &force) {
    if (force.workspace_ == nullptr) {
      throw std::runtime_error(
          "glst_force_test_access::workspace: Workspace is not initialized");
    }

    return *(force.workspace_);
  }

  static const glst_plan &plan(const glst_force &force) {
    if (force.plan_ == nullptr) {
      throw std::runtime_error(
          "glst_force_test_access::plan: Plan is not initialized");
    }

    return *(force.plan_);
  }
};

namespace {

constexpr int test_device_count = 2;
constexpr std::size_t atom_buffer_count = 14;
constexpr std::size_t global_scratch_buffer_count = 8;

struct storage_snapshot {
  std::array<std::size_t, test_device_count> atom_capacity;
  std::array<std::size_t, test_device_count> source_atom_count;
  std::array<std::size_t, test_device_count> owned_atom_count;
  std::array<std::size_t, test_device_count> atom_growth_count;
  std::array<std::size_t, test_device_count> cub_growth_count;
  std::array<std::size_t, test_device_count> cub_buffer_size;
  std::array<const void *, test_device_count> cub_buffer;
  std::array<std::array<const void *, atom_buffer_count>, test_device_count>
      atom_buffer;
  std::array<std::size_t, global_scratch_buffer_count> global_scratch_size;
  std::array<const void *, global_scratch_buffer_count> global_scratch_buffer;
};

void require(const bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);

  return;
}

std::array<std::size_t, global_scratch_buffer_count>
global_scratch_sizes(const glst_workspace &workspace) {
  return std::array<std::size_t, global_scratch_buffer_count>{
      workspace.global_sort_key_in().size(),
      workspace.global_sort_key_out().size(),
      workspace.global_packet_in().size(),
      workspace.global_packet_out().size(),
      workspace.global_cell_atom_count().size(),
      workspace.global_cell_atom_point().size(),
      workspace.global_x_plane_atom_point().size(),
      workspace.global_max_atoms_cell().size()};
}

std::array<const void *, global_scratch_buffer_count>
global_scratch_addresses(const glst_workspace &workspace) {
  return std::array<const void *, global_scratch_buffer_count>{
      static_cast<const void *>(workspace.global_sort_key_in().data()),
      static_cast<const void *>(workspace.global_sort_key_out().data()),
      static_cast<const void *>(workspace.global_packet_in().data()),
      static_cast<const void *>(workspace.global_packet_out().data()),
      static_cast<const void *>(workspace.global_cell_atom_count().data()),
      static_cast<const void *>(workspace.global_cell_atom_point().data()),
      static_cast<const void *>(workspace.global_x_plane_atom_point().data()),
      static_cast<const void *>(workspace.global_max_atoms_cell().data())};
}

void require_no_global_classification_scratch(const glst_workspace &workspace) {
  const std::array<std::size_t, global_scratch_buffer_count> sizes =
      global_scratch_sizes(workspace);

  const std::array<const void *, global_scratch_buffer_count> buffers =
      global_scratch_addresses(workspace);

  for (std::size_t i = 0; i < global_scratch_buffer_count; i++) {
    require(sizes[i] == 0,
            "Single-device workspace allocated global classification storage");

    require(buffers[i] == nullptr,
            "Single-device workspace has a global classification pointer");
  }

  return;
}

void require_global_classification_scratch(const glst_workspace &workspace,
                                           const std::size_t natom,
                                           const std::size_t ncell,
                                           const std::size_t ncell_x) {
  const std::array<std::size_t, global_scratch_buffer_count> expected_sizes{
      natom, natom, natom, natom, ncell, ncell + 1, ncell_x + 1, 1};

  const std::array<std::size_t, global_scratch_buffer_count> observed_sizes =
      global_scratch_sizes(workspace);

  require(observed_sizes == expected_sizes,
          "Global classification scratch sizes are incorrect");

  const std::array<const void *, global_scratch_buffer_count> buffers =
      global_scratch_addresses(workspace);

  cudaCheck(cudaSetDevice(0));

  for (std::size_t i = 0; i < global_scratch_buffer_count; i++) {
    require(buffers[i] != nullptr,
            "Global classification scratch contains a null pointer");

    cudaPointerAttributes attributes{};
    cudaCheck(cudaPointerGetAttributes(&attributes, buffers[i]));

    require(attributes.device == 0,
            "Global classification scratch was not allocated on GPU 0");
  }

  std::size_t sort_size = 0;
  std::size_t scan_size = 0;
  std::size_t reduce_size = 0;
  void *tmp = nullptr;

  atom_sort_key *key_in =
      const_cast<atom_sort_key *>(workspace.global_sort_key_in().data());

  atom_sort_key *key_out =
      const_cast<atom_sort_key *>(workspace.global_sort_key_out().data());

  atom_packet *packet_in =
      const_cast<atom_packet *>(workspace.global_packet_in().data());

  atom_packet *packet_out =
      const_cast<atom_packet *>(workspace.global_packet_out().data());

  unsigned int *cell_count =
      const_cast<unsigned int *>(workspace.global_cell_atom_count().data());

  unsigned int *cell_point =
      const_cast<unsigned int *>(workspace.global_cell_atom_point().data());

  unsigned int *max_atoms =
      const_cast<unsigned int *>(workspace.global_max_atoms_cell().data());

  cub::DeviceRadixSort::SortPairs(tmp, sort_size, key_in, key_out, packet_in,
                                  packet_out, static_cast<int>(natom), 0,
                                  static_cast<int>(8 * sizeof(atom_sort_key)));

  cub::DeviceScan::ExclusiveSum(tmp, scan_size, cell_count, cell_point,
                                static_cast<int>(ncell));

  cub::DeviceReduce::Max(tmp, reduce_size, cell_count, max_atoms,
                         static_cast<int>(ncell));

  std::size_t required_cub_size = sort_size;

  if (scan_size > required_cub_size)
    required_cub_size = scan_size;

  if (reduce_size > required_cub_size)
    required_cub_size = reduce_size;

  require(workspace.cub_work_buffer()[0] != nullptr,
          "GPU-0 CUB work buffer is null");

  require(workspace.cub_work_buffer_size()[0] >= required_cub_size,
          "GPU-0 CUB work buffer is too small for global classification");

  return;
}

std::array<const void *, atom_buffer_count>
atom_buffer_addresses(const glst_workspace &workspace, const int dev) {
  return std::array<const void *, atom_buffer_count>{
      static_cast<const void *>(workspace.idx()[dev].d_array().data()),
      static_cast<const void *>(workspace.sorted_idx()[dev].d_array().data()),
      static_cast<const void *>(workspace.rx()[dev].d_array().data()),
      static_cast<const void *>(workspace.ry()[dev].d_array().data()),
      static_cast<const void *>(workspace.rz()[dev].d_array().data()),
      static_cast<const void *>(workspace.qc()[dev].d_array().data()),
      static_cast<const void *>(workspace.packets()[dev].d_array().data()),
      static_cast<const void *>(
          workspace.sorted_packets()[dev].d_array().data()),
      static_cast<const void *>(
          workspace.atom_cell_idx()[dev].d_array().data()),
      static_cast<const void *>(
          workspace.atom_cell_sorted_idx()[dev].d_array().data()),
      static_cast<const void *>(workspace.fx()[dev].d_array().data()),
      static_cast<const void *>(workspace.fy()[dev].d_array().data()),
      static_cast<const void *>(workspace.fz()[dev].d_array().data()),
      static_cast<const void *>(workspace.en()[dev].d_array().data())};
}

void require_container_capacities(const glst_workspace &workspace,
                                  const int dev) {
  const std::size_t capacity = workspace.atom_capacity(dev);
  const std::string prefix =
      "Device " + std::to_string(dev) + " container size mismatch: ";

  require(workspace.idx()[dev].size() == capacity, prefix + "idx");
  require(workspace.sorted_idx()[dev].size() == capacity,
          prefix + "sorted_idx");
  require(workspace.rx()[dev].size() == capacity, prefix + "rx");
  require(workspace.ry()[dev].size() == capacity, prefix + "ry");
  require(workspace.rz()[dev].size() == capacity, prefix + "rz");
  require(workspace.qc()[dev].size() == capacity, prefix + "qc");
  require(workspace.packets()[dev].size() == capacity, prefix + "packets");
  require(workspace.sorted_packets()[dev].size() == capacity,
          prefix + "sorted_packets");
  require(workspace.atom_cell_idx()[dev].size() == capacity,
          prefix + "atom_cell_idx");
  require(workspace.atom_cell_sorted_idx()[dev].size() == capacity,
          prefix + "atom_cell_sorted_idx");
  require(workspace.fx()[dev].size() == capacity, prefix + "fx");
  require(workspace.fy()[dev].size() == capacity, prefix + "fy");
  require(workspace.fz()[dev].size() == capacity, prefix + "fz");
  require(workspace.en()[dev].size() == capacity, prefix + "en");

  return;
}

storage_snapshot take_snapshot(const glst_force &force) {
  const glst_workspace &workspace = glst_force_test_access::workspace(force);

  storage_snapshot snapshot{};

  for (int dev = 0; dev < test_device_count; dev++) {
    const std::size_t index = static_cast<std::size_t>(dev);

    require_container_capacities(workspace, dev);

    snapshot.atom_capacity[index] = workspace.atom_capacity(dev);
    snapshot.source_atom_count[index] = workspace.source_atom_count(dev);
    snapshot.owned_atom_count[index] = workspace.owned_atom_count(dev);
    snapshot.atom_growth_count[index] =
        workspace.atom_storage_growth_count(dev);
    snapshot.cub_growth_count[index] =
        workspace.cub_work_buffer_growth_count(dev);
    snapshot.cub_buffer_size[index] = workspace.cub_work_buffer_size()[dev];
    snapshot.cub_buffer[index] = workspace.cub_work_buffer()[dev];
    snapshot.atom_buffer[index] = atom_buffer_addresses(workspace, dev);
  }

  snapshot.global_scratch_size = global_scratch_sizes(workspace);
  snapshot.global_scratch_buffer = global_scratch_addresses(workspace);

  return snapshot;
}

void require_device_allocation_unchanged(const storage_snapshot &before,
                                         const storage_snapshot &after,
                                         const int dev,
                                         const std::string &label) {
  const std::size_t index = static_cast<std::size_t>(dev);
  const std::string prefix = label + ", device " + std::to_string(dev) + ": ";

  require(after.atom_capacity[index] == before.atom_capacity[index],
          prefix + "atom capacity changed");

  require(after.atom_growth_count[index] == before.atom_growth_count[index],
          prefix + "atom growth count changed");

  require(after.cub_growth_count[index] == before.cub_growth_count[index],
          prefix + "CUB growth count changed");

  require(after.cub_buffer_size[index] == before.cub_buffer_size[index],
          prefix + "CUB capacity changed");

  require(after.cub_buffer[index] == before.cub_buffer[index],
          prefix + "CUB pointer changed");

  require(after.atom_buffer[index] == before.atom_buffer[index],
          prefix + "at least one atom-array pointer changed");

  if (dev == 0) {
    require(after.global_scratch_size == before.global_scratch_size,
            prefix + "global classification sizes changed");

    require(after.global_scratch_buffer == before.global_scratch_buffer,
            prefix + "global classification pointers changed");
  }

  return;
}

void require_allocation_unchanged(const storage_snapshot &before,
                                  const storage_snapshot &after,
                                  const std::string &label) {
  for (int dev = 0; dev < test_device_count; dev++)
    require_device_allocation_unchanged(before, after, dev, label);

  return;
}

void require_active_counts(const storage_snapshot &snapshot,
                           const std::size_t source0, const std::size_t owned0,
                           const std::size_t source1, const std::size_t owned1,
                           const std::string &label) {
  require(snapshot.source_atom_count[0] == source0,
          label + ": unexpected device-0 source count");
  require(snapshot.owned_atom_count[0] == owned0,
          label + ": unexpected device-0 owned count");
  require(snapshot.source_atom_count[1] == source1,
          label + ": unexpected device-1 source count");
  require(snapshot.owned_atom_count[1] == owned1,
          label + ": unexpected device-1 owned count");

  for (int dev = 0; dev < test_device_count; dev++) {
    const std::size_t index = static_cast<std::size_t>(dev);

    require(snapshot.owned_atom_count[index] <=
                snapshot.source_atom_count[index],
            label + ": owned count exceeds source count");

    require(snapshot.source_atom_count[index] <= snapshot.atom_capacity[index],
            label + ": source count exceeds capacity");
  }

  return;
}

void set_distribution(cuda_container<double> &rx, cuda_container<double> &ry,
                      cuda_container<double> &rz, cuda_container<double> &qc,
                      const std::array<unsigned int, 4> &x_cell_counts) {
  constexpr double cell_width = 12.0;

  std::size_t atom = 0;

  for (unsigned int xcell = 0; xcell < x_cell_counts.size(); xcell++) {
    for (unsigned int i = 0; i < x_cell_counts[xcell]; i++) {
      const double sequence = static_cast<double>(atom + 1);

      rx[atom] = static_cast<double>(xcell) * cell_width + 1.0 +
                 std::fmod(0.137 * sequence, 10.0);

      ry[atom] = 1.0 + std::fmod(0.173 * sequence, 10.0);
      rz[atom] = 1.0 + std::fmod(0.197 * sequence, 10.0);
      qc[atom] = ((atom % 2) == 0) ? 1.0 : -1.0;

      atom++;
    }
  }

  require(atom == rx.size(), "Input x-cell counts do not sum to natom");

  cudaCheck(cudaSetDevice(0));
  rx.transfer_to_device();
  ry.transfer_to_device();
  rz.transfer_to_device();
  qc.transfer_to_device();

  return;
}

void assign_distribution(glst_force &force, cuda_container<double> &rx,
                         cuda_container<double> &ry, cuda_container<double> &rz,
                         cuda_container<double> &qc,
                         const std::array<unsigned int, 4> &x_cell_counts) {
  set_distribution(rx, ry, rz, qc, x_cell_counts);

  force.assign_atoms(rx.d_array().data(), ry.d_array().data(),
                     rz.d_array().data(), qc.d_array().data());

  for (int dev = 0; dev < test_device_count; dev++) {
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaDeviceSynchronize());
  }

  return;
}

} // namespace

int main(void) {
  try {
    int cuda_count = 0;
    cudaCheck(cudaGetDeviceCount(&cuda_count));

    require(cuda_count == test_device_count,
            "atom_storage_capacity requires exactly two visible GPUs");

    constexpr unsigned int natom = 12000;
    constexpr double tol = 1.0e-4;
    constexpr double box = 48.0;
    constexpr double rcut = 12.0;

    cudaCheck(cudaSetDevice(0));

    cuda_container<double> rx(natom);
    cuda_container<double> ry(natom);
    cuda_container<double> rz(natom);
    cuda_container<double> qc(natom);

    glst_force force;
    force.set_gpu_layout(2, 1);
    force.init(natom, tol, box, box, box, rcut);

    {
      glst_workspace single_device_workspace(
          glst_force_test_access::plan(force), 1);

      require_no_global_classification_scratch(single_device_workspace);
    }

    require_global_classification_scratch(
        glst_force_test_access::workspace(force), natom, 64, 4);

    const storage_snapshot initial = take_snapshot(force);

    require(initial.atom_capacity[0] == 6000,
            "Unexpected initial device-0 capacity");
    require(initial.atom_capacity[1] == 6000,
            "Unexpected initial device-1 capacity");
    require(initial.source_atom_count[0] == 0,
            "Initial device-0 source count is not zero");
    require(initial.source_atom_count[1] == 0,
            "Initial device-1 source count is not zero");
    require(initial.owned_atom_count[0] == 0,
            "Initial device-0 owned count is not zero");
    require(initial.owned_atom_count[1] == 0,
            "Initial device-1 owned count is not zero");
    require(initial.atom_growth_count[0] == 0,
            "Initial device-0 atom growth count is not zero");
    require(initial.atom_growth_count[1] == 0,
            "Initial device-1 atom growth count is not zero");
    require(initial.cub_growth_count[0] == 0,
            "Initial device-0 CUB growth count is not zero");
    require(initial.cub_growth_count[1] == 0,
            "Initial device-1 CUB growth count is not zero");

    /*
     * Partition 0 owns x cells 0-1 and sources x cells 0-2.
     * Partition 1 owns x cells 2-3 and sources x cells 2-3, then x cell 1.
     */
    const std::array<unsigned int, 4> distribution_a{2000, 1000, 2000, 7000};

    assign_distribution(force, rx, ry, rz, qc, distribution_a);
    const storage_snapshot first = take_snapshot(force);

    require_active_counts(first, 5000, 3000, 10000, 9000, "First assignment");

    require_device_allocation_unchanged(initial, first, 0, "First assignment");

    require(first.atom_capacity[1] == 12288,
            "First assignment did not apply 4096-atom growth quantization");

    require(first.atom_growth_count[1] == 1,
            "First assignment did not record one device-1 atom growth");

    require(first.atom_buffer[1] != initial.atom_buffer[1],
            "First assignment did not replace device-1 atom buffers");

    assign_distribution(force, rx, ry, rz, qc, distribution_a);
    const storage_snapshot repeated = take_snapshot(force);

    require_active_counts(repeated, 5000, 3000, 10000, 9000,
                          "Repeated identical assignment");

    require_allocation_unchanged(first, repeated,
                                 "Repeated identical assignment");

    const std::array<unsigned int, 4> distribution_b{2500, 1000, 2000, 6500};

    assign_distribution(force, rx, ry, rz, qc, distribution_b);
    const storage_snapshot below_capacity = take_snapshot(force);

    require_active_counts(below_capacity, 5500, 3500, 9500, 8500,
                          "Changed counts below capacity");

    require_allocation_unchanged(repeated, below_capacity,
                                 "Changed counts below capacity");

    const std::array<unsigned int, 4> distribution_c{3000, 2000, 3000, 4000};

    assign_distribution(force, rx, ry, rz, qc, distribution_c);
    const storage_snapshot grown = take_snapshot(force);

    require_active_counts(grown, 8000, 5000, 9000, 7000,
                          "Capacity-exceeding assignment");

    require(grown.atom_capacity[0] == 8192,
            "Device-0 capacity did not grow to 8192");

    require(grown.atom_growth_count[0] ==
                below_capacity.atom_growth_count[0] + 1,
            "Device-0 did not record exactly one atom-storage growth");

    require(grown.atom_buffer[0] != below_capacity.atom_buffer[0],
            "Device-0 atom-array pointers did not change during growth");

    require(grown.global_scratch_size == below_capacity.global_scratch_size,
            "GPU-0 global classification sizes changed during atom growth");

    require(grown.global_scratch_buffer == below_capacity.global_scratch_buffer,
            "GPU-0 global classification pointers changed during atom growth");

    require(grown.cub_buffer[0] == below_capacity.cub_buffer[0],
            "GPU-0 CUB buffer reallocated after initialization");

    require(grown.cub_growth_count[0] == below_capacity.cub_growth_count[0],
            "GPU-0 CUB growth was recorded after initialization");

    require_device_allocation_unchanged(below_capacity, grown, 1,
                                        "Capacity-exceeding assignment");

    assign_distribution(force, rx, ry, rz, qc, distribution_c);
    const storage_snapshot grown_repeated = take_snapshot(force);

    require_active_counts(grown_repeated, 8000, 5000, 9000, 7000,
                          "Repeated grown assignment");

    require_allocation_unchanged(grown, grown_repeated,
                                 "Repeated grown assignment");

    const std::array<unsigned int, 4> distribution_d{4000, 0, 0, 8000};

    assign_distribution(force, rx, ry, rz, qc, distribution_d);
    const storage_snapshot shrunk = take_snapshot(force);

    require_active_counts(shrunk, 4000, 4000, 8000, 8000,
                          "Shrunken active counts");

    require_allocation_unchanged(grown_repeated, shrunk,
                                 "Shrunken active counts");

    std::cout << "atom_storage_capacity: PASS" << std::endl;
    std::cout << "  device 0 capacity: " << shrunk.atom_capacity[0]
              << ", atom growth events: " << shrunk.atom_growth_count[0]
              << ", CUB growth events: " << shrunk.cub_growth_count[0]
              << std::endl;

    std::cout << "  device 1 capacity: " << shrunk.atom_capacity[1]
              << ", atom growth events: " << shrunk.atom_growth_count[1]
              << ", CUB growth events: " << shrunk.cub_growth_count[1]
              << std::endl;

    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "atom_storage_capacity: FAIL: " << error.what() << std::endl;

    return EXIT_FAILURE;
  }
}
