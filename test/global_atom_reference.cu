// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include <atom_packet.hcu>
#include <cuda_container.hcu>
#include <cuda_utils.hcu>
#include <glst_force.hcu>
#include <glst_plan.hcu>
#include <glst_workspace.hcu>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
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

  static const glst_workspace &workspace(const glst_force &force) {
    if (force.workspace_ == nullptr) {
      throw std::runtime_error(
          "glst_force_test_access::workspace: Workspace is not initialized");
    }

    return *(force.workspace_);
  }

  static void build_global_atom_reference(glst_force &force, const double *d_rx,
                                          const double *d_ry,
                                          const double *d_rz,
                                          const double *d_qc) {
    force.build_global_atom_reference(d_rx, d_ry, d_rz, d_qc);
    return;
  }

  static void synchronize_global_atom_reference(glst_force &force) {
    if (force.comp_streams_.empty()) {
      throw std::runtime_error(
          "glst_force_test_access::synchronize_global_atom_reference: "
          "GPU 0 compute stream is not initialized");
    }

    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaStreamSynchronize(force.comp_streams_[0]));

    return;
  }
};

namespace {

constexpr unsigned int repeat_count = 8;
constexpr std::size_t scratch_buffer_count = 8;

struct position {
  double x;
  double y;
  double z;
};

struct reference_snapshot {
  std::vector<atom_sort_key> key;
  std::vector<atom_packet> packet;
  std::vector<unsigned int> cell_count;
  std::vector<unsigned int> cell_point;
  std::vector<unsigned int> x_plane_point;
};

struct storage_snapshot {
  std::array<std::size_t, scratch_buffer_count> scratch_size;
  std::array<const void *, scratch_buffer_count> scratch_pointer;
  const void *cub_pointer;
  std::size_t cub_size;
  std::size_t cub_growth_count;
};

void require(const bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);

  return;
}

atom_sort_key encode_key(const unsigned int cell,
                         const unsigned int atom_index) {
  constexpr unsigned int atom_index_bits =
      static_cast<unsigned int>(8u * sizeof(unsigned int));

  return (static_cast<atom_sort_key>(cell) << atom_index_bits) |
         static_cast<atom_sort_key>(atom_index);
}

bool packets_equal(const atom_packet &lhs, const atom_packet &rhs) {
  return (lhs.i == rhs.i) && (lhs.cell == rhs.cell) && (lhs.x == rhs.x) &&
         (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.q == rhs.q);
}

std::vector<position> make_test_positions(void) {
  const double infinity = std::numeric_limits<double>::infinity();

  const double b12_lo = std::nextafter(12.0, 0.0);
  const double b12_hi = std::nextafter(12.0, infinity);

  const double b24_lo = std::nextafter(24.0, 0.0);
  const double b24_hi = std::nextafter(24.0, infinity);

  const double b36_lo = std::nextafter(36.0, 0.0);
  const double b36_hi = std::nextafter(36.0, infinity);

  const double b48_lo = std::nextafter(48.0, 0.0);
  const double b48_hi = std::nextafter(48.0, infinity);

  std::vector<position> positions;

  // Cell interiors and deliberately interleaved cell IDs.
  positions.push_back(position{1.0, 1.0, 1.0});
  positions.push_back(position{12.0, 1.0, 1.0});
  positions.push_back(position{2.0, 2.0, 2.0});
  positions.push_back(position{1.0, 12.0, 1.0});
  positions.push_back(position{1.0, 1.0, 12.0});
  positions.push_back(position{24.0, 1.0, 1.0});
  positions.push_back(position{1.0, 24.0, 1.0});

  // Exact and immediately adjacent values around the first internal boundary.
  positions.push_back(position{b12_lo, 1.0, 1.0});
  positions.push_back(position{b12_hi, 1.0, 1.0});
  positions.push_back(position{1.0, b12_lo, 1.0});
  positions.push_back(position{1.0, b12_hi, 1.0});
  positions.push_back(position{1.0, 1.0, b12_lo});
  positions.push_back(position{1.0, 1.0, b12_hi});

  // Exact and immediately adjacent values around the second boundary.
  positions.push_back(position{b24_lo, 1.0, 1.0});
  positions.push_back(position{b24_hi, 1.0, 1.0});
  positions.push_back(position{1.0, b24_lo, 1.0});
  positions.push_back(position{1.0, b24_hi, 1.0});
  positions.push_back(position{1.0, 1.0, 24.0});
  positions.push_back(position{1.0, 1.0, b24_lo});
  positions.push_back(position{1.0, 1.0, b24_hi});

  // The z dimension has a third internal boundary.
  positions.push_back(position{1.0, 1.0, 36.0});
  positions.push_back(position{1.0, 1.0, b36_lo});
  positions.push_back(position{1.0, 1.0, b36_hi});

  // Negative coordinates before clamping.
  positions.push_back(position{-0.25, 2.0, 2.0});
  positions.push_back(position{-12.25, 3.0, 3.0});
  positions.push_back(position{2.0, -12.25, 2.0});
  positions.push_back(position{2.0, 2.0, -12.25});

  // Exact, immediately adjacent, and far beyond upper box edges.
  positions.push_back(position{36.0, 36.0, 48.0});
  positions.push_back(position{b36_lo, b36_lo, b48_lo});
  positions.push_back(position{b36_hi, b36_hi, b48_hi});
  positions.push_back(position{72.0, 72.0, 96.0});

  // Additional interiors. Several cells intentionally remain empty.
  positions.push_back(position{25.0, 25.0, 37.0});
  positions.push_back(position{13.0, 13.0, 13.0});
  positions.push_back(position{25.0, 1.0, 25.0});
  positions.push_back(position{3.0, 3.0, 3.0});
  positions.push_back(position{4.0, 4.0, 4.0});
  positions.push_back(position{13.0, 1.0, 37.0});
  positions.push_back(position{25.0, 25.0, 13.0});
  positions.push_back(position{1.0, 25.0, 25.0});

  return positions;
}

reference_snapshot make_host_snapshot(const glst_force &force) {
  const glst_plan &plan = glst_force_test_access::plan(force);
  const glst_workspace &workspace = glst_force_test_access::workspace(force);

  const std::size_t natom = static_cast<std::size_t>(plan.natom());
  const std::size_t ncell = static_cast<std::size_t>(plan.ncell());
  const std::size_t ncell_x = static_cast<std::size_t>(plan.ncell_x());
  const unsigned int yz_cell_count = plan.ncell_y() * plan.ncell_z();

  require(workspace.owned_atom_count(0) == natom,
          "Host reference does not own every atom");

  require(workspace.source_atom_count(0) == natom,
          "Host reference unexpectedly contains halo atoms");

  const std::vector<atom_packet> &host_packets =
      workspace.sorted_packets()[0].h_array();

  const std::vector<unsigned int> &host_cell_count =
      workspace.cell_atom_count()[0].h_array();

  const std::vector<unsigned int> &host_cell_point =
      workspace.cell_atom_point()[0].h_array();

  require(host_packets.size() >= natom,
          "Host packet storage is smaller than natom");

  require(host_cell_count.size() == ncell,
          "Host cell-count size does not match ncell");

  require(host_cell_point.size() == ncell,
          "Host cell-point size does not match ncell");

  reference_snapshot snapshot;

  snapshot.key.resize(natom);
  snapshot.packet.resize(natom);
  snapshot.cell_count.resize(ncell);
  snapshot.cell_point.resize(ncell + 1u);
  snapshot.x_plane_point.resize(ncell_x + 1u);

  for (std::size_t atom = 0; atom < natom; atom++) {
    snapshot.packet[atom] = host_packets[atom];
    snapshot.key[atom] =
        encode_key(host_packets[atom].cell, host_packets[atom].i);
  }

  for (std::size_t cell = 0; cell < ncell; cell++) {
    snapshot.cell_count[cell] = host_cell_count[cell];
    snapshot.cell_point[cell] = host_cell_point[cell];
  }

  snapshot.cell_point[ncell] = plan.natom();

  for (std::size_t x = 0; x <= ncell_x; x++) {
    const std::size_t first_cell = x * static_cast<std::size_t>(yz_cell_count);

    snapshot.x_plane_point[x] = snapshot.cell_point[first_cell];
  }

  return snapshot;
}

reference_snapshot make_gpu_snapshot(const glst_force &force) {
  const glst_plan &plan = glst_force_test_access::plan(force);
  const glst_workspace &workspace = glst_force_test_access::workspace(force);

  const std::size_t natom = static_cast<std::size_t>(plan.natom());
  const std::size_t ncell = static_cast<std::size_t>(plan.ncell());
  const std::size_t x_plane_count =
      static_cast<std::size_t>(plan.ncell_x()) + 1u;

  reference_snapshot snapshot;

  snapshot.key.resize(natom);
  snapshot.packet.resize(natom);
  snapshot.cell_count.resize(ncell);
  snapshot.cell_point.resize(ncell + 1u);
  snapshot.x_plane_point.resize(x_plane_count);

  cudaCheck(cudaSetDevice(0));

  cudaCheck(cudaMemcpy(
      static_cast<void *>(snapshot.key.data()),
      static_cast<const void *>(workspace.global_sort_key_out().data()),
      natom * sizeof(atom_sort_key), cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(
      static_cast<void *>(snapshot.packet.data()),
      static_cast<const void *>(workspace.global_packet_out().data()),
      natom * sizeof(atom_packet), cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(
      static_cast<void *>(snapshot.cell_count.data()),
      static_cast<const void *>(workspace.global_cell_atom_count().data()),
      ncell * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(
      static_cast<void *>(snapshot.cell_point.data()),
      static_cast<const void *>(workspace.global_cell_atom_point().data()),
      (ncell + 1u) * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  cudaCheck(cudaMemcpy(
      static_cast<void *>(snapshot.x_plane_point.data()),
      static_cast<const void *>(workspace.global_x_plane_atom_point().data()),
      x_plane_count * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  return snapshot;
}

void validate_snapshot(const reference_snapshot &snapshot,
                       const std::vector<position> &positions,
                       const glst_plan &plan, const std::string &label) {
  const std::size_t natom = positions.size();
  const std::size_t ncell = static_cast<std::size_t>(plan.ncell());
  const std::size_t ncell_x = static_cast<std::size_t>(plan.ncell_x());
  const unsigned int yz_cell_count = plan.ncell_y() * plan.ncell_z();

  require(snapshot.key.size() == natom,
          label + ": key count does not match natom");

  require(snapshot.packet.size() == natom,
          label + ": packet count does not match natom");

  require(snapshot.cell_count.size() == ncell,
          label + ": cell-count size does not match ncell");

  require(snapshot.cell_point.size() == ncell + 1u,
          label + ": cell-point size does not include the terminal point");

  require(snapshot.x_plane_point.size() == ncell_x + 1u,
          label + ": x-plane-point size is incorrect");

  std::vector<unsigned int> observed_cell_count(ncell, 0u);

  bool found_empty_cell = false;
  bool found_multi_atom_cell = false;
  std::size_t running_total = 0;

  for (std::size_t cell = 0; cell < ncell; cell++) {
    const unsigned int count = snapshot.cell_count[cell];

    if (count == 0)
      found_empty_cell = true;

    if (count > 1)
      found_multi_atom_cell = true;

    require(snapshot.cell_point[cell] ==
                static_cast<unsigned int>(running_total),
            label + ": cell point is not the exclusive prefix sum at cell " +
                std::to_string(cell));

    running_total += static_cast<std::size_t>(count);
  }

  require(running_total == natom,
          label + ": sum of cell counts does not equal natom");

  require(snapshot.cell_point[ncell] == plan.natom(),
          label + ": terminal cell point does not equal natom");

  for (std::size_t x = 0; x <= ncell_x; x++) {
    const std::size_t first_cell = x * static_cast<std::size_t>(yz_cell_count);

    require(snapshot.x_plane_point[x] == snapshot.cell_point[first_cell],
            label + ": x-plane point is incorrect at x = " + std::to_string(x));
  }

  require(snapshot.x_plane_point[ncell_x] == plan.natom(),
          label + ": terminal x-plane point does not equal natom");

  for (std::size_t sorted_atom = 0; sorted_atom < natom; sorted_atom++) {
    const atom_packet &packet = snapshot.packet[sorted_atom];

    require(packet.i < plan.natom(),
            label + ": packet original index is out of range");

    require(packet.cell < plan.ncell(),
            label + ": packet global cell is out of range");

    const atom_sort_key expected_key = encode_key(packet.cell, packet.i);

    require(snapshot.key[sorted_atom] == expected_key,
            label + ": packet and key disagree at sorted atom " +
                std::to_string(sorted_atom));

    if (sorted_atom > 0) {
      require(snapshot.key[sorted_atom] > snapshot.key[sorted_atom - 1u],
              label + ": sorted keys are not strictly increasing");
    }

    const position &expected_position = positions[packet.i];

    require((packet.x == expected_position.x) &&
                (packet.y == expected_position.y) &&
                (packet.z == expected_position.z),
            label + ": packet coordinates changed for original atom " +
                std::to_string(packet.i));

    const double expected_charge = ((packet.i % 2u) == 0u) ? 1.0 : -1.0;

    require(packet.q == expected_charge,
            label + ": packet charge changed for original atom " +
                std::to_string(packet.i));

    observed_cell_count[packet.cell]++;
  }

  for (std::size_t cell = 0; cell < ncell; cell++) {
    require(observed_cell_count[cell] == snapshot.cell_count[cell],
            label + ": sorted packet count disagrees with metadata at cell " +
                std::to_string(cell));

    const std::size_t point =
        static_cast<std::size_t>(snapshot.cell_point[cell]);

    const std::size_t count =
        static_cast<std::size_t>(snapshot.cell_count[cell]);

    for (std::size_t atom = 0; atom < count; atom++) {
      require(snapshot.packet[point + atom].cell ==
                  static_cast<unsigned int>(cell),
              label + ": cell packet range contains the wrong global cell");
    }
  }

  require(found_empty_cell, label + ": test did not contain an empty cell");

  require(found_multi_atom_cell,
          label + ": test did not contain a multi-atom cell");

  return;
}

void compare_snapshots(const reference_snapshot &expected,
                       const reference_snapshot &observed,
                       const std::string &label) {
  require(expected.key.size() == observed.key.size(),
          label + ": key vector sizes differ");

  require(expected.packet.size() == observed.packet.size(),
          label + ": packet vector sizes differ");

  require(expected.cell_count.size() == observed.cell_count.size(),
          label + ": cell-count vector sizes differ");

  require(expected.cell_point.size() == observed.cell_point.size(),
          label + ": cell-point vector sizes differ");

  require(expected.x_plane_point.size() == observed.x_plane_point.size(),
          label + ": x-plane-point vector sizes differ");

  for (std::size_t i = 0; i < expected.key.size(); i++) {
    require(expected.key[i] == observed.key[i],
            label + ": sorted key differs at index " + std::to_string(i));

    require(packets_equal(expected.packet[i], observed.packet[i]),
            label + ": sorted packet differs at index " + std::to_string(i));
  }

  for (std::size_t i = 0; i < expected.cell_count.size(); i++) {
    require(expected.cell_count[i] == observed.cell_count[i],
            label + ": cell count differs at cell " + std::to_string(i));
  }

  for (std::size_t i = 0; i < expected.cell_point.size(); i++) {
    require(expected.cell_point[i] == observed.cell_point[i],
            label + ": cell point differs at index " + std::to_string(i));
  }

  for (std::size_t i = 0; i < expected.x_plane_point.size(); i++) {
    require(expected.x_plane_point[i] == observed.x_plane_point[i],
            label + ": x-plane point differs at index " + std::to_string(i));
  }

  return;
}

storage_snapshot take_storage_snapshot(const glst_workspace &workspace) {
  storage_snapshot snapshot{};

  snapshot.scratch_size = std::array<std::size_t, scratch_buffer_count>{
      workspace.global_sort_key_in().size(),
      workspace.global_sort_key_out().size(),
      workspace.global_packet_in().size(),
      workspace.global_packet_out().size(),
      workspace.global_cell_atom_count().size(),
      workspace.global_cell_atom_point().size(),
      workspace.global_x_plane_atom_point().size(),
      workspace.global_max_atoms_cell().size()};

  snapshot.scratch_pointer = std::array<const void *, scratch_buffer_count>{
      static_cast<const void *>(workspace.global_sort_key_in().data()),
      static_cast<const void *>(workspace.global_sort_key_out().data()),
      static_cast<const void *>(workspace.global_packet_in().data()),
      static_cast<const void *>(workspace.global_packet_out().data()),
      static_cast<const void *>(workspace.global_cell_atom_count().data()),
      static_cast<const void *>(workspace.global_cell_atom_point().data()),
      static_cast<const void *>(workspace.global_x_plane_atom_point().data()),
      static_cast<const void *>(workspace.global_max_atoms_cell().data())};

  require(!workspace.cub_work_buffer().empty(),
          "CUB work-buffer array is empty");

  require(!workspace.cub_work_buffer_size().empty(),
          "CUB work-buffer-size array is empty");

  snapshot.cub_pointer =
      static_cast<const void *>(workspace.cub_work_buffer()[0]);

  snapshot.cub_size = workspace.cub_work_buffer_size()[0];
  snapshot.cub_growth_count = workspace.cub_work_buffer_growth_count(0);

  return snapshot;
}

void validate_storage_snapshot(const storage_snapshot &snapshot,
                               const glst_plan &plan) {
  const std::array<std::size_t, scratch_buffer_count> expected_size{
      static_cast<std::size_t>(plan.natom()),
      static_cast<std::size_t>(plan.natom()),
      static_cast<std::size_t>(plan.natom()),
      static_cast<std::size_t>(plan.natom()),
      static_cast<std::size_t>(plan.ncell()),
      static_cast<std::size_t>(plan.ncell()) + 1u,
      static_cast<std::size_t>(plan.ncell_x()) + 1u,
      1u};

  require(snapshot.scratch_size == expected_size,
          "Global reference scratch sizes are incorrect");

  for (std::size_t i = 0; i < scratch_buffer_count; i++) {
    require(snapshot.scratch_pointer[i] != nullptr,
            "Global reference scratch contains a null pointer");
  }

  require(snapshot.cub_pointer != nullptr, "GPU 0 CUB work buffer is null");
  require(snapshot.cub_size > 0, "GPU 0 CUB work-buffer size is zero");

  return;
}

void compare_storage_snapshots(const storage_snapshot &expected,
                               const storage_snapshot &observed,
                               const std::string &label) {
  require(observed.scratch_size == expected.scratch_size,
          label + ": global scratch sizes changed");

  require(observed.scratch_pointer == expected.scratch_pointer,
          label + ": global scratch pointers changed");

  require(observed.cub_pointer == expected.cub_pointer,
          label + ": CUB work-buffer pointer changed");

  require(observed.cub_size == expected.cub_size,
          label + ": CUB work-buffer size changed");

  require(observed.cub_growth_count == expected.cub_growth_count,
          label + ": CUB growth count changed");

  return;
}

} // namespace

int main(void) {
  try {
    int cuda_count = 0;
    cudaCheck(cudaGetDeviceCount(&cuda_count));

    if (cuda_count < 2) {
      std::cout << "global_atom_reference: SKIPPED "
                   "(requires at least two visible GPUs)"
                << std::endl;
      return EXIT_SUCCESS;
    }

    constexpr double tol = 1.0e-6;
    constexpr double box_dim_x = 36.0;
    constexpr double box_dim_y = 36.0;
    constexpr double box_dim_z = 48.0;
    constexpr double rcut = 12.0;

    const std::vector<position> positions = make_test_positions();

    require(positions.size() <= static_cast<std::size_t>(
                                    std::numeric_limits<unsigned int>::max()),
            "Test atom count exceeds unsigned int range");

    const unsigned int natom = static_cast<unsigned int>(positions.size());

    cudaCheck(cudaSetDevice(0));

    cuda_container<double> rx(natom);
    cuda_container<double> ry(natom);
    cuda_container<double> rz(natom);
    cuda_container<double> qc(natom);

    for (unsigned int atom = 0; atom < natom; atom++) {
      rx[atom] = positions[atom].x;
      ry[atom] = positions[atom].y;
      rz[atom] = positions[atom].z;
      qc[atom] = ((atom % 2u) == 0u) ? 1.0 : -1.0;
    }

    rx.transfer_to_device();
    ry.transfer_to_device();
    rz.transfer_to_device();
    qc.transfer_to_device();

    glst_force force;

    // One cell partition makes the existing host path produce one complete
    // globally cell-ordered reference. More than one tile partition forces the
    // current multi-GPU assignment path without introducing halo atoms.
    force.set_gpu_layout(1u, static_cast<unsigned int>(cuda_count));

    force.init(natom, tol, box_dim_x, box_dim_y, box_dim_z, rcut);

    const glst_plan &plan = glst_force_test_access::plan(force);

    require(plan.ncell_x() == 3u, "Unexpected ncell_x");
    require(plan.ncell_y() == 3u, "Unexpected ncell_y");
    require(plan.ncell_z() == 4u, "Unexpected ncell_z");
    require(plan.ncell() == 36u, "Unexpected total cell count");

    // Run the unchanged production host classification.
    force.assign_atoms(rx.d_array().data(), ry.d_array().data(),
                       rz.d_array().data(), qc.d_array().data());

    const reference_snapshot host_reference = make_host_snapshot(force);

    validate_snapshot(host_reference, positions, plan, "host reference");

    const glst_workspace &workspace = glst_force_test_access::workspace(force);

    const storage_snapshot storage_before = take_storage_snapshot(workspace);

    validate_storage_snapshot(storage_before, plan);

    reference_snapshot first_gpu_reference;

    for (unsigned int repeat = 0; repeat < repeat_count; repeat++) {
      glst_force_test_access::build_global_atom_reference(
          force, rx.d_array().data(), ry.d_array().data(), rz.d_array().data(),
          qc.d_array().data());

      // One synchronization after the complete pipeline. There are no
      // synchronizations between reset, classify, sort, scan, and finalization.
      glst_force_test_access::synchronize_global_atom_reference(force);

      // These copies are test-only result inspection. They are not part of the
      // GPU reference pipeline.
      const reference_snapshot gpu_reference = make_gpu_snapshot(force);

      const std::string repeat_label =
          "GPU reference repeat " + std::to_string(repeat);

      validate_snapshot(gpu_reference, positions, plan, repeat_label);

      compare_snapshots(host_reference, gpu_reference,
                        repeat_label + " versus host");

      if (repeat == 0)
        first_gpu_reference = gpu_reference;
      else
        compare_snapshots(first_gpu_reference, gpu_reference,
                          repeat_label + " versus first GPU run");

      const storage_snapshot storage_after = take_storage_snapshot(workspace);

      compare_storage_snapshots(storage_before, storage_after, repeat_label);
    }

    std::cout << "global_atom_reference: PASSED " << repeat_count
              << " deterministic repetitions" << std::endl;

    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "global_atom_reference: FAILED: " << error.what() << std::endl;

    return EXIT_FAILURE;
  }
}
