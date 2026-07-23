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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
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

  static void scatter_sorted_atom_packets(glst_force &force, const double *d_rx,
                                          const double *d_ry,
                                          const double *d_rz,
                                          const double *d_qc) {
    force.scatter_sorted_atom_packets(d_rx, d_ry, d_rz, d_qc);
    return;
  }

  static void synchronize_compute_streams(glst_force &force) {
    for (int dev = 0; dev < force.cuda_count_; dev++) {
      cudaCheck(cudaSetDevice(dev));
      cudaCheck(cudaStreamSynchronize(force.comp_streams_[dev]));
    }

    return;
  }
};

namespace {

constexpr unsigned int repeat_count = 8u;
constexpr unsigned int test_ncell_x = 8u;
constexpr unsigned int test_ncell_y = 3u;
constexpr unsigned int test_ncell_z = 4u;
constexpr unsigned int atoms_per_cell = 2u;

constexpr double rcut = 12.0;
constexpr double box_dim_x = static_cast<double>(test_ncell_x) * rcut;
constexpr double box_dim_y = static_cast<double>(test_ncell_y) * rcut;
constexpr double box_dim_z = static_cast<double>(test_ncell_z) * rcut;
constexpr double tol = 1.0e-6;

struct position {
  double x;
  double y;
  double z;
};

struct partition_snapshot {
  std::size_t owned_atom_count = 0;
  std::size_t source_atom_count = 0;
  std::vector<atom_packet> packet;
};

struct storage_snapshot {
  std::vector<std::size_t> atom_capacity;
  std::vector<std::size_t> atom_growth_count;
  std::vector<std::size_t> cub_growth_count;
  std::vector<std::size_t> cub_size;

  std::vector<const void *> packet_pointer;
  std::vector<const void *> cub_pointer;

  const void *global_packet_pointer = nullptr;
  const void *global_x_plane_pointer = nullptr;
};

void require(const bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);

  return;
}

bool packets_equal(const atom_packet &lhs, const atom_packet &rhs) {
  return (lhs.i == rhs.i) && (lhs.cell == rhs.cell) && (lhs.x == rhs.x) &&
         (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.q == rhs.q);
}

std::vector<position> make_positions(void) {
  std::vector<position> cell_ordered;

  cell_ordered.reserve(static_cast<std::size_t>(test_ncell_x) *
                       static_cast<std::size_t>(test_ncell_y) *
                       static_cast<std::size_t>(test_ncell_z) *
                       static_cast<std::size_t>(atoms_per_cell));

  for (unsigned int x = 0; x < test_ncell_x; x++) {
    for (unsigned int y = 0; y < test_ncell_y; y++) {
      for (unsigned int z = 0; z < test_ncell_z; z++) {
        for (unsigned int atom = 0; atom < atoms_per_cell; atom++) {
          const double offset = 1.0 + 4.0 * static_cast<double>(atom);

          cell_ordered.push_back(
              position{static_cast<double>(x) * rcut + offset,
                       static_cast<double>(y) * rcut + offset + 1.0,
                       static_cast<double>(z) * rcut + offset + 2.0});
        }
      }
    }
  }

  // Deterministically interleave the original atom order. 37 is coprime to
  // 192, so this visits every generated atom exactly once.
  require(cell_ordered.size() == 192u, "Unexpected generated atom count");

  constexpr std::size_t stride = 37u;

  std::vector<position> positions;
  positions.reserve(cell_ordered.size());

  for (std::size_t atom = 0; atom < cell_ordered.size(); atom++) {
    const std::size_t source = (atom * stride) % cell_ordered.size();

    positions.push_back(cell_ordered[source]);
  }

  return positions;
}

std::vector<partition_snapshot> make_host_reference(
    const glst_plan &target_plan, const unsigned int natom,
    const cuda_container<double> &rx, const cuda_container<double> &ry,
    const cuda_container<double> &rz, const cuda_container<double> &qc) {
  require(rx.size() == static_cast<std::size_t>(natom),
          "Host-reference rx size is incorrect");

  require(ry.size() == static_cast<std::size_t>(natom),
          "Host-reference ry size is incorrect");

  require(rz.size() == static_cast<std::size_t>(natom),
          "Host-reference rz size is incorrect");

  require(qc.size() == static_cast<std::size_t>(natom),
          "Host-reference qc size is incorrect");

  std::vector<std::vector<atom_packet>> cell_packets(target_plan.ncell());

  for (unsigned int atom = 0u; atom < natom; atom++) {
    int cx = static_cast<int>(rx[atom] / target_plan.cell_dim_x());

    int cy = static_cast<int>(ry[atom] / target_plan.cell_dim_y());

    int cz = static_cast<int>(rz[atom] / target_plan.cell_dim_z());

    cx = (cx >= static_cast<int>(target_plan.ncell_x()))
             ? static_cast<int>(target_plan.ncell_x() - 1u)
             : cx;

    cy = (cy >= static_cast<int>(target_plan.ncell_y()))
             ? static_cast<int>(target_plan.ncell_y() - 1u)
             : cy;

    cz = (cz >= static_cast<int>(target_plan.ncell_z()))
             ? static_cast<int>(target_plan.ncell_z() - 1u)
             : cz;

    cx = (cx < 0) ? 0 : cx;
    cy = (cy < 0) ? 0 : cy;
    cz = (cz < 0) ? 0 : cz;

    const unsigned int global_cell =
        (static_cast<unsigned int>(cx) * target_plan.ncell_y() +
         static_cast<unsigned int>(cy)) *
            target_plan.ncell_z() +
        static_cast<unsigned int>(cz);

    cell_packets[global_cell].push_back(
        atom_packet(atom, global_cell, rx[atom], ry[atom], rz[atom], qc[atom]));
  }

  for (unsigned int cell = 0u; cell < target_plan.ncell(); cell++) {
    std::vector<atom_packet> &packets = cell_packets[cell];

    std::sort(packets.begin(), packets.end(),
              [](const atom_packet &lhs, const atom_packet &rhs) {
                return lhs.i < rhs.i;
              });
  }

  std::vector<partition_snapshot> snapshot(target_plan.cell_partition_count());

  for (unsigned int partition = 0u;
       partition < target_plan.cell_partition_count(); partition++) {
    partition_snapshot &partition_data = snapshot[partition];

    const std::vector<unsigned int> &owned_cells =
        target_plan.partition_cell_idx(partition);

    for (std::size_t i = 0u; i < owned_cells.size(); i++) {
      partition_data.owned_atom_count += cell_packets[owned_cells[i]].size();
    }

    const std::vector<unsigned int> &source_cells =
        target_plan.partition_sr_source_cell_idx(partition);

    for (std::size_t i = 0u; i < source_cells.size(); i++) {
      const std::vector<atom_packet> &packets = cell_packets[source_cells[i]];

      partition_data.packet.insert(partition_data.packet.end(), packets.begin(),
                                   packets.end());
    }

    partition_data.source_atom_count = partition_data.packet.size();
  }

  return snapshot;
}

std::vector<partition_snapshot>
read_device_snapshot(const glst_workspace &workspace, const int cuda_count) {
  std::vector<partition_snapshot> snapshot(
      static_cast<std::size_t>(cuda_count));

  for (int dev = 0; dev < cuda_count; dev++) {
    partition_snapshot &partition = snapshot[dev];

    partition.owned_atom_count = workspace.owned_atom_count(dev);

    partition.source_atom_count = workspace.source_atom_count(dev);

    partition.packet.resize(partition.source_atom_count);

    require(partition.source_atom_count <= workspace.atom_capacity(dev),
            "Device source count exceeds atom capacity");

    if (partition.source_atom_count == 0u)
      continue;

    cudaCheck(cudaSetDevice(dev));

    // Test-only result inspection. Production packet scatter performs no
    // atom-sized device-to-host transfer.
    cudaCheck(cudaMemcpy(static_cast<void *>(partition.packet.data()),
                         static_cast<const void *>(
                             workspace.sorted_packets()[dev].d_array().data()),
                         partition.source_atom_count * sizeof(atom_packet),
                         cudaMemcpyDeviceToHost));
  }

  return snapshot;
}

void compare_partition_snapshot(const partition_snapshot &expected,
                                const partition_snapshot &observed,
                                const std::string &label) {
  require(expected.owned_atom_count == observed.owned_atom_count,
          label + ": owned atom counts differ");

  require(expected.source_atom_count == observed.source_atom_count,
          label + ": source atom counts differ");

  require(expected.packet.size() == observed.packet.size(),
          label + ": packet-vector sizes differ");

  for (std::size_t atom = 0; atom < expected.packet.size(); atom++) {
    require(packets_equal(expected.packet[atom], observed.packet[atom]),
            label + ": packet differs at source index " + std::to_string(atom));
  }

  return;
}

void validate_sorted_segment(const std::vector<atom_packet> &packet,
                             const std::size_t begin, const std::size_t count,
                             const std::string &label) {
  require(begin <= packet.size(), label + ": segment begin is out of range");

  require(count <= packet.size() - begin,
          label + ": segment count is out of range");

  for (std::size_t i = 1; i < count; i++) {
    const atom_packet &previous = packet[begin + i - 1u];
    const atom_packet &current = packet[begin + i];

    require(current.cell >= previous.cell,
            label + ": global cells are not ordered");

    if (current.cell == previous.cell) {
      require(current.i > previous.i,
              label + ": original indices are not ordered");
    }
  }

  return;
}

void validate_partition_layout(const partition_snapshot &snapshot,
                               const glst_plan &plan,
                               const glst_workspace &workspace,
                               const unsigned int partition) {
  const atom_partition_range &range =
      workspace.partition_atom_range()[partition];

  const unsigned int x_point = plan.cell_partition_x_point()[partition];

  const unsigned int x_count = plan.cell_partition_x_count()[partition];

  const unsigned int x_end = x_point + x_count;

  const std::size_t atoms_per_x_plane =
      static_cast<std::size_t>(atoms_per_cell) *
      static_cast<std::size_t>(plan.ncell_y()) *
      static_cast<std::size_t>(plan.ncell_z());

  const std::size_t expected_owned_offset =
      static_cast<std::size_t>(x_point) * atoms_per_x_plane;

  const std::size_t expected_owned_count =
      static_cast<std::size_t>(x_count) * atoms_per_x_plane;

  const std::size_t expected_left_count =
      (x_point > 0u) ? atoms_per_x_plane : 0u;

  const std::size_t expected_right_count =
      (x_end < plan.ncell_x()) ? atoms_per_x_plane : 0u;

  const std::size_t expected_source_count =
      expected_owned_count + expected_left_count + expected_right_count;

  require(range.owned_source_offset == expected_owned_offset,
          "Owned source offset is incorrect for partition " +
              std::to_string(partition));

  require(range.owned_atom_count == expected_owned_count,
          "Owned atom count is incorrect for partition " +
              std::to_string(partition));

  require(range.left_atom_count == expected_left_count,
          "Left-halo atom count is incorrect for partition " +
              std::to_string(partition));

  require(range.right_atom_count == expected_right_count,
          "Right-halo atom count is incorrect for partition " +
              std::to_string(partition));

  require(range.source_atom_count == expected_source_count,
          "Source atom count is incorrect for partition " +
              std::to_string(partition));

  if (x_point > 0u) {
    require(range.left_source_offset ==
                static_cast<std::size_t>(x_point - 1u) * atoms_per_x_plane,
            "Left-halo source offset is incorrect for partition " +
                std::to_string(partition));
  }

  if (x_end < plan.ncell_x()) {
    require(range.right_source_offset ==
                static_cast<std::size_t>(x_end) * atoms_per_x_plane,
            "Right-halo source offset is incorrect for partition " +
                std::to_string(partition));
  }

  require(snapshot.owned_atom_count == range.owned_atom_count,
          "Active owned count disagrees with descriptor");

  require(snapshot.source_atom_count == range.source_atom_count,
          "Active source count disagrees with descriptor");

  require(snapshot.packet.size() == range.source_atom_count,
          "Active packet range disagrees with descriptor");

  const std::size_t left_destination = range.owned_atom_count;

  const std::size_t right_destination =
      range.owned_atom_count + range.left_atom_count;

  validate_sorted_segment(snapshot.packet, 0u, range.owned_atom_count,
                          "Owned segment for partition " +
                              std::to_string(partition));

  validate_sorted_segment(
      snapshot.packet, left_destination, range.left_atom_count,
      "Left-halo segment for partition " + std::to_string(partition));

  validate_sorted_segment(
      snapshot.packet, right_destination, range.right_atom_count,
      "Right-halo segment for partition " + std::to_string(partition));

  for (std::size_t atom = 0; atom < range.owned_atom_count; atom++) {
    require(plan.cell_partition_idx(snapshot.packet[atom].cell) == partition,
            "Owned packet belongs to the wrong partition");
  }

  for (std::size_t atom = 0; atom < range.left_atom_count; atom++) {
    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int z = 0;

    plan.global_cell_coords(x, y, z,
                            snapshot.packet[left_destination + atom].cell);

    require((x_point > 0u) && (x == x_point - 1u),
            "Left-halo packet is not on the adjacent x plane");
  }

  for (std::size_t atom = 0; atom < range.right_atom_count; atom++) {
    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int z = 0;

    plan.global_cell_coords(x, y, z,
                            snapshot.packet[right_destination + atom].cell);

    require((x_end < plan.ncell_x()) && (x == x_end),
            "Right-halo packet is not on the adjacent x plane");
  }

  return;
}

void validate_owner_coverage(const std::vector<partition_snapshot> &snapshot,
                             const unsigned int natom) {
  std::vector<unsigned int> owner_count(natom, 0u);

  for (std::size_t partition = 0; partition < snapshot.size(); partition++) {
    for (std::size_t atom = 0; atom < snapshot[partition].owned_atom_count;
         atom++) {
      const unsigned int original_index = snapshot[partition].packet[atom].i;

      require(original_index < natom,
              "Owned packet original index is out of range");

      owner_count[original_index]++;
    }
  }

  for (unsigned int atom = 0; atom < natom; atom++) {
    require(owner_count[atom] == 1u, "Atom " + std::to_string(atom) +
                                         " does not have exactly one owner");
  }

  return;
}

storage_snapshot take_storage_snapshot(const glst_workspace &workspace,
                                       const int cuda_count) {
  storage_snapshot snapshot;

  const std::size_t device_count = static_cast<std::size_t>(cuda_count);

  snapshot.atom_capacity.resize(device_count);
  snapshot.atom_growth_count.resize(device_count);
  snapshot.cub_growth_count.resize(device_count);
  snapshot.cub_size.resize(device_count);
  snapshot.packet_pointer.resize(device_count);
  snapshot.cub_pointer.resize(device_count);

  require(workspace.cub_work_buffer().size() == device_count,
          "CUB work-buffer count does not match device count");

  require(workspace.cub_work_buffer_size().size() == device_count,
          "CUB work-buffer-size count does not match device count");

  for (int dev = 0; dev < cuda_count; dev++) {
    snapshot.atom_capacity[dev] = workspace.atom_capacity(dev);

    snapshot.atom_growth_count[dev] = workspace.atom_storage_growth_count(dev);

    snapshot.cub_growth_count[dev] =
        workspace.cub_work_buffer_growth_count(dev);

    snapshot.cub_size[dev] = workspace.cub_work_buffer_size()[dev];

    snapshot.packet_pointer[dev] = static_cast<const void *>(
        workspace.sorted_packets()[dev].d_array().data());

    snapshot.cub_pointer[dev] =
        static_cast<const void *>(workspace.cub_work_buffer()[dev]);

    require(snapshot.packet_pointer[dev] != nullptr,
            "Root packet pointer is null");

    require(snapshot.cub_pointer[dev] != nullptr,
            "CUB work-buffer pointer is null");
  }

  snapshot.global_packet_pointer =
      static_cast<const void *>(workspace.global_packet_out().data());

  snapshot.global_x_plane_pointer =
      static_cast<const void *>(workspace.global_x_plane_atom_point().data());

  require(snapshot.global_packet_pointer != nullptr,
          "Global packet pointer is null");

  require(snapshot.global_x_plane_pointer != nullptr,
          "Global x-plane pointer is null");

  return snapshot;
}

void compare_storage_snapshot(const storage_snapshot &expected,
                              const storage_snapshot &observed,
                              const std::string &label) {
  require(observed.atom_capacity == expected.atom_capacity,
          label + ": atom capacities changed");

  require(observed.atom_growth_count == expected.atom_growth_count,
          label + ": atom-storage growth count changed");

  require(observed.cub_growth_count == expected.cub_growth_count,
          label + ": CUB growth count changed");

  require(observed.cub_size == expected.cub_size,
          label + ": CUB work-buffer size changed");

  require(observed.packet_pointer == expected.packet_pointer,
          label + ": root packet pointer changed");

  require(observed.cub_pointer == expected.cub_pointer,
          label + ": CUB work-buffer pointer changed");

  require(observed.global_packet_pointer == expected.global_packet_pointer,
          label + ": global packet pointer changed");

  require(observed.global_x_plane_pointer == expected.global_x_plane_pointer,
          label + ": global x-plane pointer changed");

  return;
}

} // namespace

int main(void) {
  try {
    int cuda_count = 0;
    cudaCheck(cudaGetDeviceCount(&cuda_count));

    if ((cuda_count != 2) && (cuda_count != 4) && (cuda_count != 8)) {
      std::cout << "global_atom_scatter: SKIPPED "
                << "(expose exactly 2, 4, or 8 GPUs)" << std::endl;

      return EXIT_SUCCESS;
    }

    const std::vector<position> positions = make_positions();

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

    force.set_gpu_layout(static_cast<unsigned int>(cuda_count), 1u);
    force.init(natom, tol, box_dim_x, box_dim_y, box_dim_z, rcut);

    const glst_plan &plan = glst_force_test_access::plan(force);

    const std::vector<partition_snapshot> host_reference =
        make_host_reference(plan, natom, rx, ry, rz, qc);

    require(plan.ncell_x() == test_ncell_x, "Unexpected ncell_x");

    require(plan.ncell_y() == test_ncell_y, "Unexpected ncell_y");

    require(plan.ncell_z() == test_ncell_z, "Unexpected ncell_z");

    storage_snapshot stable_storage;

    for (unsigned int repeat = 0; repeat < repeat_count; repeat++) {
      glst_force_test_access::scatter_sorted_atom_packets(
          force, rx.d_array().data(), ry.d_array().data(), rz.d_array().data(),
          qc.d_array().data());

      glst_force_test_access::synchronize_compute_streams(force);

      const glst_workspace &workspace =
          glst_force_test_access::workspace(force);

      const std::vector<partition_snapshot> observed =
          read_device_snapshot(workspace, cuda_count);

      require(observed.size() == host_reference.size(),
              "Observed/reference partition counts differ");

      for (unsigned int partition = 0;
           partition < static_cast<unsigned int>(cuda_count); partition++) {
        const std::string label = "G_cell=" + std::to_string(cuda_count) +
                                  ", repeat=" + std::to_string(repeat) +
                                  ", partition=" + std::to_string(partition);

        compare_partition_snapshot(host_reference[partition],
                                   observed[partition], label);

        validate_partition_layout(observed[partition], plan, workspace,
                                  partition);
      }

      validate_owner_coverage(observed, natom);

      const storage_snapshot current_storage =
          take_storage_snapshot(workspace, cuda_count);

      if (repeat == 0u)
        stable_storage = current_storage;
      else {
        compare_storage_snapshot(stable_storage, current_storage,
                                 "G_cell=" + std::to_string(cuda_count) +
                                     ", repeat=" + std::to_string(repeat));
      }
    }

    std::cout << "global_atom_scatter: PASSED G_cell=" << cuda_count
              << ", G_tile=1, " << repeat_count << " deterministic repetitions"
              << std::endl;

    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "global_atom_scatter: FAILED: " << error.what() << std::endl;

    return EXIT_FAILURE;
  }
}
