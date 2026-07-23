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

  static int device_count(const glst_force &force) { return force.cuda_count_; }

  static unsigned int cell_partition(const glst_force &force, const int dev) {
    if ((dev < 0) ||
        (static_cast<std::size_t>(dev) >= force.dev_cell_partition_.size())) {
      throw std::runtime_error(
          "glst_force_test_access::cell_partition: Device is out of range");
    }

    return force.dev_cell_partition_[dev];
  }

  static unsigned int tile_partition(const glst_force &force, const int dev) {
    if ((dev < 0) ||
        (static_cast<std::size_t>(dev) >= force.dev_tile_partition_.size())) {
      throw std::runtime_error(
          "glst_force_test_access::tile_partition: Device is out of range");
    }

    return force.dev_tile_partition_[dev];
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

constexpr unsigned int repeat_count = 4u;
constexpr unsigned int test_ncell_x = 8u;
constexpr unsigned int test_ncell_y = 3u;
constexpr unsigned int test_ncell_z = 4u;
constexpr unsigned int atoms_per_cell = 2u;
constexpr unsigned int test_natom =
    test_ncell_x * test_ncell_y * test_ncell_z * atoms_per_cell;

constexpr double rcut = 12.0;
constexpr double box_dim_x = static_cast<double>(test_ncell_x) * rcut;
constexpr double box_dim_y = static_cast<double>(test_ncell_y) * rcut;
constexpr double box_dim_z = static_cast<double>(test_ncell_z) * rcut;
constexpr double tol = 1.0e-6;

struct atom_site {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int atom_in_cell;
};

struct position {
  double x;
  double y;
  double z;
};

struct partition_reference {
  std::size_t owned_atom_count = 0;
  std::size_t source_atom_count = 0;
  unsigned int max_atoms_cell = 0u;

  std::vector<unsigned int> source_cell;
  std::vector<unsigned int> source_cell_atom_count;
  std::vector<unsigned int> source_cell_atom_point;
  std::vector<unsigned int> owned_cell_atom_count;
  std::vector<unsigned int> owned_cell_atom_point;

  std::vector<atom_packet> packet;
};

struct partition_snapshot {
  unsigned int cell_partition = 0u;
  std::size_t owned_atom_count = 0;
  std::size_t source_atom_count = 0;
  unsigned int max_atoms_cell = 0u;

  std::vector<unsigned int> source_cell_atom_count;
  std::vector<unsigned int> source_cell_atom_point;
  std::vector<unsigned int> owned_cell_atom_count;
  std::vector<unsigned int> owned_cell_atom_point;

  std::vector<atom_packet> packet;
  std::vector<unsigned int> sorted_idx;
  std::vector<double> rx;
  std::vector<double> ry;
  std::vector<double> rz;
  std::vector<double> qc;
};

struct device_storage_snapshot {
  std::size_t atom_capacity = 0;
  std::size_t atom_growth_count = 0;
  std::size_t cub_growth_count = 0;
  std::size_t cub_size = 0;

  const void *sorted_packet = nullptr;
  const void *sorted_idx = nullptr;
  const void *rx = nullptr;
  const void *ry = nullptr;
  const void *rz = nullptr;
  const void *qc = nullptr;
  const void *cell_atom_count = nullptr;
  const void *cell_atom_point = nullptr;
  const void *source_cell_atom_count = nullptr;
  const void *source_cell_atom_point = nullptr;
  const void *cub = nullptr;
};

struct storage_snapshot {
  std::vector<device_storage_snapshot> device;

  const void *global_sort_key_in = nullptr;
  const void *global_sort_key_out = nullptr;
  const void *global_packet_in = nullptr;
  const void *global_packet_out = nullptr;
  const void *global_cell_atom_count = nullptr;
  const void *global_cell_atom_point = nullptr;
  const void *global_x_plane_atom_point = nullptr;
  const void *global_max_atoms_cell = nullptr;
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

std::vector<atom_site> make_atom_sites(void) {
  std::vector<atom_site> cell_ordered;

  cell_ordered.reserve(static_cast<std::size_t>(test_natom));

  for (unsigned int x = 0; x < test_ncell_x; x++) {
    for (unsigned int y = 0; y < test_ncell_y; y++) {
      for (unsigned int z = 0; z < test_ncell_z; z++) {
        for (unsigned int atom = 0; atom < atoms_per_cell; atom++) {
          cell_ordered.push_back(atom_site{x, y, z, atom});
        }
      }
    }
  }

  require(cell_ordered.size() == static_cast<std::size_t>(test_natom),
          "Unexpected generated atom-site count");

  // 37 is coprime to 192, so the source index visits every generated site
  // exactly once while strongly interleaving the original atom order.
  constexpr std::size_t stride = 37u;

  std::vector<atom_site> sites;
  sites.reserve(cell_ordered.size());

  for (std::size_t atom = 0; atom < cell_ordered.size(); atom++) {
    const std::size_t source = (atom * stride) % cell_ordered.size();
    sites.push_back(cell_ordered[source]);
  }

  return sites;
}

std::vector<position> make_positions(const std::vector<atom_site> &sites,
                                     const bool pair_atoms_into_odd_z_cells) {
  std::vector<position> positions;
  positions.reserve(sites.size());

  for (std::size_t atom = 0; atom < sites.size(); atom++) {
    const atom_site &site = sites[atom];

    unsigned int target_z = site.z;
    if (pair_atoms_into_odd_z_cells && ((site.z % 2u) == 0u))
      target_z = site.z + 1u;

    require(target_z < test_ncell_z, "Generated target z-cell is out of range");

    // The parity term keeps atoms moved from adjacent z cells at distinct
    // coordinates. Every local coordinate remains strictly inside its cell.
    const double offset = 1.0 + 3.0 * static_cast<double>(site.atom_in_cell) +
                          0.25 * static_cast<double>(site.z % 2u);

    positions.push_back(
        position{static_cast<double>(site.x) * rcut + offset,
                 static_cast<double>(site.y) * rcut + offset + 1.0,
                 static_cast<double>(target_z) * rcut + offset + 2.0});
  }

  return positions;
}

std::vector<double> make_charges(const std::size_t natom) {
  std::vector<double> charge(natom, 0.0);

  for (std::size_t atom = 0; atom < natom; atom++)
    charge[atom] = ((atom % 2u) == 0u) ? 1.0 : -1.0;

  return charge;
}

void upload_input(cuda_container<double> &rx, cuda_container<double> &ry,
                  cuda_container<double> &rz, cuda_container<double> &qc,
                  const std::vector<position> &positions,
                  const std::vector<double> &charge) {
  require(positions.size() == charge.size(),
          "Input position and charge counts differ");

  require(rx.size() == positions.size(), "Input rx size is incorrect");
  require(ry.size() == positions.size(), "Input ry size is incorrect");
  require(rz.size() == positions.size(), "Input rz size is incorrect");
  require(qc.size() == positions.size(), "Input qc size is incorrect");

  for (std::size_t atom = 0; atom < positions.size(); atom++) {
    rx[atom] = positions[atom].x;
    ry[atom] = positions[atom].y;
    rz[atom] = positions[atom].z;
    qc[atom] = charge[atom];
  }

  cudaCheck(cudaSetDevice(0));

  rx.transfer_to_device();
  ry.transfer_to_device();
  rz.transfer_to_device();
  qc.transfer_to_device();

  return;
}

unsigned int global_cell_for_position(const glst_plan &plan,
                                      const position &atom_position) {
  int cx = static_cast<int>(atom_position.x / plan.cell_dim_x());
  int cy = static_cast<int>(atom_position.y / plan.cell_dim_y());
  int cz = static_cast<int>(atom_position.z / plan.cell_dim_z());

  cx = (cx >= static_cast<int>(plan.ncell_x()))
           ? static_cast<int>(plan.ncell_x() - 1u)
           : cx;
  cy = (cy >= static_cast<int>(plan.ncell_y()))
           ? static_cast<int>(plan.ncell_y() - 1u)
           : cy;
  cz = (cz >= static_cast<int>(plan.ncell_z()))
           ? static_cast<int>(plan.ncell_z() - 1u)
           : cz;

  cx = (cx < 0) ? 0 : cx;
  cy = (cy < 0) ? 0 : cy;
  cz = (cz < 0) ? 0 : cz;

  return (static_cast<unsigned int>(cx) * plan.ncell_y() +
          static_cast<unsigned int>(cy)) *
             plan.ncell_z() +
         static_cast<unsigned int>(cz);
}

std::vector<std::vector<atom_packet>>
make_cell_packets(const glst_plan &plan, const std::vector<position> &positions,
                  const std::vector<double> &charge) {
  require(positions.size() == charge.size(),
          "Host-reference position and charge counts differ");

  require(positions.size() <= static_cast<std::size_t>(
                                  std::numeric_limits<unsigned int>::max()),
          "Host-reference atom count exceeds unsigned int range");

  std::vector<std::vector<atom_packet>> cell_packets(plan.ncell());

  for (std::size_t atom = 0; atom < positions.size(); atom++) {
    const unsigned int global_cell =
        global_cell_for_position(plan, positions[atom]);

    cell_packets[global_cell].push_back(atom_packet(
        static_cast<unsigned int>(atom), global_cell, positions[atom].x,
        positions[atom].y, positions[atom].z, charge[atom]));
  }

  for (unsigned int cell = 0; cell < plan.ncell(); cell++) {
    std::vector<atom_packet> &packets = cell_packets[cell];

    std::sort(packets.begin(), packets.end(),
              [](const atom_packet &lhs, const atom_packet &rhs) {
                return lhs.i < rhs.i;
              });
  }

  return cell_packets;
}

std::vector<partition_reference> make_partition_reference(
    const glst_plan &plan,
    const std::vector<std::vector<atom_packet>> &cell_packets) {
  require(cell_packets.size() == static_cast<std::size_t>(plan.ncell()),
          "Host-reference cell-packet count does not match ncell");

  std::vector<partition_reference> reference(plan.cell_partition_count());

  for (unsigned int partition = 0; partition < plan.cell_partition_count();
       partition++) {
    partition_reference &partition_reference_data = reference[partition];

    const std::vector<unsigned int> &source_cells =
        plan.partition_sr_source_cell_idx(partition);

    const std::size_t local_cell_count =
        static_cast<std::size_t>(plan.local_cell_count(partition));

    require(source_cells.size() >= local_cell_count,
            "Source-cell list is smaller than owned-cell count");

    partition_reference_data.source_cell = source_cells;
    partition_reference_data.source_cell_atom_count.assign(source_cells.size(),
                                                           0u);
    partition_reference_data.source_cell_atom_point.assign(source_cells.size(),
                                                           0u);
    partition_reference_data.owned_cell_atom_count.assign(local_cell_count, 0u);
    partition_reference_data.owned_cell_atom_point.assign(local_cell_count, 0u);

    std::size_t atom_point = 0;

    for (std::size_t source_local_cell = 0;
         source_local_cell < source_cells.size(); source_local_cell++) {
      const unsigned int global_cell = source_cells[source_local_cell];

      require(global_cell < plan.ncell(),
              "Host-reference source cell is out of range");

      const std::vector<atom_packet> &packets = cell_packets[global_cell];

      require(packets.size() <= static_cast<std::size_t>(
                                    std::numeric_limits<unsigned int>::max()),
              "Host-reference cell population exceeds unsigned int range");

      require(atom_point <= static_cast<std::size_t>(
                                std::numeric_limits<unsigned int>::max()),
              "Host-reference atom point exceeds unsigned int range");

      const unsigned int point = static_cast<unsigned int>(atom_point);
      const unsigned int count = static_cast<unsigned int>(packets.size());

      partition_reference_data.source_cell_atom_point[source_local_cell] =
          point;
      partition_reference_data.source_cell_atom_count[source_local_cell] =
          count;

      if (source_local_cell < local_cell_count) {
        partition_reference_data.owned_cell_atom_point[source_local_cell] =
            point;
        partition_reference_data.owned_cell_atom_count[source_local_cell] =
            count;

        partition_reference_data.owned_atom_count += packets.size();

        if (count > partition_reference_data.max_atoms_cell)
          partition_reference_data.max_atoms_cell = count;
      }

      partition_reference_data.packet.insert(
          partition_reference_data.packet.end(), packets.begin(),
          packets.end());

      atom_point += packets.size();
    }

    partition_reference_data.source_atom_count =
        partition_reference_data.packet.size();

    require(atom_point == partition_reference_data.source_atom_count,
            "Host-reference source counts do not sum to source atoms");

    require(partition_reference_data.owned_atom_count <=
                partition_reference_data.source_atom_count,
            "Host-reference owned count exceeds source count");
  }

  return reference;
}

partition_snapshot read_partition_snapshot(const glst_force &force,
                                           const int dev) {
  const glst_plan &plan = glst_force_test_access::plan(force);
  const glst_workspace &workspace = glst_force_test_access::workspace(force);

  partition_snapshot snapshot;

  snapshot.cell_partition = glst_force_test_access::cell_partition(force, dev);

  require(snapshot.cell_partition < plan.cell_partition_count(),
          "Observed cell partition is out of range");

  const std::size_t local_cell_count =
      static_cast<std::size_t>(plan.local_cell_count(snapshot.cell_partition));

  const std::size_t source_cell_count =
      plan.partition_sr_source_cell_idx(snapshot.cell_partition).size();

  require(local_cell_count == workspace.cell_capacity(dev),
          "Observed local-cell count does not match workspace capacity");

  require(source_cell_count == workspace.sr_source_cell_capacity(dev),
          "Observed source-cell count does not match workspace capacity");

  snapshot.owned_atom_count = workspace.owned_atom_count(dev);
  snapshot.source_atom_count = workspace.source_atom_count(dev);
  snapshot.max_atoms_cell = workspace.max_atoms_cell()[dev];

  require(snapshot.owned_atom_count <= snapshot.source_atom_count,
          "Observed owned atom count exceeds source atom count");

  require(snapshot.source_atom_count <= workspace.atom_capacity(dev),
          "Observed source atom count exceeds workspace capacity");

  snapshot.source_cell_atom_count.resize(source_cell_count);
  snapshot.source_cell_atom_point.resize(source_cell_count);
  snapshot.owned_cell_atom_count.resize(local_cell_count);
  snapshot.owned_cell_atom_point.resize(local_cell_count);

  snapshot.packet.resize(snapshot.source_atom_count);
  snapshot.sorted_idx.resize(snapshot.source_atom_count);
  snapshot.rx.resize(snapshot.source_atom_count);
  snapshot.ry.resize(snapshot.source_atom_count);
  snapshot.rz.resize(snapshot.source_atom_count);
  snapshot.qc.resize(snapshot.source_atom_count);

  cudaCheck(cudaSetDevice(dev));

  if (snapshot.source_atom_count > 0u) {
    const std::size_t packet_bytes =
        snapshot.source_atom_count * sizeof(atom_packet);
    const std::size_t index_bytes =
        snapshot.source_atom_count * sizeof(unsigned int);
    const std::size_t value_bytes = snapshot.source_atom_count * sizeof(double);

    cudaCheck(cudaMemcpy(static_cast<void *>(snapshot.packet.data()),
                         static_cast<const void *>(
                             workspace.sorted_packets()[dev].d_array().data()),
                         packet_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(
        static_cast<void *>(snapshot.sorted_idx.data()),
        static_cast<const void *>(workspace.sorted_idx()[dev].d_array().data()),
        index_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(
        static_cast<void *>(snapshot.rx.data()),
        static_cast<const void *>(workspace.rx()[dev].d_array().data()),
        value_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(
        static_cast<void *>(snapshot.ry.data()),
        static_cast<const void *>(workspace.ry()[dev].d_array().data()),
        value_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(
        static_cast<void *>(snapshot.rz.data()),
        static_cast<const void *>(workspace.rz()[dev].d_array().data()),
        value_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(
        static_cast<void *>(snapshot.qc.data()),
        static_cast<const void *>(workspace.qc()[dev].d_array().data()),
        value_bytes, cudaMemcpyDeviceToHost));
  }

  if (source_cell_count > 0u) {
    const std::size_t source_cell_bytes =
        source_cell_count * sizeof(unsigned int);

    cudaCheck(cudaMemcpy(
        static_cast<void *>(snapshot.source_cell_atom_count.data()),
        static_cast<const void *>(
            workspace.sr_source_cell_atom_count()[dev].d_array().data()),
        source_cell_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(
        static_cast<void *>(snapshot.source_cell_atom_point.data()),
        static_cast<const void *>(
            workspace.sr_source_cell_atom_point()[dev].d_array().data()),
        source_cell_bytes, cudaMemcpyDeviceToHost));
  }

  if (local_cell_count > 0u) {
    const std::size_t local_cell_bytes =
        local_cell_count * sizeof(unsigned int);

    cudaCheck(
        cudaMemcpy(static_cast<void *>(snapshot.owned_cell_atom_count.data()),
                   static_cast<const void *>(
                       workspace.cell_atom_count()[dev].d_array().data()),
                   local_cell_bytes, cudaMemcpyDeviceToHost));

    cudaCheck(
        cudaMemcpy(static_cast<void *>(snapshot.owned_cell_atom_point.data()),
                   static_cast<const void *>(
                       workspace.cell_atom_point()[dev].d_array().data()),
                   local_cell_bytes, cudaMemcpyDeviceToHost));
  }

  return snapshot;
}

void compare_partition_snapshot(const partition_reference &expected,
                                const partition_snapshot &observed,
                                const std::string &label) {
  require(observed.owned_atom_count == expected.owned_atom_count,
          label + ": owned atom count differs");

  require(observed.source_atom_count == expected.source_atom_count,
          label + ": source atom count differs");

  require(observed.max_atoms_cell == expected.max_atoms_cell,
          label + ": max_atoms_cell differs");

  require(observed.source_cell_atom_count == expected.source_cell_atom_count,
          label + ": source cell atom counts differ");

  require(observed.source_cell_atom_point == expected.source_cell_atom_point,
          label + ": source cell atom points differ");

  require(observed.owned_cell_atom_count == expected.owned_cell_atom_count,
          label + ": owned cell atom counts differ");

  require(observed.owned_cell_atom_point == expected.owned_cell_atom_point,
          label + ": owned cell atom points differ");

  require(observed.packet.size() == expected.packet.size(),
          label + ": packet count differs");

  require(observed.sorted_idx.size() == expected.packet.size(),
          label + ": sorted-index count differs");

  require(observed.rx.size() == expected.packet.size(),
          label + ": rx count differs");
  require(observed.ry.size() == expected.packet.size(),
          label + ": ry count differs");
  require(observed.rz.size() == expected.packet.size(),
          label + ": rz count differs");
  require(observed.qc.size() == expected.packet.size(),
          label + ": qc count differs");

  for (std::size_t source_local_cell = 0;
       source_local_cell < expected.source_cell.size(); source_local_cell++) {
    const unsigned int expected_cell = expected.source_cell[source_local_cell];

    const unsigned int point =
        observed.source_cell_atom_point[source_local_cell];

    const unsigned int count =
        observed.source_cell_atom_count[source_local_cell];

    require(static_cast<std::size_t>(point) + static_cast<std::size_t>(count) <=
                observed.source_atom_count,
            label + ": source cell range is out of bounds");

    unsigned int previous_index = 0u;

    for (unsigned int atom = 0; atom < count; atom++) {
      const atom_packet &packet = observed.packet[point + atom];

      require(packet.cell == expected_cell,
              label + ": packet is stored under the wrong source cell");

      if (atom > 0u) {
        require(packet.i > previous_index,
                label + ": original atom order is not deterministic");
      }

      previous_index = packet.i;
    }
  }

  for (std::size_t atom = 0; atom < expected.packet.size(); atom++) {
    const atom_packet &expected_packet = expected.packet[atom];
    const atom_packet &observed_packet = observed.packet[atom];

    require(packets_equal(observed_packet, expected_packet),
            label + ": sorted packet differs at source atom " +
                std::to_string(atom));

    require(observed.sorted_idx[atom] == expected_packet.i,
            label + ": sorted_idx differs at source atom " +
                std::to_string(atom));

    require(observed.rx[atom] == expected_packet.x,
            label + ": rx differs at source atom " + std::to_string(atom));

    require(observed.ry[atom] == expected_packet.y,
            label + ": ry differs at source atom " + std::to_string(atom));

    require(observed.rz[atom] == expected_packet.z,
            label + ": rz differs at source atom " + std::to_string(atom));

    require(observed.qc[atom] == expected_packet.q,
            label + ": qc differs at source atom " + std::to_string(atom));
  }

  if (!observed.source_cell_atom_count.empty()) {
    const std::size_t last = observed.source_cell_atom_count.size() - 1u;

    const std::size_t terminal =
        static_cast<std::size_t>(observed.source_cell_atom_point[last]) +
        static_cast<std::size_t>(observed.source_cell_atom_count[last]);

    require(terminal == observed.source_atom_count,
            label + ": source metadata does not end at source_atom_count");
  } else {
    require(observed.source_atom_count == 0u,
            label + ": no source cells but source atoms are active");
  }

  if (!observed.owned_cell_atom_count.empty()) {
    const std::size_t last = observed.owned_cell_atom_count.size() - 1u;

    const std::size_t terminal =
        static_cast<std::size_t>(observed.owned_cell_atom_point[last]) +
        static_cast<std::size_t>(observed.owned_cell_atom_count[last]);

    require(terminal == observed.owned_atom_count,
            label + ": owned metadata does not end at owned_atom_count");
  } else {
    require(observed.owned_atom_count == 0u,
            label + ": no owned cells but owned atoms are active");
  }

  return;
}

void validate_owner_coverage(const std::vector<partition_snapshot> &snapshot,
                             const glst_plan &plan, const unsigned int natom,
                             const std::string &label) {
  std::vector<unsigned int> owner_count(natom, 0u);

  for (std::size_t dev = 0; dev < snapshot.size(); dev++) {
    const partition_snapshot &partition = snapshot[dev];

    require(partition.owned_atom_count <= partition.packet.size(),
            label + ": owned count exceeds packet storage");

    for (std::size_t atom = 0; atom < partition.owned_atom_count; atom++) {
      const atom_packet &packet = partition.packet[atom];

      require(packet.i < natom,
              label + ": original atom index is out of range");

      require(plan.cell_partition_idx(packet.cell) == partition.cell_partition,
              label + ": owned atom is on the wrong cell partition");

      owner_count[packet.i]++;
    }

    for (std::size_t atom = partition.owned_atom_count;
         atom < partition.packet.size(); atom++) {
      const atom_packet &packet = partition.packet[atom];

      require(plan.cell_partition_idx(packet.cell) != partition.cell_partition,
              label + ": halo atom is owned by its destination partition");
    }
  }

  for (unsigned int atom = 0; atom < natom; atom++) {
    require(owner_count[atom] == 1u, label + ": atom " + std::to_string(atom) +
                                         " does not have exactly one owner");
  }

  return;
}

storage_snapshot take_storage_snapshot(const glst_workspace &workspace,
                                       const int cuda_count) {
  storage_snapshot snapshot;
  snapshot.device.resize(static_cast<std::size_t>(cuda_count));

  require(workspace.cub_work_buffer().size() ==
              static_cast<std::size_t>(cuda_count),
          "CUB work-buffer count does not match device count");

  require(workspace.cub_work_buffer_size().size() ==
              static_cast<std::size_t>(cuda_count),
          "CUB work-buffer-size count does not match device count");

  for (int dev = 0; dev < cuda_count; dev++) {
    device_storage_snapshot &device = snapshot.device[dev];

    device.atom_capacity = workspace.atom_capacity(dev);
    device.atom_growth_count = workspace.atom_storage_growth_count(dev);
    device.cub_growth_count = workspace.cub_work_buffer_growth_count(dev);
    device.cub_size = workspace.cub_work_buffer_size()[dev];

    device.sorted_packet = static_cast<const void *>(
        workspace.sorted_packets()[dev].d_array().data());
    device.sorted_idx =
        static_cast<const void *>(workspace.sorted_idx()[dev].d_array().data());
    device.rx = static_cast<const void *>(workspace.rx()[dev].d_array().data());
    device.ry = static_cast<const void *>(workspace.ry()[dev].d_array().data());
    device.rz = static_cast<const void *>(workspace.rz()[dev].d_array().data());
    device.qc = static_cast<const void *>(workspace.qc()[dev].d_array().data());

    device.cell_atom_count = static_cast<const void *>(
        workspace.cell_atom_count()[dev].d_array().data());
    device.cell_atom_point = static_cast<const void *>(
        workspace.cell_atom_point()[dev].d_array().data());
    device.source_cell_atom_count = static_cast<const void *>(
        workspace.sr_source_cell_atom_count()[dev].d_array().data());
    device.source_cell_atom_point = static_cast<const void *>(
        workspace.sr_source_cell_atom_point()[dev].d_array().data());

    device.cub = workspace.cub_work_buffer()[dev];

    require(device.sorted_packet != nullptr,
            "Sorted-packet storage pointer is null");
    require(device.sorted_idx != nullptr,
            "Sorted-index storage pointer is null");
    require(device.rx != nullptr, "rx storage pointer is null");
    require(device.ry != nullptr, "ry storage pointer is null");
    require(device.rz != nullptr, "rz storage pointer is null");
    require(device.qc != nullptr, "qc storage pointer is null");
    require(device.cell_atom_count != nullptr,
            "Owned cell-count pointer is null");
    require(device.cell_atom_point != nullptr,
            "Owned cell-point pointer is null");
    require(device.source_cell_atom_count != nullptr,
            "Source cell-count pointer is null");
    require(device.source_cell_atom_point != nullptr,
            "Source cell-point pointer is null");
    require(device.cub != nullptr, "CUB work-buffer pointer is null");
  }

  snapshot.global_sort_key_in =
      static_cast<const void *>(workspace.global_sort_key_in().data());
  snapshot.global_sort_key_out =
      static_cast<const void *>(workspace.global_sort_key_out().data());
  snapshot.global_packet_in =
      static_cast<const void *>(workspace.global_packet_in().data());
  snapshot.global_packet_out =
      static_cast<const void *>(workspace.global_packet_out().data());
  snapshot.global_cell_atom_count =
      static_cast<const void *>(workspace.global_cell_atom_count().data());
  snapshot.global_cell_atom_point =
      static_cast<const void *>(workspace.global_cell_atom_point().data());
  snapshot.global_x_plane_atom_point =
      static_cast<const void *>(workspace.global_x_plane_atom_point().data());
  snapshot.global_max_atoms_cell =
      static_cast<const void *>(workspace.global_max_atoms_cell().data());

  require(snapshot.global_sort_key_in != nullptr,
          "Global input-key pointer is null");
  require(snapshot.global_sort_key_out != nullptr,
          "Global output-key pointer is null");
  require(snapshot.global_packet_in != nullptr,
          "Global input-packet pointer is null");
  require(snapshot.global_packet_out != nullptr,
          "Global output-packet pointer is null");
  require(snapshot.global_cell_atom_count != nullptr,
          "Global cell-count pointer is null");
  require(snapshot.global_cell_atom_point != nullptr,
          "Global cell-point pointer is null");
  require(snapshot.global_x_plane_atom_point != nullptr,
          "Global x-plane-point pointer is null");
  require(snapshot.global_max_atoms_cell != nullptr,
          "Global max-atoms pointer is null");

  return snapshot;
}

void compare_storage_snapshot(const storage_snapshot &expected,
                              const storage_snapshot &observed,
                              const std::string &label) {
  require(observed.device.size() == expected.device.size(),
          label + ": device storage count changed");

  for (std::size_t dev = 0; dev < expected.device.size(); dev++) {
    const device_storage_snapshot &lhs = expected.device[dev];
    const device_storage_snapshot &rhs = observed.device[dev];

    const std::string prefix = label + ", device " + std::to_string(dev) + ": ";

    require(rhs.atom_capacity == lhs.atom_capacity,
            prefix + "atom capacity changed");
    require(rhs.atom_growth_count == lhs.atom_growth_count,
            prefix + "atom growth count changed");
    require(rhs.cub_growth_count == lhs.cub_growth_count,
            prefix + "CUB growth count changed");
    require(rhs.cub_size == lhs.cub_size,
            prefix + "CUB work-buffer size changed");

    require(rhs.sorted_packet == lhs.sorted_packet,
            prefix + "sorted-packet pointer changed");
    require(rhs.sorted_idx == lhs.sorted_idx,
            prefix + "sorted-index pointer changed");
    require(rhs.rx == lhs.rx, prefix + "rx pointer changed");
    require(rhs.ry == lhs.ry, prefix + "ry pointer changed");
    require(rhs.rz == lhs.rz, prefix + "rz pointer changed");
    require(rhs.qc == lhs.qc, prefix + "qc pointer changed");
    require(rhs.cell_atom_count == lhs.cell_atom_count,
            prefix + "owned cell-count pointer changed");
    require(rhs.cell_atom_point == lhs.cell_atom_point,
            prefix + "owned cell-point pointer changed");
    require(rhs.source_cell_atom_count == lhs.source_cell_atom_count,
            prefix + "source cell-count pointer changed");
    require(rhs.source_cell_atom_point == lhs.source_cell_atom_point,
            prefix + "source cell-point pointer changed");
    require(rhs.cub == lhs.cub, prefix + "CUB work-buffer pointer changed");
  }

  require(observed.global_sort_key_in == expected.global_sort_key_in,
          label + ": global input-key pointer changed");
  require(observed.global_sort_key_out == expected.global_sort_key_out,
          label + ": global output-key pointer changed");
  require(observed.global_packet_in == expected.global_packet_in,
          label + ": global input-packet pointer changed");
  require(observed.global_packet_out == expected.global_packet_out,
          label + ": global output-packet pointer changed");
  require(observed.global_cell_atom_count == expected.global_cell_atom_count,
          label + ": global cell-count pointer changed");
  require(observed.global_cell_atom_point == expected.global_cell_atom_point,
          label + ": global cell-point pointer changed");
  require(observed.global_x_plane_atom_point ==
              expected.global_x_plane_atom_point,
          label + ": global x-plane-point pointer changed");
  require(observed.global_max_atoms_cell == expected.global_max_atoms_cell,
          label + ": global max-atoms pointer changed");

  return;
}

} // namespace

int main(void) {
  try {
    int cuda_count = 0;
    cudaCheck(cudaGetDeviceCount(&cuda_count));

    if ((cuda_count != 2) && (cuda_count != 4) && (cuda_count != 8)) {
      std::cout << "assign_atoms_hybrid: SKIPPED "
                << "(expose exactly 2, 4, or 8 GPUs)" << std::endl;
      return EXIT_SUCCESS;
    }

    const std::vector<atom_site> sites = make_atom_sites();
    const std::vector<double> charge = make_charges(sites.size());

    const std::vector<position> uniform_positions =
        make_positions(sites, false);

    const std::vector<position> paired_positions = make_positions(sites, true);

    require(uniform_positions.size() == static_cast<std::size_t>(test_natom),
            "Uniform input atom count is incorrect");

    require(paired_positions.size() == static_cast<std::size_t>(test_natom),
            "Paired input atom count is incorrect");

    cudaCheck(cudaSetDevice(0));

    cuda_container<double> rx(test_natom);
    cuda_container<double> ry(test_natom);
    cuda_container<double> rz(test_natom);
    cuda_container<double> qc(test_natom);

    glst_force force;

    force.set_gpu_layout(static_cast<unsigned int>(cuda_count), 1u);
    force.init(test_natom, tol, box_dim_x, box_dim_y, box_dim_z, rcut);

    require(glst_force_test_access::device_count(force) == cuda_count,
            "Force device count does not match CUDA device count");

    const glst_plan &plan = glst_force_test_access::plan(force);

    require(plan.natom() == test_natom, "Unexpected plan atom count");
    require(plan.ncell_x() == test_ncell_x, "Unexpected plan ncell_x");
    require(plan.ncell_y() == test_ncell_y, "Unexpected plan ncell_y");
    require(plan.ncell_z() == test_ncell_z, "Unexpected plan ncell_z");
    require(plan.ncell() == test_ncell_x * test_ncell_y * test_ncell_z,
            "Unexpected total plan cell count");

    require(plan.cell_partition_count() ==
                static_cast<unsigned int>(cuda_count),
            "Plan cell-partition count does not match visible GPUs");

    require(plan.tile_partition_count() == 1u,
            "assign_atoms_hybrid requires G_tile = 1");

    std::vector<unsigned int> partition_seen(
        static_cast<std::size_t>(cuda_count), 0u);

    for (int dev = 0; dev < cuda_count; dev++) {
      const unsigned int cell_partition =
          glst_force_test_access::cell_partition(force, dev);

      require(cell_partition < static_cast<unsigned int>(cuda_count),
              "Device cell partition is out of range");

      require(glst_force_test_access::tile_partition(force, dev) == 0u,
              "G_tile = 1 device has a nonzero tile partition");

      partition_seen[cell_partition]++;
    }

    for (unsigned int partition = 0;
         partition < static_cast<unsigned int>(cuda_count); partition++) {
      require(partition_seen[partition] == 1u,
              "Cell partition does not have exactly one root device");
    }

    const std::vector<std::vector<atom_packet>> uniform_cell_packets =
        make_cell_packets(plan, uniform_positions, charge);

    const std::vector<std::vector<atom_packet>> paired_cell_packets =
        make_cell_packets(plan, paired_positions, charge);

    const std::vector<partition_reference> uniform_reference =
        make_partition_reference(plan, uniform_cell_packets);

    const std::vector<partition_reference> paired_reference =
        make_partition_reference(plan, paired_cell_packets);

    storage_snapshot stable_storage;

    for (unsigned int repeat = 0; repeat < repeat_count; repeat++) {
      const bool use_paired_positions = ((repeat % 2u) != 0u);

      const std::vector<position> &positions =
          use_paired_positions ? paired_positions : uniform_positions;

      const std::vector<partition_reference> &reference =
          use_paired_positions ? paired_reference : uniform_reference;

      upload_input(rx, ry, rz, qc, positions, charge);

      force.assign_atoms(rx.d_array().data(), ry.d_array().data(),
                         rz.d_array().data(), qc.d_array().data());

      glst_force_test_access::synchronize_compute_streams(force);

      const glst_workspace &workspace =
          glst_force_test_access::workspace(force);

      std::vector<partition_snapshot> observed(
          static_cast<std::size_t>(cuda_count));

      for (int dev = 0; dev < cuda_count; dev++) {
        observed[dev] = read_partition_snapshot(force, dev);

        const unsigned int partition = observed[dev].cell_partition;

        require(partition < reference.size(),
                "Observed partition has no host reference");

        const std::string label =
            "G_cell=" + std::to_string(cuda_count) +
            ", G_tile=1, repeat=" + std::to_string(repeat) + ", distribution=" +
            std::string(use_paired_positions ? "paired" : "uniform") +
            ", device=" + std::to_string(dev) +
            ", partition=" + std::to_string(partition);

        compare_partition_snapshot(reference[partition], observed[dev], label);
      }

      validate_owner_coverage(observed, plan, test_natom,
                              "G_cell=" + std::to_string(cuda_count) +
                                  ", repeat=" + std::to_string(repeat));

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

    std::cout << "assign_atoms_hybrid: PASSED G_cell=" << cuda_count
              << ", G_tile=1, " << repeat_count
              << " alternating deterministic assignments" << std::endl;

    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "assign_atoms_hybrid: FAILED: " << error.what() << std::endl;

    return EXIT_FAILURE;
  }
}
