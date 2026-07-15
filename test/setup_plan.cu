// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include <cuda_utils.hcu>
#include <glst_force.hcu>
#include <glst_plan.hcu>
#include <glst_workspace.hcu>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

struct setup_case {
  const std::string_view name;
  unsigned int natom;
  double tol;
  double box_dim_x;
  double box_dim_y;
  double box_dim_z;
  double rcut;
  unsigned int expected_ncell_x;
  unsigned int expected_ncell_y;
  unsigned int expected_ncell_z;
  unsigned int expected_ngroup;
  unsigned int expected_total_nodes;
  std::vector<unsigned int> expected_group_nodes;
};

inline void check(const bool condition, const std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
  return;
}

inline void check_equal(const unsigned int observed,
                        const unsigned int expected,
                        const std::string_view label) {
  if (observed != expected) {
    std::string message(label);
    message += ": observed " + std::to_string(observed) + ", expected " +
               std::to_string(expected);
    throw std::runtime_error(message);
  }
  return;
}

inline void check_size_equal(const std::size_t observed,
                             const std::size_t expected,
                             const std::string_view label) {
  if (observed != expected) {
    std::string message(label);
    message += ": observed " + std::to_string(observed) + ", expected " +
               std::to_string(expected);
    throw std::runtime_error(message);
  }
  return;
}

void run_plan_workspace_checks(const setup_case &test_case) {
  glst_plan plan;
  plan.init_cells(test_case.natom, test_case.box_dim_x, test_case.box_dim_y,
                  test_case.box_dim_z, test_case.rcut);
  plan.init_alpha_groups(test_case.tol);
  plan.init_cubature(test_case.tol);
  plan.init_tile_schedule();
  plan.validate();

  check_equal(plan.natom(), test_case.natom,
              std::string(test_case.name) + ": natom");
  check_equal(plan.ncell_x(), test_case.expected_ncell_x,
              std::string(test_case.name) + ": ncell_x");
  check_equal(plan.ncell_y(), test_case.expected_ncell_y,
              std::string(test_case.name) + ": ncell_y");
  check_equal(plan.ncell_z(), test_case.expected_ncell_z,
              std::string(test_case.name) + ": ncell_z");
  check_equal(plan.ngroup(), test_case.expected_ngroup,
              std::string(test_case.name) + ": ngroup");

  const unsigned int expected_ncell = test_case.expected_ncell_x *
                                      test_case.expected_ncell_y *
                                      test_case.expected_ncell_z;
  check_equal(plan.ncell(), expected_ncell,
              std::string(test_case.name) + ": ncell");

  if (test_case.expected_total_nodes > 0) {
    check_equal(plan.tot_num_nodes(), test_case.expected_total_nodes,
                std::string(test_case.name) + ": total cubature nodes");
  } else {
    check(plan.tot_num_nodes() > 0,
          std::string(test_case.name) + ": total cubature nodes is zero");
  }

  const auto &points = plan.points();
  const auto &num_nodes = plan.num_nodes();

  check(!points.empty(),
        std::string(test_case.name) + ": cubature point containers are empty");
  check(!num_nodes.empty(), std::string(test_case.name) +
                                ": cubature node-count containers are empty");
  check(points.size() == num_nodes.size(),
        std::string(test_case.name) +
            ": cubature point/node-count container counts differ");

  const std::size_t cub_dev = 0;
  // const auto &points0 = points[cub_dev];
  const auto &num_nodes0 = num_nodes[cub_dev];

  std::size_t group_node_sum = 0;
  for (unsigned int group = 0; group < plan.ngroup(); group++) {
    const unsigned int group_nodes = num_nodes0[group];
    group_node_sum += static_cast<std::size_t>(group_nodes);

    if (!test_case.expected_group_nodes.empty()) {
      check_equal(group_nodes, test_case.expected_group_nodes[group],
                  std::string(test_case.name) + ": cubature nodes in group " +
                      std::to_string(group));
    }
  }

  check_size_equal(
      group_node_sum, static_cast<std::size_t>(plan.tot_num_nodes()),
      std::string(test_case.name) + ": sum of group cubature-node counts");

  std::size_t tile_node_sum = 0;
  std::vector<std::size_t> tile_node_sum_by_group(plan.ngroup(), 0);

  for (unsigned int tile = 0; tile < plan.tile_count(); tile++) {
    const unsigned int tile_group = plan.tile_group(tile);
    const unsigned int tile_point = plan.tile_node_point(tile);
    const unsigned int tile_count = plan.tile_node_count(tile);

    check(tile_group < plan.ngroup(),
          std::string(test_case.name) + ": tile group is out of range");
    check(tile_count > 0,
          std::string(test_case.name) + ": tile has zero nodes");
    check(tile_count <= plan.max_tile_nodes(),
          std::string(test_case.name) + ": tile exceeds max_tile_nodes");
    check(tile_point + tile_count <= plan.tot_num_nodes(),
          std::string(test_case.name) + ": tile range is out of bounds");

    tile_node_sum += static_cast<std::size_t>(tile_count);
    tile_node_sum_by_group[tile_group] += static_cast<std::size_t>(tile_count);
  }

  check_size_equal(
      tile_node_sum, static_cast<std::size_t>(plan.tot_num_nodes()),
      std::string(test_case.name) + ": sum of tile cubature-node counts");

  for (unsigned int group = 0; group < plan.ngroup(); group++) {
    check_size_equal(tile_node_sum_by_group[group],
                     static_cast<std::size_t>(plan.num_nodes()[0][group]),
                     std::string(test_case.name) + ": tile coverage in group " +
                         std::to_string(group));

    const unsigned int expected_tiles =
        (num_nodes0[group] + plan.max_tile_nodes() - 1) / plan.max_tile_nodes();
    check_equal(plan.tiles_in_group(group), expected_tiles,
                std::string(test_case.name) + ": tiles in group " +
                    std::to_string(group));
  }

  glst_workspace workspace(plan);

  const std::size_t expected_tile_buffer_capacity =
      static_cast<std::size_t>(plan.ncell()) *
      static_cast<std::size_t>(plan.max_tile_nodes());
  const std::size_t full_node_buffer_capacity =
      static_cast<std::size_t>(plan.ncell()) *
      static_cast<std::size_t>(plan.tot_num_nodes());

  check_size_equal(workspace.atom_capacity(),
                   static_cast<std::size_t>(plan.natom()),
                   std::string(test_case.name) + ": workspace atom capacity");
  check_size_equal(workspace.cell_capacity(),
                   static_cast<std::size_t>(plan.ncell()),
                   std::string(test_case.name) + ": workspace cell capacity");
  check_size_equal(workspace.tile_node_capacity(),
                   static_cast<std::size_t>(plan.max_tile_nodes()),
                   std::string(test_case.name) +
                       ": workspace tile node capacity");
  check_size_equal(
      workspace.tile_buffer_capacity(), expected_tile_buffer_capacity,
      std::string(test_case.name) + ": workspace tile buffer capacity");

  check(workspace.sf_re().size() == 1,
        std::string(test_case.name) + ": sf_re should be vector-of-one");
  check(workspace.sf_im().size() == 1,
        std::string(test_case.name) + ": sf_im should be vector-of-one");
  check(workspace.rmt_sum_re().size() == 1,
        std::string(test_case.name) + ": rmt_sum_re should be vector-of-one");
  check(workspace.rmt_sum_im().size() == 1,
        std::string(test_case.name) + ": rmt_sum_im should be vector-of-one");

  check_size_equal(workspace.sf_re()[0].size(), expected_tile_buffer_capacity,
                   std::string(test_case.name) + ": sf_re capacity");
  check_size_equal(workspace.sf_im()[0].size(), expected_tile_buffer_capacity,
                   std::string(test_case.name) + ": sf_im capacity");
  check_size_equal(workspace.rmt_sum_re()[0].size(),
                   expected_tile_buffer_capacity,
                   std::string(test_case.name) + ": rmt_sum_re capacity");
  check_size_equal(workspace.rmt_sum_im()[0].size(),
                   expected_tile_buffer_capacity,
                   std::string(test_case.name) + ": rmt_sum_im capacity");

  if (plan.tot_num_nodes() > plan.max_tile_nodes()) {
    check(workspace.tile_buffer_capacity() < full_node_buffer_capacity,
          std::string(test_case.name) +
              ": workspace tile buffer is not smaller than full-node buffer");
  }

  std::cout << "PASS setup metadata: " << test_case.name << std::endl;

  return;
}

void run_force_setup_smoke(const setup_case &test_case) {
  glst_force force(test_case.natom, test_case.tol, test_case.box_dim_x,
                   test_case.box_dim_y, test_case.box_dim_z, test_case.rcut);

  std::cout << "PASS glst_force setup: " << test_case.name << std::endl;

  return;
}

void run_force_setup_smoke(const setup_case &test_case,
                           const unsigned int cell_partitions,
                           const unsigned int tile_partitions) {
  glst_force force;
  force.set_gpu_layout(cell_partitions, tile_partitions);
  force.init(test_case.natom, test_case.tol, test_case.box_dim_x,
             test_case.box_dim_y, test_case.box_dim_z, test_case.rcut);

  std::cout << "PASS glst_force setup: " << test_case.name << " layout "
            << cell_partitions << " x " << tile_partitions << std::endl;

  return;
}

void run_invalid_layout_check(const setup_case &test_case,
                              const unsigned int cell_partitions,
                              const unsigned int tile_partitions) {
  bool caught = false;

  try {
    glst_force force;
    force.set_gpu_layout(cell_partitions, tile_partitions);
    force.init(test_case.natom, test_case.tol, test_case.box_dim_x,
               test_case.box_dim_y, test_case.box_dim_z, test_case.rcut);
  } catch (const std::runtime_error &) {
    caught = true;
  }

  check(caught,
        std::string(test_case.name) + ": invalid GPU layout did not throw");

  return;
}

void run_tile_partition_checks(const setup_case &test_case,
                               const unsigned int tile_partitions) {
  glst_plan plan;
  plan.init_cells(test_case.natom, test_case.box_dim_x, test_case.box_dim_y,
                  test_case.box_dim_z, test_case.rcut);
  plan.init_alpha_groups(test_case.tol);
  plan.init_cubature(test_case.tol);
  plan.init_tile_schedule();
  plan.init_tile_partitions(tile_partitions);
  plan.validate();

  check_equal(plan.tile_partition_count(), tile_partitions,
              std::string(test_case.name) + ": tile partition count");

  std::vector<unsigned int> tile_visit_count(plan.tile_count(), 0);
  std::size_t total_partition_nodes = 0;

  for (unsigned int partition = 0; partition < tile_partitions; partition++) {
    const std::vector<unsigned int> &tiles = plan.partition_tile_idx(partition);

    std::size_t observed_partition_nodes = 0;

    for (std::size_t i = 0; i < tiles.size(); i++) {
      const unsigned int tile = tiles[i];

      check(tile < plan.tile_count(),
            std::string(test_case.name) + ": partition tile is out of range");
      check_equal(plan.tile_partition_idx(tile), partition,
                  std::string(test_case.name) +
                      ": tile partition index mismatch");
      check_equal(tile % tile_partitions, partition,
                  std::string(test_case.name) +
                      ": tile modulo partition mismatch");

      if (i > 0) {
        check(tiles[i] > tiles[i - 1],
              std::string(test_case.name) +
                  ": partition tile order is not deterministic");
      }

      observed_partition_nodes +=
          static_cast<std::size_t>(plan.tile_node_count(tile));
      tile_visit_count[tile]++;
    }

    check_size_equal(
        observed_partition_nodes,
        static_cast<std::size_t>(plan.partition_tile_node_count(partition)),
        std::string(test_case.name) + ": partition node count");
    total_partition_nodes += observed_partition_nodes;
  }

  check_size_equal(
      total_partition_nodes, static_cast<std::size_t>(plan.tot_num_nodes()),
      std::string(test_case.name) + ": sum of tile partition node counts");

  for (unsigned int tile = 0; tile < plan.tile_count(); tile++) {
    check_equal(tile_visit_count[tile], 1u,
                std::string(test_case.name) + ": tile ownership visit count");
  }

  if (tile_partitions == 1) {
    const std::vector<unsigned int> &tiles = plan.partition_tile_idx(0);
    check_size_equal(tiles.size(), static_cast<std::size_t>(plan.tile_count()),
                     std::string(test_case.name) +
                         ": G_tile=1 partition tile count");

    for (unsigned int tile = 0; tile < plan.tile_count(); tile++) {
      check_equal(tiles[tile], tile,
                  std::string(test_case.name) +
                      ": G_tile=1 tile order mismatch");
    }
  }

  std::cout << "PASS tile partition metadata: " << test_case.name << " G_tile "
            << tile_partitions << std::endl;

  return;
}

int main(void) {
  int device_count = 0;
  cudaCheck(cudaGetDeviceCount(&device_count));

  try {
    const std::vector<setup_case> test_cases = {
        {"small_2959", 2959u, 1.0e-6, 32.0, 32.0, 32.0, 12.0, 3u, 3u, 3u, 1u,
         0u, std::vector<unsigned int>()},
        {"sample_197159", 197159u, 1.0e-6, 128.0, 128.0, 128.0, 12.0, 11u, 11u,
         11u, 4u, 95148u,
         std::vector<unsigned int>{52173u, 25569u, 12894u, 4512u}}};

    for (std::size_t i = 0; i < test_cases.size(); i++) {
      run_plan_workspace_checks(test_cases[i]);
      run_force_setup_smoke(test_cases[i]);

      if (device_count > 1) {
        run_force_setup_smoke(test_cases[i],
                              static_cast<unsigned int>(device_count), 1u);

        run_force_setup_smoke(test_cases[i], 1u,
                              static_cast<unsigned int>(device_count));
      }

      if (device_count == 4)
        run_force_setup_smoke(test_cases[i], 2u, 2u);

      if (device_count == 8) {
        run_force_setup_smoke(test_cases[i], 4u, 2u);
        run_force_setup_smoke(test_cases[i], 2u, 4u);
      }

      run_invalid_layout_check(test_cases[i],
                               static_cast<unsigned int>(device_count + 1), 1u);

      run_tile_partition_checks(test_cases[i], 1u);
      run_tile_partition_checks(test_cases[i], 2u);
      run_tile_partition_checks(test_cases[i], 4u);
    }
  } catch (const std::exception &e) {
    std::cerr << "FAIL setup_plan: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
