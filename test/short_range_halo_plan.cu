// BEGINLICENSE
//
// This file is part of Gauss-Legendre-Spherical-t, which is distributed under
// the BSD 3-clause license, as described in the LICENSE file in the top level
// directory of this project.
//
// Author: James E. Gonzales II
//
// ENDLICENSE

#include <glst_plan.hcu>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check_plane_x(const glst_plan &plan,
                          const std::vector<unsigned int> &cells,
                          const unsigned int expected_x) {
  for (std::size_t i = 0; i < cells.size(); i++) {
    unsigned int x = 0, y = 0, z = 0;
    plan.global_cell_coords(x, y, z, cells[i]);

    if (x != expected_x)
      throw std::runtime_error("halo plane x-coordinate mismatch");
  }

  return;
}

static void check_halo_case(const unsigned int natom, const double box,
                            const double rcut,
                            const unsigned int cell_partitions) {
  constexpr double tol = 1.0e-6;

  glst_plan plan;
  plan.init_cells(natom, box, box, box, rcut);
  plan.init_alpha_groups(tol);
  plan.init_cubature(tol);
  plan.init_tile_schedule(2048);
  plan.init_cell_partitions(cell_partitions);
  plan.validate();

  const unsigned int yz_count = plan.ncell_y() * plan.ncell_z();

  for (unsigned int part = 0; part < plan.cell_partition_count(); part++) {
    const unsigned int x_point = plan.cell_partition_x_point()[part];
    const unsigned int x_count = plan.cell_partition_x_count()[part];
    const unsigned int x_end = x_point + x_count;

    const std::vector<unsigned int> &owned_cells =
        plan.partition_cell_idx(part);
    const std::vector<unsigned int> &left_cells =
        plan.partition_left_halo_cell_idx(part);
    const std::vector<unsigned int> &right_cells =
        plan.partition_right_halo_cell_idx(part);
    const std::vector<unsigned int> &halo_cells =
        plan.partition_halo_cell_idx(part);
    const std::vector<unsigned int> &source_cells =
        plan.partition_sr_source_cell_idx(part);

    const std::size_t expected_left = ((x_count > 0) && (x_point > 0))
                                          ? static_cast<std::size_t>(yz_count)
                                          : 0;
    const std::size_t expected_right =
        ((x_count > 0) && (x_end < plan.ncell_x()))
            ? static_cast<std::size_t>(yz_count)
            : 0;

    if (left_cells.size() != expected_left)
      throw std::runtime_error("left halo count mismatch");

    if (right_cells.size() != expected_right)
      throw std::runtime_error("right halo count mismatch");

    if (halo_cells.size() != left_cells.size() + right_cells.size())
      throw std::runtime_error("combined halo count mismatch");

    if (source_cells.size() != owned_cells.size() + halo_cells.size())
      throw std::runtime_error("source cell count mismatch");

    for (std::size_t i = 0; i < owned_cells.size(); i++) {
      if (source_cells[i] != owned_cells[i])
        throw std::runtime_error("source list does not start with owned cells");
    }

    for (std::size_t i = 0; i < halo_cells.size(); i++) {
      if (source_cells[owned_cells.size() + i] != halo_cells[i])
        throw std::runtime_error("source list does not append halo cells");
    }

    if (!left_cells.empty())
      check_plane_x(plan, left_cells, x_point - 1);

    if (!right_cells.empty())
      check_plane_x(plan, right_cells, x_end);

    if (cell_partitions == 1) {
      if (!halo_cells.empty()) {
        throw std::runtime_error(
            "single partition unexpectedly has halo cells");
      }
    }

    if (x_count == 0) {
      if (!owned_cells.empty() || !halo_cells.empty() || !source_cells.empty())
        throw std::runtime_error("zero-width partition has nonempty metadata");
    }
  }

  return;
}

int main(void) {
  try {
    check_halo_case(2959, 32.0, 12.0, 1);
    check_halo_case(2959, 32.0, 12.0, 2);
    check_halo_case(2959, 32.0, 12.0, 4);

    check_halo_case(197159, 128.0, 12.0, 1);
    check_halo_case(197159, 128.0, 12.0, 2);
    check_halo_case(197159, 128.0, 12.0, 4);
    check_halo_case(197159, 128.0, 12.0, 8);
  } catch (const std::exception &e) {
    std::cerr << "FAIL short_range_halo_plan: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "PASS short_range_halo_plan" << std::endl;

  return EXIT_SUCCESS;
}
