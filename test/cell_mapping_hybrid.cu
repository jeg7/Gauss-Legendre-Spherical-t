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

static void check_mapping_case(const unsigned int natom, const double box,
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

  const unsigned int ncell_y = plan.ncell_y();
  const unsigned int ncell_z = plan.ncell_z();
  const unsigned int yz_count = ncell_y * ncell_z;

  for (unsigned int part = 0; part < plan.cell_partition_count(); part++) {
    const unsigned int x_point = plan.cell_partition_x_point()[part];
    const unsigned int x_count = plan.cell_partition_x_count()[part];
    const unsigned int expected_local_cells = x_count * yz_count;

    if (plan.local_cell_count(part) != expected_local_cells)
      throw std::runtime_error("local cell count mismatch");

    if (plan.first_global_cell(part) != x_point * yz_count)
      throw std::runtime_error("first global cell mismatch");

    const std::vector<unsigned int> &cells = plan.partition_cell_idx(part);

    if (cells.size() != static_cast<std::size_t>(expected_local_cells))
      throw std::runtime_error("partition cell list size mismatch");

    for (unsigned int local_cell = 0; local_cell < expected_local_cells;
         local_cell++) {
      const unsigned int global_cell =
          plan.global_cell_from_local_cell(part, local_cell);

      if (cells[local_cell] != global_cell)
        throw std::runtime_error("partition cell list is not local-order");

      const unsigned int observed_local_cell =
          plan.local_cell_from_global_cell(part, global_cell);

      if (observed_local_cell != local_cell)
        throw std::runtime_error("local/global round-trip mismatch");

      unsigned int x = 0, y = 0, z = 0;
      plan.global_cell_coords(x, y, z, global_cell);

      if ((x < x_point) || (x >= x_point + x_count))
        throw std::runtime_error("global cell x-coordinate outside partition");

      const unsigned int rebuilt_cell = (x * ncell_y + y) * ncell_z + z;
      if (rebuilt_cell != global_cell)
        throw std::runtime_error("coordinate rebuild mismatch");

      unsigned int lx = 0, ly = 0, lz = 0;
      plan.local_cell_coords(lx, ly, lz, part, local_cell);

      if ((lx != x) || (ly != y) || (lz != z))
        throw std::runtime_error("local cell coordinate mismatch");

      if ((cell_partitions == 1) && (global_cell != local_cell))
        throw std::runtime_error("single-partition mapping is not identity");
    }
  }
}

int main(void) {
  try {
    check_mapping_case(2959, 32.0, 12.0, 1);
    check_mapping_case(2959, 32.0, 12.0, 2);
    check_mapping_case(2959, 32.0, 12.0, 4);
    check_mapping_case(197159, 128.0, 12.0, 1);
    check_mapping_case(197159, 128.0, 12.0, 2);
    check_mapping_case(197159, 128.0, 12.0, 4);
  } catch (const std::exception &e) {
    std::cerr << "FAIL cell_mapping_hybrid: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "PASS cell_mapping_hybrid" << std::endl;

  return EXIT_SUCCESS;
}
