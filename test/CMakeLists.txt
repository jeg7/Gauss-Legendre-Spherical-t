add_executable(full_example
  ${GLST_SOURCE_DIR}/test/full_example.cu
)
target_link_libraries(full_example
  GLST
  utils
)

add_executable(assign_atoms_hybrid
  ${GLST_SOURCE_DIR}/test/assign_atoms_hybrid.cu
)
target_link_libraries(assign_atoms_hybrid
  GLST
)

add_executable(cell_mapping_hybrid
  ${GLST_SOURCE_DIR}/test/cell_mapping_hybrid.cu
)
target_link_libraries(cell_mapping_hybrid
  GLST
)

add_executable(short_range_halo_plan
  ${GLST_SOURCE_DIR}/test/short_range_halo_plan.cu
)
target_link_libraries(short_range_halo_plan
  GLST
)

add_executable(short_range_halo_exchange
  ${GLST_SOURCE_DIR}/test/short_range_halo_exchange.cu
)
target_link_libraries(short_range_halo_exchange
  GLST
)

add_executable(calc_sf_hybrid
  ${GLST_SOURCE_DIR}/test/calc_sf_hybrid.cu
)
target_link_libraries(calc_sf_hybrid
  GLST
)

add_executable(reduce_tile_partition_ef
  ${GLST_SOURCE_DIR}/test/reduce_tile_partition_ef.cu
)
target_link_libraries(reduce_tile_partition_ef
  GLST
)

# add_executable(setup_plan
#   ${GLST_SOURCE_DIR}/test/setup_plan.cu
# )
# target_link_libraries(setup_plan
#   GLST
# )

# add_test(NAME setup_plan COMMAND setup_plan)
