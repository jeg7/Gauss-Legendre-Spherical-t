add_executable(full_example
  ${GLST_SOURCE_DIR}/test/full_example.cu
)
target_link_libraries(full_example
  GLST
  utils
)

add_executable(setup_plan
  ${GLST_SOURCE_DIR}/test/setup_plan.cu
)
target_link_libraries(setup_plan
  GLST
)

add_test(NAME setup_plan COMMAND setup_plan)
