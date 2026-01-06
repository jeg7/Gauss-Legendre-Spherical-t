add_executable(full_example
  ${GLST_SOURCE_DIR}/test/full_example.cu
)
target_link_libraries(full_example
  GLST
  utils
)
