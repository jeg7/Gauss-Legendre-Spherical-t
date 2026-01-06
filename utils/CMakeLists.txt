add_library(utils
  ${GLST_SOURCE_DIR}/utils/coulomb.cu
  ${GLST_SOURCE_DIR}/utils/io.cpp
  ${GLST_SOURCE_DIR}/utils/utils.cpp
)
set_target_properties(utils PROPERTIES
  CUDA_ARCHITECTURES "60;70;80"
)
target_include_directories(utils PUBLIC ${GLST_SOURCE_DIR}/utils)
